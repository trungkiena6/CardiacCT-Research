"""Build an isolated nnU-Net v2 dataset view for ONE official ImageCAS split.

Stages (run in this order):
  train-only : imagesTr/labelsTr = the 700 train cases only, dataset.json numTraining=700
               (fingerprint + planning must run on this state so val/test never enter CT statistics)
  add-val    : add the 50 val cases to imagesTr/labelsTr; set numTraining=750 in the raw dataset.json
               and, if present, in nnUNet_preprocessed/<dataset>/dataset.json (both files, same content)
  splits     : write nnUNet_preprocessed/<dataset>/splits_final.json = [{"train": 700 ids, "val": 50 ids}]
  test       : imagesTs = 250 test images; test labels go to <work>/test_labels_s<i>/ (scorer only)
  val-dir    : <work>/val_s<i>/ = the 50 val images for nnUNetv2_predict
Images are copied unchanged (int16). Labels are re-written as uint8 {0,1} with the IMAGE header/affine
after verifying image/label shape and affine agreement within 1e-3; labels are not resampled.
Every stage writes <work>/dataset_build_s<i>.json with the case lists for provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


def case_name(cid: int) -> str:
    return f"Case{cid:04d}"


def convert_label(src_label: Path, src_image: Path, dst: Path, multiclass: bool = False) -> dict:
    img = nib.load(str(src_image))
    lab = nib.load(str(src_label))
    if img.shape != lab.shape or not np.allclose(img.affine, lab.affine, atol=1e-3, rtol=0):
        raise ValueError(
            f"{src_label}: image/label geometry differs; conversion does not resample labels"
        )
    data = np.asanyarray(lab.dataobj)
    vals = np.unique(data)
    if multiclass:
        if not set(vals.tolist()) <= {0, 1, 2}:
            raise ValueError(f"{src_label}: label values {vals[:10]} not in {{0,1,2}}")
        out = np.rint(data).astype(np.uint8)
    else:
        if not set(vals.tolist()) <= {0.0, 1.0}:
            raise ValueError(f"{src_label}: label values {vals[:10]} not in {{0,1}}")
        out = (data > 0.5).astype(np.uint8)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    if dst.exists():
        existing = nib.load(str(dst))
        if (
            existing.shape != out.shape
            or not np.allclose(existing.affine, img.affine, atol=1e-3, rtol=0)
            or existing.get_data_dtype() != np.dtype(np.uint8)
            or not np.array_equal(np.asanyarray(existing.dataobj), out)
        ):
            raise FileExistsError(f"{dst}: existing label differs; use a fresh dataset directory")
        return dict(
            src=str(src_label), dst=str(dst), fg_voxels=int(out.sum()), shape=list(out.shape)
        )
    nib.save(nib.Nifti1Image(out, img.affine, hdr), str(dst))
    return dict(src=str(src_label), dst=str(dst), fg_voxels=int(out.sum()), shape=list(out.shape))


def copy_image(src: Path, dst: Path) -> dict:
    if dst.exists():
        same = os.path.samefile(src, dst)
        if not same and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            with src.open("rb") as source, dst.open("rb") as target:
                same = True
                while chunk := source.read(1024 * 1024):
                    if chunk != target.read(len(chunk)):
                        same = False
                        break
        if not same:
            raise FileExistsError(f"{dst}: existing image differs; use a fresh dataset directory")
    else:
        try:  # same filesystem: a hard link costs no disk (content is identical)
            os.link(src, dst)
        except OSError:
            with src.open("rb") as source, dst.open("xb") as target:
                shutil.copyfileobj(source, target)
    return dict(src=str(src), dst=str(dst), bytes=dst.stat().st_size)


def job(kind: str, src_a: str, src_b: str, dst: str):
    if kind == "image":
        return copy_image(Path(src_a), Path(dst))
    return convert_label(Path(src_a), Path(src_b), Path(dst), multiclass=(kind == "label_lr"))


def write_dataset_json(path: Path, num_training: int, multiclass: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT"},
                "labels": (
                    {"background": 0, "left_coronary": 1, "right_coronary": 2}
                    if multiclass
                    else {"background": 0, "coronary_artery": 1}
                ),
                "numTraining": num_training,
                "file_ending": ".nii.gz",
                "name": path.parent.name,
                "description": "ImageCAS v2-latest official split view",
                "licence": "CC BY-NC 4.0 (ImageCAS, Zeng et al. 2023)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def prepared_case_ids(raw: Path) -> set[str]:
    images = list((raw / "imagesTr").glob("*.nii.gz"))
    if any(not path.name.endswith("_0000.nii.gz") for path in images):
        raise ValueError("Prepared CT images must have one _0000.nii.gz channel per case")
    image_ids = {path.name[:-12] for path in images}
    label_ids = {path.name[:-7] for path in (raw / "labelsTr").glob("*.nii.gz")}
    if image_ids != label_ids:
        raise ValueError(
            "Prepared image and label case sets differ; finish dataset preparation first"
        )
    return image_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--split", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--nnunet-raw", type=Path, required=True)
    ap.add_argument("--nnunet-preprocessed", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument(
        "--stage", required=True, choices=["train-only", "add-val", "splits", "test", "val-dir"]
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--excluded",
        type=Path,
        default=None,
        help="excluded_cases.csv from the audit (must be empty)",
    )
    ap.add_argument(
        "--label-cases-dir",
        type=Path,
        default=None,
        help="alternative label source <dir>/<group>/<id>.label.nii.gz (uint8 0/1/2)",
    )
    ap.add_argument(
        "--dataset-name",
        default=None,
        help="override DatasetNNN_<name> suffix (default ImageCAS_S<split>)",
    )
    ap.add_argument(
        "--skip-missing-labels",
        action="store_true",
        help="skip cases whose alternative label is missing (records them)",
    )
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(a.manifest.open(encoding="utf-8")))
    col = f"split_{a.split}"
    ids = {
        k: sorted(int(r["case_id"]) for r in rows if r[col] == k) for k in ("train", "val", "test")
    }
    assert (len(ids["train"]), len(ids["val"]), len(ids["test"])) == (700, 50, 250), {
        k: len(v) for k, v in ids.items()
    }
    assert (
        not (set(ids["train"]) & set(ids["val"]))
        and not (set(ids["train"]) & set(ids["test"]))
        and not (set(ids["val"]) & set(ids["test"]))
    )
    if a.excluded and a.excluded.exists():
        ex = list(csv.DictReader(a.excluded.open(encoding="utf-8")))
        if ex:
            raise SystemExit(
                f"{len(ex)} excluded cases in {a.excluded}; resolve before building the dataset"
            )
    src = {int(r["case_id"]): r for r in rows}
    ds_name = f"Dataset{a.dataset_id:03d}_{a.dataset_name or f'ImageCAS_S{a.split}'}"
    label_kind = "label_lr" if a.label_cases_dir else "label"
    raw = a.nnunet_raw / ds_name
    pre = a.nnunet_preprocessed / ds_name
    build_log = a.work / f"dataset_build_s{a.split}.json"
    log = json.loads(build_log.read_text(encoding="utf-8")) if build_log.exists() else {}

    def src_paths(cid):
        r = src[cid]
        lab_dir = a.label_cases_dir if a.label_cases_dir else a.cases_dir
        return (
            a.cases_dir / r["archive_group"] / f"{cid}.img.nii.gz",
            lab_dir / r["archive_group"] / f"{cid}.label.nii.gz",
        )

    def run_jobs(jobs):
        if not jobs:
            return []
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(job, *zip(*jobs)))
        return res

    if a.stage in ("train-only", "add-val"):
        subset = ids["train"] if a.stage == "train-only" else ids["val"]
        (raw / "imagesTr").mkdir(parents=True, exist_ok=True)
        (raw / "labelsTr").mkdir(parents=True, exist_ok=True)
        jobs, skipped = [], []
        for cid in subset:
            ip, lp = src_paths(cid)
            if a.skip_missing_labels and not lp.exists():
                skipped.append(cid)
                continue
            jobs.append(
                ("image", str(ip), "", str(raw / "imagesTr" / f"{case_name(cid)}_0000.nii.gz"))
            )
            jobs.append(
                (label_kind, str(lp), str(ip), str(raw / "labelsTr" / f"{case_name(cid)}.nii.gz"))
            )
        res = run_jobs(jobs)
        have = prepared_case_ids(raw)
        allowed = {
            case_name(cid)
            for cid in (ids["train"] if a.stage == "train-only" else ids["train"] + ids["val"])
        }
        if not have <= allowed or (not a.skip_missing_labels and have != allowed):
            raise ValueError("Prepared cases do not match the requested train/validation scope")
        expected = len(have)
        write_dataset_json(raw / "dataset.json", expected, multiclass=bool(a.label_cases_dir))
        if skipped:
            log[f"{a.stage}_skipped_missing_label"] = skipped
            print(f"skipped {len(skipped)} cases without alternative label: {skipped[:10]}...")
        if a.stage == "add-val" and (pre / "dataset.json").exists():
            write_dataset_json(pre / "dataset.json", expected, multiclass=bool(a.label_cases_dir))
            print(f"updated {pre / 'dataset.json'} to numTraining={expected}")
        kept_subset = [cid for cid in subset if case_name(cid) in have]
        log[a.stage] = dict(
            cases=kept_subset,
            requested_cases=subset,
            skipped_missing_label=skipped,
            n_files=len(res),
            numTraining=expected,
        )
        print(f"{a.stage}: {len(kept_subset)} prepared cases -> {raw}, numTraining={expected}")

    elif a.stage == "splits":
        have = prepared_case_ids(raw)
        allowed = {case_name(cid) for cid in ids["train"] + ids["val"]}
        if not have <= allowed or (not a.skip_missing_labels and have != allowed):
            raise ValueError(
                "Official split needs exactly 750 prepared image/label pairs; no split file written"
            )
        kept = {
            part: [cid for cid in ids[part] if case_name(cid) in have] for part in ("train", "val")
        }
        if not kept["train"] or not kept["val"]:
            raise ValueError("Training and validation must both contain prepared cases")
        splits = [{part: [case_name(cid) for cid in kept[part]] for part in ("train", "val")}]
        target = pre / "splits_final.json"
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != splits:
                raise FileExistsError(
                    f"{target}: existing split differs; use a fresh dataset directory"
                )
        else:
            pre.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                json.dump(splits, handle, indent=1)
        log["splits"] = dict(
            train=kept["train"],
            val=kept["val"],
            skipped={
                part: [cid for cid in ids[part] if cid not in kept[part]]
                for part in ("train", "val")
            },
            protocol_complete=have == allowed,
            path=str(target),
        )
        print(f"splits_final.json written: train={len(kept['train'])} val={len(kept['val'])}")

    elif a.stage == "test":
        (raw / "imagesTs").mkdir(parents=True, exist_ok=True)
        lab_dir = a.work / f"test_labels_s{a.split}"
        lab_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for cid in ids["test"]:
            ip, lp = src_paths(cid)
            jobs.append(
                ("image", str(ip), "", str(raw / "imagesTs" / f"{case_name(cid)}_0000.nii.gz"))
            )
            jobs.append(("label", str(lp), str(ip), str(lab_dir / f"{case_name(cid)}.nii.gz")))
        run_jobs(jobs)
        log["test"] = dict(cases=ids["test"], images=str(raw / "imagesTs"), labels=str(lab_dir))
        print(f"test: {len(ids['test'])} images -> {raw / 'imagesTs'}; labels -> {lab_dir}")

    elif a.stage == "val-dir":
        vd = a.work / f"val_s{a.split}"
        vd.mkdir(parents=True, exist_ok=True)
        lab_dir = a.work / f"val_labels_s{a.split}"
        lab_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for cid in ids["val"]:
            ip, lp = src_paths(cid)
            jobs.append(("image", str(ip), "", str(vd / f"{case_name(cid)}_0000.nii.gz")))
            jobs.append(("label", str(lp), str(ip), str(lab_dir / f"{case_name(cid)}.nii.gz")))
        run_jobs(jobs)
        log["val-dir"] = dict(cases=ids["val"], images=str(vd), labels=str(lab_dir))
        print(f"val-dir: {len(ids['val'])} images -> {vd}; labels -> {lab_dir}")

    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
