"""Second ImageCAS Split-1 test access.

select : predict the 50 validation cases with every available candidate, apply the
         selection rule, write test_access_2_choice.json (no test data is read).
test   : refuse unless the choice file exists and no previous test-2 output exists;
         predict the 250 test cases with the chosen configuration, post-process, score.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from prediction_cache import (
    ensemble_request,
    inference_request,
    prediction_command,
    record_cache,
    reuse_cache,
)
from score_segmentation import metric_arrays, read_evaluation_rows, require_new_files

W = Path(os.environ.get("COROSEG_WORK", "/workspace/coroseg-work"))
RES = W / "nnunet_results" / "Dataset901_ImageCAS_S1"
EVAL = Path(__file__).resolve().parents[1] / "src/coronary_tree/evaluate/segmentation.py"
CHOICE = W / "test_access_2_choice.json"
T_GRID = [0, 10, 25, 50, 100, 200]

MODELS = {
    "unet_ft500": ("nnUNetTrainer_ft500_NM", "nnUNetPlans"),
    "resenc_m": ("nnUNetTrainer_250epochs_NoMirroring", "nnUNetResEncUNetMPlans"),
    "resenc_l": ("nnUNetTrainer_250epochs_NoMirroring", "nnUNetResEncUNetL40GPlans"),
}
CANDIDATES = {
    "unet_ft500": ["unet_ft500"],
    "unet_ft500+resenc_m": ["unet_ft500", "resenc_m"],
    "resenc_l": ["resenc_l"],
    "unet_ft500+resenc_l": ["unet_ft500", "resenc_l"],
}


def available(model: str) -> bool:
    tr, pl = MODELS[model]
    return (RES / f"{tr}__{pl}__3d_fullres" / "fold_0" / "checkpoint_final.pth").is_file()


def predict(model: str, images: Path, out: Path) -> None:
    tr, pl = MODELS[model]
    fold = RES / f"{tr}__{pl}__3d_fullres" / "fold_0"
    ids, request = inference_request(images, fold, model, tr, pl, "checkpoint_final.pth", 0.5)
    if reuse_cache(out, request, ids, probabilities=True):
        return
    subprocess.run(
        prediction_command(
            request,
            [
                "-i",
                str(images),
                "-o",
                str(out),
                "-d",
                "901",
                "-c",
                "3d_fullres",
                "-f",
                "0",
                "-tr",
                tr,
                "-p",
                pl,
                "-chk",
                "checkpoint_final.pth",
                "-step_size",
                "0.5",
                "--disable_tta",
                "--save_probabilities",
                "-npp",
                "4",
                "-nps",
                "4",
            ],
        ),
        check=True,
    )
    record_cache(out, request, ids, probabilities=True)


def build(members: list[str], pred_root: Path, out: Path, t_mm3: float) -> None:
    directories = {member: pred_root / member for member in members}
    ids, request = ensemble_request(
        directories, {member: 1.0 for member in members}, t_mm3, Path(__file__)
    )
    if reuse_cache(out, request, ids, probabilities=False):
        return
    out.mkdir(parents=True, exist_ok=True)
    first = pred_root / members[0]
    for case in ids:
        ref = first / f"{case}.nii.gz"
        probabilities = []
        for member in members:
            with np.load(pred_root / member / f"{case}.npz") as data:
                probabilities.append(data["probabilities"][1])
        prob = np.mean(probabilities, axis=0)
        mask = prob >= 0.5
        img = sitk.ReadImage(str(ref))
        if t_mm3 > 0 and mask.any():
            vox = float(np.prod(img.GetSpacing()))
            lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
            sizes = ndi.sum(mask, lab, range(1, n + 1)) * vox
            mask = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= t_mm3])
        o = sitk.GetImageFromArray(mask.astype(np.uint8))
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out / ref.name))
    record_cache(out, request, ids, probabilities=False)


def score(pred: Path, gt: Path, csv_out: Path) -> dict:
    require_new_files(csv_out)
    subprocess.run(
        [
            sys.executable,
            str(EVAL),
            "--gt-dir",
            str(gt),
            "--pred-dir",
            str(pred),
            "--output",
            str(csv_out),
            "--workers",
            "32",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    expected = {path.name[:-7] for path in gt.glob("*.nii.gz")}
    rows = read_evaluation_rows(csv_out, expected)
    return {
        "n": len(rows),
        **{key: float(values.mean()) for key, values in metric_arrays(rows).items()},
    }


def select() -> None:
    require_new_files(CHOICE)
    val_in, val_gt, root = W / "val_in", W / "val_labels", W / "testaccess2" / "val"
    results = {}
    for name, members in CANDIDATES.items():
        if not all(available(m) for m in members):
            continue
        for m in members:
            predict(m, val_in, root / m)
        build(members, root, root / f"cand_{name}", 0)
        results[name] = score(root / f"cand_{name}", val_gt, root / f"cand_{name}.csv")
        print(name, results[name], flush=True)
    if not results:
        raise RuntimeError("No candidate has all required checkpoints")
    best = max(results, key=lambda k: results[k]["dice"])
    base = results[best]
    pp = {0: base}
    for t in T_GRID[1:]:
        d = root / f"pp_{best}_T{t}"
        build(CANDIDATES[best], root, d, t)
        pp[t] = score(d, val_gt, root / f"pp_{best}_T{t}.csv")
        print("T", t, pp[t], flush=True)
    ok = [
        t
        for t in T_GRID
        if base["dice"] - pp[t]["dice"] <= 0.002 and base["cldice"] - pp[t]["cldice"] <= 0.005
    ]
    t_star = min(ok, key=lambda t: pp[t]["hd95_mm"]) if ok else 0
    choice = {
        "configuration": best,
        "members": CANDIDATES[best],
        "T_mm3": t_star,
        "val_candidates": results,
        "val_postprocessing": {str(k): v for k, v in pp.items()},
        "val_chosen": pp[t_star],
    }
    with CHOICE.open("x", encoding="utf-8") as handle:
        json.dump(choice, handle, indent=2, allow_nan=False)
    print("CHOICE", best, "T", t_star, pp[t_star], flush=True)


def test() -> None:
    if not CHOICE.is_file():
        raise SystemExit("no choice file: run select first")
    out_root = W / "testaccess2" / "test"
    result_path = W / "test_access_2_result.json"
    require_new_files(result_path, out_root / "final_per_case.csv")
    if (out_root / "FINAL").exists():
        raise SystemExit("test-2 output already exists: the test set may be scored only once more")
    choice = json.loads(CHOICE.read_text())
    for m in choice["members"]:
        predict(m, W / "test_in", out_root / m)
    final = out_root / "FINAL"
    build(choice["members"], out_root, final, choice["T_mm3"])
    res = score(final, W / "test_labels", out_root / "final_per_case.csv")
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump({"choice": choice, "test": res}, handle, indent=2, allow_nan=False)
    print("TEST-2 RESULT", res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "test"])
    {"select": select, "test": test}[ap.parse_args().stage]()
