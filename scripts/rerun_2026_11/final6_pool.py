"""Final Split-1 configuration (PREREG_final_split1.md, 03/11/2026).

val       : predict every available pool member on the 50 validation cases (step 0.25, no TTA, probabilities), score each
            member alone (pool rule: Dice >= 0.8340), evaluate the pre-registered candidate ensembles (ALL, leave-one-out,
            access-3 trio), choose by val Dice, apply the gate (>= 0.8460), write W/final6/val_choice_pool<N>.json.
selfcheck : build the chosen ensemble on val through the sum-and-threshold path used for the test and compare per case
            with the subset evaluation (must agree to 1e-6). No test data.
test      : the single sixth test access: predict the chosen members on the 250 test cases, accumulate, threshold, score,
            bootstrap, paired difference to access 3, write W/test_access_6_result.json (refused if it exists).

Run on the pod (environment set by final_chain.sh):
  cd /workspace/coroseg/scripts && python3 /workspace/rerun/final6_pool.py {val|selfcheck|test}
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/workspace/coroseg/scripts")
from imagecas_ensemble_access2 import W, score  # noqa: E402
from score_segmentation import metric_arrays, read_evaluation_rows, require_new_files  # noqa: E402

RES = W / "nnunet_results" / "Dataset901_ImageCAS_S1"
OUT = W / "final6"
LOG = Path("/workspace/final6.log")
PREREG = "PREREG_final_split1.md"
PL = "nnUNetResEncUNetL40GPlans"
MEMBERS = {  # name -> (trainer, plans, extra environment for the trainer source)
    "unet_ft500": ("nnUNetTrainer_ft500_NM", "nnUNetPlans", {}),
    "resenc_l": ("nnUNetTrainer_250epochs_NoMirroring", PL, {}),
    "resenc_l_ft": ("nnUNetTrainer_ft250_NM", PL, {}),
    "pilot1000": ("nnUNetTrainer_1000epochs_NoMirroring", PL, {}),
    "expB": (
        "nnUNetTrainer_250epochs_NoMirroring_SkelRec",
        PL,
        {
            "nnUNet_extTrainer": "/workspace/rerun/skelrec/skeleton_recall/training/nnUNetTrainer",
            "PYTHONPATH": "/workspace/rerun/skelrec:/workspace/coroseg/src",
        },
    ),
    "resenc_l_s2": ("nnUNetTrainer_250epochs_NoMirroring_seed2", PL, {}),
    "resenc_l_s3": ("nnUNetTrainer_250epochs_NoMirroring_seed3", PL, {}),
}
ARCHIVED = ["unet_ft500", "resenc_l", "resenc_l_ft"]
ARCHIVED_VAL = {m: W / "testaccess3" / "val" / f"{m}__checkpoint_final__s0.25" for m in ARCHIVED}
POOL_FLOOR = 0.8340
GATE = 0.8460
TIE = 0.0005
STEP = 0.25
TRIO_TAG = "+".join(sorted(ARCHIVED))
ACCESS3_TEST_CSV = W / "testaccess3" / "test" / "final_per_case.csv"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d_%H:%M:%S')} F {msg}\n")
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_dir(m: str) -> Path:
    tr, pl, _ = MEMBERS[m]
    return RES / f"{tr}__{pl}__3d_fullres" / "fold_0"


def available(m: str) -> bool:
    return (fold_dir(m) / "checkpoint_final.pth").is_file()


def case_ids(images: Path) -> list[str]:
    return sorted(p.name[:-12] for p in images.glob("*_0000.nii.gz"))


# ----------------------------------------------------------------------------------------------- prediction
def predict_member(m: str, images: Path, out: Path) -> Path:
    """nnUNetv2_predict with probabilities; resumable; provenance.json next to the outputs."""
    ids = case_ids(images)
    if (out / ".done").is_file():
        return out
    tr, pl, extra = MEMBERS[m]
    env = dict(os.environ, **extra)
    env.setdefault("nnUNet_results", str(W / "nnunet_results"))
    env.setdefault("nnUNet_raw", str(W / "nnunet_raw"))
    env.setdefault("nnUNet_preprocessed", str(W / "nnunet_preprocessed"))
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
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
        str(STEP),
        "--disable_tta",
        "--save_probabilities",
        "--continue_prediction",
        "-npp",
        "4",
        "-nps",
        "4",
    ]
    log(f"MEMBER {m} predict {len(ids)} cases -> {out}")
    t0 = time.time()
    with (out / "predict.log").open("a") as fh:
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    missing = [
        c for c in ids if not (out / f"{c}.nii.gz").is_file() or not (out / f"{c}.npz").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{m}: {len(missing)} cases without mask+probabilities, e.g. {missing[:3]}"
        )
    prov = dict(
        member=m,
        trainer=tr,
        plans=pl,
        step=STEP,
        tta=False,
        checkpoint="checkpoint_final.pth",
        checkpoint_sha256=sha256(fold_dir(m) / "checkpoint_final.pth"),
        plans_sha256=sha256(fold_dir(m).parent / "plans.json"),
        images=str(images),
        n=len(ids),
        command=cmd,
        extra_env=extra,
        seconds=round(time.time() - t0),
        finished=time.strftime("%Y-%m-%d_%H:%M:%S"),
    )
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    (out / ".done").touch()
    log(f"MEMBER {m} done: {len(ids)} cases in {prov['seconds']} s")
    return out


def val_dir(m: str) -> Path:
    if m in ARCHIVED:
        d = ARCHIVED_VAL[m]
        if len(list(d.glob("*.npz"))) != 50:
            raise RuntimeError(f"archived val probabilities incomplete: {d}")
        return d
    return OUT / "val" / m


# ----------------------------------------------------------------------------------------------- ensembles
def ensemble_masks(
    dirs: dict[str, Path], ids: list[str], out: Path, threshold_single: bool = False
) -> None:
    """Mean of the members' foreground probabilities >= 0.5 -> masks in out (geometry copied from the first member)."""
    if (out / ".done").is_file():
        return
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(dirs.values()))
    for c in ids:
        acc = None
        for d in dirs.values():
            with np.load(d / f"{c}.npz") as z:
                p = z["probabilities"][1].astype(np.float32)
            acc = p if acc is None else acc + p
        mask = (acc / len(dirs)) >= 0.5
        ref = sitk.ReadImage(str(first / f"{c}.nii.gz"))
        o = sitk.GetImageFromArray(mask.astype(np.uint8))
        o.CopyInformation(ref)
        sitk.WriteImage(o, str(out / f"{c}.nii.gz"))
    (out / ".done").touch()


def evaluate(tag: str, dirs: dict[str, Path], ids: list[str], root: Path, gt: Path) -> dict:
    """Resumable: an existing CSV is re-read instead of re-scored."""
    csv_out = root / f"{tag}.csv"
    if csv_out.is_file():
        rows = read_evaluation_rows(csv_out, ids)
        res = {"n": len(rows), **{k: float(v.mean()) for k, v in metric_arrays(rows).items()}}
    else:
        ensemble_masks(dirs, ids, root / tag)
        res = score(root / tag, gt, csv_out)
    res = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}
    return res


def member_single(m: str, ids: list[str], root: Path, gt: Path) -> dict:
    return evaluate(f"single_{m}", {m: val_dir(m)}, ids, root, gt)


def b_verdict_ok() -> tuple[bool, str]:
    p = W / "expB" / "decision.json"
    if not p.is_file():
        return False, "no decision.json"
    d = json.loads(p.read_text())
    return float(d["delta"]["dice"]) >= 0, d["verdict"]


# ----------------------------------------------------------------------------------------------- val stage
def stage_val() -> dict:
    images, gt = W / "val_in", W / "val_labels"
    ids = case_ids(images)
    assert len(ids) == 50, len(ids)
    root = OUT / "val"
    root.mkdir(parents=True, exist_ok=True)
    log(f"VAL start: pool candidates {[m for m in MEMBERS if available(m)]}")
    pool, single, excluded = [], {}, {}
    for m in MEMBERS:
        if not available(m):
            excluded[m] = "no checkpoint_final.pth"
            continue
        if m not in ARCHIVED:
            predict_member(m, images, root / m)
        single[m] = member_single(m, ids, root, gt)
        log(f"SINGLE {m}: {single[m]}")
        if m in ARCHIVED:
            pool.append(m)
            continue
        if m == "expB":
            ok, verdict = b_verdict_ok()
            if not ok:
                excluded[m] = f"B verdict: {verdict}"
                continue
        if single[m]["dice"] < POOL_FLOOR:
            excluded[m] = f"single val dice {single[m]['dice']} < {POOL_FLOOR}"
            continue
        pool.append(m)
    log(f"POOL {pool}; excluded {excluded}")
    cands = {"+".join(sorted(pool)): sorted(pool)}
    if len(pool) > 2:
        for m in pool:
            rest = sorted(x for x in pool if x != m)
            cands["+".join(rest)] = rest
    cands[TRIO_TAG] = sorted(ARCHIVED)
    results = {}
    for tag, members in sorted(cands.items()):
        results[tag] = evaluate(f"ens_{tag}", {m: val_dir(m) for m in members}, ids, root, gt)
        log(f"ENS {tag}: {results[tag]}")
    best = max(results.values(), key=lambda r: r["dice"])["dice"]
    tied = [t for t in results if best - results[t]["dice"] <= TIE]
    chosen = sorted(tied, key=lambda t: (-len(cands[t]), t))[0]
    gate = "PASS" if results[chosen]["dice"] >= GATE else "FAIL"
    choice = dict(
        prereg=PREREG,
        pool=pool,
        excluded=excluded,
        single=single,
        candidates=results,
        chosen=chosen,
        members=cands[chosen],
        val=results[chosen],
        tied_within=TIE,
        tied=tied,
        gate_rule=f"val dice >= {GATE}",
        gate=gate,
        step=STEP,
        weights="equal",
        threshold=0.5,
        T_mm3=0,
        test_access="none in this stage",
        written=time.strftime("%Y-%m-%d_%H:%M:%S"),
    )
    path = OUT / f"val_choice_pool{len(pool)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    require_new_files(path)
    path.write_text(json.dumps(choice, indent=1))
    log(f"CHOICE {chosen}: val dice {results[chosen]['dice']} (tied {tied}) -> {path.name}")
    log(f"GATE {gate}: {results[chosen]['dice']} vs {GATE}")
    return choice


def latest_choice() -> dict:
    files = sorted(OUT.glob("val_choice_pool*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no val choice file: run val first")
    return json.loads(files[-1].read_text())


# ----------------------------------------------------------------------------------------------- sum path (test)
def accumulate(m: str, src: Path, ids: list[str], sums: Path, delete_npz: bool) -> None:
    """Per-case running probability sum with the list of members already added (atomic per case, resumable)."""
    sums.mkdir(parents=True, exist_ok=True)
    for c in ids:
        f = sums / f"{c}.npz"
        if f.is_file():
            with np.load(f) as z:
                members, acc = [str(x) for x in z["members"]], z["sum"].astype(np.float32)
            if m in members:
                continue
        else:
            members, acc = [], None
        with np.load(src / f"{c}.npz") as z:
            p = z["probabilities"][1].astype(np.float32)
        acc = p if acc is None else acc + p
        tmp = sums / f"{c}.tmp.npz"
        np.savez(tmp, sum=acc, members=np.array(members + [m]))
        tmp.rename(f)
    if delete_npz:
        for c in ids:
            (src / f"{c}.npz").unlink(missing_ok=True)
            (src / f"{c}.pkl").unlink(missing_ok=True)
        log(f"MEMBER {m}: probabilities added to the sum and deleted")


def masks_from_sums(
    members: list[str], ids: list[str], sums: Path, ref_dir: Path, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for c in ids:
        with np.load(sums / f"{c}.npz") as z:
            added, acc = sorted(str(x) for x in z["members"]), z["sum"].astype(np.float32)
        if added != sorted(members):
            raise RuntimeError(f"{c}: sum holds {added}, expected {sorted(members)}")
        mask = (acc / len(members)) >= 0.5
        ref = sitk.ReadImage(str(ref_dir / f"{c}.nii.gz"))
        o = sitk.GetImageFromArray(mask.astype(np.uint8))
        o.CopyInformation(ref)
        sitk.WriteImage(o, str(out / f"{c}.nii.gz"))


def stage_selfcheck() -> None:
    choice = latest_choice()
    members, ids = choice["members"], case_ids(W / "val_in")
    root = OUT / "selfcheck"
    if root.exists():
        shutil.rmtree(root)
    sums = root / "sums"
    for m in members:
        accumulate(m, val_dir(m), ids, sums, delete_npz=False)
    masks_from_sums(members, ids, sums, val_dir(members[0]), root / "FINAL")
    res = score(root / "FINAL", W / "val_labels", root / "final_per_case.csv")
    a = {
        r["case"]: float(r["dice"]) for r in read_evaluation_rows(root / "final_per_case.csv", ids)
    }
    b = {
        r["case"]: float(r["dice"])
        for r in read_evaluation_rows(OUT / "val" / f"ens_{choice['chosen']}.csv", ids)
    }
    worst = max(abs(a[c] - b[c]) for c in ids)
    shutil.rmtree(sums)
    if worst > 1e-6:
        log(f"LOI SELFCHECK: sum path differs from subset path by {worst}")
        raise SystemExit(1)
    log(
        f"SELFCHECK ok: sum path val dice {round(res['dice'], 6)} = subset path (max per-case diff {worst:.1e})"
    )


# ----------------------------------------------------------------------------------------------- test stage
def bootstrap(v: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def stage_test() -> None:
    choice = latest_choice()
    if choice["gate"] != "PASS":
        raise SystemExit("gate FAIL: the test set is not accessed")
    result_path = W / "test_access_6_result.json"
    root = OUT / "test"
    require_new_files(result_path, root / "FINAL", root / "final_per_case.csv")
    if not ACCESS3_TEST_CSV.is_file():
        raise SystemExit(f"access-3 per-case CSV missing: {ACCESS3_TEST_CSV}")
    images, gt = W / "test_in", W / "test_labels"
    ids = case_ids(images)
    assert len(ids) == 250, len(ids)
    members = choice["members"]
    log(
        f"TEST start: sixth Split-1 test access, members {members}, step {STEP}, equal weights, threshold 0.5, T 0"
    )
    t0 = time.time()
    prov = {}
    for m in members:
        d = predict_member(m, images, root / m)
        accumulate(m, d, ids, root / "sums", delete_npz=True)
        prov[m] = json.loads((d / "provenance.json").read_text())
    masks_from_sums(members, ids, root / "sums", root / members[0], root / "FINAL")
    res = score(root / "FINAL", gt, root / "final_per_case.csv")
    rows = {r["case"]: r for r in read_evaluation_rows(root / "final_per_case.csv", ids)}
    prev = {r["case"]: r for r in read_evaluation_rows(ACCESS3_TEST_CSV, ids)}
    ci, paired = {}, {}
    for k in ("dice", "cldice", "hd95_mm", "gt_centerline_coverage_1mm"):
        v = np.array([float(rows[c][k]) for c in ids])
        ci[k] = bootstrap(v)
        d = np.array([float(rows[c][k]) - float(prev[c][k]) for c in ids])
        paired[k] = {"mean_diff_vs_access3": float(d.mean()), "ci95": bootstrap(d)}
    out = dict(
        prereg=PREREG,
        access="Split-1 test access 6 (the last of the 11/2026 rerun)",
        choice=choice,
        members=members,
        test=res,
        ci95=ci,
        paired_vs_access3=paired,
        provenance=prov,
        seconds=round(time.time() - t0),
        finished=time.strftime("%Y-%m-%d_%T"),
    )
    with result_path.open("x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    shutil.rmtree(root / "sums")
    summ = {k: round(v, 4) for k, v in res.items() if isinstance(v, float)}
    log(
        f"TEST-DONE: {summ} ci95 dice {[round(x, 4) for x in ci['dice']]} "
        f"paired dice vs access 3 {paired['dice']} -> {result_path}"
    )


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in ("val", "selfcheck", "test"):
        raise SystemExit("usage: final6_pool.py {val|selfcheck|test}")
    try:
        {"val": stage_val, "selfcheck": stage_selfcheck, "test": stage_test}[stage]()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log(f"LOI {stage}: {type(exc).__name__}: {exc}")
        raise
