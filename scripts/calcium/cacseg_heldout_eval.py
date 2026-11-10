"""CACSeg held-out protocol: selection on the validation split, one test access.

  select : predict the validation split with every available candidate (C1 full FOV, C2 heart crop,
           C3 crop + candidate channel) and their ensembles; pick by 3D Dice over positives; sweep
           threshold / candidate-snap cap / heart gating; write cacseg_choice.json.
  test   : refuse if a test output exists; predict the test split with the chosen configuration,
           score segmentation (A) and Agatston agreement (B) with bootstrap CIs.
Run on the A100 pod: paths below are the pod layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import stats

sys.path.insert(0, "/workspace/cacseg")
from eval_cacseg_nongated import agatston  # noqa: E402

W = Path("/workspace/cacseg-work")
RES = W / "nnunet_results"
TRAINER = "nnUNetTrainer_250epochs_NoMirroring"
# name -> (dataset id, dataset folder, input kind, plans[, trainer]); candidates of the pre-registration are c1–c4,
# the others are post-hoc ablations evaluated on the validation split only.
MODELS = {
    "c1": (910, "Dataset910_COCA_CAC_HO", "all", "nnUNetPlans"),
    "c2": (911, "Dataset911_COCA_CAC_HO_CROP", "crop", "nnUNetPlans"),
    "c3": (912, "Dataset912_COCA_CAC_HO_CAND", "crop_cand", "nnUNetPlans"),
    "c4": (911, "Dataset911_COCA_CAC_HO_CROP", "crop", "nnUNetResEncUNetL40GPlans"),
    "c2_mirror": (
        911,
        "Dataset911_COCA_CAC_HO_CROP",
        "crop",
        "nnUNetPlans",
        "nnUNetTrainer_250epochs",
    ),
    "c4_seed2": (
        911,
        "Dataset911_COCA_CAC_HO_CROP",
        "crop",
        "nnUNetResEncUNetL40GPlans",
        "nnUNetTrainer_250epochs_NoMirroring_seed2",
    ),
}
CANDIDATE_NAMES = ("c1", "c2", "c3", "c4")


def trainer_of(m: str) -> str:
    return MODELS[m][4] if len(MODELS[m]) > 4 else TRAINER


THRS, CAPS, GATES = (0.3, 0.4, 0.5, 0.6), (0, 200, 400, 800, 1600), (0, 1, 2)
# gate 0 = none; 1 = heart mask dilated 10 mm; 2 = gate 1 AND outside the aorta dilated 3 mm (aortic-root calcium
# is the dominant false lesion on validation; amendment of 2026-10-08)
AORTA_DILATE_MM = 3.0
CHOICE = W / "cacseg_choice.json"
EDGES5, EDGES4 = (0, 10, 100, 400), (0, 100, 400)


def category(score: float, edges) -> int:
    return sum(score > e for e in edges)


def available(m: str) -> bool:
    _, name, _, plans = MODELS[m][:4]
    return (
        RES / name / f"{trainer_of(m)}__{plans}__3d_fullres" / "fold_0" / "checkpoint_final.pth"
    ).is_file()


def predict(m: str, split: str) -> Path:
    did, _, kind, plans = MODELS[m][:4]
    images, out = W / f"{split}_{kind}", W / "heldout" / split / m
    if not (
        out.is_dir() and len(list(out.glob("*.npz"))) == len(list(images.glob("*_0000.nii.gz")))
    ):
        subprocess.run(
            [
                "nnUNetv2_predict",
                "-i",
                str(images),
                "-o",
                str(out),
                "-d",
                str(did),
                "-c",
                "3d_fullres",
                "-f",
                "0",
                "-tr",
                trainer_of(m),
                "-p",
                plans,
                "--save_probabilities",
                "-npp",
                "4",
                "-nps",
                "4",
            ],
            check=True,
        )
    return out


def prob_full(m: str, split: str, case: str, shape) -> np.ndarray:
    p = np.load(W / "heldout" / split / m / f"{case}.npz")["probabilities"][1].astype(np.float32)
    if MODELS[m][2] == "all":
        return p
    box = json.loads((W / "crop_boxes.json").read_text())[case]
    (lx, ly, lz), (sx, sy, sz) = box["lo"], box["size"]
    full = np.zeros(shape, np.float32)
    full[lz : lz + sz, ly : ly + sy, lx : lx + sx] = p
    return full


def postprocess(
    prob: np.ndarray, hu: np.ndarray, spacing, thr: float, cap: float, gate: int, heart
) -> np.ndarray:
    mask = prob >= thr
    if cap > 0 and mask.any():
        cand, n = ndi.label(hu >= 130, structure=np.ones((3, 3, 3), bool))
        vox = float(np.prod(spacing))
        touched = np.unique(cand[mask & (cand > 0)])
        sizes = ndi.sum(np.ones_like(cand, bool), cand, touched) * vox if len(touched) else []
        keep = [c for c, s in zip(touched, sizes) if s <= cap]
        mask = mask | np.isin(cand, keep)
    if gate and heart is not None:
        mask = mask & heart
    return mask


def case_metrics(args):
    """One case: returns dict with segmentation counts and Agatston pred/ref."""
    case, split, members, thr, cap, gate = args
    gt_img = sitk.ReadImage(str(W / f"{split}_labels" / f"{case}.nii.gz"))
    gt = sitk.GetArrayFromImage(gt_img).astype(bool)
    img = sitk.ReadImage(str(W / f"{split}_all" / f"{case}_0000.nii.gz"))
    hu = sitk.GetArrayFromImage(img).astype(np.int16)
    sp = img.GetSpacing()
    prob = np.mean([prob_full(m, split, case, gt.shape) for m in members], axis=0)
    heart = None
    hp = W / "heart_flat" / f"{case}.nii.gz"
    if gate and hp.exists():
        h = sitk.GetArrayFromImage(sitk.ReadImage(str(hp))) > 0
        heart = ndi.distance_transform_edt(~h, sampling=sp[::-1]) <= 10.0
        ap = W / "aorta" / case / "aorta.nii.gz"
        if gate == 2 and ap.exists():
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(ap))) > 0
            if a.any():
                heart &= ndi.distance_transform_edt(~a, sampling=sp[::-1]) > AORTA_DILATE_MM
    pred = postprocess(prob, hu, sp, thr, cap, gate, heart)
    tp, fp, fn = int((gt & pred).sum()), int((pred & ~gt).sum()), int((gt & ~pred).sum())
    slice_d = [
        2 * (gt[z] & pred[z]).sum() / (gt[z].sum() + pred[z].sum())
        for z in range(gt.shape[0])
        if gt[z].any()
    ]
    s26 = np.ones((3, 3, 3), bool)
    glab, gn = ndi.label(gt, s26)
    plab, pn = ndi.label(pred, s26)
    hit = len(set(np.unique(glab[gt & pred])) - {0})
    fp_comp = pn - len(set(np.unique(plab[gt & pred])) - {0})
    return {
        "case": case,
        "gt_voxels": int(gt.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice3d": 2 * tp / (2 * tp + fp + fn) if gt.any() else float("nan"),
        "slice_dice": float(np.mean(slice_d)) if slice_d else float("nan"),
        "gt_lesions": gn,
        "lesions_hit": hit,
        "pred_lesions": pn,
        "pred_lesions_fp": fp_comp,
        "agatston_pred": agatston(hu, pred, sp),
        "agatston_ref": agatston(hu, gt, sp),
    }


def weighted_kappa(a, b, k: int, quadratic=False) -> float:
    o = np.zeros((k, k))
    for i, j in zip(a, b):
        o[i, j] += 1
    n = o.sum()
    e = np.outer(o.sum(1), o.sum(0)) / n
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    w = w**2 if quadratic else w
    return float(1 - (w * o).sum() / (w * e).sum())


def icc21(x, y) -> float:
    d = np.stack([x, y], 1)
    n, k = d.shape
    msr = k * d.mean(1).var(ddof=1)
    msc = n * d.mean(0).var(ddof=1)
    mse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()) ** 2).sum() / (
        (n - 1) * (k - 1)
    )
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["gt_voxels"] > 0]
    d3 = np.array([r["dice3d"] for r in pos])
    tp, fp, fn = (sum(r[k] for r in pos) for k in ("tp", "fp", "fn"))
    ap = np.array([r["agatston_pred"] for r in rows])
    ar = np.array([r["agatston_ref"] for r in rows])
    c5p, c5r = [category(v, EDGES5) for v in ap], [category(v, EDGES5) for v in ar]
    c4p, c4r = [category(v, EDGES4) for v in ap], [category(v, EDGES4) for v in ar]
    diff = ap - ar
    return {
        "n": len(rows),
        "n_positive": len(pos),
        "dice3d_mean": float(d3.mean()),
        "dice3d_median": float(np.median(d3)),
        "dice_pooled": 2 * tp / (2 * tp + fp + fn),
        "slice_dice_mean": float(np.nanmean([r["slice_dice"] for r in pos])),
        "lesion_sensitivity": sum(r["lesions_hit"] for r in pos)
        / max(1, sum(r["gt_lesions"] for r in pos)),
        "lesion_ppv": 1
        - sum(r["pred_lesions_fp"] for r in rows) / max(1, sum(r["pred_lesions"] for r in rows)),
        "kappa_lin_5cat": weighted_kappa(c5r, c5p, 5),
        "kappa_quad_5cat": weighted_kappa(c5r, c5p, 5, True),
        "kappa_lin_4cat": weighted_kappa(c4r, c4p, 4),
        "accuracy_5cat": float(np.mean(np.array(c5p) == np.array(c5r))),
        "accuracy_4cat": float(np.mean(np.array(c4p) == np.array(c4r))),
        "icc21": icc21(ap, ar),
        "pearson": float(stats.pearsonr(ap, ar)[0]),
        "spearman": float(stats.spearmanr(ap, ar)[0]),
        "bland_altman_bias": float(diff.mean()),
        "bland_altman_loa": [
            float(diff.mean() - 1.96 * diff.std(ddof=1)),
            float(diff.mean() + 1.96 * diff.std(ddof=1)),
        ],
    }


def evaluate(
    split: str, members: list[str], thr: float, cap: float, gate: int, tag: str
) -> tuple[dict, list[dict]]:
    cases = sorted(p.name[:-7] for p in (W / f"{split}_labels").glob("*.nii.gz"))
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(case_metrics, [(c, split, members, thr, cap, gate) for c in cases]))
    out = W / "heldout" / split / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    s = summarize(rows)
    print(
        tag,
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in s.items()
            if k != "bland_altman_loa"
        },
        flush=True,
    )
    return s, rows


def select() -> None:
    avail = [m for m in CANDIDATE_NAMES if available(m)]
    for m in avail:
        predict(m, "val")
    cands = [list(c) for r in range(1, len(avail) + 1) for c in combinations(avail, r)]
    results = {"+".join(c): evaluate("val", c, 0.5, 0, 0, "cand_" + "+".join(c))[0] for c in cands}
    best = max(results, key=lambda k: (results[k]["dice3d_mean"], results[k]["kappa_lin_5cat"]))
    members = best.split("+")
    base = results[best]
    sweep = {}
    for thr in THRS:
        for cap in CAPS:
            for gate in GATES:
                if (thr, cap, gate) == (0.5, 0, 0):
                    sweep["0.5_0_0"] = base
                    continue
                sweep[f"{thr}_{cap}_{gate}"] = evaluate(
                    "val", members, thr, cap, gate, f"pp_{best}_t{thr}_c{cap}_g{gate}"
                )[0]
    ok = {k: v for k, v in sweep.items() if v["kappa_lin_5cat"] >= base["kappa_lin_5cat"] - 1e-9}
    k_star = max(ok, key=lambda k: ok[k]["dice3d_mean"])
    thr, cap, gate = k_star.split("_")
    choice = {
        "configuration": best,
        "members": members,
        "threshold": float(thr),
        "snap_cap_mm3": float(cap),
        "heart_gate": int(gate),
        "val_candidates": results,
        "val_sweep": sweep,
        "val_chosen": sweep[k_star],
    }
    CHOICE.write_text(json.dumps(choice, indent=2))
    print("CACSEG-CHOICE", best, k_star, sweep[k_star], flush=True)


def bootstrap(fn, rows: list[dict], n_boot=2000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = [fn([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def test() -> None:
    if not CHOICE.is_file():
        raise SystemExit("no choice file: run select first")
    if (W / "heldout" / "test" / "FINAL.csv").exists():
        raise SystemExit(
            "test output already exists: the held-out test set may be scored only once"
        )
    ch = json.loads(CHOICE.read_text())
    for m in ch["members"]:
        predict(m, "test")
    s, rows = evaluate(
        "test", ch["members"], ch["threshold"], ch["snap_cap_mm3"], ch["heart_gate"], "FINAL"
    )
    ci = {
        k: bootstrap(lambda r, k=k: summarize(r)[k], rows)
        for k in (
            "dice3d_mean",
            "slice_dice_mean",
            "kappa_lin_5cat",
            "kappa_lin_4cat",
            "icc21",
            "accuracy_4cat",
        )
    }
    (W / "cacseg_heldout_result.json").write_text(
        json.dumps({"choice": ch, "test": s, "ci95": ci}, indent=2)
    )
    print("CACSEG-TEST RESULT", s, ci, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "test"])
    {"select": select, "test": test}[ap.parse_args().stage]()
