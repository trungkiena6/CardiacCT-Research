"""Post-processing A (PREREG_postproc_A.md, 03/11/2026): fixed grid of binarisation x anatomical gate on the Split-1
three-member ensemble probability maps of the 50 validation cases, scored with the repository evaluator; selection and
decision rules exactly as pre-registered. No test data is read. Resumable: configurations whose CSV exists are re-read.

Run on the pod:  cd /workspace/coroseg/scripts && python3 /workspace/rerun/postproc_a_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, "/workspace/coroseg/scripts")
from imagecas_ensemble_access2 import W, score  # noqa: E402
from score_segmentation import metric_arrays, read_evaluation_rows  # noqa: E402

MEMBERS = ["unet_ft500", "resenc_l", "resenc_l_ft"]
VAL = W / "testaccess3" / "val"
OUT = W / "postproc_a"
CACHE = OUT / "cache"
MASKS = OUT / "heart_masks"
GT = W / "val_labels"
LOG = Path("/workspace/postproc_a.log")
BASE_RECORDED = 0.8439028864300036  # trail.c_step_0.25 in test_access_3_choice.json
S26 = np.ones((3, 3, 3), bool)
PLAIN = [0.30, 0.35, 0.40, 0.45, 0.50]
HYST = [0.25, 0.30, 0.35, 0.40]
GATES = [0, 10, 20]
T_STEP = [10, 25]
WORKERS = 16


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d_%H:%M:%S')} A {msg}\n")
    print(msg, flush=True)


def member_dir(m: str) -> Path:
    return VAL / f"{m}__checkpoint_final__s0.25"


def ids() -> list[str]:
    return sorted(p.name[:-7] for p in GT.glob("*.nii.gz"))


# ---------------------------------------------------------------- step 0: per-case cache (mean probability + gates)
def build_cache(case: str) -> str:
    out = CACHE / f"{case}.npz"
    if out.is_file():
        return case
    probs = []
    for m in MEMBERS:
        with np.load(member_dir(m) / f"{case}.npz") as data:
            probs.append(data["probabilities"][1].astype(np.float32))
    prob = sum(probs) / len(probs)
    ref = sitk.ReadImage(str(member_dir(MEMBERS[1]) / f"{case}.nii.gz"))
    hm = sitk.ReadImage(str(MASKS / f"{case}.nii.gz"))
    heart = sitk.GetArrayFromImage(hm) == 1
    if heart.shape != prob.shape:
        raise RuntimeError(f"{case}: heart mask {heart.shape} vs probability {prob.shape}")
    if not np.allclose(hm.GetSpacing(), ref.GetSpacing(), atol=1e-4):
        raise RuntimeError(f"{case}: heart mask spacing {hm.GetSpacing()} vs {ref.GetSpacing()}")
    sp = ref.GetSpacing()[::-1]  # array order z, y, x
    dist = ndi.distance_transform_edt(~heart, sampling=sp).astype(np.float32)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, prob=prob, g10=dist <= 10.0, g20=dist <= 20.0)
    tmp.rename(out)
    return case


# ---------------------------------------------------------------- binarisation / gate / T
def binarise(prob: np.ndarray, kind: str, t: float) -> np.ndarray:
    if kind == "plain":
        return prob >= t
    low = prob >= t
    high = prob >= 0.5
    lab, n = ndi.label(low, structure=S26)
    if n == 0:
        return low
    keep = np.unique(lab[high])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def gate(mask: np.ndarray, g: np.ndarray | None) -> np.ndarray:
    if g is None or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=S26)
    idx = np.arange(1, n + 1)
    inside = ndi.sum(g, lab, idx)
    size = ndi.sum(mask, lab, idx)
    keep = idx[(inside / size) >= 0.5]
    return np.isin(lab, keep)


def remove_small(mask: np.ndarray, t_mm3: float, vox_mm3: float) -> np.ndarray:
    if t_mm3 <= 0 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=S26)
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, lab, idx) * vox_mm3
    return np.isin(lab, idx[sizes >= t_mm3])


def cfg_name(kind: str, t: float, g: int, t_mm3: float) -> str:
    return f"{kind}{t:.2f}_g{g}_T{int(t_mm3)}"


def write_case(args) -> str:
    case, kind, t, g, t_mm3, out_dir = args
    dst = Path(out_dir) / f"{case}.nii.gz"
    if dst.is_file():
        return case
    with np.load(CACHE / f"{case}.npz") as data:
        prob = data["prob"]
        gm = None if g == 0 else data[f"g{g}"]
    ref = sitk.ReadImage(str(member_dir(MEMBERS[1]) / f"{case}.nii.gz"))
    mask = remove_small(gate(binarise(prob, kind, t), gm), t_mm3, float(np.prod(ref.GetSpacing())))
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(dst), True)
    return case


def evaluate(pool: Pool, cases: list[str], kind: str, t: float, g: int, t_mm3: float) -> dict:
    name = cfg_name(kind, t, g, t_mm3)
    out_dir = OUT / f"cfg_{name}"
    csv_out = OUT / f"cfg_{name}.csv"
    if csv_out.is_file():
        rows = read_evaluation_rows(csv_out, cases)
        res = {"n": len(rows), **{k: float(v.mean()) for k, v in metric_arrays(rows).items()}}
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        pool.map(write_case, [(c, kind, t, g, t_mm3, str(out_dir)) for c in cases])
        res = score(out_dir, GT, csv_out)
    res.update(kind=kind, t=t, gate_mm=g, T_mm3=t_mm3, name=name)
    log(
        f"CFG {name}: "
        + " ".join(
            f"{k}={res[k]:.4f}"
            for k in ("dice", "cldice", "hd95_mm", "betti0_err", "coverage_1mm")
            if k in res
        )
    )
    return res


def per_case_dice(name: str, cases: list[str]) -> np.ndarray:
    rows = {
        r["case"]: float(r["dice"]) for r in read_evaluation_rows(OUT / f"cfg_{name}.csv", cases)
    }
    return np.array([rows[c] for c in cases])


def bootstrap(v: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def changes(r: dict) -> tuple:
    return (
        int(r["kind"] != "plain") + int(r["gate_mm"] != 0) + int(abs(r["t"] - 0.5) > 1e-9),
        int(r["kind"] != "plain"),
        int(r["gate_mm"] != 0),
        abs(r["t"] - 0.5),
        r["gate_mm"],
    )


def main() -> None:
    cases = ids()
    if len(cases) != 50:
        raise SystemExit(f"expected 50 validation cases, found {len(cases)}")
    for m in MEMBERS:
        n = len(list(member_dir(m).glob("*.npz")))
        if n != 50:
            raise SystemExit(f"{m}: {n} probability files, expected 50")
    missing = [c for c in cases if not (MASKS / f"{c}.nii.gz").is_file()]
    if missing:
        raise SystemExit(f"heart masks missing for {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    log(f"SWEEP start: {len(cases)} cases, grid {len(PLAIN) + len(HYST)} x {len(GATES)}")
    with Pool(WORKERS) as pool:
        pool.map(build_cache, cases)
        log("CACHE-DONE")
        results = {}
        base = evaluate(pool, cases, "plain", 0.50, 0, 0)
        results[base["name"]] = base
        if abs(base["dice"] - BASE_RECORDED) > 0.001:
            log(
                f"LOI BASE-MISMATCH: re-scored {base['dice']:.4f} vs recorded {BASE_RECORDED:.4f}; sweep stopped"
            )
            raise SystemExit(2)
        log(f"BASE-OK: {base['dice']:.4f} (recorded {BASE_RECORDED:.4f})")
        for g in GATES:
            for t in PLAIN:
                r = evaluate(pool, cases, "plain", t, g, 0)
                results[r["name"]] = r
            for t in HYST:
                r = evaluate(pool, cases, "hyst", t, g, 0)
                results[r["name"]] = r
        best_dice = max(r["dice"] for r in results.values())
        ties = [r for r in results.values() if r["dice"] >= best_dice - 0.0005]
        winner = min(ties, key=changes)
        log(
            f"WINNER {winner['name']} dice {winner['dice']:.4f} (max {best_dice:.4f}, {len(ties)} tied)"
        )
        # T step for the winner only
        chosen = winner
        t_results = {}
        for t_mm3 in T_STEP:
            r = evaluate(pool, cases, winner["kind"], winner["t"], winner["gate_mm"], t_mm3)
            t_results[str(t_mm3)] = r
            if winner["dice"] - r["dice"] <= 0.0005 and r["betti0_err"] < winner["betti0_err"]:
                chosen = r
                break
    d_dice = chosen["dice"] - base["dice"]
    d_cl = chosen["cldice"] - base["cldice"]
    adopt = d_dice >= 0.0015 and d_cl >= -0.005
    delta = per_case_dice(chosen["name"], cases) - per_case_dice(base["name"], cases)
    ci = bootstrap(delta)
    choice = {
        "prereg": "PREREG_postproc_A.md",
        "baseline": base,
        "winner_T0": winner,
        "T_step": t_results,
        "chosen": chosen,
        "delta_dice": d_dice,
        "delta_cldice": d_cl,
        "delta_dice_paired_ci95": ci,
        "rule": "adopt if delta_dice >= 0.0015 and delta_cldice >= -0.005 (val-50)",
        "adopt": adopt,
        "test_access": "none in this step",
    }
    json.dump(
        {"results": results, "T_step": t_results, "choice": choice},
        (OUT / "val_sweep.json").open("w"),
        indent=1,
    )
    json.dump(choice, (OUT / "choice.json").open("w"), indent=1)
    log(f"SWEEP-DONE: {len(results)} configurations + {len(t_results)} T steps")
    log(
        f"CHOICE {chosen['name']}: dice {chosen['dice']:.4f} vs base {base['dice']:.4f} (delta {d_dice:+.4f}, CI {ci[0]:+.4f}..{ci[1]:+.4f}; cldice {d_cl:+.4f}) -> {'ADOPT' if adopt else 'KEEP BASELINE'}"
    )


if __name__ == "__main__":
    main()
