"""Validation-only selection for the ImageCAS-X lumen access 2.

select : for every validation case, average the foreground probabilities of the candidate models, threshold at 0.5
         (p > 0.5, nnU-Net's argmax for two classes), optional hysteresis growth (p >= L and 26-connected to a seed)
         and component removal (T mm^3, evaluate/postprocess.py), scored with evaluate/segmentation.case_metrics.
         Every combination is scored and written; the choice follows Amendment 1 of the prereg (joint configuration
         and L -> T -> go/no-go), the original sequential rule is recorded next to it, and both go to
         <out>_choice.json. With fewer than three models (smoke runs) nothing is chosen.
apply  : write the chosen pipeline's masks for a folder of probability maps (<prob-root>/<prefix>_<model>/<case>.npz).
rechoose: re-apply both rules to an existing per-case csv and print the result (never writes a choice).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/workspace/coroseg/src")
from coronary_tree.evaluate.postprocess import remove_small_components  # noqa: E402
from coronary_tree.evaluate.segmentation import STRUCT26, case_metrics  # noqa: E402

MODELS = ("M1", "M2", "M3")
CONFIGS = [("M1",), ("M1", "M2"), ("M1", "M3"), ("M2", "M3"), ("M1", "M2", "M3")]
LOWS = [0.0, 0.3, 0.4]  # 0.0 = no hysteresis
TS = [0, 10, 25, 50, 100, 200]
KEYS = [
    "dice",
    "cldice",
    "hd95_mm",
    "n_components_pred",
    "n_components_gt",
    "gt_centerline_coverage_1mm",
]
PREREG = Path("/workspace/PREREG_lumen_access2.md")


def fg_prob(npz: Path) -> np.ndarray:
    with np.load(npz) as z:
        p = z["probabilities"][1]
    return np.asarray(p, dtype=np.float32).transpose(
        2, 1, 0
    )  # SimpleITK (z, y, x) -> nibabel (x, y, z)


def hysteresis(p: np.ndarray, low: float) -> np.ndarray:
    seed = p > 0.5
    if low <= 0:
        return seed
    lab, _ = ndi.label(p >= low, structure=STRUCT26)
    keep = np.unique(lab[seed])
    return np.isin(lab, keep[keep > 0])


def pipeline(probs: list[np.ndarray], low: float, t: float, spacing) -> np.ndarray:
    return remove_small_components(hysteresis(np.mean(probs, axis=0), low), spacing, t)[0]


def select_case(case: str, root: str, labels: str, models: tuple[str, ...]):
    img = nib.load(str(Path(labels) / f"{case}.nii.gz"))
    gt = np.asanyarray(img.dataobj) > 0.5
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    probs, check = {}, {}
    for m in models:
        d = Path(root) / f"val_prob_{m}"
        p = probs[m] = fg_prob(d / f"{case}.npz")
        assert p.shape == gt.shape, (case, m, p.shape, gt.shape)
        seg = np.asanyarray(nib.load(str(d / f"{case}.nii.gz")).dataobj) > 0
        a = p > 0.5
        check[m] = float(
            2 * (a & seg).sum() / max(1, a.sum() + seg.sum())
        )  # must be ~1: orientation + threshold
    union = (
        gt.copy()
    )  # lossless crop: every candidate mask lies inside {p >= 0.3} of some model, plus the GT
    for p in probs.values():
        union |= p >= 0.3
    idx = np.argwhere(union)
    sl = tuple(
        slice(a, b)
        for a, b in zip(np.maximum(idx.min(0) - 5, 0), np.minimum(idx.max(0) + 6, gt.shape))
    )
    gt_c, rows = gt[sl], []
    for cfg in CONFIGS:
        if not all(m in probs for m in cfg):
            continue
        p = np.mean([probs[m][sl] for m in cfg], axis=0)
        for low in LOWS:
            base = hysteresis(p, low)
            for t in TS:
                pred, removed, _ = remove_small_components(base, spacing, t)
                mt = case_metrics(gt_c, pred, spacing)
                rows.append(
                    dict(
                        case=case,
                        config="+".join(cfg),
                        low=low,
                        T=t,
                        removed=removed,
                        **{k: mt[k] for k in KEYS},
                    )
                )
    return rows, check


def summarize(rows):
    out = {}
    for r in rows:
        out.setdefault((r["config"], r["low"], r["T"]), []).append(r)
    s = {}
    for key, rs in out.items():
        hd = [r["hd95_mm"] for r in rs if r["hd95_mm"] is not None]
        s[key] = dict(
            n=len(rs),
            dice=float(np.mean([r["dice"] for r in rs])),
            cldice=float(np.mean([r["cldice"] or 0 for r in rs])),
            hd95_mm=float(np.mean(hd)) if hd else None,
            betti0=float(np.mean([abs(r["n_components_pred"] - r["n_components_gt"]) for r in rs])),
            coverage=float(np.mean([r["gt_centerline_coverage_1mm"] or 0 for r in rs])),
        )
    return s


def go_line(final, m1):
    go = final["dice"] - m1["dice"] >= 0.002
    return (
        go,
        f"step 4 go/no-go: chosen dice {final['dice']:.4f} vs M1 {m1['dice']:.4f} (diff {final['dice'] - m1['dice']:+.4f}, need >= +0.0020) -> {'GO' if go else 'NO-GO'}",
    )


def choose_amended(s):
    """Amendment 1: joint (configuration, L), then T by Betti-0 under Dice/clDice/HD95 limits."""
    lines = []
    pairs = [
        (c, l)
        for c in ("+".join(c) for c in CONFIGS)
        if (c, 0.0, 0) in s
        for l in LOWS
        if l == 0.0 or s[(c, 0.0, 0)]["cldice"] - s[(c, l, 0)]["cldice"] <= 0.002
    ]
    cfg, low = max(pairs, key=lambda p: s[(p[0], p[1], 0)]["dice"])
    lines.append(
        f"amended step 1-2 (configuration, L): {cfg}, L {low} ({', '.join(f'{c}/L{l} {s[(c, l, 0)]["dice"]:.4f}' for c, l in pairs)})"
    )
    ref = s[(cfg, low, 0)]
    ok = [
        t
        for t in TS
        if ref["dice"] - s[(cfg, low, t)]["dice"] <= 0.001
        and ref["cldice"] - s[(cfg, low, t)]["cldice"] <= 0.002
        and s[(cfg, low, t)]["hd95_mm"] <= 1.10 * ref["hd95_mm"]
    ]
    t = min(ok, key=lambda t: (s[(cfg, low, t)]["betti0"], t))
    lines.append(
        f"amended step 3 component removal T: {t} mm3 ({', '.join(f'T{x} dice {s[(cfg, low, x)]["dice"]:.4f} cldice {s[(cfg, low, x)]["cldice"]:.4f} hd95 {s[(cfg, low, x)]["hd95_mm"]:.2f} betti0 {s[(cfg, low, x)]["betti0"]:.2f}' for x in TS)})"
    )
    final, m1 = s[(cfg, low, t)], s[("M1", 0.0, 0)]
    go, line = go_line(final, m1)
    lines.append(line)
    return dict(config=cfg, low=low, T=t, go=bool(go), chosen=final, m1_reference=m1), lines


def choose_original(s):
    """Rule as first pre-registered (kept for the record; Amendment 1 decides)."""
    lines = []
    cfgs = [c for c in ("+".join(c) for c in CONFIGS) if (c, 0.0, 0) in s]
    cfg = max(cfgs, key=lambda c: s[(c, 0.0, 0)]["dice"])
    lines.append(
        f"step 1 configuration: {cfg} ({', '.join(f'{c} {s[(c, 0.0, 0)]["dice"]:.4f}' for c in cfgs)})"
    )
    ref = s[(cfg, 0.0, 0)]
    ok = [
        l
        for l in LOWS[1:]
        if s[(cfg, l, 0)]["dice"] - ref["dice"] >= 0.001
        and ref["cldice"] - s[(cfg, l, 0)]["cldice"] <= 0.002
    ]
    low = max(ok, key=lambda l: s[(cfg, l, 0)]["dice"]) if ok else 0.0
    lines.append(
        f"step 2 hysteresis L: {low} ({', '.join(f'L{l} dice {s[(cfg, l, 0)]["dice"]:.4f} cldice {s[(cfg, l, 0)]["cldice"]:.4f}' for l in LOWS)})"
    )
    ref = s[(cfg, low, 0)]
    ok = [
        t
        for t in TS
        if ref["dice"] - s[(cfg, low, t)]["dice"] <= 0.001
        and ref["coverage"] - s[(cfg, low, t)]["coverage"] <= 0.002
    ]
    t = min(ok, key=lambda t: (s[(cfg, low, t)]["betti0"], t))
    lines.append(
        f"step 3 component removal T: {t} mm3 ({', '.join(f'T{x} dice {s[(cfg, low, x)]["dice"]:.4f} betti0 {s[(cfg, low, x)]["betti0"]:.2f} cov {s[(cfg, low, x)]["coverage"]:.4f}' for x in TS)})"
    )
    final, m1 = s[(cfg, low, t)], s[("M1", 0.0, 0)]
    go, line = go_line(final, m1)
    lines.append(line)
    return dict(config=cfg, low=low, T=t, go=bool(go), chosen=final, m1_reference=m1), lines


def choose(s):
    """Both rules; the amended one decides, the original one is recorded."""
    orig, lo = choose_original(s)
    amended, la = choose_amended(s)
    amended["original_rule"] = {k: orig[k] for k in ("config", "low", "T", "go", "chosen")}
    return amended, ["original rule (recorded):", *lo, "amended rule (decides):", *la]


def read_rows(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        r = dict(r, low=float(r["low"]), T=int(r["T"]))
        for k in KEYS:
            r[k] = None if r[k] in ("", "None") else float(r[k])
        rows.append(r)
    return rows


def cmd_rechoose(a):
    """Re-apply both rules to an existing per-case csv (prints only, never writes a choice)."""
    s = summarize(read_rows(a.per_case))
    print("\n".join(choose(s)[1]))


def cmd_select(a):
    models = tuple(m for m in MODELS if (a.root / f"val_prob_{m}" / ".done").exists())
    cases = sorted(p.name[:-7] for p in a.labels.glob("*.nii.gz"))[: a.max_cases or None]
    rows, checks = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(select_case, c, str(a.root), str(a.labels), models): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            r, chk = f.result()
            rows.extend(r)
            checks[futs[f]] = chk
            if i % 10 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} cases", flush=True)
    rows.sort(key=lambda r: (r["case"], r["config"], r["low"], r["T"]))
    with open(f"{a.out}_per_case.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    L = [
        "| configuration | L | T (mm³) | Dice | clDice | HD95 mm | Betti-0 err | coverage@1mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (c, l, t), v in sorted(s.items()):
        L.append(
            f"| {c} | {l} | {t} | {v['dice']:.4f} | {v['cldice']:.4f} | {v['hd95_mm'] if v['hd95_mm'] is None else round(v['hd95_mm'], 2)} | {v['betti0']:.2f} | {v['coverage']:.4f} |"
        )
    min_chk = min(min(c.values()) for c in checks.values())
    L.append(
        f"\nModels: {', '.join(models)}; cases: {len(cases)}; min Dice(own threshold, nnU-Net seg) = {min_chk:.5f}"
    )
    if len(models) == len(MODELS) and not a.max_cases:
        choice, lines = choose(s)
        L += ["", *lines]
        choice.update(
            models=list(models),
            n_cases=len(cases),
            min_threshold_check=min_chk,
            written=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest()
            if PREREG.exists()
            else None,
            log=lines,
        )
        Path(f"{a.out}_choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    Path(f"{a.out}_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-6:]))


def apply_case(case: str, root: str, prefix: str, choice: dict, out: str):
    cfg = choice["config"].split("+")
    ref = nib.load(str(Path(root) / f"{prefix}_{cfg[0]}" / f"{case}.nii.gz"))
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    probs = [fg_prob(Path(root) / f"{prefix}_{m}" / f"{case}.npz") for m in cfg]
    pred = pipeline(probs, choice["low"], choice["T"], spacing)
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(pred.astype(np.uint8), ref.affine, hdr), str(Path(out) / f"{case}.nii.gz")
    )
    return case


def cmd_apply(a):
    choice = json.loads(a.choice.read_text())
    cfg = choice["config"].split("+")
    cases = sorted(p.name[:-4] for p in (a.prob_root / f"{a.prefix}_{cfg[0]}").glob("*.npz"))
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        done = list(
            ex.map(
                apply_case,
                cases,
                [str(a.prob_root)] * len(cases),
                [a.prefix] * len(cases),
                [choice] * len(cases),
                [str(a.out)] * len(cases),
            )
        )
    print(
        f"wrote {len(done)} masks to {a.out} with {choice['config']}, L {choice['low']}, T {choice['T']}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", type=Path, default=Path("/workspace/xwork"))
    s.add_argument("--labels", type=Path, default=Path("/workspace/xwork/val_labels"))
    s.add_argument("--out", default="/workspace/xwork/lumen_access2")
    s.add_argument("--workers", type=int, default=48)
    s.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="smoke run on the first N cases (never writes a choice)",
    )
    p = sub.add_parser("apply")
    p.add_argument("--choice", type=Path, required=True)
    p.add_argument("--prob-root", type=Path, required=True)
    p.add_argument("--prefix", default="test_prob")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    r = sub.add_parser("rechoose")
    r.add_argument("--per-case", type=Path, required=True)
    a = ap.parse_args()
    {"select": cmd_select, "apply": cmd_apply, "rechoose": cmd_rechoose}[a.cmd](a)


if __name__ == "__main__":
    main()
