"""Component-size post-processing, selected on validation data only.
Component pruning is measured together with the GT centreline length it removes.

sweep : for each threshold T (mm^3) remove 26-connected components with volume < T from every prediction,
        recompute all metrics against GT and the GT-skeleton length lost relative to the raw prediction;
        write per-case + per-threshold tables and pick T with the validation rule:
          candidates = T whose mean Dice >= raw mean Dice - 0.005 and mean GT-centerline coverage >= raw - 0.005
          choose the candidate with the highest mean clDice; keep T = 0 (no post-processing) unless the gain in
          mean clDice is >= 0.005; ties -> smaller T.
apply : write post-processed masks (uint8, same header) for a prediction folder with a locked T.
Raw predictions are always kept; post-processed masks go to a sibling folder '<pred>_pp<T>'.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow running as a plain script
from coronary_tree.evaluate.segmentation import STRUCT26, case_metrics, load_pair, write_csv_rows  # noqa: E402

THRESHOLDS_MM3 = [0, 5, 10, 20, 50, 100, 200]


def remove_small_components(pred: np.ndarray, spacing, min_mm3: float):
    if min_mm3 <= 0:
        return pred, 0, 0.0
    vox_mm3 = float(np.prod(spacing[:3]))
    lab, n = ndi.label(pred, structure=STRUCT26)
    if n == 0:
        return pred, 0, 0.0
    sizes = ndi.sum(pred, lab, index=np.arange(1, n + 1)) * vox_mm3
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_mm3
    out = keep[lab]
    removed = int((~keep[1:]).sum())
    return out, removed, float(sizes[~keep[1:]].sum() / 1000.0)


def sweep_case(case: str, gt_path: str, pred_path: str, thresholds):
    thresholds = list(thresholds)
    base = dict(case=case, gt_path=gt_path, pred_path=pred_path, status="ok", reason_codes="")

    def failed_rows(status, reason):
        return [dict(base, threshold_mm3=t, status=status, reason_codes=reason) for t in thresholds]

    if not Path(pred_path).is_file():
        return failed_rows("missing_prediction", "missing_prediction")
    try:
        gt_img, pr_img, problems = load_pair(Path(gt_path), Path(pred_path))
        if problems:
            return failed_rows("invalid_geometry", ";".join(problems))
        gt = np.asanyarray(gt_img.dataobj) > 0.5
        pred = np.asanyarray(pr_img.dataobj) > 0.5
        spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
        raw = case_metrics(gt, pred, spacing)
    except Exception as exc:  # retain the case for review rather than selecting on a partial sweep
        return failed_rows("error", f"exception:{type(exc).__name__}:{exc}")
    rows = []
    for t in thresholds:
        try:
            pp, removed, removed_ml = remove_small_components(pred, spacing, t)
            m = raw if t == 0 else case_metrics(gt, pp, spacing)
            rows.append(
                dict(
                    base,
                    threshold_mm3=t,
                    removed_components=removed,
                    removed_volume_ml=removed_ml,
                    gt_skel_len_lost_mm=max(
                        0.0, m["uncovered_gt_skeleton_len_mm"] - raw["uncovered_gt_skeleton_len_mm"]
                    ),
                    **m,
                )
            )
        except Exception as exc:
            rows.append(
                dict(
                    base,
                    threshold_mm3=t,
                    status="error",
                    reason_codes=f"exception:{type(exc).__name__}:{exc}",
                )
            )
    return rows


def summarize(rows, thresholds):
    if 0 not in thresholds:
        raise ValueError("sweep summaries require the raw threshold 0")
    if not rows or any(row.get("status", "ok") != "ok" for row in rows):
        raise ValueError("cannot select a threshold from an empty or invalid sweep")
    out = {}
    for t in thresholds:
        r = [x for x in rows if x["threshold_mm3"] == t]
        hd = [x["hd95_mm"] for x in r if x["hd95_mm"] is not None]
        msd = [x["msd_mm"] for x in r if x["msd_mm"] is not None]
        out[t] = dict(
            n=len(r),
            dice=float(np.mean([x["dice"] for x in r])),
            cldice=float(np.mean([x["cldice"] or 0 for x in r])),
            coverage=float(np.mean([x["gt_centerline_coverage_1mm"] or 0 for x in r])),
            hd95_mm=float(np.mean(hd)) if hd else None,
            n_hd95=len(hd),
            msd_mm=float(np.mean(msd)) if msd else None,
            n_components=float(np.mean([x["n_components_pred"] for x in r])),
            fp_ml=float(np.mean([x["fp_volume_ml"] for x in r])),
            fn_ml=float(np.mean([x["fn_volume_ml"] for x in r])),
            bridge_mm=float(np.mean([x["apparent_bridge_len_mm"] for x in r])),
            gt_len_lost_mm=float(np.mean([x["gt_skel_len_lost_mm"] for x in r])),
            cases_gt_len_lost_gt5mm=int(sum(x["gt_skel_len_lost_mm"] > 5 for x in r)),
            empty_predictions=int(sum(bool(x["pred_empty"]) for x in r)),
        )
    raw = out[0]
    cands = [
        t
        for t in thresholds
        if out[t]["dice"] >= raw["dice"] - 0.005 and out[t]["coverage"] >= raw["coverage"] - 0.005
    ]
    best = max(cands, key=lambda t: (out[t]["cldice"], -t))
    chosen = best if out[best]["cldice"] - raw["cldice"] >= 0.005 else 0
    return out, chosen


def _thresholds(values, include_raw=False):
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("thresholds must be finite and nonnegative")
    return sorted(set(values + ([0.0] if include_raw else [])))


def _write_text(path, content, overwrite):
    with Path(path).open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def cmd_sweep(a):
    a.thresholds = _thresholds(a.thresholds, include_raw=True)
    overwrite = getattr(a, "overwrite", False)
    csv_path, json_path, md_path = [
        Path(f"{a.out_prefix}{suffix}")
        for suffix in ("_per_case.csv", "_summary.json", "_summary.md")
    ]
    if not overwrite and any(path.exists() for path in (csv_path, json_path, md_path)):
        raise FileExistsError("sweep output already exists; use --overwrite to replace it")
    names = sorted(p.name[:-7] for p in a.gt_dir.glob("*.nii.gz"))
    if not names:
        raise ValueError("no ground-truth cases selected for the sweep")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {
            ex.submit(
                sweep_case,
                n,
                str(a.gt_dir / f"{n}.nii.gz"),
                str(a.pred_dir / f"{n}.nii.gz"),
                a.thresholds,
            ): n
            for n in names
        }
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 10 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} cases swept", flush=True)
    rows.sort(key=lambda r: (r["case"], r["threshold_mm3"]))
    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(csv_path, rows, overwrite)
    invalid = [row for row in rows if row["status"] != "ok"]
    if invalid:
        result = dict(
            status="invalid",
            n_cases=len(names),
            n_invalid_rows=len(invalid),
            chosen_threshold_mm3=None,
            thresholds_mm3=a.thresholds,
            per_case_csv=str(csv_path),
        )
        _write_text(json_path, json.dumps(result, indent=1, allow_nan=False) + "\n", overwrite)
        _write_text(
            md_path,
            "Sweep failed: inspect status and reason_codes in the per-case CSV. No threshold was selected.\n",
            overwrite,
        )
        print(f"sweep failed: {len(invalid)} invalid rows retained in {csv_path}", file=sys.stderr)
        return 1
    summary, chosen = summarize(rows, a.thresholds)
    res = dict(
        pred_dir=str(a.pred_dir),
        gt_dir=str(a.gt_dir),
        thresholds_mm3=a.thresholds,
        rule=__doc__.strip(),
        chosen_threshold_mm3=chosen,
        per_threshold=summary,
    )
    _write_text(json_path, json.dumps(res, indent=1, allow_nan=False) + "\n", overwrite)
    L = [
        "| T (mm³) | Dice | clDice | coverage@1mm | HD95 mm | components | FP mL | bridge mm | GT length lost mm | cases lost>5mm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in a.thresholds:
        s = summary[t]
        hd95 = "NA" if s["hd95_mm"] is None else f"{s['hd95_mm']:.2f}"
        L.append(
            f"| {t} | {s['dice']:.4f} | {s['cldice']:.4f} | {s['coverage']:.4f} | {hd95} | {s['n_components']:.1f} | "
            f"{s['fp_ml']:.2f} | {s['bridge_mm']:.1f} | {s['gt_len_lost_mm']:.2f} | {s['cases_gt_len_lost_gt5mm']} |"
        )
    L.append(f"\nChosen threshold (validation rule): **{chosen} mm³**")
    _write_text(md_path, "\n".join(L) + "\n", overwrite)
    print("\n".join(L))
    return 0


def apply_case(src: str, dst: str, min_mm3: float):
    img = nib.load(src)
    pred = np.asanyarray(img.dataobj) > 0.5
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    pp, removed, removed_ml = remove_small_components(pred, spacing, min_mm3)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(pp.astype(np.uint8), img.affine, hdr), dst)
    return dict(case=Path(src).name[:-7], removed_components=removed, removed_volume_ml=removed_ml)


def cmd_apply(a):
    threshold = _thresholds([a.threshold])[0]
    label = str(int(threshold)) if threshold.is_integer() else repr(threshold)
    out = a.pred_dir.parent / f"{a.pred_dir.name}_pp{label}"
    overwrite = getattr(a, "overwrite", False)
    files = sorted(a.pred_dir.glob("*.nii.gz"))
    if not files:
        raise ValueError("no prediction masks selected for post-processing")
    if overwrite and out.is_dir():
        unexpected = {path.name for path in out.glob("*.nii.gz")} - {path.name for path in files}
        if unexpected:
            raise ValueError(
                "existing output contains masks absent from the current source: "
                + ", ".join(sorted(unexpected))
                + "; use a new output directory"
            )
    out.mkdir(parents=True, exist_ok=overwrite)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(
            ex.map(
                apply_case,
                [str(p) for p in files],
                [str(out / p.name) for p in files],
                [a.threshold] * len(files),
            )
        )
    _write_text(
        out / "POSTPROCESS.json",
        json.dumps(dict(source=str(a.pred_dir), min_component_mm3=threshold, cases=res), indent=1)
        + "\n",
        overwrite,
    )
    print(
        f"wrote {len(res)} post-processed masks to {out}; mean removed components {np.mean([r['removed_components'] for r in res]):.2f}"
    )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--gt-dir", type=Path, required=True)
    s.add_argument("--pred-dir", type=Path, required=True)
    s.add_argument("--out-prefix", type=Path, required=True)
    s.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS_MM3)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument(
        "--overwrite", action="store_true", help="explicitly replace existing sweep outputs"
    )
    p = sub.add_parser("apply")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly reuse the post-processed output directory",
    )
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("workers must be positive")
    try:
        return cmd_sweep(a) if a.cmd == "sweep" else cmd_apply(a)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
