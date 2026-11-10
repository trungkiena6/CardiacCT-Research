"""Per-case mask and topology metrics on the native image grid.

Metrics per case (all computed after geometry verification, prediction on the same grid as GT):
  dice                       voxel Dice; GT non-empty and prediction empty -> 0
  cldice, cl_precision, cl_sensitivity   clDice (Shit et al., CVPR 2021) with 26-connected 3D skeletons
  hd95_mm, msd_mm            95th-percentile symmetric Hausdorff and mean surface distance in mm (null if pred empty)
  fp_volume_ml, fn_volume_ml, gt_volume_ml, pred_volume_ml
  n_components_pred, n_components_gt     26-connectivity components
  gt_centerline_coverage_1mm fraction of GT skeleton voxels within 1 mm of the predicted mask (length proxy)
  pred_centerline_precision_1mm          fraction of predicted skeleton voxels within 1 mm of the GT mask
  gt_skeleton_len_mm, uncovered_gt_skeleton_len_mm, apparent_bridge_len_mm (proxy: predicted skeleton > 1 mm outside GT)
Skeletons: skimage.morphology.skeletonize on the 3D binary volume (Lee 1994 thinning), 26-connectivity.
Distances: scipy EDT with physical spacing, inside a bounding box that contains both masks plus margin
(exact for surface/skeleton points because every candidate nearest surface voxel lies inside the box).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

STRUCT26 = np.ones((3, 3, 3), dtype=bool)
TOL_MM = 1.0
AFFINE_TOL = 1e-3
MINIMUM_CSV_FIELDS = (
    "case",
    "status",
    "dice",
    "cldice",
    "hd95_mm",
    "n_components_gt",
    "n_components_pred",
    "gt_path",
    "pred_path",
    "reason_codes",
)


def _bbox(mask_a, mask_b, margin):
    both = mask_a | mask_b
    if not both.any():
        return None
    idx = np.where(both)
    lo = [max(int(i.min()) - margin, 0) for i in idx]
    hi = [min(int(i.max()) + margin + 1, s) for i, s in zip(idx, both.shape)]
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def _surface(mask):
    return mask & ~ndi.binary_erosion(
        mask, structure=ndi.generate_binary_structure(3, 1), border_value=0
    )


def _skeleton_length_mm(skel, spacing):
    """Approximate polyline length: sum over skeleton voxels of half the distance to each 26-neighbour."""
    if not skel.any():
        return 0.0
    total = 0.0
    sp = np.asarray(spacing, dtype=float)
    padded = np.pad(skel, 1)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                shifted = padded[
                    1 + dz : padded.shape[0] - 1 + dz,
                    1 + dy : padded.shape[1] - 1 + dy,
                    1 + dx : padded.shape[2] - 1 + dx,
                ]
                n_pairs = int((skel & shifted).sum())
                total += n_pairs * 0.5 * float(np.sqrt(((np.array([dz, dy, dx]) * sp) ** 2).sum()))
    return total  # each undirected pair counted twice x 0.5


def case_metrics(gt: np.ndarray, pred: np.ndarray, spacing, tol_mm: float = TOL_MM) -> dict:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    spacing = tuple(float(s) for s in spacing[:3])
    vox_ml = float(np.prod(spacing)) / 1000.0
    m = {}
    n_gt, n_pred, n_inter = int(gt.sum()), int(pred.sum()), int((gt & pred).sum())
    m["gt_voxels"], m["pred_voxels"] = n_gt, n_pred
    m["gt_volume_ml"], m["pred_volume_ml"] = n_gt * vox_ml, n_pred * vox_ml
    m["fp_volume_ml"] = int((pred & ~gt).sum()) * vox_ml
    m["fn_volume_ml"] = int((gt & ~pred).sum()) * vox_ml
    m["dice"] = (2.0 * n_inter / (n_gt + n_pred)) if (n_gt + n_pred) else 1.0
    m["pred_empty"] = n_pred == 0
    _, m["n_components_gt"] = ndi.label(gt, structure=STRUCT26)
    _, m["n_components_pred"] = ndi.label(pred, structure=STRUCT26)

    margin = int(np.ceil(max(tol_mm, 5.0) / min(spacing))) + 2
    box = _bbox(gt, pred, margin)
    if box is None:
        m.update(
            cldice=None,
            cl_precision=None,
            cl_sensitivity=None,
            hd95_mm=None,
            msd_mm=None,
            gt_centerline_coverage_1mm=None,
            pred_centerline_precision_1mm=None,
            gt_skeleton_len_mm=0.0,
            uncovered_gt_skeleton_len_mm=0.0,
            apparent_bridge_len_mm=0.0,
        )
        return m
    g, p = gt[box], pred[box]

    skel_g = skeletonize(g).astype(bool) if g.any() else np.zeros_like(g)
    skel_p = skeletonize(p).astype(bool) if p.any() else np.zeros_like(p)
    n_sg, n_sp = int(skel_g.sum()), int(skel_p.sum())
    tprec = int((skel_p & g).sum()) / n_sp if n_sp else 0.0
    tsens = int((skel_g & p).sum()) / n_sg if n_sg else 0.0
    m["cl_precision"], m["cl_sensitivity"] = tprec, tsens
    m["cldice"] = (2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) > 0 else 0.0

    # distance maps (mm) to the masks and to the surfaces
    dist_to_gt = ndi.distance_transform_edt(~g, sampling=spacing)
    m["gt_skeleton_len_mm"] = _skeleton_length_mm(skel_g, spacing)
    if n_sp:
        dist_to_pred = ndi.distance_transform_edt(~p, sampling=spacing)
        cover = dist_to_pred[skel_g] <= tol_mm
        m["gt_centerline_coverage_1mm"] = float(cover.mean()) if n_sg else None
        m["uncovered_gt_skeleton_len_mm"] = (
            m["gt_skeleton_len_mm"] * (1.0 - float(cover.mean())) if n_sg else 0.0
        )
        outside = dist_to_gt[skel_p] > tol_mm
        m["pred_centerline_precision_1mm"] = float(1.0 - outside.mean())
        m["apparent_bridge_len_mm"] = _skeleton_length_mm(skel_p & (dist_to_gt > tol_mm), spacing)
        sg, sp_ = _surface(g), _surface(p)
        d_gt_surf = ndi.distance_transform_edt(~sg, sampling=spacing)
        d_pred_surf = ndi.distance_transform_edt(~sp_, sampling=spacing)
        d_p2g = d_gt_surf[sp_]
        d_g2p = d_pred_surf[sg]
        m["hd95_mm"] = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
        m["msd_mm"] = float((d_p2g.sum() + d_g2p.sum()) / (d_p2g.size + d_g2p.size))
    else:
        m["gt_centerline_coverage_1mm"] = 0.0 if n_sg else None
        m["uncovered_gt_skeleton_len_mm"] = m["gt_skeleton_len_mm"]
        m["pred_centerline_precision_1mm"] = None
        m["apparent_bridge_len_mm"] = 0.0
        m["hd95_mm"] = None
        m["msd_mm"] = None
    return m


def load_pair(gt_path: Path, pred_path: Path):
    gt_img = nib.load(str(gt_path))
    pr_img = nib.load(str(pred_path))
    problems = []
    if gt_img.shape != pr_img.shape:
        problems.append(f"shape_mismatch gt={gt_img.shape} pred={pr_img.shape}")
    if not np.allclose(gt_img.affine, pr_img.affine, atol=AFFINE_TOL):
        problems.append("affine_mismatch")
    return gt_img, pr_img, problems


def evaluate_case(case: str, gt_path: str, pred_path: str, tol_mm: float) -> dict:
    row = dict(case=case, gt_path=gt_path, pred_path=pred_path, status="ok", reason_codes="")
    try:
        if not Path(pred_path).exists():
            row.update(
                status="missing_prediction",
                reason_codes="missing_prediction",
                dice=0.0,
                pred_empty=True,
            )
            return row
        gt_img, pr_img, problems = load_pair(Path(gt_path), Path(pred_path))
        if problems:
            row.update(status="invalid_geometry", reason_codes=";".join(problems), dice=0.0)
            return row
        gt = np.asanyarray(gt_img.dataobj) > 0.5
        pred = np.asanyarray(pr_img.dataobj) > 0.5
        row.update(case_metrics(gt, pred, gt_img.header.get_zooms(), tol_mm))
        if row["pred_empty"]:
            row["reason_codes"] = "empty_mask"
    except Exception as e:  # noqa: BLE001
        row.update(status="error", reason_codes=f"exception:{type(e).__name__}:{e}")
    return row


def write_csv_rows(path: Path, rows: list[dict], overwrite: bool = False):
    """Keep the evaluation schema even when every case failed; default to exclusive creation."""
    path = Path(path)
    fields = list(MINIMUM_CSV_FIELDS)
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", type=Path, required=True, help="CaseNNNN.nii.gz uint8 labels")
    ap.add_argument("--pred-dir", type=Path, required=True, help="CaseNNNN.nii.gz predictions")
    ap.add_argument("--output", type=Path, required=True, help="per-case CSV")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tol-mm", type=float, default=TOL_MM)
    ap.add_argument(
        "--cases", type=Path, default=None, help="optional text file with one case name per line"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="explicitly replace an existing output CSV"
    )
    a = ap.parse_args(argv)
    if a.output.exists() and not a.overwrite:
        ap.error(f"output already exists: {a.output}; use --overwrite to replace it")
    if a.workers < 1 or not math.isfinite(a.tol_mm) or a.tol_mm < 0:
        ap.error("workers must be positive and tol-mm must be finite and nonnegative")
    if a.cases:
        try:
            names = [
                line.strip()
                for line in a.cases.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError) as exc:
            ap.error(f"cannot read UTF-8 case list: {exc}")
    else:
        names = sorted(p.name[:-7] for p in a.gt_dir.glob("*.nii.gz"))
    if not names:
        ap.error("no cases selected; check gt-dir and the case list")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        ap.error(f"duplicate case IDs: {', '.join(duplicates)}")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {
            ex.submit(
                evaluate_case,
                n,
                str(a.gt_dir / f"{n}.nii.gz"),
                str(a.pred_dir / f"{n}.nii.gz"),
                a.tol_mm,
            ): n
            for n in names
        }
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} cases evaluated", flush=True)
    rows.sort(key=lambda r: r["case"])
    try:
        write_csv_rows(a.output, rows, a.overwrite)
    except OSError as exc:
        ap.error(f"cannot write evaluation CSV: {exc}")
    ok = [r for r in rows if r["status"] == "ok"]
    complete = len(ok) == len(rows)
    cldice_values = [r["cldice"] for r in ok if r.get("cldice") is not None]
    print(
        json.dumps(
            dict(
                n=len(rows),
                n_ok=len(ok),
                mean_dice=float(np.mean([r["dice"] for r in rows])) if complete else None,
                mean_cldice=float(np.mean(cldice_values)) if complete and cldice_values else None,
            ),
            indent=1,
            allow_nan=False,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
