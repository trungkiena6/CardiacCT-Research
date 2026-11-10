"""Post hoc paired mask/shell sensitivity with cached expert centrelines, no inference.

All three configurations run on both sources. Outputs contain measurements only;
CT volumes and masks remain at their original restricted local paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Each case is already parallelized by process; avoid BLAS oversubscription.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coronary_tree.pcat.compare import icc21
from coronary_tree.pcat.pcat import FAT_HU, SEGMENTS, _resample, _trunk_polyline

X = Path("D:/imagecas-work/imagecas-x")
CASES = Path("D:/imagecas-work/imagecas/cases")
WORK = Path("D:/imagecas-work/imagecas/work")
CONFIGS = ("diameter_wall05", "diameter_wall00", "surface05_35")


def protocol():
    return {
        "analysis_intent": "post hoc exploratory sensitivity; specified before generating new shell results",
        "mask_sources": ["ImageCAS-X expert lumen", "original ImageCAS whole-vessel"],
        "diameter_wall05": "r + 0.5 < d <= 3r + 0.5 mm (original routine)",
        "diameter_wall00": "r < d <= 3r mm",
        "surface05_35": "0.5 < distance to nearest mask voxel centre <= 3.5 mm, intersected with d <= r + 3.5 mm selected-centreline envelope",
        "surface_distance_caveat": "EDT distance uses voxel centres, not a subvoxel reconstructed surface. The envelope restricts neighbouring branches but does not isolate identical tissue when masks differ.",
        "held_fixed": "CT, expert tree, selected segment and resampling, original 12 mm crop, HU window [-190,-30], radius floor 0.6 mm, heart/aorta exclusion (51/52), exclusion of all coronary-mask voxels",
        "notation": "d is distance to selected resampled expert centreline; r is inner-mask EDT sampled at the nearest centreline point and floored at 0.6 mm",
        "endpoints": [
            "paired whole-minus-lumen FAI bias and two-sided t 95% CI",
            "SD and 95% Bland-Altman limits (bias +/- 1.96 SD)",
            "absolute agreement ICC(2,1) for FAI and volume",
            "valid paired counts",
            "counts crossing either direction of exploratory -70.1 HU reference",
        ],
        "validity": "finite FAI values from both masks; retain short segments as in baseline and report flags; count errors and missing outputs",
        "interpretation": "Report every configuration regardless of direction. Persistence supports robustness only to tested shell choices; attenuation of bias limits interpretation to routine-specific geometry. Neither isolates label convention from annotation source. Threshold crossings are descriptive, not clinical reclassification; no kVp or FAI correction is available.",
        "training_or_inference": "none; use locally cached expert tree.json and existing CT/masks/frame",
    }


def measure_one(job):
    cid, grp = job
    started = time.perf_counter()
    img = nib.load(str(CASES / grp / f"{cid}.img.nii.gz"))
    ct = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    vox_ml = float(np.prod(spacing)) / 1000.0
    inv = np.linalg.inv(img.affine)
    ha = np.asanyarray(nib.load(str(WORK / cid / "heart-aorta.nii.gz")).dataobj)
    excluded = (ha == 51) | (ha == 52)
    del ha
    masks = {
        "lumen": np.asanyarray(
            nib.load(
                str(X / "ImageCAS-X_dataset" / "segmentations" / f"{cid}.coronary.nii.gz")
            ).dataobj
        )
        > 0,
        "wholevessel": np.asanyarray(nib.load(str(CASES / grp / f"{cid}.label.nii.gz")).dataobj)
        > 0,
    }
    tree = json.loads((X / "pcat_expert" / cid / "tree.json").read_text(encoding="utf-8"))
    rows = []
    for vessel, (a, b) in SEGMENTS.items():
        tr = tree["trees"]["right" if vessel == "RCA" else "left"]
        Q, t = _resample(_trunk_polyline(tr, vessel))
        sel = (t >= a) & (t <= b)
        if len(Q) < 4 or sel.sum() < 4:
            raise ValueError(f"{cid}/{vessel}: no usable expert segment")
        Qs = Q[sel]
        achieved = float(t[sel][-1] - t[sel][0])
        vox = (inv[:3, :3] @ Qs.T).T + inv[:3, 3]
        margin = np.ceil(12.0 / spacing).astype(int)
        lo = np.maximum(np.floor(vox.min(0)).astype(int) - margin, 0)
        hi = np.minimum(np.ceil(vox.max(0)).astype(int) + margin + 1, np.array(ct.shape))
        box = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        gx, gy, gz = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
        G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(float)
        Gmm = (img.affine[:3, :3] @ G.T).T + img.affine[:3, 3]
        d, idx = cKDTree(Qs).query(Gmm)
        shape = ct[box].shape
        for source, mask in masks.items():
            m_c = mask[box]
            inside = ndi.distance_transform_edt(m_c, sampling=spacing)
            vidx = np.clip(np.round(vox).astype(int) - lo, 0, np.array(shape) - 1)
            radius = np.maximum(inside[tuple(vidx.T)], 0.6)
            r = radius[idx]
            outside = ndi.distance_transform_edt(~m_c, sampling=spacing)
            exclusions = m_c | excluded[box]
            shells = {
                "diameter_wall05": ((d > r + 0.5) & (d <= r + 0.5 + 2.0 * r)).reshape(shape),
                "diameter_wall00": ((d > r) & (d <= r + 2.0 * r)).reshape(shape),
                "surface05_35": (outside > 0.5) & (outside <= 3.5) & (d <= r + 3.5).reshape(shape),
            }
            for config, shell in shells.items():
                shell &= ~exclusions
                hu = ct[box][shell]
                fat = (hu >= FAT_HU[0]) & (hu <= FAT_HU[1])
                rows.append(
                    dict(
                        case=cid,
                        vessel=vessel,
                        mask_source=source,
                        config=config,
                        fai_hu=float(hu[fat].mean()) if fat.any() else None,
                        pcat_volume_ml=float(fat.sum() * vox_ml),
                        n_fat_voxels=int(fat.sum()),
                        n_shell_voxels=int(shell.sum()),
                        achieved_mm=achieved,
                        mean_radius_mm=float(radius.mean()),
                        short_segment=bool(achieved < 0.5 * (b - a)),
                        status="ok" if fat.any() else "failed",
                    )
                )
    return rows, time.perf_counter() - started


def summarize(rows):
    result = {}
    for config in CONFIGS:
        result[config] = {}
        for vessel in SEGMENTS:
            selected = [r for r in rows if r["config"] == config and r["vessel"] == vessel]
            sources = {
                s: {
                    r["case"]: r
                    for r in selected
                    if r["mask_source"] == s and r["fai_hu"] is not None
                }
                for s in ("lumen", "wholevessel")
            }
            ids = sorted(set(sources["lumen"]) & set(sources["wholevessel"]))
            if len(ids) < 2:
                continue
            x, y = [
                np.array([sources[s][cid]["fai_hu"] for cid in ids])
                for s in ("lumen", "wholevessel")
            ]
            vx, vy = [
                np.array([sources[s][cid]["pcat_volume_ml"] for cid in ids])
                for s in ("lumen", "wholevessel")
            ]
            delta = y - x
            bias, sd = float(delta.mean()), float(delta.std(ddof=1))
            half = float(student_t.ppf(0.975, len(ids) - 1) * sd / np.sqrt(len(ids)))
            result[config][vessel] = dict(
                n_paired=len(ids),
                n_lumen=len(sources["lumen"]),
                n_wholevessel=len(sources["wholevessel"]),
                lumen_mean_hu=float(x.mean()),
                wholevessel_mean_hu=float(y.mean()),
                bias_hu=bias,
                bias_ci95_hu=[bias - half, bias + half],
                sd_difference_hu=sd,
                loa95_hu=[bias - 1.96 * sd, bias + 1.96 * sd],
                fai_icc21=icc21(x, y),
                volume_icc21=icc21(vx, vy),
                lumen_volume_mean_ml=float(vx.mean()),
                wholevessel_volume_mean_ml=float(vy.mean()),
                lumen_ge_reference=int((x >= -70.1).sum()),
                wholevessel_ge_reference=int((y >= -70.1).sum()),
                crossed_reference=int(((x >= -70.1) != (y >= -70.1)).sum()),
                cross_lumen_high_whole_low=int(((x >= -70.1) & (y < -70.1)).sum()),
                cross_lumen_low_whole_high=int(((x < -70.1) & (y >= -70.1)).sum()),
                n_short_lumen=sum(sources["lumen"][cid]["short_segment"] for cid in ids),
            )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--protocol-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    if args.protocol_only:
        return
    grp = {
        r["case_id"]: r["archive_group"]
        for r in csv.DictReader((ROOT / "configs/imagecas_v2_split_manifest.csv").open())
    }
    ids = args.ids or sorted(
        [
            p.name.split(".")[0]
            for p in (X / "ImageCAS-X_dataset/segmentations").glob("*.coronary.nii.gz")
        ],
        key=int,
    )
    began = time.perf_counter()
    rows, errors, timings = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(measure_one, (cid, grp[cid])): cid for cid in ids}
        for f in as_completed(futs):
            cid = futs[f]
            try:
                measured, seconds = f.result()
                rows.extend(measured)
                timings.append(seconds)
            except Exception as exc:
                errors.append(dict(case=cid, error=repr(exc)))
            n_done = len(timings) + len(errors)
            if n_done % 10 == 0 or n_done == len(ids):
                print(
                    f"{n_done}/{len(ids)}; elapsed {time.perf_counter() - began:.1f}s; errors {len(errors)}",
                    flush=True,
                )
    rows.sort(key=lambda r: (int(r["case"]), r["vessel"], r["mask_source"], r["config"]))
    if rows:
        with (args.out / "per_vessel.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    parity = []
    for source, filename in (
        ("lumen", "pcat_expert_800_per_vessel.csv"),
        ("wholevessel", "pcat_expert_wholevessel_800_per_vessel.csv"),
    ):
        reference = {
            (r["case"], r["vessel"]): r
            for r in csv.DictReader((ROOT / "results/imagecas-x" / filename).open())
        }
        for r in rows:
            if r["mask_source"] != source or r["config"] != "diameter_wall05":
                continue
            ref = reference[(r["case"], r["vessel"])]
            parity.append(
                dict(
                    case=r["case"],
                    vessel=r["vessel"],
                    mask_source=source,
                    fai_absolute_difference=abs(r["fai_hu"] - float(ref["fai_hu"])),
                    volume_absolute_difference=abs(
                        r["pcat_volume_ml"] - float(ref["pcat_volume_ml"])
                    ),
                    shell_count_difference=r["n_shell_voxels"] - int(ref["n_shell_voxels"]),
                )
            )
    run = dict(
        n_requested=len(ids),
        n_completed=len(timings),
        errors=errors,
        workers=args.workers,
        elapsed_seconds=time.perf_counter() - began,
        mean_case_seconds=float(np.mean(timings)) if timings else None,
        max_case_seconds=max(timings) if timings else None,
        baseline_parity_max_fai_hu=max(
            (p["fai_absolute_difference"] for p in parity), default=None
        ),
        baseline_parity_max_volume_ml=max(
            (p["volume_absolute_difference"] for p in parity), default=None
        ),
        baseline_parity_max_abs_shell_voxels=max(
            (abs(p["shell_count_difference"]) for p in parity), default=None
        ),
    )
    (args.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (args.out / "baseline_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
