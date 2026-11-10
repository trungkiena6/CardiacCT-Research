"""External check of the rule-based EAT estimator on the Visual Lab cardiac-fat database (Rodrigues et al., CMPB 2016;123:109-128;
https://visual.ic.uff.br/en/cardio/ctfat/): 20 non-contrast cardiac CT patients (9 Philips, 11 Siemens), 878 axial slices in all
(median 42 per patient), epicardial (red) and mediastinal (green) fat labelled by a physician and a computer scientist on 8-bit
images windowed to [-200, 500] HU ("Higher Range" set); their epicardial fat volumes: mean 103.6 mL, median 92.2, range 35.6-226.2.

Only the 8-bit images are available (the DICOM originals are no longer downloadable), so HU is reconstructed linearly from the
8-bit level (HU = -200 + v * 700 / 255; values outside [-200, 500] are clipped, lungs become -200) and the voxel spacing is
assumed (--spacing, default 0.35 0.35 3.0 mm: the paper rescales every slice to a 0.35 mm pixel spacing; the slice thickness is
not given). Evaluation is therefore 2D/pixel based (per-patient Dice over all labelled slices), and the mm-based rule parameters
depend on the assumed spacing.

Stages:
  build   : BMP stacks -> <work>/<pat>.ct.nii.gz (HU int16) and <work>/<pat>.gt.nii.gz (1 EAT, 2 mediastinal, 3 pericardium line)
  totalseg: TotalSegmentator total --fast --ml on each volume -> <work>/<pat>.ts.nii.gz
  eval    : rule-based EAT on each volume (z-range = all slices, no LM landmark), Dice vs GT. Pre-registered protocol: the
            parameter grid is scored on the first 10 patients (alphabetical) only; the chosen setting and the default are then
            reported on the last 10.

  python scripts/visuallab_eat_eval.py build
  python scripts/visuallab_eat_eval.py totalseg
  python scripts/visuallab_eat_eval.py eval [--grid]

Set VISUALLAB_WORK to the private root containing Ground Truth - Higher Range
and work/. TOTAL_SEGMENTATOR_EXE can select the TotalSegmentator executable.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eat_rule_based import P, compute_eat  # noqa: E402

VL = Path(os.environ.get("VISUALLAB_WORK", "D:/imagecas-work/visuallab"))
GT_DIR = VL / "Ground Truth - Higher Range"
WORK = VL / "work"
TS_EXE = Path(
    os.environ.get("TOTAL_SEGMENTATOR_EXE", "D:/imagecas-work/tsvenv/Scripts/TotalSegmentator.exe")
)
GRID = dict(
    root_mm=(5.0, 8.0),
    sac_dilate_mm=(0.0, 2.0, 4.0),
    cap_mm=(12.0, 20.0, 30.0),
    inf_ext_mm=(0.0, 10.0),
)


def discover_patients(directory: Path | None = None) -> list[str]:
    """Delay access to private patient folders until after CLI argument parsing."""
    directory = GT_DIR if directory is None else directory
    patients = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not patients:
        raise ValueError(f"No patient directories found in {directory}")
    return patients


def build(spacing, patients=None) -> None:
    patients = discover_patients() if patients is None else patients
    WORK.mkdir(exist_ok=True)
    for pat in patients:
        files = sorted((GT_DIR / pat).glob("*.bmp"))
        hu, gt = [], []
        for f in files[
            ::-1
        ]:  # slice 0001 is the most superior -> reverse so z increases superiorly
            a = np.array(Image.open(f).convert("RGB")).astype(np.int16)
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            v = a.max(axis=-1)
            hu.append(-200 + v.astype(np.float32) * 700.0 / 255.0)
            lab = np.zeros(v.shape, np.uint8)
            lab[(r > 0) & (g == 0) & (b == 0)] = 1  # red   = epicardial fat
            lab[(g > 0) & (r == 0) & (b == 0)] = 2  # green = mediastinal fat
            lab[(r == 0) & (g > 0) & (b > 0)] = 3  # cyan  = pericardium line (blue overlay)
            lab[(b > 0) & (r == 0) & (g == 0)] = 3
            gt.append(lab)
        for name, arr in (
            ("ct", np.stack(hu).round().astype(np.int16)),
            ("gt", np.stack(gt)),
        ):  # not ".img": SimpleITK would write Analyze
            im = sitk.GetImageFromArray(arr)
            im.SetSpacing([float(s) for s in spacing])
            sitk.WriteImage(im, str(WORK / f"{pat}.{name}.nii.gz"))
        print(
            f"{pat}: {len(files)} slices, EAT px {int((np.stack(gt) == 1).sum())}, mediastinal px {int((np.stack(gt) == 2).sum())}",
            flush=True,
        )


def totalseg(patients=None) -> None:
    patients = discover_patients() if patients is None else patients
    for pat in patients:
        o = WORK / f"{pat}.ts.nii.gz"
        if o.exists():
            continue
        r = subprocess.run(
            [
                str(TS_EXE),
                "-i",
                str(WORK / f"{pat}.ct.nii.gz"),
                "-o",
                str(o),
                "--ml",
                "--fast",
                "-d",
                "gpu",
                "-ns",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        print(
            pat, "ok" if o.exists() else f"FAILED rc={r.returncode}: {r.stderr[-400:]}", flush=True
        )


def dice(a, b):
    return 2.0 * (a & b).sum() / max(1, a.sum() + b.sum())


def load(pat: str):
    hu = sitk.GetArrayFromImage(sitk.ReadImage(str(WORK / f"{pat}.ct.nii.gz")))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(WORK / f"{pat}.gt.nii.gz")))
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(WORK / f"{pat}.ts.nii.gz")))
    return hu, gt, lab, {}


def score(pat: str, vol, spacing, p: dict) -> dict:
    hu, gt, lab, cache = vol
    eat, sac, med = compute_eat(hu, lab, spacing[::-1], 0, hu.shape[0] - 1, p, cache)
    g_eat, g_med = gt == 1, gt == 2
    g_fat = g_eat | g_med
    inside = sac & g_fat  # GT fat voxels the rule places inside the pericardium
    line = gt == 3  # the drawn pericardium line hides the fat under it: ignore those pixels
    return {
        "patient": pat,
        "slices": hu.shape[0],
        "dice_eat": round(dice(eat, g_eat), 4),
        "dice_eat_ign_line": round(dice(eat & ~line, g_eat), 4),
        "dice_med": round(dice(med, g_med), 4),
        "gt_eat_px": int(g_eat.sum()),
        "rule_eat_px": int(eat.sum()),
        "vol_ratio": round(eat.sum() / max(1, g_eat.sum()), 3),
        "fat_inside_acc": round(float(((inside == g_eat) & g_fat).sum() / max(1, g_fat.sum())), 4),
        "eat_sens": round(float((eat & g_eat).sum() / max(1, g_eat.sum())), 4),
        "eat_ppv": round(float((eat & g_eat).sum() / max(1, eat.sum())), 4),
        "heart_found": bool((lab == 51).any()),
    }


def summary(tag: str, rs: list[dict]) -> str:
    m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    return (
        f"{tag}: Dice EAT {m('dice_eat'):.3f} (sens {m('eat_sens'):.3f} ppv {m('eat_ppv'):.3f}, vol-ratio {m('vol_ratio'):.2f}) | "
        f"Dice mediastinal {m('dice_med'):.3f} | fat-inside-acc {m('fat_inside_acc'):.3f} | n={len(rs)}"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _grid_job(args):
    vals, spacing, tune_patients = args
    keys = list(GRID)
    p = dict(P, **dict(zip(keys, vals)))
    rs = [score(pat, load(pat), spacing, p) for pat in tune_patients]
    return {
        "dice_eat_tune": float(np.mean([r["dice_eat"] for r in rs])),
        "ppv_tune": float(np.mean([r["eat_ppv"] for r in rs])),
        "sens_tune": float(np.mean([r["eat_sens"] for r in rs])),
        **dict(zip(keys, vals)),
    }


def evaluate(spacing, grid: bool, workers: int, patients=None) -> None:
    patients = discover_patients() if patients is None else patients
    tune_patients, test_patients = patients[:10], patients[10:]
    rows = [score(pat, load(pat), spacing, P) for pat in patients]
    write_csv(WORK / "visuallab_default_per_patient.csv", rows)
    tune = [r for r in rows if r["patient"] in tune_patients]
    test = [r for r in rows if r["patient"] in test_patients]
    print(
        "DEFAULT",
        json.dumps(P),
        "\n  ",
        summary("tune-10", tune),
        "\n  ",
        summary("TEST-10", test),
        flush=True,
    )
    if not grid:
        return
    results, keys = [], list(GRID)
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(
            _grid_job, [(v, spacing, tune_patients) for v in itertools.product(*GRID.values())]
        ):
            results.append(res)
            print(
                f"  grid { ({k: res[k] for k in keys}) }: tune Dice EAT {res['dice_eat_tune']:.4f} (sens {res['sens_tune']:.3f} ppv {res['ppv_tune']:.3f})",
                flush=True,
            )
    best = max(results, key=lambda r: r["dice_eat_tune"])
    p = dict(P, **{k: best[k] for k in keys})
    rs = [score(pat, load(pat), spacing, p) for pat in test_patients]
    write_csv(WORK / "visuallab_tuned_test_per_patient.csv", rs)
    (WORK / "visuallab_choice.json").write_text(
        json.dumps(
            {
                "chosen": p,
                "tune_dice_eat": best["dice_eat_tune"],
                "grid": results,
                "spacing_assumed": list(spacing),
            },
            indent=1,
        )
    )
    print(
        "VISUALLAB-DONE chosen",
        {k: best[k] for k in keys},
        f"(tune Dice {best['dice_eat_tune']:.4f})\n  ",
        summary("TEST-10 tuned", rs),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("build", "totalseg", "eval"))
    ap.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=(0.35, 0.35, 3.0),
        help="x y z mm (assumed; headers unavailable)",
    )
    ap.add_argument("--grid", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="parallel grid settings (each worker holds one volume at a time)",
    )
    a = ap.parse_args()
    patients = discover_patients()
    {
        "build": lambda: build(a.spacing, patients),
        "totalseg": lambda: totalseg(patients),
        "eval": lambda: evaluate(a.spacing, a.grid, a.workers, patients),
    }[a.stage]()


if __name__ == "__main__":
    main()
