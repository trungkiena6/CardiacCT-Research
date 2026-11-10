"""How the original ImageCAS labels (whole vessel) differ from the ImageCAS-X lumen labels, over all 800 re-annotated
cases: volume ratio, Dice, coverage of the lumen by the original label, and how far the extra original voxels lie from
the lumen surface (an estimate of the wall thickness the original annotation included).

  python scripts/label_convention_imagecasx.py --out D:/imagecas-work/imagecas-x/analysis/label_convention_800.csv
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

X = Path("D:/imagecas-work/imagecas-x/ImageCAS-X_dataset")
CASES = Path("D:/imagecas-work/imagecas/cases")
MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "imagecas_v2_split_manifest.csv"


def one(args):
    cid, grp = args
    x = sitk.ReadImage(str(X / "segmentations" / f"{cid}.coronary.nii.gz"))
    xa = sitk.GetArrayFromImage(x) > 0
    o = sitk.GetArrayFromImage(sitk.ReadImage(str(CASES / grp / f"{cid}.label.nii.gz"))) > 0
    sp = x.GetSpacing()
    vox = float(np.prod(sp))
    inter = int((xa & o).sum())
    d = ndi.distance_transform_edt(~xa, sampling=sp[::-1])
    extra = o & ~xa
    return {
        "case": cid,
        "orig_ml": o.sum() * vox / 1000,
        "lumen_ml": xa.sum() * vox / 1000,
        "ratio": o.sum() / max(1, xa.sum()),
        "dice": 2 * inter / max(1, o.sum() + xa.sum()),
        "orig_covers_lumen": inter / max(1, xa.sum()),
        "lumen_in_orig": inter / max(1, o.sum()),
        "extra_median_dist_mm": float(np.median(d[extra])) if extra.any() else 0.0,
        "extra_p90_dist_mm": float(np.percentile(d[extra], 90)) if extra.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    grp = {r["case_id"]: r["archive_group"] for r in csv.DictReader(open(MANIFEST))}
    ids = sorted(
        (p.name.split(".")[0] for p in (X / "segmentations").glob("*.coronary.nii.gz")), key=int
    )
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(c, grp[c]) for c in ids]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    m = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    q = lambda v: tuple(np.percentile(v, [25, 50, 75]).round(3))  # noqa: E731
    print(
        f"LABEL-CONVENTION n={len(rows)} | orig/lumen volume ratio Q1/med/Q3 {q(m('ratio'))} | Dice {q(m('dice'))} | "
        f"orig covers lumen {q(m('orig_covers_lumen'))} | extra-voxel distance to lumen: median {q(m('extra_median_dist_mm'))} mm, "
        f"p90 {q(m('extra_p90_dist_mm'))} mm",
        flush=True,
    )


if __name__ == "__main__":
    main()
