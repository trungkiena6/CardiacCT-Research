"""Rule-based epicardial adipose tissue (EAT) from CCTA, using TotalSegmentator v2 'total' labels as the anatomical frame and
the coronary tree (LM ostium) as the landmark for the superior limit. No expert pericardium labels exist for ImageCAS, so this
builds software-derived EAT labels (to be QC'd, self-trained and checked against the Visual Lab cardiac-fat database).

Protocol (v2, 12 Oct 2026):
  z-range        = from the inferior end of the heart mask up to LM ostium + `sup_mm` (proxy for the pulmonary artery bifurcation,
                   clipped at the top of the heart mask); the LM ostium comes from the tree.json of the case (expert or CoroSeg).
  roots          = aorta / pulmonary veins / SVC / IVC within `root_mm` of the heart, excluding the descending aorta (aorta within
                   `spine_mm` of the thoracic spine)
  sac            = per-slice 2D convex hull of (heart ∪ roots), dilated by `sac_dilate_mm`, within `cap_mm` of the cardiac mass,
                   minus lungs, bones, oesophagus, trachea, abdominal organs and the non-root great vessels
  EAT            = fat voxels (-190 <= HU <= -30) inside the sac that are connected to the heart surface (within 3 mm)
  mediastinal fat (reported, not EAT) = fat between the lungs, outside the sac, same z-range

  python scripts/eat_rule_based.py --out D:/imagecas-work/imagecas-x/eat_rule [--ids 1 2] [--png] [--workers 3]
Writes <out>/<id>.eat.nii.gz (1 = EAT), <out>/<id>.sac.nii.gz, <out>/eat_rule_per_case.csv and optional preview PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

CASES = Path("D:/imagecas-work/imagecas/cases")
TS = Path("D:/imagecas-work/imagecas-x/totalseg")
TREES = Path("D:/imagecas-work/imagecas-x/pcat_expert")  # <id>/tree.json (expert anatomy)
MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "imagecas_v2_split_manifest.csv"
HEART, AORTA, PV, BCT, SVC, IVC = 51, 52, 53, 54, 62, 63
VESSELS = (AORTA, PV, BCT, SVC, IVC)
LUNGS = (10, 11, 12, 13, 14)
SPINE = tuple(range(32, 51)) + (79,)  # vertebrae T12…C1 + spinal cord
EXCLUDE = (
    (1, 5, 6, 15, 16, 17, 18, 19, 20) + SPINE + tuple(range(92, 118))
)  # organs, spine, ribs, sternum, costal cartilage
FAT = (-190.0, -30.0)
P = dict(sup_mm=35.0, root_mm=8.0, spine_mm=20.0, sac_dilate_mm=4.0, cap_mm=20.0, touch_mm=3.0)


def slicewise_hull(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        if mask[z].any():
            out[z] = convex_hull_image(mask[z])
    return out


def within_mm(mask: np.ndarray, mm: float, spacing) -> np.ndarray:
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= mm if mask.any() else mask.copy()


def lm_ostium_z(cid: str, img: sitk.Image):
    p = TREES / cid / "tree.json"
    if not p.exists():
        return None
    edges = json.loads(p.read_text(encoding="utf-8"))["trees"]["left"]["edges"]
    if not edges:
        return None
    x, y, z = edges[0]["points_xyz_mm"][0]  # RAS mm -> LPS for SimpleITK
    return img.TransformPhysicalPointToIndex([float(-x), float(-y), float(z)])[2]


def compute_eat(
    hu: np.ndarray,
    lab: np.ndarray,
    sp,
    z_bot: int,
    z_top: int,
    p: dict = P,
    cache: dict | None = None,
):
    """Core rule on (z, y, x) arrays whose z index increases towards superior. Returns (eat, sac, mediastinal_fat) bool arrays.
    `cache` (per volume) keeps the parameter-independent distance maps so a parameter grid re-runs cheaply."""
    cache = {} if cache is None else cache
    heart = lab == HEART
    if "d_heart" not in cache:
        cache["d_heart"] = ndi.distance_transform_edt(~heart, sampling=sp)
        spine = np.isin(lab, SPINE)
        cache["d_spine"] = (
            ndi.distance_transform_edt(~spine, sampling=sp)
            if spine.any()
            else np.full(heart.shape, np.inf, np.float32)
        )
    d_heart, d_spine = cache["d_heart"], cache["d_spine"]
    zmask = np.zeros_like(heart)
    zmask[z_bot : z_top + 1] = True
    near = d_heart <= p["root_mm"]
    desc = (d_spine <= p["spine_mm"]) & (lab == AORTA)
    roots = np.isin(lab, VESSELS) & near & ~desc
    mass = heart | roots
    hull = slicewise_hull(mass & zmask)
    inf_ext = int(round(p.get("inf_ext_mm", 0.0) / sp[0]))
    if inf_ext > 0:  # continue the sac below the heart mask onto the diaphragm
        zh = int(
            np.nonzero((heart & zmask).any(axis=(1, 2)))[0].min()
        )  # lowest heart slice inside the z-range
        ref = hull[min(z_top, zh + int(round(10.0 / sp[0])))]
        for z in range(max(0, zh - inf_ext), zh):
            hull[z] = ref
        zmask[max(0, zh - inf_ext) : zh] = True
    sac = within_mm(hull, p["sac_dilate_mm"], sp) & within_mm(mass, p["cap_mm"], sp) & zmask
    sac &= ~np.isin(lab, LUNGS) & ~np.isin(lab, EXCLUDE) & ~(np.isin(lab, VESSELS) & ~roots)
    fat = (hu >= FAT[0]) & (hu <= FAT[1])
    eat = sac & fat
    comp, n = ndi.label(eat, structure=np.ones((3, 3, 3), bool))
    if n:
        touch = np.unique(comp[(d_heart <= p["touch_mm"]) & eat])
        eat = np.isin(comp, touch[touch > 0])
    lungs = np.isin(lab, LUNGS)
    med = (
        slicewise_hull((lungs | sac) & zmask) & ~lungs & ~sac & zmask & fat & ~np.isin(lab, EXCLUDE)
    )
    return eat, sac, med


def eat_case(cid: str, grp: str, out: Path, png: bool = False, p: dict = P) -> dict:
    img = sitk.ReadImage(str(CASES / grp / f"{cid}.img.nii.gz"))
    hu = sitk.GetArrayFromImage(img).astype(np.int16)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(TS / f"{cid}.nii.gz")))
    if lab.shape != hu.shape:
        raise RuntimeError(f"{cid}: TotalSeg grid {lab.shape} != image {hu.shape}")
    d = img.GetDirection()
    flip = d[8] < 0  # make index z increase towards superior
    if flip:
        hu, lab = hu[::-1], lab[::-1]
    sp = img.GetSpacing()[::-1]  # z, y, x
    vox_ml = float(np.prod(sp)) / 1000.0
    heart = lab == HEART
    if not heart.any():
        raise RuntimeError(f"{cid}: no heart label")
    zs = np.nonzero(heart.any(axis=(1, 2)))[0]
    z_lm = lm_ostium_z(cid, img)
    if z_lm is not None and flip:
        z_lm = hu.shape[0] - 1 - z_lm
    z_top = (
        int(min(zs.max(), z_lm + round(p["sup_mm"] / sp[0]))) if z_lm is not None else int(zs.max())
    )
    z_bot = int(zs.min())
    eat, sac, med = compute_eat(hu, lab, sp, z_bot, z_top, p)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in (("eat", eat), ("sac", sac)):
        a = arr[::-1] if flip else arr
        o = sitk.GetImageFromArray(a.astype(np.uint8))
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out / f"{cid}.{name}.nii.gz"))
    n_comp = int(ndi.label(eat, structure=np.ones((3, 3, 3), bool))[1])
    row = {
        "case": cid,
        "eat_ml": round(eat.sum() * vox_ml, 2),
        "eat_mean_hu": round(float(hu[eat].mean()), 1) if eat.any() else None,
        "mediastinal_fat_ml": round(med.sum() * vox_ml, 2),
        "sac_ml": round(sac.sum() * vox_ml, 1),
        "heart_ml": round(heart.sum() * vox_ml, 1),
        "z_lm_found": z_lm is not None,
        "z_range_mm": round((z_top - z_bot + 1) * sp[0], 1),
        "heart_above_top_mm": round((zs.max() - z_top) * sp[0], 1),
        "n_eat_components": n_comp,
    }
    if png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        zl = [z_bot + int((z_top - z_bot) * f) for f in (0.15, 0.4, 0.65, 0.9)]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5.4))
        ant_up = d[4] < 0  # index y increases towards anterior -> show anterior at top
        for ax, z in zip(axs, zl):
            bg = np.clip((hu[z] + 160) / 560, 0, 1)
            rgb = np.stack([bg] * 3, -1)
            m = sac[z] & ~eat[z]
            rgb[m] = rgb[m] * 0.6 + np.array([0.0, 0.25, 0.45])
            rgb[eat[z]] = (1.0, 0.85, 0.1)
            rgb[med[z]] = (0.2, 0.8, 0.3)
            edge = heart[z] ^ ndi.binary_erosion(heart[z])
            rgb[edge] = (1.0, 0.2, 0.2)
            ax.imshow(rgb[::-1] if ant_up else rgb)
            ax.set_title(f"{cid} z={z} ({(z - z_bot) * sp[0]:.0f} mm above heart bottom)")
            ax.axis("off")
        fig.suptitle(
            f"case {cid}: yellow EAT {row['eat_ml']:.0f} mL | green mediastinal fat {row['mediastinal_fat_ml']:.0f} mL | blue sac | red heart edge | anterior at top"
        )
        fig.savefig(str(out / f"{cid}.preview.png"), dpi=70, bbox_inches="tight")
        plt.close(fig)
    return row


def _job(a):
    return eat_case(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--png-ids", nargs="*", default=[])
    a = ap.parse_args()
    grp = {r["case_id"]: r["archive_group"] for r in csv.DictReader(open(MANIFEST))}
    ids = a.ids or sorted(
        (p.name[:-7] for p in TS.glob("*.nii.gz") if (TREES / p.name[:-7] / "tree.json").exists()),
        key=int,
    )
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_job, (c, grp[c], a.out, a.png or c in a.png_ids)): c for c in ids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:  # noqa: BLE001
                rows.append({"case": futs[f], "error": repr(e)[:160]})
            if i % 25 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: int(r["case"]))
    fields = list({k: None for r in rows for k in r})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "eat_rule_per_case.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "eat_ml" in r]
    if ok:
        v = np.array([r["eat_ml"] for r in ok])
        m = np.array([r["mediastinal_fat_ml"] for r in ok])
        print(
            f"EAT-RULE-DONE n={len(ok)} errors={len(rows) - len(ok)} | EAT mL median {np.median(v):.0f} IQR {np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f} "
            f"range {v.min():.0f}-{v.max():.0f} | mean HU {np.mean([r['eat_mean_hu'] for r in ok if r['eat_mean_hu'] is not None]):.1f} | "
            f"mediastinal median {np.median(m):.0f} mL | z-range median {np.median([r['z_range_mm'] for r in ok]):.0f} mm | "
            f"LM found {sum(r['z_lm_found'] for r in ok)}/{len(ok)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
