"""Ostium (coronary origin) candidates from a coronary mask and the TotalSegmentator heart (51) / aorta (52) masks.

Method (no learning, no GT):
  1. aortic root region = aorta voxels within `root_reach_mm` of the heart mask (excludes arch / descending aorta)
  2. candidate voxels = coronary voxels within `touch_mm` of that region; 26-connected clusters >= `min_voxels`
  3. each cluster: centroid in RAS mm; local aorta centre = centroid of aorta voxels within +-5 slices of the cluster z;
     side score s = (dx + dy)/sqrt(2) in RAS (right ostium sits right-anterior of the root, left ostium left-posterior)
     -> side = right if s > 0 else left; confidence = min(1, |s| / 10 mm)
Returns a list of candidates (side, xyz_mm, xyz_vox, n_voxels, confidence, side_score_mm, dist_to_root_mm).
Selection among several candidates on one side is done later with the skeleton (longest attached tree wins).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def aortic_root_region(
    heart: np.ndarray, aorta: np.ndarray, spacing, root_reach_mm: float = 12.0
) -> np.ndarray:
    if not heart.any() or not aorta.any():
        return np.zeros_like(aorta, dtype=bool)
    d_heart = ndi.distance_transform_edt(~heart, sampling=spacing)
    return aorta & (d_heart <= root_reach_mm)


def local_root_centre(
    root: np.ndarray, cluster: np.ndarray, spacing, xy_margin_mm: float = 25.0, z_half: int = 5
):
    """Centroid of the aortic-root component nearest to the candidate cluster, inside a slab of +-z_half slices
    around the cluster and +-xy_margin_mm in-plane (excludes the descending aorta seen in the same slices)."""
    idx = np.where(cluster)
    if len(idx[0]) == 0:
        return None
    mx, my = int(np.ceil(xy_margin_mm / spacing[0])), int(np.ceil(xy_margin_mm / spacing[1]))
    x0, x1 = max(idx[0].min() - mx, 0), min(idx[0].max() + mx + 1, root.shape[0])
    y0, y1 = max(idx[1].min() - my, 0), min(idx[1].max() + my + 1, root.shape[1])
    zc = int(round(idx[2].mean()))
    z0, z1 = max(zc - z_half, 0), min(zc + z_half + 1, root.shape[2])
    box = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    sub_root, sub_c = root[box], cluster[box]
    if not sub_root.any():
        return None
    lab, n = ndi.label(sub_root, structure=STRUCT26)
    d = ndi.distance_transform_edt(~sub_c, sampling=spacing)
    best, best_d = None, np.inf
    for k in range(1, n + 1):
        dk = d[lab == k].min()
        if dk < best_d:
            best, best_d = k, dk
    c = np.array(ndi.center_of_mass(lab == best))
    return c + np.array([x0, y0, z0], dtype=float)


def ostium_candidates(
    mask: np.ndarray,
    heart: np.ndarray,
    aorta: np.ndarray,
    spacing,
    affine: np.ndarray,
    touch_mm: float = 2.5,
    min_voxels: int = 30,
    root_reach_mm: float = 12.0,
) -> dict:
    spacing = tuple(float(s) for s in spacing[:3])
    root = aortic_root_region(heart, aorta, spacing, root_reach_mm)
    out = dict(root_region_voxels=int(root.sum()), candidates=[], reason_codes=[])
    if not root.any():
        out["reason_codes"].append("no_aortic_root_region")
        return out
    d_root = ndi.distance_transform_edt(~root, sampling=spacing)
    cand = mask & (d_root <= touch_mm)
    lab, n = ndi.label(cand, structure=STRUCT26)
    if n == 0:
        out["reason_codes"].append("no_ostium_candidate")
        return out
    sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1))
    for i in range(n):
        if sizes[i] < min_voxels:
            continue
        c_vox = np.array(ndi.center_of_mass(cand, lab, i + 1))
        c_mm = (affine @ np.r_[c_vox, 1.0])[:3]
        a_vox = local_root_centre(root, lab == i + 1, spacing)
        if a_vox is None:
            a_vox = np.array(ndi.center_of_mass(root))
        a_mm = (affine @ np.r_[a_vox, 1.0])[:3]
        dx, dy, dz = (c_mm - a_mm).tolist()
        s = (dx + dy) / np.sqrt(2.0)
        side = "right" if s > 0 else "left"
        out["candidates"].append(
            dict(
                side=side,
                xyz_mm=[float(v) for v in c_mm],
                xyz_vox=[float(v) for v in c_vox],
                n_voxels=int(sizes[i]),
                side_score_mm=float(s),
                confidence=float(min(1.0, abs(s) / 10.0)),
                offset_from_aorta_centre_mm=[float(dx), float(dy), float(dz)],
                dist_to_root_mm=float(d_root[tuple(np.round(c_vox).astype(int))]),
            )
        )
    out["candidates"].sort(key=lambda c: -c["n_voxels"])
    if not out["candidates"]:
        out["reason_codes"].append("no_ostium_candidate")
    return out
