"""G2c prototype - pericoronary adipose tissue (PCAT) attenuation / volume per vessel from a CoroTree tree.json with
branch-name suggestions (RCA / LM / LAD / LCx) and the CT.

Definition (after Antonopoulos 2017 / CRISP-CT): along the vessel centreline, voxels whose distance to the centreline
lies in (r_outer, r_outer + 2 r_lumen], where r_lumen is the local lumen radius from the segmentation (EDT) and
r_outer = r_lumen + wall_mm (0.5 mm, the wall is not segmented), excluding any coronary-mask voxel, heart and
aorta voxels (TotalSegmentator 51/52); adipose voxels are those with -190 <= HU <= -30. Segments: RCA 10-50 mm from
the ostium (root), LAD and LCx 0-40 mm from the LM bifurcation (end of the LM chain). FAI = mean HU of the adipose
voxels; PCAT volume = adipose voxel count x voxel volume. kVp is not available in ImageCAS NIfTI, so no kVp
correction is applied. Everything here is a research prototype (review_required = true); branch names are rule
suggestions unless confirmed by a reader.
Output: <out>/<case>_pcat.json and <out>/pcat_per_vessel.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FAT_HU = (-190.0, -30.0)
WALL_MM = 0.5
SEGMENTS = dict(RCA=(10.0, 50.0), LAD=(0.0, 40.0), LCx=(0.0, 40.0))


def _trunk_polyline(tree: dict, name: str):
    """Ordered points (mm) of the chain of edges named `name`, starting at the root (RCA) or where LM ends (LAD/LCx)."""
    edges = tree["edges"]
    by_parent = defaultdict(list)
    for e in edges:
        by_parent[e["parent_node_id"]].append(e)
    if name == "RCA":
        node = tree["root_node_id"]
    else:
        node = tree["root_node_id"]
        # follow LM edges to their end
        while True:
            nxt = [e for e in by_parent.get(node, []) if e.get("suggested_name") == "LM"]
            if not nxt:
                break
            node = nxt[0]["child_node_id"]
    pts = []
    while True:
        nxt = [e for e in by_parent.get(node, []) if e.get("suggested_name") == name]
        if not nxt:
            break
        e = max(nxt, key=lambda x: x["radius_mm"])
        P = np.asarray(e["points_xyz_mm"])
        pts.append(P if not pts else P[1:])
        node = e["child_node_id"]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _resample(P: np.ndarray, step: float = 0.25):
    if len(P) < 2:
        return P, np.zeros(len(P))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1], step)
    Q = np.stack([np.interp(t, s, P[:, i]) for i in range(3)], 1)
    return Q, t


def pcat_case(
    case: str, image_p: str, mask_p: str, tree_p: str, ha_p: str, out_dir: str
) -> list[dict]:
    img = nib.load(image_p)
    ct = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = np.array([float(s) for s in img.header.get_zooms()[:3]])
    vox_ml = float(np.prod(spacing)) / 1000.0
    inv = np.linalg.inv(img.affine)
    mask = np.asanyarray(nib.load(mask_p).dataobj) > 0.5
    ha = np.asanyarray(nib.load(ha_p).dataobj)
    excl = mask | (ha == 51) | (ha == 52)
    tree = json.loads(Path(tree_p).read_text(encoding="utf-8"))
    rows = []
    for vessel, (a, b) in SEGMENTS.items():
        side = "right" if vessel == "RCA" else "left"
        row = dict(case=case, vessel=vessel, requested_mm=[a, b], status="ok", reason="")
        tr = tree["trees"].get(side)
        if not tr:
            row.update(status="failed", reason=f"missing_{side}_tree")
            rows.append(row)
            continue
        P = _trunk_polyline(tr, vessel)
        Q, t = _resample(P)
        if len(Q) < 4:
            row.update(status="failed", reason="vessel_not_named_or_too_short")
            rows.append(row)
            continue
        sel = (t >= a) & (t <= b)
        if sel.sum() < 4:
            row.update(status="failed", reason="segment_out_of_range", trunk_len_mm=float(t[-1]))
            rows.append(row)
            continue
        Qs = Q[sel]
        achieved = float(t[sel][-1] - t[sel][0])
        # crop around the segment
        vox = (inv[:3, :3] @ Qs.T).T + inv[:3, 3]
        margin = np.ceil(12.0 / spacing).astype(int)
        lo = np.maximum(np.floor(vox.min(0)).astype(int) - margin, 0)
        hi = np.minimum(np.ceil(vox.max(0)).astype(int) + margin + 1, np.array(mask.shape))
        box = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        m_c, ct_c, ex_c = mask[box], ct[box], excl[box]
        edt = ndi.distance_transform_edt(
            m_c, sampling=spacing
        )  # inside-mask distance = local lumen radius
        vidx = np.clip(np.round(vox).astype(int) - lo, 0, np.array(m_c.shape) - 1)
        r_lumen = edt[tuple(vidx.T)]
        r_lumen = np.maximum(r_lumen, 0.6)  # thin/partial-volume floor (mm)
        # distance of every crop voxel to the segment centreline
        gx, gy, gz = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
        G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(float)
        Gmm = (img.affine[:3, :3] @ G.T).T + img.affine[:3, 3]
        d, idx = cKDTree(Qs).query(Gmm)
        r_out = r_lumen[idx] + WALL_MM
        shell = (d > r_out) & (d <= r_out + 2.0 * r_lumen[idx])
        shell = shell.reshape(m_c.shape) & ~ex_c
        hu = ct_c[shell]
        fat = (hu >= FAT_HU[0]) & (hu <= FAT_HU[1])
        row.update(
            trunk_len_mm=float(t[-1]),
            achieved_mm=achieved,
            n_shell_voxels=int(shell.sum()),
            n_fat_voxels=int(fat.sum()),
            fat_fraction=float(fat.mean()) if shell.any() else None,
            fai_hu=float(hu[fat].mean()) if fat.any() else None,
            fai_sd_hu=float(hu[fat].std()) if fat.any() else None,
            pcat_volume_ml=float(fat.sum() * vox_ml),
            mean_lumen_radius_mm=float(r_lumen.mean()),
            short_segment=bool(achieved < 0.5 * (b - a)),
        )
        if row["short_segment"]:
            row["status"] = "needs_review"
            row["reason"] = "short_segment"
        if row["fai_hu"] is None:
            row["status"] = "failed"
            row["reason"] = "no_adipose_voxels"
        rows.append(row)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{case}_pcat.json").write_text(
        json.dumps(
            dict(
                case=case,
                review_required=True,
                method=__doc__.strip().splitlines()[0],
                wall_mm=WALL_MM,
                fat_hu=FAT_HU,
                kvp="unknown (NIfTI)",
                vessels=rows,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--trees", type=Path, required=True)
    ap.add_argument(
        "--heart-aorta-root",
        type=Path,
        required=True,
        help="context-mask root containing <numeric-case-id>/heart-aorta.nii.gz",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    cases = a.cases or sorted(
        p.name for p in a.trees.iterdir() if p.is_dir() and p.name.startswith("Case")
    )
    jobs = [
        (
            c,
            str(a.images / f"{c}_0000.nii.gz"),
            str(a.pred / f"{c}.nii.gz"),
            str(a.trees / c / "tree.json"),
            str(a.heart_aorta_root / str(int(c[4:])) / "heart-aorta.nii.gz"),
            str(a.out),
        )
        for c in cases
    ]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(pcat_case, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.extend(f.result())
            except Exception as e:  # noqa: BLE001
                rows.append(dict(case="?", vessel="?", status="error", reason=str(e)))
            if i % 10 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: (r["case"], r["vessel"]))
    a.out.mkdir(parents=True, exist_ok=True)
    fields = list({k: None for r in rows for k in r})
    with (a.out / "pcat_per_vessel.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for v in SEGMENTS:
        ok = [r for r in rows if r["vessel"] == v and r.get("fai_hu") is not None]
        if ok:
            fai = np.array([r["fai_hu"] for r in ok])
            vol = np.array([r["pcat_volume_ml"] for r in ok])
            print(
                f"{v}: n={len(ok)}/{sum(1 for r in rows if r['vessel'] == v)} FAI mean {fai.mean():.1f} HU (sd {fai.std():.1f}, range {fai.min():.0f}..{fai.max():.0f}); "
                f"PCAT volume mean {vol.mean():.2f} mL; achieved length mean {np.mean([r['achieved_mm'] for r in ok]):.1f} mm; short {sum(r['short_segment'] for r in ok)}"
            )


if __name__ == "__main__":
    main()
