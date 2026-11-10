"""Paired expert-mask control: PCAT from the ImageCAS-X expert centrelines combined
with the original ImageCAS whole-vessel labels instead of the ImageCAS-X lumen
masks. The contrast holds the CT scans, expert centrelines and measurement routine
fixed, but retains both annotation and label-convention differences; it does not
isolate convention alone. Derived from pcat_expert_reference.py: masks + centrelines (segment_label per
point) are converted into the CoroTree tree.json shape that pcat.pcat_case expects, so the same G2c measurement runs on
expert and on automatic anatomy and the two can be compared (ICC / Bland-Altman per vessel).

  python scripts/pcat_expert_wholevessel.py --out D:/imagecas-work/imagecas-x/pcat_expert_wholevessel [--ids 878 879] [--workers 6]

VTK centerlines are LPS mm; pcat works in the NIfTI (RAS) frame, so x and y are negated. Labels: 1 LM, 2 LAD, 3 LCx, 9 RCA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coronary_tree.pcat.pcat import pcat_case  # noqa: E402

X = Path("D:/imagecas-work/imagecas-x/ImageCAS-X_dataset")
CASES = Path("D:/imagecas-work/imagecas/cases")
WORK = Path("D:/imagecas-work/imagecas/work")
MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "imagecas_v2_split_manifest.csv"
VESSELS = {"left": {"LM": 1, "LAD": 2, "LCx": 3}, "right": {"RCA": 9}}


def read_polylines(path: Path):
    """[(points_ras_mm (n,3), segment_label (n,))] per polyline in the VTK."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    r = vtk.vtkPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    pd = r.GetOutput()
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float) * np.array([-1.0, -1.0, 1.0])
    lab = vtk_to_numpy(pd.GetPointData().GetArray("segment_label")).astype(int)
    lines, idl, out = pd.GetLines(), vtk.vtkIdList(), []
    lines.InitTraversal()
    while lines.GetNextCell(idl):
        ids = [idl.GetId(i) for i in range(idl.GetNumberOfIds())]
        out.append((pts[ids], lab[ids]))
    return out


def chain(polys, label: int, start_xyz):
    """Ordered points of every polyline carrying `label`, chained outward from start_xyz by endpoint adjacency."""
    parts = []
    for P, L in polys:
        sel = L == label
        if sel.sum() >= 2:
            parts.append(P[sel])
    if not parts:
        return np.zeros((0, 3))
    ordered, cur = [], np.asarray(start_xyz, float)
    while parts:
        best = min(
            range(len(parts)),
            key=lambda i: min(
                np.linalg.norm(parts[i][0] - cur), np.linalg.norm(parts[i][-1] - cur)
            ),
        )
        P = parts.pop(best)
        if np.linalg.norm(P[-1] - cur) < np.linalg.norm(P[0] - cur):
            P = P[::-1]
        ordered.append(P)
        cur = P[-1]
    return np.concatenate(ordered)


def build_tree(cid: str, edt_radius) -> dict:
    trees = {}
    for side, names in VESSELS.items():
        polys = read_polylines(X / "centerlines" / f"{cid}.coronary_{side}_centerline.vtk")
        allP = np.concatenate([P for P, _ in polys])
        allL = np.concatenate([L for _, L in polys])
        if side == "left":
            lm = allP[allL == 1]
            # the LM starts at the ostium: take the LM point farthest from the LAD/LCx mass as the root
            dist_mass = allP[(allL == 2) | (allL == 3)]
            if len(lm) and len(dist_mass):
                root = lm[np.argmax(np.linalg.norm(lm - dist_mass.mean(0), axis=1))]
            elif len(lm):
                root = lm[0]
            else:
                root = allP[0]
            lm_chain = chain(polys, 1, root)
            lm_end = lm_chain[-1] if len(lm_chain) else root
            edges, nid = [], 1
            if len(lm_chain):
                edges.append(
                    dict(
                        parent_node_id=0,
                        child_node_id=nid,
                        suggested_name="LM",
                        radius_mm=edt_radius(lm_chain),
                        points_xyz_mm=lm_chain.tolist(),
                    )
                )
                lm_node, nid = nid, nid + 1
            else:
                lm_node = 0
            for name, lab in (("LAD", 2), ("LCx", 3)):
                c = chain(polys, lab, lm_end)
                if len(c) >= 2:
                    edges.append(
                        dict(
                            parent_node_id=lm_node,
                            child_node_id=nid,
                            suggested_name=name,
                            radius_mm=edt_radius(c),
                            points_xyz_mm=c.tolist(),
                        )
                    )
                    nid += 1
        else:
            rca = allP[allL == 9]
            root = (
                rca[np.argmax(rca[:, 2])] if len(rca) else allP[0]
            )  # ostium = most cranial RCA point (RAS z up)
            c = chain(polys, 9, root)
            edges = (
                [
                    dict(
                        parent_node_id=0,
                        child_node_id=1,
                        suggested_name="RCA",
                        radius_mm=edt_radius(c),
                        points_xyz_mm=c.tolist(),
                    )
                ]
                if len(c) >= 2
                else []
            )
        trees[side] = dict(root_node_id=0, edges=edges)
    return {
        "case": cid,
        "source": "ImageCAS-X expert centerlines + original whole-vessel label",
        "trees": trees,
    }


def one(args):
    cid, grp, out_dir = args
    out = Path(out_dir) / cid
    out.mkdir(parents=True, exist_ok=True)
    seg = nib.load(str(CASES / grp / f"{cid}.label.nii.gz"))
    lumen = np.asanyarray(seg.dataobj) > 0
    mask_p = out / "lumen.nii.gz"
    if not mask_p.exists():
        nib.save(nib.Nifti1Image(lumen.astype(np.uint8), seg.affine), str(mask_p))
    spacing = np.array(seg.header.get_zooms()[:3], float)
    edt = ndi.distance_transform_edt(lumen, sampling=spacing)
    inv = np.linalg.inv(seg.affine)

    def edt_radius(P):
        v = np.clip(
            np.round((inv[:3, :3] @ P.T).T + inv[:3, 3]).astype(int), 0, np.array(lumen.shape) - 1
        )
        return float(np.maximum(edt[tuple(v.T)], 0.6).mean())

    tree = build_tree(cid, edt_radius)
    tree_p = out / "tree.json"
    tree_p.write_text(json.dumps(tree), encoding="utf-8")
    rows = pcat_case(
        f"X{cid}",
        str(CASES / grp / f"{cid}.img.nii.gz"),
        str(mask_p),
        str(tree_p),
        str(WORK / cid / "heart-aorta.nii.gz"),
        str(out),
    )
    for r in rows:
        r["case"] = cid
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    grp = {r["case_id"]: r["archive_group"] for r in csv.DictReader(open(MANIFEST))}
    ids = a.ids or sorted(
        (p.name.split(".")[0] for p in (X / "segmentations").glob("*.coronary.nii.gz")), key=int
    )
    rows, done = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, (c, grp[c], str(a.out))): c for c in ids}
        for f in as_completed(futs):
            done += 1
            try:
                rows.extend(f.result())
            except Exception as e:  # noqa: BLE001
                rows.append(dict(case=futs[f], vessel="?", status="error", reason=repr(e)[:200]))
            if done % 25 == 0 or done == len(futs):
                print(f"{done}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: (int(r["case"]), r["vessel"]))
    fields = list({k: None for r in rows for k in r})
    with (a.out / "pcat_expert_per_vessel.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for v in ("RCA", "LAD", "LCx"):
        ok = [r for r in rows if r.get("vessel") == v and r.get("fai_hu") not in (None, "")]
        n_all = sum(1 for r in rows if r.get("vessel") == v)
        if ok:
            fai = np.array([float(r["fai_hu"]) for r in ok])
            print(
                f"{v}: ok {len(ok)}/{n_all} | FAI mean {fai.mean():.1f} HU sd {fai.std():.1f} | PCAT vol mean {np.mean([r['pcat_volume_ml'] for r in ok]):.2f} mL | achieved {np.mean([r['achieved_mm'] for r in ok]):.1f} mm"
            )
    print(
        "PCAT-EXPERT-DONE",
        len(ids),
        "cases;",
        sum(1 for r in rows if r.get("status") == "error"),
        "errors",
        flush=True,
    )


if __name__ == "__main__":
    main()
