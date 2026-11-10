"""Branch-name check on ImageCAS-X: rule-based CoroTree names on the out-of-sample CoroSeg predictions vs the expert segment labels.

For every automatic edge with a name, its centreline points are looked up in the expert multi-label lumen mask (nearest labelled voxel
within `near_mm`); the edge's expert label is the majority over its points. Reported per automatic name: number of edges, share whose
majority expert label matches (precision). Reported per expert segment: share of its expert centreline points within 1 mm of any
automatic tree point (coverage) and within 1 mm of a correctly named edge (named coverage), and the share of cases where the segment is
"found" (named coverage >= 0.5). Per case: whether all four main vessels (RCA, LM, LAD, LCx) are named correctly.

  python scripts/branch_name_check_imagecasx.py --trees D:/imagecas-work/imagecas-x/oos_tree --out results/imagecas-x [--workers 8]
Writes <out>/branch_names_800_edges.csv, _segments.csv and branch_names_800.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

X = Path("D:/imagecas-work/imagecas-x/ImageCAS-X_dataset")
EXPERT = {
    1: "LM",
    2: "LAD",
    3: "LCx",
    4: "D1",
    5: "D2",
    6: "OM1",
    7: "OM2",
    8: "Ramus",
    9: "RCA",
    10: "R-PDA",
    11: "R-PLA",
    12: "L-PDA",
    13: "L-PLA",
    14: "Other",
}
AUTO_TO_EXPERT = {
    "RCA": {9},
    "LM": {1},
    "LAD": {2},
    "LCx": {3},
    "D1": {4},
    "D2": {5},
    "OM1": {6},
    "OM2": {7},
    "Ramus": {8},
    "PDA": {10, 12},
}
MAIN = ("RCA", "LM", "LAD", "LCx")


def expert_centerline(cid: str):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pts, labs = [], []
    for side in ("left", "right"):
        r = vtk.vtkPolyDataReader()
        r.SetFileName(str(X / "centerlines" / f"{cid}.coronary_{side}_centerline.vtk"))
        r.Update()
        pd = r.GetOutput()
        pts.append(
            vtk_to_numpy(pd.GetPoints().GetData()).astype(float) * np.array([-1.0, -1.0, 1.0])
        )  # LPS -> RAS
        labs.append(vtk_to_numpy(pd.GetPointData().GetArray("segment_label")).astype(int))
    return np.concatenate(pts), np.concatenate(labs)


def one(args):
    case, tree_p, near_mm = args
    cid = str(int(case[4:]))
    seg = nib.load(str(X / "segmentations" / f"{cid}.coronary.nii.gz"))
    lab = np.asanyarray(seg.dataobj).astype(np.int16)
    idx = np.argwhere(lab > 0)
    vox_mm = (seg.affine[:3, :3] @ idx.T).T + seg.affine[:3, 3]
    kd = cKDTree(vox_mm)
    vox_lab = lab[tuple(idx.T)]
    tree = json.loads(Path(tree_p).read_text(encoding="utf-8"))
    edges_out, named_pts = [], {}  # expert label -> points of correctly named auto edges
    all_pts = []
    for side in ("left", "right"):
        tr = tree["trees"].get(side)
        if not tr:
            continue
        for e in tr["edges"]:
            P = np.asarray(e["points_xyz_mm"], float)
            if len(P) < 2:
                continue
            all_pts.append(P)
            name = e.get("suggested_name") or ""
            key = name.split("?")[0]
            if key not in AUTO_TO_EXPERT:
                continue
            d, j = kd.query(P)
            ok = d <= near_mm
            maj = int(Counter(vox_lab[j[ok]].tolist()).most_common(1)[0][0]) if ok.any() else 0
            correct = maj in AUTO_TO_EXPERT[key]
            edges_out.append(
                dict(
                    case=case,
                    side=side,
                    auto_name=name,
                    auto_key=key,
                    n_points=len(P),
                    near_frac=round(float(ok.mean()), 3),
                    expert_majority=EXPERT.get(maj, "none" if maj == 0 else str(maj)),
                    correct=correct,
                    confidence=e.get("name_confidence"),
                    length_mm=round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 1),
                )
            )
            if correct:
                named_pts.setdefault(maj, []).append(P)
    E, L = expert_centerline(cid)
    kd_all = cKDTree(np.concatenate(all_pts)) if all_pts else None
    seg_rows = []
    for l in sorted(set(L.tolist())):
        if l not in EXPERT or l == 14:
            continue
        pts = E[L == l]
        cov = float((kd_all.query(pts)[0] <= 1.0).mean()) if kd_all is not None else 0.0
        covn = (
            float((cKDTree(np.concatenate(named_pts[l])).query(pts)[0] <= 1.0).mean())
            if l in named_pts
            else 0.0
        )
        seg_rows.append(
            dict(
                case=case,
                expert_segment=EXPERT[l],
                n_expert_points=len(pts),
                coverage_any=round(cov, 3),
                coverage_named=round(covn, 3),
                found=covn >= 0.5,
            )
        )
    main_ok = {m: any(r["auto_key"] == m and r["correct"] for r in edges_out) for m in MAIN}
    return (
        edges_out,
        seg_rows,
        dict(
            case=case,
            **{f"{m}_correct": v for m, v in main_ok.items()},
            all_main_correct=all(main_ok.values()),
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trees", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--near-mm", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cases = sorted(p.name for p in a.trees.iterdir() if p.is_dir() and (p / "tree.json").exists())
    edges, segs, per_case, errors = [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, (c, str(a.trees / c / "tree.json"), a.near_mm)): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                e, s, pc = f.result()
                edges += e
                segs += s
                per_case.append(pc)
            except Exception as ex_:  # noqa: BLE001
                errors.append((futs[f], repr(ex_)[:120]))
            if i % 50 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("branch_names_800_edges.csv", edges),
        ("branch_names_800_segments.csv", segs),
        ("branch_names_800_per_case.csv", per_case),
    ):
        with (a.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    L = [
        f"# Branch names: CoroTree rule-based names on out-of-sample CoroSeg predictions vs ImageCAS-X expert segment labels ({len(per_case)} cases, {len(errors)} errors)",
        "",
        "## Precision per automatic name (edge majority expert label matches)",
        "",
        "| automatic name | edges | correct | precision |",
        "|---|---:|---:|---:|",
    ]
    for key in AUTO_TO_EXPERT:
        es = [r for r in edges if r["auto_key"] == key]
        if es:
            L.append(
                f"| {key} | {len(es)} | {sum(r['correct'] for r in es)} | {np.mean([r['correct'] for r in es]):.3f} |"
            )
    L += [
        "",
        "## Expert segments: coverage by the automatic tree (within 1 mm)",
        "",
        "| expert segment | cases with segment | mean coverage (any name) | mean coverage (correct name) | found (named coverage >= 0.5) |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in EXPERT.values():
        ss = [r for r in segs if r["expert_segment"] == lab]
        if ss:
            L.append(
                f"| {lab} | {len(ss)} | {np.mean([r['coverage_any'] for r in ss]):.3f} | {np.mean([r['coverage_named'] for r in ss]):.3f} | {np.mean([r['found'] for r in ss]):.3f} |"
            )
    L += ["", "## Main vessels per case", "", "| | share of cases |", "|---|---:|"]
    for m in MAIN:
        L.append(f"| {m} named correctly | {np.mean([r[f'{m}_correct'] for r in per_case]):.3f} |")
    L.append(f"| all four correct | {np.mean([r['all_main_correct'] for r in per_case]):.3f} |")
    L += [
        "",
        f"Method: automatic edge points matched to the nearest expert-labelled voxel within {a.near_mm} mm, majority label per edge; expert centreline points from the",
        "ImageCAS-X VTK files (LPS converted to RAS). An edge named 'X?' counts under X. PDA accepts R-PDA or L-PDA. Automatic names other than the ten mapped ones are not scored.",
        "The CoroSeg predictions are out-of-sample (official 4-split models, each case predicted by the split whose test set contains it).",
    ]
    if errors:
        L += ["", "Errors: " + "; ".join(f"{c}: {e}" for c, e in errors[:10])]
    (a.out / "branch_names_800.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))
    print("BRANCH-NAMES-DONE", flush=True)


if __name__ == "__main__":
    main()
