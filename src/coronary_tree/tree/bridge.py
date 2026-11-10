"""G2.1 - evidence-based bridging of unassigned skeleton fragments.

Evidence volume E in [0, 1] = max(model probability, CT lumen evidence), where the CT term is a soft ramp of the
Hounsfield value between `hu_lo` and `hu_hi` (contrast-filled lumen is bright; the model's softmax is saturated
to 0 in confident misses, so the image itself must supply the evidence. For each fragment (graph component
without a root) the geodesic minimal-cost path to the nearest
assigned tree is computed on cost = 1 / (E + 0.02) with E < e_min impassable, inside a search box (fragment +
tree points within `search_mm`). The 'gap' is the part of the path outside the mask.
Features recorded per candidate: gap_mm, ev_gap_mean, p_gap_mean, hu_gap_mean, align_cos (fragment direction vs
path direction at the fragment end), radius_frag_mm, radius_tree_mm, fragment_len_mm, path_len_mm.
Acceptance rule for validation: gap_mm <= gap_max_mm and ev_gap_mean >= ev_min_mean and align_cos >= align_min.
Accepted bridges become graph edges flagged `bridged` with confidence = ev_gap_mean; the tree construction then runs
unchanged. Against GT (validation only): 'true' bridge = >= 90 % of the path within 1.5 mm of the GT mask;
frag_gt_frac = fraction of the fragment skeleton within 1.5 mm of GT (FP fragments have ~0).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from coronary_tree.tree.build import snap_root
from coronary_tree.tree.skeleton_graph import vox_to_mm

BRIDGE_P = dict(
    e_min=0.05,
    search_mm=25.0,
    margin_mm=5.0,
    gap_max_mm=10.0,
    ev_min_mean=0.3,
    align_min=-1.0,
    min_fragment_mm=2.0,
    max_iter=2,
    hu_lo=100.0,
    hu_hi=350.0,
)


def load_probability(npz_path) -> np.ndarray:
    """nnU-Net validation npz: probabilities (classes, z, y, x) -> foreground probability as (x, y, z)."""
    z = np.load(npz_path)
    return np.ascontiguousarray(z["probabilities"][1].transpose(2, 1, 0)).astype(np.float32)


def evidence_volume(
    prob: np.ndarray | None, ct: np.ndarray | None, hu_lo: float, hu_hi: float
) -> np.ndarray:
    ev = None
    if ct is not None:
        ev = np.clip((ct.astype(np.float32) - hu_lo) / (hu_hi - hu_lo), 0.0, 1.0)
    if prob is not None:
        ev = prob.astype(np.float32) if ev is None else np.maximum(ev, prob)
    return ev


def _component_voxels(G, comp):
    pts = [np.round(d["pts_vox"]).astype(int) for u, v, d in G.edges(comp, data=True)]
    return np.unique(np.concatenate(pts), axis=0) if pts else np.zeros((0, 3), int)


def _direction(pts_mm: np.ndarray, at: np.ndarray, k: int = 6):
    """Unit direction of the k points nearest to `at` (principal axis), oriented away from `at`'s neighbourhood mean."""
    if len(pts_mm) < 3:
        return None
    d = np.linalg.norm(pts_mm - at, axis=1)
    sel = pts_mm[np.argsort(d)[:k]]
    c = sel.mean(0)
    u, s, vt = np.linalg.svd(sel - c, full_matrices=False)
    v = vt[0]
    return v / (np.linalg.norm(v) + 1e-9)


def find_bridge(
    G, frag_nodes, tree_vox_by_side, ev, prob, ct, mask, edt_mm, spacing, affine, params
):
    sp = np.asarray(spacing[:3], float)
    fv = _component_voxels(G, frag_nodes)
    if len(fv) == 0:
        return None
    fv_mm = vox_to_mm(affine, fv)
    near_tv, lo, hi = {}, fv.min(0), fv.max(0)
    best_d = np.inf
    for side, tv in tree_vox_by_side.items():
        if len(tv) == 0:
            continue
        d, _ = cKDTree(fv_mm).query(vox_to_mm(affine, tv))
        best_d = min(best_d, float(d.min()))
        sel = tv[d <= params["search_mm"]]
        if len(sel):
            near_tv[side] = sel
            lo, hi = np.minimum(lo, sel.min(0)), np.maximum(hi, sel.max(0))
    if not near_tv:
        return dict(found=False, nearest_tree_mm=best_d)
    m = np.ceil(params["margin_mm"] / sp).astype(int)
    lo = np.maximum(lo - m, 0)
    hi = np.minimum(hi + m + 1, np.array(ev.shape))
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    e = ev[box].astype(np.float64)
    cost = 1.0 / (e + 0.02)
    cost[e < params["e_min"]] = np.inf
    starts = [tuple(v) for v in (fv - lo)]
    ends, end_side = [], []
    for side, sel in near_tv.items():
        for v in sel - lo:
            ends.append(tuple(v))
            end_side.append(side)
    mcp = MCP_Geometric(cost, sampling=tuple(sp))
    try:
        cum, _ = mcp.find_costs(starts, ends, find_all_ends=False)
    except Exception:  # noqa: BLE001
        return dict(found=False, nearest_tree_mm=best_d, error="mcp")
    end_costs = np.array([cum[t] for t in ends])
    if not np.isfinite(end_costs).any():
        return dict(found=False, nearest_tree_mm=best_d)
    k = int(np.argmin(end_costs))
    path = np.array(mcp.traceback(ends[k])) + lo
    pm = vox_to_mm(affine, path)
    seg = np.linalg.norm(np.diff(pm, axis=0), axis=1)
    idx = tuple(path.T)
    in_mask = mask[idx]
    gap = ~in_mask
    gap_mm = float(seg[gap[1:]].sum()) if len(seg) else 0.0
    ev_path, p_path = ev[idx], (prob[idx] if prob is not None else np.full(len(path), np.nan))
    hu_path = ct[idx] if ct is not None else np.full(len(path), np.nan)
    # alignment: fragment direction at the start vs. path direction leaving the fragment
    fdir = _direction(fv_mm, pm[0])
    n_lead = min(8, len(pm) - 1)
    pdir = pm[n_lead] - pm[0]
    pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
    align = float(abs(np.dot(fdir, pdir))) if fdir is not None and n_lead > 0 else float("nan")
    fidx = tuple(fv.T)
    frag_hu = float(ct[fidx].mean()) if ct is not None else float("nan")
    frag_p = float(prob[fidx].mean()) if prob is not None else float("nan")
    frag_ev = float(ev[fidx].mean())
    # attachment point: distance from the path end to the nearest tree END node (degree <= 1) of that side
    end_mm = pm[-1]
    side_nodes = [n for n in G.nodes if G.degree(n) <= 1]
    d_end = min(
        (np.linalg.norm(G.nodes[n]["mm"] - end_mm) for n in side_nodes), default=float("nan")
    )
    return dict(
        found=True,
        side=end_side[k],
        path_vox=path,
        path_mm=pm,
        path_len_mm=float(seg.sum()),
        gap_mm=gap_mm,
        frag_hu_mean=frag_hu,
        frag_p_mean=frag_p,
        frag_ev_mean=frag_ev,
        attach_end_dist_mm=float(d_end),
        ev_gap_mean=float(ev_path[gap].mean()) if gap.any() else 1.0,
        p_gap_mean=float(np.nanmean(p_path[gap])) if gap.any() else 1.0,
        hu_gap_mean=float(np.nanmean(hu_path[gap])) if gap.any() else float("nan"),
        ev_min=float(ev_path.min()),
        n_gap_voxels=int(gap.sum()),
        align_cos=align,
        radius_frag_mm=float(edt_mm[tuple(path[0])]),
        radius_tree_mm=float(edt_mm[tuple(path[-1])]),
        fragment_len_mm=float(sum(d["length_mm"] for u, v, d in G.edges(frag_nodes, data=True))),
        nearest_tree_mm=best_d,
    )


def gt_inlier_fraction(
    pts_mm: np.ndarray, gt_edt_mm: np.ndarray, affine: np.ndarray, tol_mm: float = 1.5
) -> float:
    inv = np.linalg.inv(affine)
    vox = np.round((inv[:3, :3] @ pts_mm.T).T + inv[:3, 3]).astype(int)
    vox = np.clip(vox, 0, np.array(gt_edt_mm.shape) - 1)
    return float((gt_edt_mm[tuple(vox.T)] <= tol_mm).mean())


def add_bridge_edge(G, br: dict, edt_mm: np.ndarray):
    a = snap_root(G, br["path_mm"][0], 3.0)
    b = snap_root(G, br["path_mm"][-1], 3.0)
    if a is None or b is None or a == b:
        return None
    pts = br["path_vox"]
    r = np.maximum(edt_mm[tuple(pts.T)].astype(np.float32), 0.3)
    G.add_edge(
        a,
        b,
        path_id=f"bridge{G.number_of_edges()}",
        pts_vox=pts.astype(float),
        pts_mm=br["path_mm"],
        length_mm=br["path_len_mm"],
        radius_mm=float(np.median(r)),
        radius_min_mm=float(r.min()),
        radius_samples=r,
        skeleton_id=-1,
        branch_type=-1,
        bridged=True,
        conf=br["ev_gap_mean"],
        gap_mm=br["gap_mm"],
    )
    return (a, b)


def accept_rule(br: dict, params: dict) -> bool:
    return (
        br["gap_mm"] <= params["gap_max_mm"]
        and br["ev_gap_mean"] >= params["ev_min_mean"]
        and (np.isnan(br["align_cos"]) or br["align_cos"] >= params["align_min"])
    )


def bridge_all(
    G,
    result,
    ev,
    prob,
    ct,
    mask,
    edt_mm,
    spacing,
    affine,
    params=BRIDGE_P,
    gt_edt_mm=None,
    accept=True,
):
    """One bridging pass over the unassigned fragments of `result`. Mutates G when accept=True and the rule holds."""
    tree_vox = {
        s: _component_voxels(G, set(nodes)) for s, nodes in result.get("_side_nodes", {}).items()
    }
    records = []
    for comp in result.get("_unknown_comps", []):
        frag_len = float(sum(d["length_mm"] for u, v, d in G.edges(comp, data=True)))
        if frag_len < params["min_fragment_mm"]:
            continue
        br = find_bridge(
            G, set(comp), tree_vox, ev, prob, ct, mask, edt_mm, spacing, affine, params
        )
        rec = dict(
            fragment_len_mm=frag_len,
            found=bool(br and br.get("found")),
            accepted=False,
            added=False,
        )
        if gt_edt_mm is not None:
            rec["frag_gt_frac"] = gt_inlier_fraction(
                vox_to_mm(affine, _component_voxels(G, set(comp))), gt_edt_mm, affine
            )
        if br:
            rec["nearest_tree_mm"] = br.get("nearest_tree_mm")
        if br and br.get("found"):
            rec.update(
                {
                    k: br[k]
                    for k in (
                        "side",
                        "gap_mm",
                        "ev_gap_mean",
                        "p_gap_mean",
                        "hu_gap_mean",
                        "ev_min",
                        "n_gap_voxels",
                        "align_cos",
                        "radius_frag_mm",
                        "radius_tree_mm",
                        "path_len_mm",
                        "frag_hu_mean",
                        "frag_p_mean",
                        "frag_ev_mean",
                        "attach_end_dist_mm",
                    )
                }
            )
            rec["path_mm"] = np.round(
                br["path_mm"][::2], 1
            ).tolist()  # subsampled geometry for reviewers
            if gt_edt_mm is not None:
                rec["gt_inlier_frac"] = gt_inlier_fraction(br["path_mm"], gt_edt_mm, affine)
                rec["gt_inlier_frac_3mm"] = gt_inlier_fraction(
                    br["path_mm"], gt_edt_mm, affine, tol_mm=3.0
                )
                rec["gt_true_strict"] = bool(rec["gt_inlier_frac"] >= 0.9)
                # 'true' bridge: the fragment is real (>= 50 % on GT) and the path mostly follows GT (>= 50 % within 1.5 mm)
                rec["gt_true"] = bool(
                    rec.get("frag_gt_frac", 0.0) >= 0.5 and rec["gt_inlier_frac"] >= 0.5
                )
            ok = accept_rule(br, params)
            rec["accepted"] = bool(ok)
            if ok and accept:
                rec["added"] = add_bridge_edge(G, br, edt_mm) is not None
        records.append(rec)
    return records
