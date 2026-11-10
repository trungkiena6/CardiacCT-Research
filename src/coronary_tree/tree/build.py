"""CoroTree-0.1: rooted left/right coronary trees from a mask (+ ostium candidates), with QC and exports.

Pipeline per case:
  root nodes  : nearest skeleton node (or split point on an edge) to each ostium candidate within `root_snap_mm`
  side choice : per side keep the candidate whose graph component carries the most skeleton length
  bridge      : if both roots share one component, cut the thinnest edge on the root-to-root path (logged, needs_review)
  tree        : BFS from the root; non-tree edges are cycles (removed, logged); leaf edges < `spur_mm` pruned (logged)
  simplify    : pass-through nodes merged into one polyline edge; nodes typed root / junction / end
  unknown     : components without a root are reported as fragments (never deleted silently)
  invariants  : 2 roots, every node reaches its root, no cycles, no left-right edge, polylines inside the mask
                (<= one voxel diagonal), finite positive lengths/radii
  status      : failed (empty mask / no skeleton) | needs_review (reason codes) | complete_candidate
Confidence of an edge is an uncalibrated algorithm score: min(1, length_mm / 10) * min(1, radius_mm / 0.7).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "1.0"
THRESHOLDS_VERSION = (
    "corotree-0.2: root_snap 6 mm, spur 2 mm, min_tree 40 mm, tolerance 1 mm; complete_candidate requires "
    "unassigned-length fraction <= 0.20 and <= 8 unassigned fragments (calibrated on Split-1 val, 2026-09-25)"
)
P = dict(
    root_snap_mm=6.0,
    spur_mm=2.0,
    min_tree_mm=40.0,
    unknown_frac_max=0.20,
    unknown_count_max=8,
    tol_mm=1.0,
)


def _edge_conf(length_mm: float, radius_mm: float) -> float:
    return float(min(1.0, length_mm / 10.0) * min(1.0, radius_mm / 0.7))


def _component_length(G: nx.MultiGraph, node) -> float:
    comp = nx.node_connected_component(G, node)
    return float(
        sum(d["length_mm"] for u, v, d in G.edges(comp, data=True) if u in comp and v in comp)
    )


def snap_root(G: nx.MultiGraph, xyz_mm: np.ndarray, snap_mm: float):
    """Nearest node, or split the nearest edge at its closest polyline point. Returns node id or None."""
    if G.number_of_nodes() == 0:
        return None
    nodes = list(G.nodes)
    d = [np.linalg.norm(G.nodes[n]["mm"] - xyz_mm) for n in nodes]
    j = int(np.argmin(d))
    best_node, best_d = nodes[j], d[j]
    best_edge, best_i, best_ed = None, None, best_d
    for u, v, k, data in G.edges(keys=True, data=True):
        dd = np.linalg.norm(data["pts_mm"] - xyz_mm, axis=1)
        i = int(np.argmin(dd))
        if dd[i] < best_ed - 0.5 and 1 <= i <= len(dd) - 2:
            best_edge, best_i, best_ed = (u, v, k), i, dd[i]
    if best_edge is not None and best_ed < best_d - 0.5:
        u, v, k = best_edge
        data = G.get_edge_data(u, v, k)
        nid = max(G.nodes) + 1
        G.add_node(nid, vox=data["pts_vox"][best_i].astype(float), mm=data["pts_mm"][best_i].copy())
        G.remove_edge(u, v, k)
        for a, b, sl in ((u, nid, slice(0, best_i + 1)), (nid, v, slice(best_i, None))):
            seg_v, seg_m, seg_r = (
                data["pts_vox"][sl],
                data["pts_mm"][sl],
                data["radius_samples"][sl],
            )
            L = float(np.linalg.norm(np.diff(seg_m, axis=0), axis=1).sum())
            G.add_edge(
                a,
                b,
                path_id=f"{data['path_id']}{'a' if a == u else 'b'}",
                pts_vox=seg_v,
                pts_mm=seg_m,
                length_mm=L,
                radius_mm=float(np.median(seg_r)),
                radius_min_mm=float(seg_r.min()),
                radius_samples=seg_r,
                skeleton_id=data["skeleton_id"],
                branch_type=data["branch_type"],
            )
        best_node, best_d = nid, best_ed
    return best_node if best_d <= snap_mm else None


def build_trees(
    G: nx.MultiGraph, ostia: dict, mask: np.ndarray, spacing, affine: np.ndarray, params: dict = P
) -> dict:
    log = dict(
        pruned_edges=[],
        cycles_removed=[],
        bridges_cut=[],
        extra_ostium_candidates=[],
        reason_codes=list(ostia.get("reason_codes", [])),
    )
    roots = {}
    # --- root selection per side --------------------------------------------------------------------------------
    for side in ("left", "right"):
        cands = [c for c in ostia.get("candidates", []) if c["side"] == side]
        scored = []
        for c in cands:
            n = snap_root(G, np.asarray(c["xyz_mm"]), params["root_snap_mm"])
            if n is None:
                continue
            scored.append((_component_length(G, n), n, c))
        scored.sort(key=lambda t: -t[0])
        if scored:
            L, n, c = scored[0]
            roots[side] = dict(node=n, candidate=c, attached_length_mm=L)
            comp_root = nx.node_connected_component(G, n)
            for L2, n2, c2 in scored[1:]:
                far = np.linalg.norm(np.asarray(c2["xyz_mm"]) - np.asarray(c["xyz_mm"])) > 10.0
                same_tree = n2 in comp_root
                log["extra_ostium_candidates"].append(
                    dict(
                        side=side,
                        xyz_mm=c2["xyz_mm"],
                        attached_length_mm=L2,
                        far_from_chosen=bool(far),
                        same_tree_as_chosen=bool(same_tree),
                    )
                )
                if far and L2 > 20.0 and not same_tree:
                    log["reason_codes"].append("anatomy_variant")
        else:
            log["reason_codes"].append(f"missing_{side}_root")
    # --- bridge between left and right ----------------------------------------------------------------------------
    if "left" in roots and "right" in roots:
        rl, rr = roots["left"]["node"], roots["right"]["node"]
        if rl == rr:
            log["reason_codes"].append("apparent_bridge")
        elif nx.has_path(G, rl, rr):
            path = nx.shortest_path(
                G, rl, rr, weight=lambda u, v, d: min(x["length_mm"] for x in d.values())
            )
            thinnest, best = None, np.inf
            for u, v in zip(path[:-1], path[1:]):
                for k, d in G.get_edge_data(u, v).items():
                    if d["radius_min_mm"] < best:
                        thinnest, best = (u, v, k, d), d["radius_min_mm"]
            if thinnest:
                u, v, k, d = thinnest
                G.remove_edge(u, v, k)
                log["bridges_cut"].append(
                    dict(
                        edge_path_id=str(d["path_id"]),
                        length_mm=d["length_mm"],
                        radius_min_mm=d["radius_min_mm"],
                    )
                )
                log["reason_codes"].append("apparent_bridge")
    # --- per-side trees --------------------------------------------------------------------------------------------
    trees, used_nodes, side_nodes = {}, set(), {}
    for side, r in roots.items():
        root = r["node"]
        comp = nx.node_connected_component(G, root)
        H = G.subgraph(comp)
        T = nx.bfs_tree(nx.Graph(H), root)
        tree_edges = {}
        for u, v in T.edges():
            cands_e = G.get_edge_data(u, v)
            k, d = min(cands_e.items(), key=lambda kv: kv[1]["length_mm"])
            tree_edges[(u, v)] = (k, d)
        tree_keys = {(u, v, k) for (u, v), (k, d) in tree_edges.items()} | {
            (v, u, k) for (u, v), (k, d) in tree_edges.items()
        }
        for u, v, k, d in G.edges(comp, keys=True, data=True):
            if (u, v, k) not in tree_keys:
                log["cycles_removed"].append(
                    dict(side=side, length_mm=d["length_mm"], radius_mm=d["radius_mm"])
                )
        # prune spurs iteratively
        changed = True
        while changed:
            changed = False
            leaves = [n for n in T.nodes if T.out_degree(n) == 0 and n != root]
            for leaf in leaves:
                (parent,) = T.predecessors(leaf)
                k, d = tree_edges[(parent, leaf)]
                if d["length_mm"] < params["spur_mm"] and T.out_degree(parent) >= 2:
                    T.remove_node(leaf)
                    del tree_edges[(parent, leaf)]
                    log["pruned_edges"].append(
                        dict(
                            side=side,
                            length_mm=d["length_mm"],
                            radius_mm=d["radius_mm"],
                            reason="spur<%.1fmm" % params["spur_mm"],
                        )
                    )
                    changed = True
        # simplify pass-through nodes (out_degree 1, non-root)
        nodes_out, edges_out = {}, []

        def kind(n):
            return "root" if n == root else ("end" if T.out_degree(n) == 0 else "junction")

        keep = [n for n in T.nodes if n == root or T.out_degree(n) != 1]
        for n in keep:
            nodes_out[n] = dict(
                id=f"{side[0].upper()}n{n}",
                kind=kind(n),
                xyz_mm=[float(x) for x in G.nodes[n]["mm"]],
            )
        for n in keep:
            for child in list(T.successors(n)):
                m, pts, rad, L = child, None, None, 0.0
                k, d = tree_edges[(n, child)]
                p = d["pts_mm"]
                if np.linalg.norm(p[0] - G.nodes[n]["mm"]) > np.linalg.norm(
                    p[-1] - G.nodes[n]["mm"]
                ):
                    p = p[::-1]
                pts, rad, L, segs = [p], [d["radius_samples"]], d["length_mm"], [d]
                while T.out_degree(m) == 1 and m != root:
                    (c2,) = T.successors(m)
                    k2, d2 = tree_edges[(m, c2)]
                    p2 = d2["pts_mm"]
                    if np.linalg.norm(p2[0] - G.nodes[m]["mm"]) > np.linalg.norm(
                        p2[-1] - G.nodes[m]["mm"]
                    ):
                        p2 = p2[::-1]
                    pts.append(p2[1:])
                    rad.append(d2["radius_samples"])
                    L += d2["length_mm"]
                    m = c2
                    segs.append(d2)
                P_ = np.concatenate(pts)
                R_ = np.concatenate(rad)
                eid = f"{side[0].upper()}e{len(edges_out):03d}"
                bridged = any(sg.get("bridged") for sg in segs)
                conf = min(
                    [_edge_conf(L, float(np.median(R_)))]
                    + [float(sg["conf"]) for sg in segs if sg.get("bridged")]
                )
                edges_out.append(
                    dict(
                        id=eid,
                        parent_node_id=nodes_out[n]["id"],
                        child_node_id=nodes_out[m]["id"],
                        points_xyz_mm=[[float(x) for x in q] for q in P_],
                        length_mm=float(L),
                        radius_mm=float(np.median(R_)),
                        confidence=conf,
                        bridged=bridged,
                        bridge_gap_mm=float(
                            sum(sg.get("gap_mm", 0.0) for sg in segs if sg.get("bridged"))
                        ),
                    )
                )
        trees[side] = dict(
            root_node_id=nodes_out[root]["id"],
            nodes=list(nodes_out.values()),
            edges=edges_out,
            length_mm=float(sum(e["length_mm"] for e in edges_out)),
            n_pruned=sum(1 for x in log["pruned_edges"] if x["side"] == side),
        )
        used_nodes |= set(comp)
        side_nodes[side] = sorted(comp)
    # --- unknown fragments -----------------------------------------------------------------------------------------
    unknown = dict(nodes=[], edges=[], fragments=[])
    unknown_comps = []
    for comp in nx.connected_components(G):
        if comp & used_nodes:
            continue
        unknown_comps.append(sorted(comp))
        L = float(sum(d["length_mm"] for u, v, d in G.edges(comp, data=True)))
        unknown["fragments"].append(dict(length_mm=L, n_edges=G.subgraph(comp).number_of_edges()))
        for n in comp:
            unknown["nodes"].append(
                dict(
                    id=f"Un{n}",
                    kind="end" if G.degree(n) <= 1 else "junction",
                    xyz_mm=[float(x) for x in G.nodes[n]["mm"]],
                )
            )
        for u, v, k, d in G.edges(comp, keys=True, data=True):
            unknown["edges"].append(
                dict(
                    id=f"Ue{len(unknown['edges']):03d}",
                    parent_node_id=f"Un{u}",
                    child_node_id=f"Un{v}",
                    points_xyz_mm=[[float(x) for x in q] for q in d["pts_mm"]],
                    length_mm=float(d["length_mm"]),
                    radius_mm=float(d["radius_mm"]),
                    confidence=_edge_conf(d["length_mm"], d["radius_mm"]),
                )
            )
    unknown_len = float(sum(f["length_mm"] for f in unknown["fragments"]))
    # --- invariants + status ----------------------------------------------------------------------------------------
    violations = check_invariants(trees, mask, spacing, affine, params["tol_mm"])
    rc = log["reason_codes"]
    for side in ("left", "right"):
        if side in trees and trees[side]["length_mm"] < params["min_tree_mm"]:
            rc.append("short_tree")
    assigned_len = float(sum(t["length_mm"] for t in trees.values()))
    unknown_frac = (
        unknown_len / (unknown_len + assigned_len) if (unknown_len + assigned_len) > 0 else 0.0
    )
    if (
        unknown_frac > params["unknown_frac_max"]
        or len(unknown["fragments"]) > params["unknown_count_max"]
    ):
        rc.append("disconnected_fragment")
    if unknown_frac > 0.5:
        rc.append("suspected_low_coverage")
    if violations:
        rc.append("invalid_geometry")
    if not mask.any() or (G.number_of_edges() == 0):
        status = "failed"
        rc.append("empty_mask")
    elif ("left" in trees and "right" in trees) and not rc:
        status = "complete_candidate"
    else:
        status = "needs_review"
    rc = sorted(set(rc))
    return dict(
        status=status,
        reason_codes=rc,
        trees=dict(
            left=trees.get("left"),
            right=trees.get("right"),
            unknown=dict(nodes=unknown["nodes"], edges=unknown["edges"]),
        ),
        roots={
            s: dict(
                node_id=trees[s]["root_node_id"],
                confidence=r["candidate"]["confidence"],
                xyz_mm=r["candidate"]["xyz_mm"],
            )
            for s, r in roots.items()
            if s in trees
        },
        unknown_fragments=unknown["fragments"],
        unknown_length_mm=unknown_len,
        unknown_fraction=unknown_frac,
        log=log,
        violations=violations,
        _side_nodes=side_nodes,
        _unknown_comps=unknown_comps,
    )


def check_invariants(trees: dict, mask: np.ndarray, spacing, affine: np.ndarray, tol_mm: float):
    v = []
    inv = np.linalg.inv(affine)
    diag = float(np.linalg.norm(np.asarray(spacing[:3])))
    for side, t in trees.items():
        if t is None:
            continue
        ids = {n["id"] for n in t["nodes"]}
        if t["root_node_id"] not in ids:
            v.append(f"{side}: root not in nodes")
        if not t["edges"]:
            v.append(f"{side}: no edges")
        g = nx.DiGraph()
        for e in t["edges"]:
            g.add_edge(e["parent_node_id"], e["child_node_id"])
            if not (
                np.isfinite(e["length_mm"])
                and e["length_mm"] > 0
                and np.isfinite(e["radius_mm"])
                and e["radius_mm"] > 0
            ):
                v.append(f"{side}: non-positive length/radius on {e['id']}")
            if len(e["points_xyz_mm"]) < 2:
                v.append(f"{side}: polyline < 2 points on {e['id']}")
            if e["parent_node_id"] not in ids or e["child_node_id"] not in ids:
                v.append(f"{side}: edge {e['id']} references unknown node")
            if e.get("bridged"):
                continue
            pts = np.asarray(e["points_xyz_mm"])
            vox = np.round((inv[:3, :3] @ pts.T).T + inv[:3, 3]).astype(int)
            inside = np.all((vox >= 0) & (vox < np.array(mask.shape)), axis=1)
            ok = inside.copy()
            ok[inside] = mask[tuple(vox[inside].T)]
            if not ok.all():
                # allow <= one voxel diagonal: check distance to mask via neighbours of failing points
                bad = pts[~ok]
                if len(bad) > 0.02 * len(pts) + 2:
                    v.append(
                        f"{side}: {int((~ok).sum())} polyline points outside mask on {e['id']}"
                    )
        if g.number_of_nodes():
            if not nx.is_directed_acyclic_graph(g):
                v.append(f"{side}: cycle")
            reach = nx.descendants(g, t["root_node_id"]) | {t["root_node_id"]}
            if reach != set(g.nodes):
                v.append(f"{side}: {len(set(g.nodes) - reach)} nodes not reachable from root")
    if trees.get("left") and trees.get("right"):
        ln = {n["id"] for n in trees["left"]["nodes"]}
        rn = {n["id"] for n in trees["right"]["nodes"]}
        if ln & rn:
            v.append("left-right shared node")
    return v


# ---------------------------------------------------------------- exports -------------------------------------------
def write_tree_json(path: Path, result: dict, model_sha: str, input_sha: str):
    doc = dict(
        schema_version=SCHEMA,
        coordinate_system="RAS_mm",
        status=result["status"],
        trees={
            s: (
                None
                if result["trees"][s] is None
                else {k: result["trees"][s][k] for k in ("root_node_id", "nodes", "edges")}
            )
            for s in ("left", "right")
        },
        model_sha256=model_sha,
        input_sha256=input_sha,
    )
    doc["trees"]["unknown"] = result["trees"]["unknown"]
    path.write_text(json.dumps(doc), encoding="utf-8")


def write_qc_json(
    path: Path, result: dict, model_sha: str, input_sha: str, ostia: dict, n_comp_raw: int
):
    t = result["trees"]
    doc = dict(
        schema_version=SCHEMA,
        status=result["status"],
        review_required=True,
        reason_codes=result["reason_codes"],
        input_sha256=input_sha,
        model_sha256=model_sha,
        thresholds_version=THRESHOLDS_VERSION,
        component_count_raw=int(n_comp_raw),
        component_count_post=int(
            sum(1 for s in ("left", "right") if t[s]) + len(result["unknown_fragments"])
        ),
        tree_length_mm_by_side={
            s: (t[s]["length_mm"] if t[s] else None) for s in ("left", "right")
        },
        root_confidence_by_side={
            s: (result["roots"][s]["confidence"] if s in result["roots"] else None)
            for s in ("left", "right")
        },
        confidence_method="edge: min(1, length_mm/10) * min(1, radius_mm/0.7); root: ostium side-score |dx+dy|/sqrt2 / 10 mm; uncalibrated",
        unknown_fragments=result["unknown_fragments"],
        unknown_length_mm=result["unknown_length_mm"],
        unknown_fraction=result.get("unknown_fraction"),
        pruned_edges=result["log"]["pruned_edges"],
        cycles_removed=result["log"]["cycles_removed"],
        bridges_cut=result["log"]["bridges_cut"],
        extra_ostium_candidates=result["log"]["extra_ostium_candidates"],
        ostium_candidates=ostia.get("candidates", []),
        invariant_violations=result["violations"],
    )
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def write_vtp(path: Path, result: dict):
    pts, lines, offsets, edge_id, side_id, conf, radius = [], [], [], [], [], [], []
    off = 0
    for si, side in enumerate(("left", "right", "unknown")):
        t = result["trees"][side]
        if t is None:
            continue
        for ei, e in enumerate(t["edges"]):
            P_ = e["points_xyz_mm"]
            lines.extend(range(off, off + len(P_)))
            off += len(P_)
            offsets.append(off)
            pts.extend(P_)
            radius.extend([e["radius_mm"]] * len(P_))
            edge_id.append(int(e["id"][2:]))
            side_id.append(si)
            conf.append(e["confidence"])
    n_pts, n_lines = len(pts), len(offsets)

    def arr(vals):
        return " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals)

    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">',
        "<PolyData>",
        f'<Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="{n_lines}" NumberOfStrips="0" NumberOfPolys="0">',
        '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        arr([c for p in pts for c in p]),
        "</DataArray></Points>",
        '<PointData Scalars="radius_mm"><DataArray type="Float32" Name="radius_mm" format="ascii">',
        arr(radius),
        "</DataArray></PointData>",
        '<CellData Scalars="edge_id"><DataArray type="Int32" Name="edge_id" format="ascii">',
        arr(edge_id),
        "</DataArray>",
        '<DataArray type="Int32" Name="side" format="ascii">',
        arr(side_id),
        "</DataArray>",
        '<DataArray type="Float32" Name="confidence" format="ascii">',
        arr(conf),
        "</DataArray></CellData>",
        '<Lines><DataArray type="Int64" Name="connectivity" format="ascii">',
        arr(lines),
        "</DataArray>",
        '<DataArray type="Int64" Name="offsets" format="ascii">',
        arr(offsets),
        "</DataArray></Lines>",
        "</Piece></PolyData></VTKFile>",
    ]
    path.write_text("\n".join(xml), encoding="utf-8")
    # note: CellData edge_id is the integer part of edges.id ("Le012" -> 12); side 0=left 1=right 2=unknown (metadata: RAS mm)


def write_preview(path: Path, result: dict, mask: np.ndarray, affine: np.ndarray):
    from PIL import Image, ImageDraw

    inv = np.linalg.inv(affine)
    mip = mask.max(axis=2)  # (x, y)
    img = np.transpose(mip[:, ::-1])  # rows = y (anterior top), cols = x
    base = Image.fromarray((img.astype(np.uint8) * 90)).convert("RGB")
    scale = 2
    base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(base)
    colors = dict(left=(255, 176, 0), right=(39, 192, 255), unknown=(255, 92, 240))
    for side in ("left", "right", "unknown"):
        t = result["trees"][side]
        if t is None:
            continue
        for e in t["edges"]:
            pts = np.asarray(e["points_xyz_mm"])
            vox = (inv[:3, :3] @ pts.T).T + inv[:3, 3]
            xy = [(float(v[0]) * scale, float(mask.shape[1] - 1 - v[1]) * scale) for v in vox]
            if len(xy) >= 2:
                d.line(xy, fill=colors[side], width=2)
    for side, r in result["roots"].items():
        v = inv[:3, :3] @ np.asarray(r["xyz_mm"]) + inv[:3, 3]
        x, y = float(v[0]) * scale, float(mask.shape[1] - 1 - v[1]) * scale
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255), width=2)
    d.text(
        (6, 6),
        f"{result['status']}  L={colors['left']} R={colors['right']} U={colors['unknown']}",
        fill=(255, 255, 255),
    )
    base.save(path)


def tree_points(result: dict, sides=("left", "right")) -> np.ndarray:
    pts = [
        np.asarray(e["points_xyz_mm"])
        for s in sides
        if result["trees"][s]
        for e in result["trees"][s]["edges"]
    ]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def coverage_of(reference_pts: np.ndarray, pts: np.ndarray, tol_mm: float) -> float:
    if len(reference_pts) == 0:
        return float("nan")
    if len(pts) == 0:
        return 0.0
    d, _ = cKDTree(pts).query(reference_pts)
    return float((d <= tol_mm).mean())
