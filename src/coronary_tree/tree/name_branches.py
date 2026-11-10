"""G2b - rule-based branch-name SUGGESTIONS for a CoroTree tree.json (RAS mm). Names are hypotheses for a reader to
confirm (requirements FR-12: names are released only with expert confirmation); each carries an uncalibrated confidence.

Rules (RAS: +x right, +y anterior, +z superior):
  right tree : trunk (largest-radius continuation at every junction) = RCA; at the last junction with two
               substantial children the more anterior/inferior child = PDA, the other = PLV; other side branches =
               "RCA branch" (acute marginal / conus / SA-nodal are not separated).
  left tree  : edges from the root up to the first junction with >= 2 substantial children = LM; at that junction the
               child heading more anterior over its first 30 mm = LAD, the more posterior = LCx, a third (middle)
               child = Ramus; along LAD the trunk continues by radius, side branches = D1, D2, ...; along LCx side
               branches = OM1, OM2, ...
  A child is 'substantial' when its subtree length >= 10 mm. Confidence: LM/RCA 0.9; LAD/LCx 0.5 + 0.4 * |anterior
  separation| / 20 mm (capped at 0.9); PDA/PLV 0.6; diagonals/marginals 0.6; generic branches 0.5.
Usage: name_branches.py --run <tree run dir> [--cases ...]  -> adds suggested_name / name_confidence to every edge of
tree.json and a `branch_names` summary to qc.json; prints a table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBSTANTIAL_MM = 10.0
LOOK_MM = 30.0


def _tree_graph(tree):
    edges = {e["id"]: e for e in tree["edges"]}
    children = defaultdict(list)
    for e in tree["edges"]:
        children[e["parent_node_id"]].append(e)
    return edges, children


def _subtree_length(children, edge):
    total = edge["length_mm"]
    stack = list(children.get(edge["child_node_id"], []))
    while stack:
        e = stack.pop()
        total += e["length_mm"]
        stack.extend(children.get(e["child_node_id"], []))
    return total


def _path_points(children, edge, look_mm=LOOK_MM):
    """Points along the trunk (largest radius continuation) starting at `edge`, up to look_mm."""
    pts, acc, e = [], 0.0, edge
    while e is not None and acc < look_mm:
        P = np.asarray(e["points_xyz_mm"])
        pts.append(P)
        acc += e["length_mm"]
        kids = children.get(e["child_node_id"], [])
        e = max(kids, key=lambda k: k["radius_mm"]) if kids else None
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _direction_score(children, edge, origin):
    P = _path_points(children, edge)
    if len(P) == 0:
        return dict(dy=0.0, dz=0.0, dx=0.0)
    d = P.mean(0) - origin
    return dict(dx=float(d[0]), dy=float(d[1]), dz=float(d[2]))


def _walk_trunk(
    children,
    start_edge,
    name,
    side_prefix,
    names,
    conf,
    side_branch_names=None,
    min_side_mm=SUBSTANTIAL_MM,
):
    """Follow the largest-radius continuation from start_edge; name trunk edges `name`, side branches sequentially."""
    e, k = start_edge, 0
    while e is not None:
        names[e["id"]] = (name, conf)
        kids = children.get(e["child_node_id"], [])
        if not kids:
            break
        kids_sorted = sorted(kids, key=lambda x: -x["radius_mm"])
        trunk = kids_sorted[0]
        for sb in kids_sorted[1:]:
            L = _subtree_length(children, sb)
            if L >= min_side_mm:
                k += 1
                sname = (
                    side_branch_names[k - 1]
                    if side_branch_names and k - 1 < len(side_branch_names)
                    else f"{side_prefix} branch {k}"
                )
                _label_subtree(children, sb, sname, 0.6 if side_branch_names else 0.5, names)
            else:
                _label_subtree(children, sb, f"{side_prefix} twig", 0.4, names)
        e = trunk
    return e


def _label_subtree(children, edge, name, conf, names):
    stack = [edge]
    while stack:
        e = stack.pop()
        names.setdefault(e["id"], (name, conf))
        stack.extend(children.get(e["child_node_id"], []))


def name_tree(tree: dict, side: str) -> dict:
    """Return {edge_id: (name, confidence)} for one rooted tree."""
    names = {}
    if not tree or not tree.get("edges"):
        return names
    edges, children = _tree_graph(tree)
    root = tree["root_node_id"]
    first = children.get(root, [])
    if not first:
        return names
    start = max(first, key=lambda e: e["radius_mm"])
    for extra in first:
        if extra is not start:
            _label_subtree(
                children,
                extra,
                f"{'right' if side == 'right' else 'left'} ostial branch",
                0.4,
                names,
            )
    if side == "right":
        # RCA trunk with PDA/PLV at the last substantial bifurcation
        e = start
        trunk_edges = []
        while e is not None:
            trunk_edges.append(e)
            kids = children.get(e["child_node_id"], [])
            subs = [k for k in kids if _subtree_length(children, k) >= SUBSTANTIAL_MM]
            if not kids:
                break
            e = max(kids, key=lambda k: k["radius_mm"])
        # find the last junction on the trunk with >= 2 substantial children -> PDA / PLV
        crux = None
        for te in trunk_edges:
            kids = children.get(te["child_node_id"], [])
            subs = [k for k in kids if _subtree_length(children, k) >= SUBSTANTIAL_MM]
            if len(subs) >= 2:
                crux = (te, subs)
        if crux:
            te, subs = crux
            origin = np.asarray(te["points_xyz_mm"][-1])
            scored = sorted(
                subs,
                key=lambda k: (
                    -(
                        _direction_score(children, k, origin)["dy"]
                        - _direction_score(children, k, origin)["dz"]
                    )
                ),
            )
            pda, plv = scored[0], scored[-1]
            # RCA up to the crux (side branches generic), then PDA / PLV subtrees
            e, k = start, 0
            while e is not None and e is not te:
                names[e["id"]] = ("RCA", 0.9)
                kids = sorted(children.get(e["child_node_id"], []), key=lambda x: -x["radius_mm"])
                if not kids:
                    break
                for sb in kids[1:]:
                    k += 1
                    _label_subtree(
                        children,
                        sb,
                        f"RCA branch {k}"
                        if _subtree_length(children, sb) >= SUBSTANTIAL_MM
                        else "RCA twig",
                        0.5,
                        names,
                    )
                e = kids[0]
            names[te["id"]] = ("RCA", 0.9)
            _label_subtree(children, pda, "PDA", 0.6, names)
            _label_subtree(children, plv, "PLV", 0.6, names)
            for sb in children.get(te["child_node_id"], []):
                if sb is not pda and sb is not plv:
                    _label_subtree(children, sb, "RCA branch", 0.4, names)
        else:
            _walk_trunk(children, start, "RCA", "RCA", names, 0.9)
    else:
        # LM until the first junction with >= 2 substantial children
        e, lm_edges = start, []
        while e is not None:
            lm_edges.append(e)
            kids = children.get(e["child_node_id"], [])
            subs = [k for k in kids if _subtree_length(children, k) >= SUBSTANTIAL_MM]
            if len(subs) >= 2 or not kids:
                break
            for sb in kids:
                if sb not in subs and sb is not max(kids, key=lambda x: x["radius_mm"]):
                    _label_subtree(children, sb, "LM twig", 0.4, names)
            e = max(kids, key=lambda x: x["radius_mm"])
        for le in lm_edges:
            names[le["id"]] = ("LM", 0.9)
        if e is None:
            return names
        kids = children.get(e["child_node_id"], [])
        subs = [k for k in kids if _subtree_length(children, k) >= SUBSTANTIAL_MM]
        if len(subs) < 2:
            # no bifurcation: the continuation is LAD-like but uncertain
            if kids:
                _walk_trunk(
                    children, max(kids, key=lambda x: x["radius_mm"]), "LAD?", "LAD", names, 0.4
                )
            return names
        origin = np.asarray(e["points_xyz_mm"][-1])
        sc = {k["id"]: _direction_score(children, k, origin) for k in subs}
        order = sorted(subs, key=lambda k: -sc[k["id"]]["dy"])  # most anterior first
        lad, lcx = order[0], order[-1]
        sep = sc[lad["id"]]["dy"] - sc[lcx["id"]]["dy"]
        conf = float(min(0.9, 0.5 + 0.4 * min(1.0, abs(sep) / 20.0)))
        _walk_trunk(children, lad, "LAD", "LAD", names, conf, side_branch_names=["D1", "D2", "D3"])
        _walk_trunk(
            children, lcx, "LCx", "LCx", names, conf, side_branch_names=["OM1", "OM2", "OM3"]
        )
        for mid in order[1:-1]:
            _label_subtree(children, mid, "Ramus", 0.5, names)
        for sb in kids:
            if sb not in subs:
                _label_subtree(children, sb, "LM twig", 0.4, names)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    cases = a.cases or sorted(
        p.name for p in a.run.iterdir() if p.is_dir() and p.name.startswith("Case")
    )
    rows = []
    for case in cases:
        tp, qp = a.run / case / "tree.json", a.run / case / "qc.json"
        if not tp.exists():
            continue
        t = json.loads(tp.read_text(encoding="utf-8"))
        summary = {}
        for side in ("left", "right"):
            tr = t["trees"].get(side)
            if not tr:
                continue
            names = name_tree(tr, side)
            for e in tr["edges"]:
                nm, cf = names.get(e["id"], ("unnamed", 0.0))
                e["suggested_name"], e["name_confidence"] = nm, cf
            for nm in set(n for n, c in names.values()):
                summary[nm] = round(
                    sum(
                        e["length_mm"] for e in tr["edges"] if names.get(e["id"], ("", 0))[0] == nm
                    ),
                    1,
                )
        tp.write_text(json.dumps(t), encoding="utf-8")
        if qp.exists():
            q = json.loads(qp.read_text(encoding="utf-8"))
            q["branch_names"] = dict(
                method="rule-based suggestions (name_branches.py); not expert-confirmed",
                length_mm_by_name=summary,
            )
            qp.write_text(json.dumps(q, indent=1), encoding="utf-8")
        rows.append((case, summary))
        main4 = {k: summary.get(k) for k in ("RCA", "LM", "LAD", "LCx") if k in summary}
        print(
            case,
            main4,
            "| PDA" if "PDA" in summary else "",
            "| Ramus" if "Ramus" in summary else "",
        )
    n = len(rows)
    print(
        f"named {n} cases; with RCA {sum('RCA' in s for c, s in rows)}, LM {sum('LM' in s for c, s in rows)}, LAD {sum('LAD' in s for c, s in rows)}, "
        f"LCx {sum('LCx' in s for c, s in rows)}, PDA {sum('PDA' in s for c, s in rows)}, Ramus {sum('Ramus' in s for c, s in rows)}"
    )


if __name__ == "__main__":
    main()
