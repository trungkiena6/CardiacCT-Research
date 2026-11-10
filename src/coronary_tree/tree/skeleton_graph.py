"""Skeleton -> undirected multigraph with physical polylines and radii.

Nodes: skan junction/end points (voxel + RAS mm). Edges: skan paths with the ordered voxel path, RAS polyline,
length in mm (skan, spacing-aware) and radius = median Euclidean distance transform of the mask along the path.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skan import Skeleton, summarize
from skimage.morphology import skeletonize


def vox_to_mm(affine: np.ndarray, pts_vox: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_vox, dtype=float)
    return (affine[:3, :3] @ pts.T).T + affine[:3, 3]


def skeleton_graph(mask: np.ndarray, spacing, affine: np.ndarray):
    """Return (graph, skeleton_bool, edt_mm). Graph is a networkx MultiGraph; edge key = skan path id."""
    spacing = tuple(float(s) for s in spacing[:3])
    G = nx.MultiGraph()
    sk = skeletonize(mask.astype(bool)) if mask.any() else np.zeros_like(mask, dtype=bool)
    edt = (
        ndi.distance_transform_edt(mask, sampling=spacing)
        if mask.any()
        else np.zeros(mask.shape, dtype=np.float32)
    )
    if int(sk.sum()) < 2:
        return G, sk.astype(bool), edt
    S = Skeleton(sk.astype(np.uint8), spacing=spacing)
    df = summarize(S, separator="_")
    for pid, row in df.iterrows():
        pid = int(pid)
        src, dst = int(row["node_id_src"]), int(row["node_id_dst"])
        pts_vox = np.asarray(S.path_coordinates(pid))
        if pts_vox.ndim != 2 or len(pts_vox) < 2:
            continue
        pts_mm = vox_to_mm(affine, pts_vox)
        idx = tuple(np.round(pts_vox).astype(int).T)
        radius = edt[idx]
        G.add_edge(
            src,
            dst,
            key=pid,
            path_id=pid,
            pts_vox=pts_vox,
            pts_mm=pts_mm,
            length_mm=float(row["branch_distance"]),
            radius_mm=float(np.median(radius)),
            radius_min_mm=float(radius.min()),
            radius_samples=radius.astype(np.float32),
            skeleton_id=int(row["skeleton_id"]),
            branch_type=int(row["branch_type"]),
        )
    for n in G.nodes:
        v = np.asarray(S.coordinates[n], dtype=float)
        G.nodes[n]["vox"] = v
        G.nodes[n]["mm"] = vox_to_mm(affine, v[None, :])[0]
    return G, sk.astype(bool), edt
