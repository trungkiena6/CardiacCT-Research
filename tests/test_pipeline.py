"""Regression tests: split isolation, metric behaviour on empty/perfect predictions, geometry gate."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coronary_tree.evaluate.segmentation import case_metrics, evaluate_case  # noqa: E402

MANIFEST = ROOT / "configs" / "imagecas_v2_split_manifest.csv"


def test_split_isolation():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    assert len(rows) == 1000 and sorted(int(r["case_id"]) for r in rows) == list(range(1, 1001))
    for i in range(1, 5):
        col = f"split_{i}"
        parts = {
            k: {int(r["case_id"]) for r in rows if r[col] == k} for k in ("train", "val", "test")
        }
        assert (len(parts["train"]), len(parts["val"]), len(parts["test"])) == (700, 50, 250)
        assert (
            not (parts["train"] & parts["val"])
            and not (parts["train"] & parts["test"])
            and not (parts["val"] & parts["test"])
        )
    # every case is a test case in exactly one split
    for r in rows:
        assert sum(r[f"split_{i}"] == "test" for i in range(1, 5)) == 1


def _tube(shape=(40, 40, 40), axis_len=30, radius=2):
    z, y, x = np.indices(shape)
    return ((y - 20) ** 2 + (x - 20) ** 2 <= radius**2) & (z >= 5) & (z < 5 + axis_len)


def test_perfect_prediction():
    gt = _tube()
    m = case_metrics(gt, gt.copy(), (0.5, 0.4, 0.4))
    assert m["dice"] == 1.0 and m["cldice"] == 1.0
    assert m["hd95_mm"] == 0.0 and m["msd_mm"] == 0.0
    assert m["fp_volume_ml"] == 0.0 and m["fn_volume_ml"] == 0.0
    assert m["gt_centerline_coverage_1mm"] == 1.0 and m["n_components_pred"] == 1


def test_empty_prediction():
    gt = _tube()
    m = case_metrics(gt, np.zeros_like(gt), (0.5, 0.4, 0.4))
    assert m["dice"] == 0.0 and m["cldice"] == 0.0 and m["pred_empty"]
    assert m["hd95_mm"] is None and m["msd_mm"] is None
    assert m["gt_centerline_coverage_1mm"] == 0.0
    assert m["fn_volume_ml"] > 0 and m["fp_volume_ml"] == 0.0


def test_small_shift_within_tolerance():
    gt = _tube()
    pred = np.roll(
        gt, 3, axis=1
    )  # 3 voxels * 0.4 mm = 1.2 mm; tube radius 0.8 mm -> GT centreline 0.4 mm from pred mask
    m = case_metrics(gt, pred, (0.5, 0.4, 0.4))
    assert 0 < m["dice"] < 1
    assert 1.0 <= m["hd95_mm"] <= 1.3
    assert m["gt_centerline_coverage_1mm"] == 1.0 and m["pred_centerline_precision_1mm"] == 1.0
    assert m["apparent_bridge_len_mm"] == 0.0 and m["n_components_pred"] == 1


def test_large_shift_outside_tolerance():
    gt = _tube()
    pred = np.roll(
        gt, 6, axis=1
    )  # 2.4 mm shift: no overlap, centrelines 1.6 mm from the other mask
    m = case_metrics(gt, pred, (0.5, 0.4, 0.4))
    assert m["dice"] == 0.0 and not m["pred_empty"]
    assert 2.2 <= m["hd95_mm"] <= 2.6
    assert m["gt_centerline_coverage_1mm"] == 0.0 and m["pred_centerline_precision_1mm"] == 0.0
    assert (
        m["apparent_bridge_len_mm"] > 10.0
    )  # whole predicted skeleton (~15 mm) counts as outside GT
    assert abs(m["uncovered_gt_skeleton_len_mm"] - m["gt_skeleton_len_mm"]) < 1e-9


def test_geometry_gate(tmp_path):
    gt = _tube().astype(np.uint8)
    aff = np.diag([0.4, 0.4, 0.5, 1.0])
    nib.save(nib.Nifti1Image(gt, aff), str(tmp_path / "gt.nii.gz"))
    nib.save(nib.Nifti1Image(gt[:, :, :-1], aff), str(tmp_path / "bad_shape.nii.gz"))
    aff2 = aff.copy()
    aff2[0, 3] = 5.0
    nib.save(nib.Nifti1Image(gt, aff2), str(tmp_path / "bad_affine.nii.gz"))
    r = evaluate_case("c", str(tmp_path / "gt.nii.gz"), str(tmp_path / "bad_shape.nii.gz"), 1.0)
    assert r["status"] == "invalid_geometry" and r["dice"] == 0.0
    r = evaluate_case("c", str(tmp_path / "gt.nii.gz"), str(tmp_path / "bad_affine.nii.gz"), 1.0)
    assert r["status"] == "invalid_geometry"
    r = evaluate_case("c", str(tmp_path / "gt.nii.gz"), str(tmp_path / "missing.nii.gz"), 1.0)
    assert r["status"] == "missing_prediction" and r["dice"] == 0.0
    r = evaluate_case("c", str(tmp_path / "gt.nii.gz"), str(tmp_path / "gt.nii.gz"), 1.0)
    assert r["status"] == "ok" and r["dice"] == 1.0
