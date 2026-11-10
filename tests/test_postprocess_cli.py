"""Geometry, baseline and output-preservation checks on small disposable masks."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coronary_tree.evaluate.postprocess import sweep_case  # noqa: E402

SCRIPT = ROOT / "src/coronary_tree/evaluate/postprocess.py"


@pytest.fixture
def masks(tmp_path):
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:7, 4, 4] = 1
    for directory in (gt, pred):
        nib.save(nib.Nifti1Image(mask, np.eye(4)), directory / "Case0001.nii.gz")
    return gt, pred, mask


def run_cli(*options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, options), "--workers", "1"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_sweep_geometry_failure_retains_rows_and_selects_no_threshold(masks, tmp_path):
    gt, pred, mask = masks
    affine = np.eye(4)
    affine[0, 3] = 5
    nib.save(nib.Nifti1Image(mask, affine), pred / "Case0001.nii.gz")
    prefix = tmp_path / "sweep"
    result = run_cli(
        "sweep",
        "--gt-dir",
        gt,
        "--pred-dir",
        pred,
        "--out-prefix",
        prefix,
        "--thresholds",
        "0",
        "10",
    )
    with (tmp_path / "sweep_per_case.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((tmp_path / "sweep_summary.json").read_text(encoding="utf-8"))
    assert result.returncode != 0 and len(rows) == 2
    assert all(
        row["status"] == "invalid_geometry" and "affine_mismatch" in row["reason_codes"]
        for row in rows
    )
    assert all("cldice" in row and row["dice"] == "" for row in rows)
    assert summary["status"] == "invalid" and summary["chosen_threshold_mm3"] is None


def test_sweep_raw_baseline_does_not_depend_on_threshold_order(masks):
    gt, pred, _ = masks
    args = ("Case0001", str(gt / "Case0001.nii.gz"), str(pred / "Case0001.nii.gz"))
    forward = {row["threshold_mm3"]: row for row in sweep_case(*args, [0, 10])}
    reverse = {row["threshold_mm3"]: row for row in sweep_case(*args, [10, 0])}
    assert forward[10]["gt_skel_len_lost_mm"] == pytest.approx(4.0)
    assert reverse[10]["gt_skel_len_lost_mm"] == forward[10]["gt_skel_len_lost_mm"]
    assert reverse[0]["gt_skel_len_lost_mm"] == 0


def test_sweep_includes_raw_and_handles_undefined_surface_metrics(masks, tmp_path):
    gt, pred, _ = masks
    prefix = tmp_path / "sweep"
    result = run_cli(
        "sweep", "--gt-dir", gt, "--pred-dir", pred, "--out-prefix", prefix, "--thresholds", "10"
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["thresholds_mm3"] == [0, 10]
    assert summary["per_threshold"]["10.0"]["hd95_mm"] is None
    assert summary["per_threshold"]["10.0"]["msd_mm"] is None
    original = (tmp_path / "sweep_per_case.csv").read_bytes()
    second = run_cli("sweep", "--gt-dir", gt, "--pred-dir", pred, "--out-prefix", prefix)
    assert second.returncode != 0 and (tmp_path / "sweep_per_case.csv").read_bytes() == original


def test_fractional_thresholds_use_distinct_outputs_and_refuse_reuse(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[2:13, 5, 5] = 1
    nib.save(nib.Nifti1Image(mask, np.diag([1, 1, 0.5, 1])), pred / "Case0001.nii.gz")
    for threshold in ("5.1", "5.9"):
        result = run_cli("apply", "--pred-dir", pred, "--threshold", threshold)
        assert result.returncode == 0, result.stderr
    kept, removed = tmp_path / "pred_pp5.1/Case0001.nii.gz", tmp_path / "pred_pp5.9/Case0001.nii.gz"
    assert np.asanyarray(nib.load(kept).dataobj).sum() == 11
    assert np.asanyarray(nib.load(removed).dataobj).sum() == 0
    original = hashlib.sha256(kept.read_bytes()).hexdigest()
    repeated = run_cli("apply", "--pred-dir", pred, "--threshold", "5.1")
    assert repeated.returncode != 0 and hashlib.sha256(kept.read_bytes()).hexdigest() == original
    explicit = run_cli("apply", "--pred-dir", pred, "--threshold", "5.1", "--overwrite")
    assert explicit.returncode == 0, explicit.stderr


def test_explicit_overwrite_rejects_stale_case_ids_without_changing_outputs(masks, tmp_path):
    _, pred, mask = masks
    second = pred / "Case0002.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), second)
    first = run_cli("apply", "--pred-dir", pred, "--threshold", "1")
    assert first.returncode == 0, first.stderr
    output = tmp_path / "pred_pp1"
    original = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    second.unlink()
    nib.save(nib.Nifti1Image(np.zeros_like(mask), np.eye(4)), pred / "Case0001.nii.gz")
    changed = run_cli("apply", "--pred-dir", pred, "--threshold", "1", "--overwrite")
    assert changed.returncode != 0 and "Case0002.nii.gz" in changed.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == original
