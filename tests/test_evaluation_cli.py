"""Evaluation CLI contracts exercised with disposable masks, never study outputs."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/coronary_tree/evaluate/segmentation.py"
REQUIRED = {"case", "status", "dice", "cldice", "hd95_mm", "n_components_gt", "n_components_pred"}


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


def run_cli(gt, pred, output, *options):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gt-dir",
            str(gt),
            "--pred-dir",
            str(pred),
            "--output",
            str(output),
            "--workers",
            "1",
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return set(reader.fieldnames), list(reader)


def test_missing_predictions_retain_schema_and_reason_but_fail(masks, tmp_path):
    gt, _, _ = masks
    missing = tmp_path / "missing"
    missing.mkdir()
    output = tmp_path / "result.csv"
    result = run_cli(gt, missing, output)
    fields, rows = read_csv(output)
    assert result.returncode == 1 and REQUIRED.issubset(fields)
    assert len(rows) == 1 and rows[0]["status"] == "missing_prediction"
    assert rows[0]["reason_codes"] == "missing_prediction"
    summary = json.loads(result.stdout[result.stdout.index("{") :])
    assert summary["n_ok"] == 0 and summary["mean_dice"] is None


@pytest.mark.parametrize("invalid_geometry", [False, True])
def test_invalid_cases_keep_error_rows_with_fixed_columns(masks, tmp_path, invalid_geometry):
    gt, pred, mask = masks
    if invalid_geometry:
        affine = np.eye(4)
        affine[0, 3] = 5
        nib.save(nib.Nifti1Image(mask, affine), pred / "Case0001.nii.gz")
    else:
        (pred / "Case0001.nii.gz").write_bytes(b"not a NIfTI file")
    output = tmp_path / "result.csv"
    result = run_cli(gt, pred, output)
    fields, rows = read_csv(output)
    assert result.returncode == 1 and REQUIRED.issubset(fields)
    assert rows[0]["status"] == ("invalid_geometry" if invalid_geometry else "error")
    assert rows[0]["reason_codes"]


def test_existing_csv_needs_explicit_overwrite(masks, tmp_path):
    gt, pred, _ = masks
    output = tmp_path / "result.csv"
    output.write_text("keep this original\n", encoding="utf-8")
    result = run_cli(gt, pred, output)
    assert result.returncode != 0 and output.read_text(encoding="utf-8") == "keep this original\n"
    result = run_cli(gt, pred, output, "--overwrite")
    _, rows = read_csv(output)
    assert result.returncode == 0 and rows[0]["status"] == "ok"
    assert float(rows[0]["dice"]) == 1.0


@pytest.mark.parametrize(
    "contents, message",
    [
        (b"Case0001\nCase0001\n", "duplicate case IDs"),
        (b"Case0001\xff\n", "UTF-8"),
        (b"\n", "no cases selected"),
    ],
)
def test_invalid_case_lists_fail_before_writing(masks, tmp_path, contents, message):
    gt, pred, _ = masks
    cases = tmp_path / "cases.txt"
    cases.write_bytes(contents)
    output = tmp_path / "result.csv"
    result = run_cli(gt, pred, output, "--cases", str(cases))
    assert result.returncode != 0 and message in result.stderr
    assert not output.exists()


def test_partial_success_does_not_report_a_complete_mean(masks, tmp_path):
    gt, pred, mask = masks
    nib.save(nib.Nifti1Image(mask, np.eye(4)), gt / "Case0002.nii.gz")
    output = tmp_path / "result.csv"
    result = run_cli(gt, pred, output)
    _, rows = read_csv(output)
    summary = json.loads(result.stdout[result.stdout.index("{") :])
    assert result.returncode == 1 and len(rows) == 2
    assert summary["n_ok"] == 1 and summary["mean_dice"] is None and summary["mean_cldice"] is None
