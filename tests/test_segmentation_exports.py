"""Exercise rejection of incomplete/invalid exports using disposable test data only."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "segmentation_exports",
    Path(__file__).resolve().parents[1] / "scripts/validate_segmentation_exports.py",
)
exports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exports)


def write_csv(path, cases, **overrides):
    fields = sorted(exports.REQUIRED) + ["reason_codes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = dict(
                case=case,
                status="ok",
                dice="0.8",
                cldice="0.9",
                hd95_mm="2.5",
                n_components_gt="2",
                n_components_pred="3",
                reason_codes="",
            )
            row.update(overrides)
            writer.writerow(row)


def test_complete_scope_has_arithmetic_means_and_preserves_source(tmp_path):
    path = tmp_path / "export.csv"
    write_csv(path, ["Case0751", "Case0752"])
    original = path.read_bytes()
    result = exports.inspect_csv(path, {"Case0751", "Case0752"})
    assert result["status"] == "valid" and result["mean_dice"] == pytest.approx(0.8)
    assert result["mean_cldice"] == pytest.approx(0.9)
    assert path.read_bytes() == original


def test_duplicate_with_missing_case_never_produces_mean(tmp_path):
    path = tmp_path / "export.csv"
    write_csv(path, ["Case0751", "Case0751"])
    result = exports.inspect_csv(path, {"Case0751", "Case0752"})
    assert result["status"] == "invalid" and result["mean_dice"] is None
    assert any("Duplicate" in error for error in result["errors"])
    assert any("Missing case IDs" in error for error in result["errors"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"dice": "84.16"},
        {"dice": "nan"},
        {"cldice": "inf"},
        {"hd95_mm": "-1"},
        {"n_components_gt": "1.5"},
        {"status": "error", "reason_codes": "load_failed"},
        {"reason_codes": "empty_mask"},
    ],
)
def test_bad_units_nonfinite_values_status_or_reason_require_review(tmp_path, overrides):
    path = tmp_path / "export.csv"
    write_csv(path, ["100"], **overrides)
    result = exports.inspect_csv(path, {"100"})
    assert result["status"] == "invalid" and result["mean_dice"] is None
    assert result["errors"]


def test_schema_and_encoding_are_required(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("case,status,dice\n100,ok,0.8\n", encoding="utf-8")
    assert "Missing required columns" in exports.inspect_csv(path, {"100"})["errors"][0]
    path.write_bytes(b"case,status,\xff\n")
    assert "UTF-8" in exports.inspect_csv(path, {"100"})["errors"][0]


def test_correct_row_count_cannot_replace_official_scope_or_id_format(tmp_path):
    path = tmp_path / "export.csv"
    write_csv(path, ["751", "752"])
    result = exports.inspect_csv(path, {"Case0751", "Case0752"})
    assert result["n_rows"] == result["expected_n"] == 2
    assert result["status"] == "invalid" and result["mean_dice"] is None
    assert any("Unexpected case IDs" in error for error in result["errors"])


def test_absent_export_is_not_zero(tmp_path):
    result = exports.inspect_csv(tmp_path / "absent.csv", {"100"})
    assert result["status"] == "missing"
    assert result["mean_dice"] is None and result["n_rows"] is None


def test_malformed_row_is_reported_without_crashing(tmp_path):
    path = tmp_path / "export.csv"
    write_csv(path, ["100"])
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n100\n", encoding="utf-8")
    result = exports.inspect_csv(path, {"100"})
    assert result["status"] == "invalid" and result["errors"]


def test_report_refuses_overwrite_and_never_turns_missing_exports_into_zero(tmp_path):
    test_list = Path(__file__).resolve().parents[1] / "configs/imagecasx_filelists/test.txt"
    report_path = tmp_path / "report.json"
    args = [
        "--imagecasx-test-list",
        str(test_list),
        "--results-root",
        str(tmp_path),
        "--report",
        str(report_path),
    ]
    assert exports.main(args) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evidence_scope"] == "csv_only" and report["provenance_verified"] is False
    assert len(report["runs"]) == 10
    assert all(run["mean_dice"] is None and run["status"] == "missing" for run in report["runs"])
    original = report_path.read_bytes()
    with pytest.raises(SystemExit) as error:
        exports.main(args)
    assert error.value.code == 2 and report_path.read_bytes() == original


def test_default_ten_and_explicit_nine_exact_names():
    splits = {number: {f"Case{number:04d}"} for number in range(1, 5)}
    nine = exports.export_specs(splits, {"100"}, False)
    ten = exports.export_specs(splits, {"100"})
    assert len(nine) == 9 and len(ten) == 10
    assert not any(name == "imagecas/split1_ensemble_per_case.csv" for _, name, _ in nine)
    assert any(name == "imagecas/split1_ensemble_per_case.csv" for _, name, _ in ten)
    assert nine[-1][1] == "imagecas-x/m1_postprocess_updated_per_case.csv"


def official_paths():
    repo = Path(__file__).resolve().parents[1]
    return (
        repo / "configs/imagecas_v2_split_manifest.csv",
        repo / "configs/imagecasx_filelists/test.txt",
    )


def test_complete_official_partition_is_available_locally():
    splits, lumen = exports.load_scopes(*official_paths())
    assert all(len(cases) == 250 for cases in splits.values())
    assert len(set.union(*splits.values())) == 1000 and len(lumen) == 160


def test_repeated_test_partitions_and_incorrect_train_val_counts_are_rejected(tmp_path):
    manifest, test_list = official_paths()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, original = reader.fieldnames, list(reader)
    for mutation in ("repeated_test", "bad_train_val"):
        rows = [dict(row) for row in original]
        if mutation == "repeated_test":
            for row in rows:
                for split in (2, 3, 4):
                    row[f"split_{split}"] = row["split_1"]
        else:
            next(row for row in rows if row["split_1"] == "train")["split_1"] = "val"
        path = tmp_path / f"{mutation}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError):
            exports.load_scopes(path, test_list)


def test_wrong_160_id_list_is_rejected_even_with_correct_cardinality(tmp_path):
    manifest, _ = official_paths()
    path = tmp_path / "wrong_test.txt"
    path.write_text("\n".join(str(case) for case in range(10001, 10161)), encoding="utf-8")
    with pytest.raises(ValueError, match="official test-160"):
        exports.load_scopes(manifest, path)


@pytest.mark.parametrize("contents", ["", "100\n" * 160, "not-an-id\n"])
def test_empty_duplicate_or_malformed_scope_lists_are_rejected(tmp_path, contents):
    manifest, _ = official_paths()
    path = tmp_path / "bad_test.txt"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        exports.load_scopes(manifest, path)
