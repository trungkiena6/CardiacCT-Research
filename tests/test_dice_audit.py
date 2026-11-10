"""Meaningful audit regressions; any constructed metrics live only in tmp_path."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dice_audit", REPO / "scripts/audit_dice_evidence.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def by_claim(report):
    return {row["claim_id"]: row for row in report["summaries"]}


def write_test_exports(root, source_set, score):
    """Disposable routing fixtures, not research evaluation evidence."""
    splits, lumen = audit.load_scopes(
        REPO / "configs/imagecas_v2_split_manifest.csv",
        REPO / "configs/imagecasx_filelists/test.txt",
    )
    for spec in audit.export_specs(source_set):
        ids = splits[spec.split] if spec.split else lumen
        path = root / spec.relative_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case",
                    "status",
                    "dice",
                    "cldice",
                    "hd95_mm",
                    "n_components_gt",
                    "n_components_pred",
                ],
            )
            writer.writeheader()
            writer.writerows(
                dict(
                    case=case,
                    status="ok",
                    dice=score,
                    cldice=score,
                    hd95_mm=1,
                    n_components_gt=1,
                    n_components_pred=1,
                )
                for case in sorted(ids)
            )


def test_archived_results_are_reproduced_with_ensemble_scope_and_ci_separate():
    report = audit.audit()
    rows = by_claim(report)
    assert report["source_set"] == "archived" and report["valid_exports"] is True
    assert report["provenance_verified"] is False
    assert rows["T3_SINGLE"]["recomputed_saved_mean_dice_percent"] == pytest.approx(
        83.50231796437062
    )
    ensemble = rows["T3_ENSEMBLE"]
    assert ensemble["status"] == "scope_mismatch" and ensemble["expected_case_n"] == 250
    assert ensemble["reported_matches_saved_mean_to_2dp"] is None
    assert ensemble["recomputed_saved_mean_dice_percent"] == pytest.approx(84.37219214152418)
    assert ensemble["ci95_dice_percent"][1] == pytest.approx(84.90475849995905)
    for claim in ("T3_ENSEMBLE", "T4_M1", "T4_M1_PP"):
        assert rows[claim]["manuscript_number_equals_saved_upper_ci_to_2dp"] is True
        assert (
            rows[claim]["recomputed_saved_mean_dice_percent"]
            != rows[claim]["manuscript_reference_dice_percent"]
        )


def test_updated_mode_consumes_ten_new_files_and_never_archived_fallbacks(tmp_path):
    write_test_exports(tmp_path, "archived", 0.2)
    write_test_exports(tmp_path, "updated", 0.8)
    updated = audit.audit(tmp_path, "updated")
    archived = audit.audit(tmp_path, "archived")
    assert updated["valid_exports"] and archived["valid_exports"]
    assert all(
        row["recomputed_saved_mean_dice_percent"] == pytest.approx(80)
        for row in updated["summaries"]
    )
    assert all(
        row["recomputed_saved_mean_dice_percent"] == pytest.approx(20)
        for row in archived["summaries"]
    )
    assert sum(len(records) for records in updated["sources"].values()) == 10
    missing = tmp_path / "imagecas/split1_ensemble_per_case.csv"
    missing.unlink()
    report = audit.audit(tmp_path, "updated")
    ensemble = by_claim(report)["T3_ENSEMBLE"]
    assert not report["valid_exports"] and ensemble["status"] == "incomplete"
    assert ensemble["available_saved_case_n"] == 750
    assert ensemble["recomputed_saved_mean_dice_percent"] is None
    assert ensemble["ci95_dice_percent"] is None


def test_updated_missing_inputs_return_nonzero_and_stdout_report_only(tmp_path, capsys):
    before = list(tmp_path.iterdir())
    assert audit.main(["--source-set", "updated", "--results-root", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["source_set"] == "updated" and report["valid_exports"] is False
    assert all(row["recomputed_saved_mean_dice_percent"] is None for row in report["summaries"])
    assert list(tmp_path.iterdir()) == before


def test_existing_report_is_never_overwritten(tmp_path, capsys):
    output = tmp_path / "report.json"
    output.write_text("original", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        audit.main(["--report", str(output)])
    assert exc.value.code == 2 and output.read_text(encoding="utf-8") == "original"
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "segmentation_dice_audit.csv").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit.write_reports(audit.audit(), output_dir)
    assert not (output_dir / "segmentation_dice_audit.json").exists()


def test_incorrect_lumen_ids_do_not_produce_scope_mean(tmp_path):
    write_test_exports(tmp_path, "updated", 0.8)
    for relative in (
        "imagecas-x/m1_updated_per_case.csv",
        "imagecas-x/m1_postprocess_updated_per_case.csv",
    ):
        path = tmp_path / relative
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for index, row in enumerate(rows):
            row["case"] = str(10001 + index)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report = audit.audit(tmp_path, "updated")
    assert not report["valid_exports"]
    for claim in ("T4_M1", "T4_M1_PP"):
        assert by_claim(report)[claim]["recomputed_saved_mean_dice_percent"] is None


def test_saved_interval_is_not_reused_with_inconsistent_summary(tmp_path):
    directory = tmp_path / "imagecas-x"
    directory.mkdir()
    (directory / "xbench_ours_result.json").write_text(
        json.dumps({"n": 160, "dice": {"mean": 0.2, "ci95": [0.1, 0.3]}}), encoding="utf-8"
    )
    result = audit.saved_interval(tmp_path, "T4_M1", 160, 0.8)
    assert result["ci95_dice_percent"] is None
    assert result["ci95_status"] == "unavailable_or_inconsistent"
