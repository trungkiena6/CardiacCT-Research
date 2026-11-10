"""Audit one explicit source set without changing results or configurations.

Default archived mode reads seven historical exports. Updated mode reads ten
canonical new exports without historical fallbacks. Invalid/missing CSVs exit 1.
Default output is JSON on stdout; optional report files are created exclusively.
Manuscript references, confidence intervals and empirical means remain separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coronary_tree.evaluate.export_contract import export_specs, inspect_csv, load_scopes  # noqa: E402

# Frozen manuscript references, not empirical inputs used to adjust any score.
MANUSCRIPT_REFERENCES = {
    "T3_SINGLE": 84.16,
    "T3_ENSEMBLE": 84.90,
    "T4_M1": 91.40,
    "T4_M1_PP": 91.72,
}
ARCHIVED_INTERVALS = {
    "T3_ENSEMBLE": ("imagecas/test_access_3_result.json", "test"),
    "T4_M1": ("imagecas-x/xbench_ours_result.json", "dice"),
    "T4_M1_PP": ("imagecas-x/lumen_access2_test_result.json", "dice"),
}


def saved_interval(results_root, claim, case_n, mean_fraction):
    """Expose a saved CI only when its saved n/mean agree with the CSV scope."""
    if claim not in ARCHIVED_INTERVALS:
        return dict(ci95_dice_percent=None, ci95_status="not_stored_for_aggregate")
    relative, kind = ARCHIVED_INTERVALS[claim]
    path = Path(results_root) / relative
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if kind == "test":
            n, value, bounds = data["test"]["n"], data["test"]["dice"], data["ci95"]["dice"]
        else:
            n, value, bounds = data["n"], data["dice"]["mean"], data["dice"]["ci95"]
        if n != case_n or not math.isclose(float(value), mean_fraction, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved summary n/mean do not match this CSV scope")
        if len(bounds) != 2 or any(
            not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in bounds
        ):
            raise ValueError("Invalid stored confidence bounds")
        if not float(bounds[0]) <= float(value) <= float(bounds[1]):
            raise ValueError("Stored bounds do not contain the saved mean")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return dict(
            ci95_dice_percent=None,
            ci95_status="unavailable_or_inconsistent",
            ci95_source=relative,
            ci95_error=str(exc),
        )
    return dict(
        ci95_dice_percent=[float(x) * 100 for x in bounds],
        ci95_status="saved_summary_matches_csv_n_and_mean",
        ci95_source=relative,
        ci95_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def audit(results_root=None, source_set="archived", split_manifest=None, imagecasx_test_list=None):
    results_root = Path(results_root) if results_root is not None else REPO / "results"
    split_manifest = (
        Path(split_manifest)
        if split_manifest is not None
        else REPO / "configs/imagecas_v2_split_manifest.csv"
    )
    imagecasx_test_list = (
        Path(imagecasx_test_list)
        if imagecasx_test_list is not None
        else REPO / "configs/imagecasx_filelists/test.txt"
    )
    splits, lumen = load_scopes(split_manifest, imagecasx_test_list)
    sources = {claim: [] for claim in MANUSCRIPT_REFERENCES}
    for spec in export_specs(source_set):
        result = inspect_csv(
            results_root / spec.relative_csv, splits[spec.split] if spec.split else lumen
        )
        result.update(
            run_id=spec.run_id,
            relative_csv=spec.relative_csv,
            split=spec.split,
            dataset=spec.dataset,
        )
        sources[spec.claim_id].append(result)
    summaries = []
    for claim, records in sources.items():
        complete = all(record["status"] == "valid" for record in records)
        available = [record for record in records if record["status"] == "valid"]
        expected_n = sum(record["expected_n"] for record in records)
        available_n = sum(record["n_rows"] for record in available)
        mean_fraction = (
            math.fsum(record["mean_dice"] * record["n_rows"] for record in records) / expected_n
            if complete
            else None
        )
        subset_mean = (
            math.fsum(record["mean_dice"] * record["n_rows"] for record in available) / available_n
            if available_n
            else None
        )
        scope_matches = not (source_set == "archived" and claim == "T3_ENSEMBLE")
        reference = MANUSCRIPT_REFERENCES[claim]
        matches = (
            f"{mean_fraction * 100:.2f}" == f"{reference:.2f}"
            if complete and scope_matches
            else None
        )
        summary = dict(
            claim_id=claim,
            source_set=source_set,
            evaluation_scope=(
                "official Split 1 only"
                if not scope_matches
                else "official Splits 1-4"
                if claim.startswith("T3")
                else "official test-160"
            ),
            expected_case_n=expected_n,
            available_saved_case_n=available_n,
            recomputed_saved_mean_dice_percent=mean_fraction * 100 if complete else None,
            available_subset_mean_dice_percent=subset_mean * 100
            if subset_mean is not None
            else None,
            manuscript_reference_dice_percent=reference,
            scope_matches_manuscript=scope_matches,
            reported_matches_saved_mean_to_2dp=matches,
            status="incomplete"
            if not complete
            else "scope_mismatch"
            if not scope_matches
            else "matches_saved_csv"
            if matches
            else "mismatch",
        )
        interval = (
            saved_interval(results_root, claim, expected_n, mean_fraction)
            if complete and source_set == "archived"
            else dict(ci95_dice_percent=None, ci95_status="not_provided_for_selected_scope")
        )
        summary.update(interval)
        bounds = interval["ci95_dice_percent"]
        summary["manuscript_number_equals_saved_upper_ci_to_2dp"] = (
            f"{bounds[1]:.2f}" == f"{reference:.2f}" if bounds else None
        )
        summaries.append(summary)
    return dict(
        evidence_level="arithmetic_reaggregation_of_saved_csv_only",
        source_set=source_set,
        provenance_verified=False,
        predictions_or_checkpoints_verified=False,
        valid_exports=all(
            record["status"] == "valid" for records in sources.values() for record in records
        ),
        aggregation="Unweighted arithmetic mean of per-case Dice; incomplete selected scopes have no full-scope mean.",
        interval_interpretation="Stored CI bounds are distinct from means; numerical correspondence does not establish copying or author intent.",
        manuscript_reference_source="Frozen JBHI_CCTA_PCAT_2026_11_01.pdf, Tables III-IV; not empirical inputs",
        split_manifest_sha256=hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        imagecasx_test_list_sha256=hashlib.sha256(imagecasx_test_list.read_bytes()).hexdigest(),
        summaries=summaries,
        sources=sources,
    )


def write_reports(report, output_dir):
    """Create report files exclusively; never replace a saved snapshot."""
    output_dir = Path(output_dir)
    paths = [output_dir / f"segmentation_dice_audit.{suffix}" for suffix in ("json", "csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Audit output exists; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with paths[1].open("x", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in report["summaries"] for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["summaries"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-set", choices=("archived", "updated"), default="archived")
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument(
        "--split-manifest", type=Path, default=REPO / "configs/imagecas_v2_split_manifest.csv"
    )
    parser.add_argument(
        "--imagecasx-test-list", type=Path, default=REPO / "configs/imagecasx_filelists/test.txt"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output-dir", type=Path, help="Create JSON/CSV reports; refuse existing reports"
    )
    output.add_argument("--report", type=Path, help="Create a JSON report; refuse an existing path")
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.results_root, args.source_set, args.split_manifest, args.imagecasx_test_list
        )
        if args.output_dir:
            write_reports(report, args.output_dir)
        if args.report:
            with args.report.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["valid_exports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
