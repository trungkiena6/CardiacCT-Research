"""Summarize the MultiD4CAD external evaluation as cohort-level statistics.

The per-patient tables stay outside the repository under the dataset's data-use agreement.
Only counts, means, agreement statistics and AUCs enter the JSON report.

Usage:
  python scripts/summarize_multid4cad.py --reports <directory with the per-patient CSVs and
    clinical report> --tree-summary <tree/summary.csv>
    --output results/multid4cad/multid4cad_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np

EAT_TABLE = "multid4cad_eat_per_patient.csv"
PAT_TABLE = "multid4cad_pat_per_patient.csv"
CLINICAL_REPORT = "multid4cad_clinical.md"
CLINICAL_ROW = re.compile(
    r"^\| (?P<feature>[^|]+?) \| (?P<n>\d+) \| (?P<auc>[0-9.]+) \| (?P<odds>[0-9.]+) "
    r"\| \[(?P<low>[0-9.]+), (?P<high>[0-9.]+)\] \| (?P<p>[0-9.]+) \|$"
)
PATIENT_ID = re.compile(r"\bP\d{3}\b")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def agreement(measured: np.ndarray, reference: np.ndarray) -> dict:
    difference = measured - reference
    bias = float(difference.mean())
    half_width = 1.96 * float(difference.std(ddof=1))
    return dict(
        n=len(difference),
        bias=bias,
        loa=[bias - half_width, bias + half_width],
        pearson_r=float(np.corrcoef(measured, reference)[0, 1]),
    )


def summarize_eat(rows: list[dict]) -> dict:
    reference = values(rows, "ref_ml")
    return dict(
        n=len(rows),
        dice_rule=float(values(rows, "dice_rule_refrange").mean()),
        dice_rule_fat_window=float(values(rows, "dice_rule_refrange_fatwin").mean()),
        dice_learned=float(values(rows, "dice_model_refrange").mean()),
        volume_rule_ml=agreement(values(rows, "ours_in_ref_range_ml"), reference),
        volume_learned_ml=agreement(values(rows, "model_in_ref_range_ml"), reference),
        lm_landmark_found=sum(row["lm_found"] == "True" for row in rows),
        sac_coverage_fraction=float(values(rows, "sac_covers_ref").mean()),
        reference_fraction_outside_fat_window=float(1 - values(rows, "ref_frac_in_fatwin").mean()),
    )


def summarize_pat(rows: list[dict]) -> dict:
    paired = [row for row in rows if row["lad_fai_auto"]]
    automatic = values(paired, "lad_fai_auto")
    return dict(
        n_cohort=len(rows),
        available={
            vessel: sum(bool(row[f"{vessel.lower()}_fai_auto"]) for row in rows)
            for vessel in ("RCA", "LAD", "LCx")
        },
        lad_vs_reference_hu=agreement(automatic, values(paired, "pat_ref_hu")),
        lad_vs_reference_fat_window_hu=agreement(automatic, values(paired, "pat_ref_hu_fatwin")),
    )


def summarize_tree(rows: list[dict]) -> dict:
    return dict(
        n=len(rows),
        median_components=statistics.median(float(row["n_components_raw"]) for row in rows),
        both_ostia_found=sum(
            row["left_found"] == "True" and row["right_found"] == "True" for row in rows
        ),
        complete_candidates=sum(row["status"] == "complete_candidate" for row in rows),
    )


def parse_clinical(report: str) -> list[dict]:
    matches = (CLINICAL_ROW.match(line.strip()) for line in report.splitlines())
    rows = [
        dict(
            feature=match["feature"],
            n=int(match["n"]),
            auc=float(match["auc"]),
            odds_ratio_per_sd=float(match["odds"]),
            odds_ratio_ci95=[float(match["low"]), float(match["high"])],
            p=float(match["p"]),
        )
        for match in matches
        if match
    ]
    if not rows:
        raise ValueError("No CAD association rows found in the clinical report")
    return rows


def build_summary(reports: Path, tree_summary: Path) -> dict:
    summary = dict(
        boundary="Cohort-level statistics only; per-patient tables are not redistributed.",
        eat=summarize_eat(read_rows(reports / EAT_TABLE)),
        pat=summarize_pat(read_rows(reports / PAT_TABLE)),
        tree=summarize_tree(read_rows(tree_summary)),
        cad_association=parse_clinical((reports / CLINICAL_REPORT).read_text(encoding="utf-8")),
    )
    if PATIENT_ID.search(json.dumps(summary)):
        raise ValueError("Summary must not contain patient identifiers")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--tree-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Create a new JSON report; never overwrite")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path")
    text = json.dumps(build_summary(args.reports, args.tree_summary), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
