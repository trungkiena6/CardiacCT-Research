"""The paper comparison must stay computed from saved files, never copied from the paper."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_values", REPO / "scripts/check_paper_values.py"
)
paper_values = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper_values)


@pytest.mark.parametrize(
    ("paper", "value", "narrower_scope", "expected"),
    [
        ("91.32", 91.3204, False, "match"),
        ("4.97", 4.975, False, "match"),
        ("93.80", 93.82, False, "precision"),
        ("91.40", 91.0075, False, "mismatch"),
        ("84.16", 83.5023, False, "mismatch"),
        ("89.81", 89.8138, True, "scope"),
        ("84.90", 84.3722, True, "mismatch"),
        ("2.90", 2.86, True, "mismatch"),
        ("2.2", None, False, "not_archived"),
    ],
)
def test_classification(paper, value, narrower_scope, expected):
    assert paper_values.classify(paper, value, narrower_scope) == expected


def test_saved_report_is_current():
    with (REPO / "results/paper_values.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved == paper_values.build_rows()


def test_every_computed_row_names_an_existing_source():
    for row in paper_values.build_rows():
        if row["status"] != "not_archived":
            assert (REPO / "results" / row["source"]).exists() or (REPO / row["source"]).exists()


def test_multid4cad_summary_has_no_patient_rows():
    text = (REPO / "results/multid4cad/multid4cad_summary.json").read_text(encoding="utf-8")
    assert re.search(r"\bP\d{3}\b", text) is None
    assert json.loads(text)["eat"]["n"] == 108
