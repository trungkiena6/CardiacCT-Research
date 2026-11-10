"""Evaluate one mask directory and summarize complete, valid per-case metrics.

The CSV is retained when evaluation fails. Existing CSV/summary files are never
overwritten, and invalid or undefined endpoints require review before aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

EVALUATOR = Path(__file__).resolve().parents[1] / "src/coronary_tree/evaluate/segmentation.py"
METRICS = (
    "dice",
    "cldice",
    "hd95_mm",
    "n_components_gt",
    "n_components_pred",
    "gt_centerline_coverage_1mm",
)


def require_new_files(*paths: Path) -> None:
    conflicts = [str(path) for path in paths if Path(path).exists()]
    if conflicts:
        raise FileExistsError("Existing outputs preserved; use new paths: " + ", ".join(conflicts))


def read_evaluation_rows(path: Path, expected_case_ids=None) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        if not {"case", "status", *METRICS}.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError(
                f"{path}: missing or duplicate metric columns; CSV retained for review"
            )
        rows = list(reader)
    ids = [row.get("case") for row in rows]
    if not rows or any(not case for case in ids) or len(ids) != len(set(ids)):
        raise ValueError(
            f"{path}: empty evaluation or duplicate/blank case IDs; CSV retained for review"
        )
    if expected_case_ids is not None and set(ids) != set(expected_case_ids):
        raise ValueError(
            f"{path}: evaluated case IDs differ from the required scope; CSV retained for review"
        )
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: malformed row for {row['case']}; CSV retained for review")
        if row["status"] != "ok" or (row.get("reason_codes") or "").strip():
            raise ValueError(
                f"{path}: {row['case']} status={row['status']} reason={row.get('reason_codes', '')}; CSV retained for review"
            )
        for key in METRICS:
            try:
                value = float(row[key])
                valid = math.isfinite(value) and value >= 0
                if key in ("dice", "cldice", "gt_centerline_coverage_1mm"):
                    valid = valid and value <= 1
                if key.startswith("n_components_"):
                    valid = valid and re.fullmatch(r"[0-9]+", row[key]) is not None
                if not valid:
                    raise ValueError("outside valid range")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: {row['case']} has undefined/invalid {key}; CSV retained for review"
                ) from exc
    return rows


def metric_arrays(rows: list[dict]) -> dict:
    values = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
    return {
        "dice": values["dice"],
        "cldice": values["cldice"],
        "hd95_mm": values["hd95_mm"],
        "betti0_err": np.abs(values["n_components_pred"] - values["n_components_gt"]),
        "coverage_1mm": values["gt_centerline_coverage_1mm"],
    }


def summarize_csv(
    path: Path, expected_case_ids=None, *, n_boot: int = 10000, seed: int = 0
) -> dict:
    rows = read_evaluation_rows(path, expected_case_ids)
    rng = np.random.default_rng(seed)
    result = {"n": len(rows)}
    for key, values in metric_arrays(rows).items():
        samples = np.array(
            [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
        )
        result[key] = {
            "mean": float(values.mean()),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "median": float(np.median(values)),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred", type=Path)
    parser.add_argument("gt", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    csv_out = args.out.with_suffix(".per_case.csv")
    try:
        require_new_files(args.out, csv_out)
        if args.cases:
            ids = args.cases.read_text(encoding="utf-8-sig").split()
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("--cases must contain distinct, nonempty case IDs")
        else:
            ids = sorted(path.name[:-7] for path in args.gt.glob("*.nii.gz"))
        if not ids:
            raise ValueError("No ground-truth cases selected")
        command = [
            sys.executable,
            str(EVALUATOR),
            "--gt-dir",
            str(args.gt),
            "--pred-dir",
            str(args.pred),
            "--output",
            str(csv_out),
            "--workers",
            str(args.workers),
        ]
        if args.cases:
            command.extend(["--cases", str(args.cases)])
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise ValueError(
                f"Evaluator exited {completed.returncode}; any CSV at {csv_out} was retained for review"
            )
        result = summarize_csv(csv_out, ids)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=1, allow_nan=False)
            handle.write("\n")
        print(
            "SCORE",
            args.out.name,
            {
                key: round(value["mean"], 4) if isinstance(value, dict) else value
                for key, value in result.items()
            },
            flush=True,
        )
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
