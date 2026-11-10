"""Validate complete CSV exports; execution provenance remains unverified.

Updated mode checks ten canonical exports by default. --exclude-ensemble-split1
selects an explicit nine-export subset. Archived mode checks seven saved exports.
Neither mode falls back to files from the other source set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coronary_tree.evaluate.export_contract import (  # noqa: E402
    REQUIRED_COLUMNS,
    inspect_csv,
    load_scopes,
)
from coronary_tree.evaluate.export_contract import (
    export_specs as contract_specs,
)

REQUIRED = REQUIRED_COLUMNS


def export_specs(splits, lumen, include_ensemble_split1=True, source_set="updated"):
    """Expose the shared contract with its expected case sets."""
    return [
        (spec.run_id, spec.relative_csv, splits[spec.split] if spec.split else lumen)
        for spec in contract_specs(source_set, include_ensemble_split1)
    ]


def validate_exports(
    results_root,
    split_manifest,
    imagecasx_test_list,
    source_set="updated",
    include_ensemble_split1=True,
):
    splits, lumen = load_scopes(split_manifest, imagecasx_test_list)
    runs = [
        dict(run_id=run_id, **inspect_csv(Path(results_root) / name, ids))
        for run_id, name, ids in export_specs(splits, lumen, include_ensemble_split1, source_set)
    ]
    return dict(
        evidence_scope="csv_only",
        provenance_verified=False,
        source_set=source_set,
        excludes_updated_ensemble_split1=source_set == "updated" and not include_ensemble_split1,
        aggregation="Unweighted arithmetic mean over each complete official case set",
        valid=all(run["status"] == "valid" for run in runs),
        runs=runs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-set", choices=("archived", "updated"), default="updated")
    parser.add_argument(
        "--imagecasx-test-list", type=Path, default=REPO / "configs/imagecasx_filelists/test.txt"
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=REPO / "configs/imagecas_v2_split_manifest.csv"
    )
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--exclude-ensemble-split1", action="store_true")
    subset.add_argument(
        "--include-ensemble-split1",
        action="store_true",
        help="Explicitly select the default ten-export scope",
    )
    parser.add_argument(
        "--report", type=Path, help="Create a JSON report exclusively; never overwrite"
    )
    args = parser.parse_args(argv)
    if args.source_set == "archived" and args.exclude_ensemble_split1:
        parser.error("--exclude-ensemble-split1 applies only to updated exports")
    try:
        report = validate_exports(
            args.results_root,
            args.split_manifest,
            args.imagecasx_test_list,
            args.source_set,
            not args.exclude_ensemble_split1,
        )
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.report:
        try:
            with args.report.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
        except OSError as exc:
            parser.error(f"Cannot create report without overwriting: {exc}")
    for run in report["runs"]:
        detail = (
            f"mean Dice={run['mean_dice']:.8f}"
            if run["status"] == "valid"
            else "; ".join(run["errors"][:2])
        )
        print(f"{run['run_id']}: {run['status']}; {detail}")
    print(
        f"CSV-only validation ({args.source_set}): {sum(run['status'] == 'valid' for run in report['runs'])}/{len(report['runs'])} complete exports; provenance not verified."
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
