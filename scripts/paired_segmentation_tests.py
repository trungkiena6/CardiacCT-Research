"""Paired comparisons between our own segmentation configurations scored on the same scans.

For each pair of per-case CSVs (same test population) the script reports, per metric, the mean paired
difference (second minus first), its 95% percentile-bootstrap interval over cases (10 000 resamples,
seed 0), a two-sided Wilcoxon signed-rank test and the number of scans that improved or worsened.
Published comparators have no per-case outputs and are therefore never tested here.

  python scripts/paired_segmentation_tests.py                      # writes results/imagecas/paired_segmentation_tests.json
  python scripts/paired_segmentation_tests.py --out <file.json>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ("dice", "cldice", "hd95_mm", "n_components_pred")
PAIRS = {
    "imagecasx_test160_m1_postprocess_vs_m1": (
        "imagecas-x/xbench_ours_per_case.csv",
        "imagecas-x/lumen_access2_test_result.per_case.csv",
    ),
    "imagecas_split1_three_member_vs_single": (
        "imagecas/split1_resencl_single_per_case.csv",
        "imagecas/test_access_3_per_case.csv",
    ),
    "imagecas_split1_four_member_vs_single": (
        "imagecas/split1_resencl_single_per_case.csv",
        "imagecas/test_access_6_per_case.csv",
    ),
    "imagecas_split1_four_member_vs_three_member": (
        "imagecas/test_access_3_per_case.csv",
        "imagecas/test_access_6_per_case.csv",
    ),
}


def load(relative: str) -> dict[str, dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]
    return {row["case"]: row for row in rows}


def paired(first, second, metric: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    cases = sorted(set(first) & set(second))
    delta = np.array([float(second[c][metric]) - float(first[c][metric]) for c in cases])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(delta, delta.size).mean() for _ in range(n_boot)])
    nonzero = delta[delta != 0]
    p_value = (
        float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue) if nonzero.size else None
    )
    return {
        "n": int(delta.size),
        "mean_difference": float(delta.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "wilcoxon_two_sided_p": p_value,
        "improved": int((delta > 0).sum()),
        "worsened": int((delta < 0).sum()),
    }


def run() -> dict:
    report = {
        "method": "second minus first on the same scans; 95% percentile bootstrap over cases "
        "(10000 resamples, seed 0); two-sided Wilcoxon signed-rank on non-zero differences",
        "direction": "dice and cldice higher is better; hd95_mm and n_components_pred lower is better",
        "comparisons": {},
    }
    for name, (first_file, second_file) in PAIRS.items():
        first, second = load(first_file), load(second_file)
        report["comparisons"][name] = {
            "first": first_file,
            "second": second_file,
            "metrics": {
                m: paired(first, second, m)
                for m in METRICS
                if m in next(iter(first.values())) and m in next(iter(second.values()))
            },
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=RESULTS / "imagecas/paired_segmentation_tests.json"
    )
    args = parser.parse_args(argv)
    report = run()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, comparison in report["comparisons"].items():
        dice = comparison["metrics"]["dice"]
        print(
            f"{name}: dice {dice['mean_difference']:+.4f} "
            f"[{dice['ci95'][0]:+.4f}, {dice['ci95'][1]:+.4f}] p={dice['wilcoxon_two_sided_p']:.2e} "
            f"n={dice['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
