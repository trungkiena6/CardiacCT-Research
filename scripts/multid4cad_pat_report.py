"""Render the LAD PAT report from saved patient rows, without opening CT images.

python scripts/multid4cad_pat_report.py --csv INPUT.csv --out REPORT.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CUTOFF_HU = -70.1
REFERENCE_FIELDS = (
    ("all_voxels", "pat_ref_hu", "mean HU of the radiologist PAT mask (all voxels)"),
    ("fat_window", "pat_ref_hu_fatwin", "mean HU of the radiologist mask inside -190…-30"),
)


def _values(rows, field):
    values = np.array([float(row[field]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite saved values in {field}")
    return values


def summarize_pat(rows):
    """Use HU for all threshold comparisons and retain the paired-case denominator."""
    paired = [row for row in rows if row.get("lad_fai_auto") not in (None, "", "None")]
    result = {"n_patients": len(rows), "n_paired": len(paired)}
    if len(paired) < 3:
        return result

    automatic_hu = _values(paired, "lad_fai_auto")
    automatic_high = automatic_hu >= CUTOFF_HU
    result.update(
        automatic_mean_hu=float(automatic_hu.mean()),
        automatic_ge_cutoff=int(automatic_high.sum()),
        references={},
    )
    for key, field, _ in REFERENCE_FIELDS:
        reference_hu = _values(paired, field)
        difference_hu = automatic_hu - reference_hu
        bias_hu = float(difference_hu.mean())
        sd_hu = float(difference_hu.std(ddof=1))
        reference_high = reference_hu >= CUTOFF_HU
        result["references"][key] = dict(
            mean_hu=float(reference_hu.mean()),
            r=float(np.corrcoef(reference_hu, automatic_hu)[0, 1]),
            bias_hu=bias_hu,
            sd_hu=sd_hu,
            loa_hu=[bias_hu - 1.96 * sd_hu, bias_hu + 1.96 * sd_hu],
            ge_cutoff=int(reference_high.sum()),
            same_side=int((automatic_high == reference_high).sum()),
        )

    reference_ml = _values(paired, "pat_ref_ml")
    automatic_ml = _values(paired, "lad_pcat_ml_auto")
    result["volume"] = dict(
        reference_median_ml=float(np.median(reference_ml)),
        automatic_median_ml=float(np.median(automatic_ml)),
        r=float(np.corrcoef(reference_ml, automatic_ml)[0, 1]),
    )
    result["all_reference"] = dict(
        mean_hu=float(_values(rows, "pat_ref_hu").mean()),
        mean_fat_fraction=float(_values(rows, "pat_ref_frac_fatwin").mean()),
        median_p5_hu=float(np.median(_values(rows, "pat_ref_p5"))),
        median_p95_hu=float(np.median(_values(rows, "pat_ref_p95"))),
    )
    return result


def render_pat_report(rows):
    summary = summarize_pat(rows)
    n = summary["n_paired"]
    lines = [
        f"# Pericoronary fat around the LAD: automatic PCAT/FAI vs the MultiD4CAD radiologist PAT mask ({summary['n_patients']} patients, automatic LAD available in {n})",
        "",
        "The radiologist PAT volume is a reformatted sub-volume along the LAD whose header carries the full volume's affine, so no voxel-wise overlap is",
        "possible; the comparison is at patient level. Their mask covers the LAD over an unspecified length; ours covers 0–40 mm from the LM end,",
        "shell r+0.5 < d ≤ 3r+0.5 mm (width 2r), fat -190…-30 HU. All scans are 120 kVp. FAI here is uncorrected mean PCAT attenuation.",
        "The published -70.1 HU threshold is used only as an exploratory reference; these comparisons do not establish clinical risk categories or calibrated clinical FAI.",
        "",
    ]
    if n < 3:
        lines.append(
            "Fewer than three paired automatic LAD measurements are available; agreement is not summarized."
        )
        return "\n".join(lines) + "\n"

    for key, _, label in REFERENCE_FIELDS:
        stats = summary["references"][key]
        lines.append(
            f"- FAI automatic vs {label}: r {stats['r']:.3f}, bias {stats['bias_hu']:+.1f} HU, SD {stats['sd_hu']:.1f}, LoA [{stats['loa_hu'][0]:+.1f}, {stats['loa_hu'][1]:+.1f}] (n {n})."
        )
    references = summary["references"]
    lines.append(
        f"- Paired mean attenuation (n {n}): automatic {summary['automatic_mean_hu']:.2f} HU; radiologist all-mask {references['all_voxels']['mean_hu']:.2f} HU; radiologist fat-window {references['fat_window']['mean_hu']:.2f} HU."
    )
    volume = summary["volume"]
    lines.append(
        f"- Volume: radiologist median {volume['reference_median_ml']:.2f} mL vs automatic LAD 0–40 mm median {volume['automatic_median_ml']:.2f} mL, r {volume['r']:.3f} (extents differ by design)."
    )
    cohort = summary["all_reference"]
    lines.append(
        f"- Radiologist mask HU (all {summary['n_patients']} references): mean {cohort['mean_hu']:.1f}, fraction inside -190…-30 window {cohort['mean_fat_fraction']:.3f}, p5/p95 median {cohort['median_p5_hu']:.0f}/{cohort['median_p95_hu']:.0f}."
    )
    lines += [
        "",
        "## Exploratory threshold comparison on paired cases",
        "",
        "All rows below use the same patients with automatic LAD FAI available. Counts compare HU, never PCAT volume.",
        "",
        "| Measurement | ≥ -70.1 HU | Share |",
        "|---|---:|---:|",
    ]
    for label, count in (
        ("Automatic LAD FAI", summary["automatic_ge_cutoff"]),
        ("Radiologist all-mask mean HU", references["all_voxels"]["ge_cutoff"]),
        ("Radiologist fat-window mean HU", references["fat_window"]["ge_cutoff"]),
    ):
        lines.append(f"| {label} | {count}/{n} | {100 * count / n:.1f} % |")
    lines += [
        "",
        "| Automatic and reference on the same side of -70.1 HU | Cases | Share |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("all_voxels", "Radiologist all-mask mean"),
        ("fat_window", "Radiologist fat-window mean"),
    ):
        count = references[key]["same_side"]
        lines.append(f"| {label} | {count}/{n} | {100 * count / n:.1f} % |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    args.out.write_text(render_pat_report(rows), encoding="utf-8")
    print(f"Regenerated {args.out} from {args.csv}; no image scoring or inference.")


if __name__ == "__main__":
    main()
