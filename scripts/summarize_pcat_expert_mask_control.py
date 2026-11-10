"""Summarize completed expert-mask PCAT outputs without inference or image scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from audit_eat_pcat_provenance import bootstrap_ci, rows, sha256

EXPLORATORY_THRESHOLD_HU = -70.1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Create a new JSON report; never overwrite a saved result"
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from coronary_tree.pcat.compare import icc21

    folder = repo / "results/imagecas-x"
    lumen_path = folder / "pcat_expert_800_per_vessel.csv"
    whole_path = folder / "pcat_expert_wholevessel_800_per_vessel.csv"
    lumen_rows, whole_rows = rows(lumen_path), rows(whole_path)
    lumen = {(int(r["case"]), r["vessel"]): r for r in lumen_rows}
    whole = {(int(r["case"]), r["vessel"]): r for r in whole_rows}
    source_paths = [
        repo / "scripts" / name
        for name in ("pcat_expert_reference.py", "pcat_expert_wholevessel.py")
    ]
    result = dict(
        boundary="Reaggregation of completed saved outputs; no new training, inference or image scoring.",
        contrast="original ImageCAS whole-vessel expert mask minus ImageCAS-X lumen expert mask",
        interpretation="Paired mask-source contrast, retaining both label-convention and annotation differences; not an isolated convention effect or a comparison of segmentation models.",
        held_fixed="Same 800 CT scans, expert labelled centrelines, segment definitions, LPS-to-RAS conversion, heart/aorta frame and pcat_case routine.",
        source_comparison="pcat_expert_wholevessel.py differs from pcat_expert_reference.py only in its docstring, tree source string and segmentation input path. Coordinate transforms and centreline assembly are unchanged. Local EDT radii and coronary-exclusion masks vary with the expert mask.",
        endpoint_definition="fai_hu is the raw, uncorrected mean HU of fat-window voxels selected by the same implemented PCAT routine; it is not a calibrated clinical FAI score.",
        reference_scope="Expert coronary masks and centrelines are inputs to the same routine; no independently manually contoured PCAT ground truth is available in this control. All non-missing paired measurements are retained, including needs_review statuses.",
        threshold_analysis=dict(
            threshold_hu=EXPLORATORY_THRESHOLD_HU,
            positive_definition="raw mean attenuation >= threshold; equality retained in upper group",
            crossing_direction="lumen-mask measurement to whole-vessel-mask measurement",
            intent="exploratory descriptive numerical sensitivity only",
            clinical_scope="No validation of an artery-independent clinical cutoff, inflammation diagnosis, or risk reclassification; no technical, anatomical or biological calibration applied.",
        ),
        remaining_limitations="Mask-source contrast retains annotation differences as well as label-convention differences. This original comparison does not itself test shell sensitivity; the separate shell_sensitivity_800 analysis evaluates three definitions. Independent pericoronary tissue validation remains untested.",
        inputs={
            str(path.relative_to(repo)): sha256(path)
            for path in [lumen_path, whole_path] + source_paths
        },
        bootstrap=dict(
            n_resamples=10000,
            seed=0,
            unit="patient within each artery",
            method="percentile 2.5/97.5; paired rows retained together; RNG reset per endpoint",
            multiplicity_adjustment="none; descriptive",
        ),
        vessels={},
    )
    for vessel in ("RCA", "LAD", "LCx"):
        keys = sorted(
            key
            for key in lumen
            if key[1] == vessel
            and key in whole
            and lumen[key].get("fai_hu") not in ("", None)
            and whole[key].get("fai_hu") not in ("", None)
        )
        x = np.array([float(lumen[key]["fai_hu"]) for key in keys])
        y = np.array([float(whole[key]["fai_hu"]) for key in keys])
        vx = np.array([float(lumen[key]["pcat_volume_ml"]) for key in keys])
        vy = np.array([float(whole[key]["pcat_volume_ml"]) for key in keys])
        d = y - x
        paired = np.column_stack([x, y])
        upper_lumen = x >= EXPLORATORY_THRESHOLD_HU
        upper_whole = y >= EXPLORATORY_THRESHOLD_HU
        info = dict(
            n_paired=len(keys),
            n_lumen_rows=sum(r["vessel"] == vessel for r in lumen_rows),
            n_whole_rows=sum(r["vessel"] == vessel for r in whole_rows),
            statuses_lumen={
                status: sum(lumen[key]["status"] == status for key in keys)
                for status in {lumen[key]["status"] for key in keys}
            },
            statuses_whole={
                status: sum(whole[key]["status"] == status for key in keys)
                for status in {whole[key]["status"] for key in keys}
            },
            fai_lumen_mean_hu=float(x.mean()),
            fai_whole_mean_hu=float(y.mean()),
            fai_difference_mean_hu=float(d.mean()),
            fai_difference_sd_hu=float(d.std(ddof=1)),
            fai_difference_mean_ci95_hu=bootstrap_ci(d, np.mean),
            fai_loa95_hu=[
                float(d.mean() - 1.96 * d.std(ddof=1)),
                float(d.mean() + 1.96 * d.std(ddof=1)),
            ],
            fai_icc21=icc21(x, y),
            fai_icc21_ci95=bootstrap_ci(paired, lambda a: icc21(a[:, 0], a[:, 1])),
            fai_abs_difference_gt5_fraction=float((np.abs(d) > 5).mean()),
            threshold_counts=dict(
                n_paired=len(keys),
                lumen_upper=int(upper_lumen.sum()),
                whole_upper=int(upper_whole.sum()),
                upper_to_lower=int((upper_lumen & ~upper_whole).sum()),
                lower_to_upper=int((~upper_lumen & upper_whole).sum()),
                any_crossing=int((upper_lumen != upper_whole).sum()),
            ),
            pcat_volume_lumen_mean_ml=float(vx.mean()),
            pcat_volume_whole_mean_ml=float(vy.mean()),
            pcat_volume_difference_mean_ml=float((vy - vx).mean()),
            pcat_volume_icc21=icc21(vx, vy),
            pcat_volume_icc21_ci95=bootstrap_ci(
                np.column_stack([vx, vy]), lambda a: icc21(a[:, 0], a[:, 1])
            ),
            pcat_volume_mean_patient_ratio=float((vy / vx).mean()),
            local_radius_lumen_mean_mm=float(
                np.mean([float(lumen[key]["mean_lumen_radius_mm"]) for key in keys])
            ),
            local_radius_whole_mean_mm=float(
                np.mean([float(whole[key]["mean_lumen_radius_mm"]) for key in keys])
            ),
            centreline_extent_max_abs_difference_mm={
                field: float(
                    max(abs(float(whole[key][field]) - float(lumen[key][field])) for key in keys)
                )
                for field in ("trunk_len_mm", "achieved_mm")
            },
        )
        result["vessels"][vessel] = info
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
