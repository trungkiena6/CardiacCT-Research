"""Audit saved EAT/PCAT outputs; never run inference or change source data.

Run with a Python environment containing numpy and SimpleITK:
  python audit_eat_pcat_provenance.py --work-root D:/imagecas-work

The optional local-data audit reads the ten masks missing from the published
per-patient EAT CSV and their saved TotalSegmentator labels. Included references
are established from the existing nonzero ref_ml values, not re-evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(
        n=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        q25=float(np.percentile(values, 25)),
        q75=float(np.percentile(values, 75)),
    )


def nifti_payload(path):
    """Check uncompressed NIfTI-1 payload length before a tolerant reader loads it."""
    with path.open("rb") as handle:
        header = handle.read(348)
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    dims = struct.unpack(endian + "8h", header[40:56])
    bitpix = struct.unpack(endian + "h", header[72:74])[0]
    offset = struct.unpack(endian + "f", header[108:112])[0]
    required = int(offset) + int(np.prod(dims[1 : dims[0] + 1])) * bitpix // 8
    actual = path.stat().st_size
    return dict(actual_bytes=actual, required_bytes=required, payload_complete=actual >= required)


def bootstrap_ci(values, statistic, n_boot=10000, seed=0):
    """Percentile CI; resample complete patient rows (paired measurements stay paired)."""
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array(
        [statistic(data[rng.integers(0, len(data), len(data))]) for _ in range(n_boot)]
    )
    return [float(v) for v in np.percentile(samples[np.isfinite(samples)], [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("D:/imagecas-work"))
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/multid4cad/audit_eat_pcat_provenance.json",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "results"
    sys.path.insert(0, str(repo / "src"))
    from coronary_tree.pcat.compare import icc21

    eat_csv = artifacts / "multid4cad/multid4cad_eat_per_patient.csv"
    auto_csv = artifacts / "imagecas-x/pcat_lumen_160_per_vessel.csv"
    expert_csv = artifacts / "imagecas-x/pcat_expert_800_per_vessel.csv"
    eat = rows(eat_csv)
    included = {row["pid"] for row in eat}
    cohort = {f"P{i:03d}" for i in range(1, 119)}
    omitted = sorted(cohort - included)
    x = np.array([float(row["ref_ml"]) for row in eat])
    y = np.array([float(row["ours_in_ref_range_ml"]) for row in eat])
    d, m = y - x, (x + y) / 2
    slope, intercept = np.polyfit(m, d, 1)
    result = dict(
        audit_boundary="Saved outputs and omitted-case source masks only; no new predictions, training, or inference.",
        inputs={str(p.relative_to(repo)): sha256(p) for p in (eat_csv, auto_csv, expert_csv)},
        eat=dict(
            cohort_n=118,
            scored_n=len(eat),
            included_references_all_positive=bool((x > 0).all()),
            omitted=omitted,
            mean_dice_rule_refrange=float(
                np.mean([float(row["dice_rule_refrange"]) for row in eat])
            ),
            mean_dice_rule_refrange_fatwin=float(
                np.mean([float(row["dice_rule_refrange_fatwin"]) for row in eat])
            ),
            rule_bias_ml=float(d.mean()),
            rule_difference_sd_ml=float(d.std(ddof=1)),
            rule_loa_ml=[
                float(d.mean() - 1.96 * d.std(ddof=1)),
                float(d.mean() + 1.96 * d.std(ddof=1)),
            ],
            rule_volume_correlation=float(np.corrcoef(x, y)[0, 1]),
            ba_difference_vs_mean_slope=float(slope),
            ba_intercept_ml=float(intercept),
            rule_volume_correlation_ci95=bootstrap_ci(
                np.column_stack([x, y]), lambda a: np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            ),
            model_volume_correlation_ci95=bootstrap_ci(
                np.column_stack([x, [float(row["model_in_ref_range_ml"]) for row in eat]]),
                lambda a: np.corrcoef(a[:, 0], a[:, 1])[0, 1],
            ),
            ratio_rule_to_reference_in_ref_range=describe(y / x),
            ratio_sums_in_ref_range=float(y.sum() / x.sum()),
            ratio_rule_to_reference_full_protocol=describe(
                [float(row["ours_ml"]) / float(row["ref_ml"]) for row in eat]
            ),
            absolute_axial_extent_difference_mm=describe(
                [abs(float(row["our_range_mm"]) - float(row["ref_slices_mm"])) for row in eat]
            ),
            reference_above_rule_top_mm=describe(
                [float(row["ref_above_our_top_mm"]) for row in eat]
            ),
            interpretation="The BA slope relates difference to pair mean and does not imply a constant volume ratio. Report ratios directly, with their domain.",
        ),
    )
    auto, expert = rows(auto_csv), rows(expert_csv)
    auto_cases = {row["case"] for row in auto}
    expert_keys = {(f"Case{int(row['case']):04d}", row["vessel"]): row for row in expert}
    result["pcat"] = dict(
        auto_cohort_n=len(auto_cases),
        expert_pool_n=len({row["case"] for row in expert}),
        vessels={},
    )
    result["new_saved_output_bootstrap"] = dict(
        n_resamples=10000,
        seed=0,
        method="percentile 2.5 and 97.5; patient resampling, paired values retained together; RNG reinitialized for each endpoint",
        status="Reproducible reaggregation of existing per-case outputs for submission revision; not a new prediction or a reconstruction of undocumented previous bootstrap runs.",
    )
    for vessel in ("RCA", "LAD", "LCx"):
        current = [row for row in auto if row["vessel"] == vessel]
        valid = [row for row in current if row["fai_hu"] not in ("", "None")]
        pairs = [
            row
            for row in valid
            if expert_keys.get((row["case"], vessel), {}).get("fai_hu") not in (None, "", "None")
        ]
        result["pcat"]["vessels"][vessel] = dict(
            n_rows=len(current),
            n_auto_valid=len(valid),
            n_paired=len(pairs),
            availability=float(len(valid) / len(auto_cases)),
            failures=dict(
                Counter(
                    f"{row['status']}/{row['reason']}"
                    for row in current
                    if row["fai_hu"] in ("", "None")
                )
            ),
            fai_icc21_ci95=bootstrap_ci(
                np.array(
                    [
                        [float(expert_keys[(row["case"], vessel)]["fai_hu"]), float(row["fai_hu"])]
                        for row in pairs
                    ]
                ),
                lambda a: icc21(a[:, 0], a[:, 1]),
            ),
        )
    visual_input = repo / "results/visual-lab/visuallab_default_per_patient.csv"
    if not visual_input.exists():
        visual_input = args.work_root / "visuallab/work/visuallab_default_per_patient.csv"
    if visual_input.exists():
        visual_all = sorted(rows(visual_input), key=lambda row: row["patient"])
        visual_test = visual_all[10:]
        result["visuallab"] = dict(
            input=str(visual_input),
            sha256=sha256(visual_input),
            split="last 10 of 20 patients in alphabetical order; default rule, not tuned variant",
            test_patients=[row["patient"] for row in visual_test],
            dice=describe([row["dice_eat"] for row in visual_test]),
            dice_ci95=bootstrap_ci([float(row["dice_eat"]) for row in visual_test], np.mean),
            dice_ign_line=describe([row["dice_eat_ign_line"] for row in visual_test]),
            dice_ign_line_ci95=bootstrap_ci(
                [float(row["dice_eat_ign_line"]) for row in visual_test], np.mean
            ),
        )
    if not args.skip_local:
        import SimpleITK as sitk

        base = args.work_root / "multid4cad"
        refs = base / "MultiD4CAD_Dataset_v1/Epicardial Adipose Tissue Segmentations (118 patients)"
        checks = []
        for pid in omitted:
            item = dict(pid=pid)
            for tag, path, label in (
                ("reference_original", refs / pid / "epicardialDatasetNIFTI_mask.nii", None),
                ("reference_staged", base / "stage/eat_ref" / f"{pid}.nii.gz", None),
                ("saved_anatomical_frame", base / "totalseg" / f"{pid}.nii.gz", 51),
            ):
                info = dict(path=str(path))
                try:
                    if path.suffix == ".nii":
                        info.update(nifti_payload(path))
                        if not info["payload_complete"]:
                            info["status"] = "truncated_payload"
                            item[tag] = info
                            continue
                    image = sitk.ReadImage(str(path))
                    data = sitk.GetArrayFromImage(image)
                    info.update(
                        size=list(image.GetSize()),
                        positive_voxels=int(np.count_nonzero(data > 0)),
                        status="readable",
                    )
                    if label is not None:
                        info["heart_label_51_voxels"] = int(np.count_nonzero(data == label))
                    del data, image
                except Exception as error:
                    info.update(status="read_error", error=str(error))
                item[tag] = info
            checks.append(item)
        result["eat"]["omitted_case_source_audit"] = checks
        plan_candidates = [
            base / "lumen_pred/plans.json",
            args.work_root
            / "imagecas-x/pod_backup/xwork/nnunet_results/Dataset906_ImageCASX_lumen/nnUNetTrainer_250epochs_NoMirroring__nnUNetResEncUNetL40GPlans__3d_fullres/plans.json",
            args.work_root / "nnunet_preprocessed/Dataset901_ImageCAS_S1/nnUNetPlans.json",
        ]
        plans = []
        for path in plan_candidates:
            if not path.exists():
                continue
            plan = json.loads(path.read_text(encoding="utf-8"))
            configuration = plan["configurations"]["3d_fullres"]
            plans.append(
                dict(
                    path=str(path),
                    sha256=sha256(path),
                    dataset_name=plan.get("dataset_name"),
                    plans_name=plan.get("plans_name"),
                    image_reader_writer=plan.get("image_reader_writer"),
                    transpose_forward=plan.get("transpose_forward"),
                    configuration={
                        key: configuration.get(key)
                        for key in (
                            "spacing",
                            "patch_size",
                            "resampling_fn_data",
                            "resampling_fn_data_kwargs",
                            "resampling_fn_seg",
                            "resampling_fn_seg_kwargs",
                            "resampling_fn_probabilities",
                            "resampling_fn_probabilities_kwargs",
                            "normalization_schemes",
                        )
                    },
                    intensity_properties=plan.get("foreground_intensity_properties_per_channel"),
                )
            )
        result["saved_plans"] = plans
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    print(
        json.dumps(
            {
                "scored_n": len(eat),
                "omitted": omitted,
                "pcat_cohort_n": len(auto_cases),
                "ba_slope": float(slope),
                "ratio_mean": float((y / x).mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
