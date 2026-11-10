"""Compare the values printed in the final paper with the saved results in this repository.

Paper values are the strings printed in JBHI_CCTA_PCAT_2026_11_01.pdf. Repository values are
always computed from a saved file; none is copied from the paper.

Status:
  match         the saved value rounds to the printed value
  precision     the printed value has a trailing zero the saved value does not support
  scope         the saved value matches, but for a narrower population than the paper states
  mismatch      the saved value differs from the printed value
  not_archived  no saved file holds the value
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIELDS = ("location", "quantity", "paper_value", "repository_value", "status", "source", "note")
# Published comparators quoted in Tables III-IV; margins are recomputed against them.
IMAGECAS_BASELINE, SEG2REFINENET, DIFFCAS, CAS_NET = 82.96, 83.13, 84.59, 91.20
SPLIT_1_ONLY = "Saved ensemble results cover Split 1 only; none exist for Splits 2-4"
VESSELS = ("RCA", "LAD", "LCx")


def classify(paper: str, value: float | None, narrower_scope: bool = False) -> str:
    if value is None:
        return "not_archived"
    decimals = len(paper.partition(".")[2])
    distance = abs(float(paper) - value)
    tolerance = 0.5 * 10**-decimals + 1e-9
    if distance <= tolerance:
        return "scope" if narrower_scope else "match"
    if not narrower_scope and decimals and paper.endswith("0") and distance <= 10 * tolerance:
        return "precision"
    return "mismatch"


def load_json(relative: str):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict]:
    with (RESULTS / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def limits(bias: float, sd: float) -> tuple[float, float]:
    return bias - 1.96 * sd, bias + 1.96 * sd


def side_branch_precision(edges: list[dict]) -> list[float]:
    named = Counter(), Counter()
    for edge in edges:
        if edge["auto_name"] not in ("RCA", "LM", "LAD", "LCx", "Ramus"):
            named[0][edge["auto_name"]] += 1
            named[1][edge["auto_name"]] += edge["correct"] == "True"
    return [100 * named[1][name] / count for name, count in named[0].items() if count >= 10]


def build_rows() -> list[dict]:
    rows = []

    def add(location, quantity, paper, value, source, *, narrower_scope=False, note=""):
        shown = "" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")
        status = classify(paper, value, narrower_scope)
        rows.append(dict(zip(FIELDS, (location, quantity, paper, shown, status, source, note))))

    add_segmentation(add)
    add_validation_selection(add)
    add_trees(add)
    add_pcat(add)
    add_eat(add)
    add_calcium(add)
    add_external(add)
    return rows


def add_segmentation(add):
    four_path = "imagecas/foursplit_summary.json"
    ensemble_path = "imagecas/test_access_3_result.json"
    m1_path = "imagecas-x/xbench_ours_result.json"
    post_path = "imagecas-x/lumen_access2_test_result.json"
    authors_m1_path = "imagecas-x/imagecasx_evaluate_py_results.json"
    authors_post_path = "imagecas-x/imagecasx_evaluate_py_results_access2.json"
    four = load_json(four_path)["mean_sd"]
    ensemble = load_json(ensemble_path)["test"]
    m1, post = load_json(m1_path), load_json(post_path)
    hd95_m1 = load_json(authors_m1_path)["summary"]["hd95"]["mean"]
    hd95_post = load_json(authors_post_path)["summary"]["hd95"]["mean"]
    dice_single, dice_ensemble = 100 * four["dice"][0], 100 * ensemble["dice"]
    dice_m1, dice_post = 100 * m1["dice"]["mean"], 100 * post["dice"]["mean"]
    errors = m1["betti0_err"]["mean"], post["betti0_err"]["mean"]

    split1 = dict(narrower_scope=True, note=SPLIT_1_ONLY)
    add("Abstract", "ImageCAS-X Dice, M1 + post-processing", "0.9172", dice_post / 100, post_path)
    add(
        "Abstract",
        "ImageCAS-X clDice, M1 + post-processing",
        "0.940",
        post["cldice"]["mean"],
        post_path,
    )
    add(
        "Abstract",
        "ImageCAS Dice, ensemble, four splits",
        "0.8490",
        dice_ensemble / 100,
        ensemble_path,
        **split1,
    )
    add(
        "Contributions",
        "ImageCAS clDice, ensemble, four splits",
        "0.8981",
        ensemble["cldice"],
        ensemble_path,
        **split1,
    )

    table = "Table III"
    add(table, "Single ResEnc-L Dice, four splits", "84.16", dice_single, four_path)
    add(table, "Single ResEnc-L clDice, four splits", "89.24", 100 * four["cldice"][0], four_path)
    add(table, "Single ResEnc-L component-count error", "2.90", four["betti0_err"][0], four_path)
    add("Section IV-D", "Single ResEnc-L HD95 (mm)", "8.90", four["hd95_mm"][0], four_path)
    add(table, "Ensemble Dice, four splits", "84.90", dice_ensemble, ensemble_path, **split1)
    add(
        table,
        "Ensemble clDice, four splits",
        "89.81",
        100 * ensemble["cldice"],
        ensemble_path,
        **split1,
    )
    add(
        table,
        "Ensemble component-count error",
        "2.90",
        ensemble["betti0_err"],
        ensemble_path,
        **split1,
    )
    add("Section IV-D", "Ensemble HD95 (mm)", "8.37", ensemble["hd95_mm"], ensemble_path, **split1)
    add(
        "Section IV-D",
        "Ensemble margin over DiffCAS",
        "0.31",
        dice_ensemble - DIFFCAS,
        ensemble_path,
        **split1,
    )
    add(
        "Section IV-D",
        "Ensemble margin over ImageCAS baseline",
        "1.94",
        dice_ensemble - IMAGECAS_BASELINE,
        ensemble_path,
        **split1,
    )
    add(
        "Section IV-D",
        "Ensemble margin over Seg2RefineNet",
        "1.77",
        dice_ensemble - SEG2REFINENET,
        ensemble_path,
        **split1,
    )

    table = "Table IV"
    add(table, "M1 Dice", "91.40", dice_m1, m1_path)
    add(table, "M1 clDice", "93.90", 100 * m1["cldice"]["mean"], m1_path)
    add(table, "M1 HD95 (mm), ImageCAS-X evaluator", "2.39", hd95_m1, authors_m1_path)
    add(table, "M1 component-count error", "3.26", errors[0], m1_path)
    add(table, "M1 + post-processing Dice", "91.72", dice_post, post_path)
    add(table, "M1 + post-processing clDice", "94.00", 100 * post["cldice"]["mean"], post_path)
    add(
        table,
        "M1 + post-processing HD95 (mm), ImageCAS-X evaluator",
        "2.47",
        hd95_post,
        authors_post_path,
    )
    add(table, "M1 + post-processing component-count error", "0.96", errors[1], post_path)
    add(
        "Section IV-D",
        "Post-processing margin over CAS-Net",
        "0.52",
        dice_post - CAS_NET,
        post_path,
    )
    add(
        "Section IV-D",
        "Component-count error reduction (%)",
        "70.6",
        100 * (errors[0] - errors[1]) / errors[0],
        post_path,
    )
    add("Section IV-D", "HD95 penalty (mm)", "0.08", hd95_post - hd95_m1, authors_post_path)


def add_validation_selection(add):
    path = "imagecas-x/validation_selection.csv"
    printed = {
        "M1": ("91.32", "93.80", "2.38", "3.23", "95.30"),
        "M1 + hysteresis": ("91.56", "93.80", "2.44", "2.76", "95.40"),
        "M1 + M3": ("91.45", "93.70", "2.38", "4.97", "95.70"),
        "M1 + M3 + component filter": ("91.50", "93.90", "2.61", "1.15", "94.70"),
        "M1 + hysteresis + component filter": ("91.61", "94.00", "2.60", "0.90", "94.90"),
    }
    metrics = (
        ("Dice", "dice", 100),
        ("clDice", "cldice", 100),
        ("HD95 (mm)", "hd95_mm", 1),
        ("component-count error", "component_count_error", 1),
        ("coverage", "coverage_1mm", 100),
    )
    for row in load_csv(path):
        for paper, (label, key, scale) in zip(printed[row["configuration"]], metrics):
            add(
                "Table VI", f"{row['configuration']}: {label}", paper, scale * float(row[key]), path
            )


def add_trees(add):
    per_case_path = "imagecas-x/branch_names_800_per_case.csv"
    summary_path = "imagecas-x/oos_tree_summary_800.csv"
    lumen_path = "imagecas-x/branch_names_lumen_160_per_case.csv"
    label_path = "imagecas-x/label_convention_800.csv"
    edges_path = "imagecas-x/branch_names_800_edges.csv"
    named = {row["case"]: row["all_main_correct"] == "True" for row in load_csv(per_case_path)}
    summary = load_csv(summary_path)
    complete = [named[row["case"]] for row in summary if row["status"] == "complete_candidate"]
    flagged = [named[row["case"]] for row in summary if row["status"] != "complete_candidate"]
    lumen = [row["all_main_correct"] == "True" for row in load_csv(lumen_path)]
    labels = load_csv(label_path)
    precision = side_branch_precision(load_csv(edges_path))
    test = (REPO / "configs/imagecasx_filelists/test.txt").read_text(encoding="utf-8").split()
    manifest = "configs/imagecas_v2_split_manifest.csv"
    with (REPO / manifest).open(encoding="utf-8-sig", newline="") as handle:
        train = {row["case_id"] for row in csv.DictReader(handle) if row["split_1"] == "train"}

    add(
        "Section IV-A",
        "ImageCAS-X test cases in Split 1 training",
        "107",
        len(set(test) & train),
        manifest,
    )
    add(
        "Section IV-A",
        "Whole-vessel to lumen volume ratio (median)",
        "3.4",
        statistics.median(float(r["ratio"]) for r in labels),
        label_path,
    )
    add("Section IV-A", "Whole-vessel to lumen Dice", "0.42", mean(labels, "dice"), label_path)
    both = sum(r["left_found"] == "True" and r["right_found"] == "True" for r in summary)
    add("Table V", "Both ostia found, of 800", "799", both, summary_path)
    add("Table V", "All four vessels correct, of 800", "697", sum(named.values()), per_case_path)
    add(
        "Table V",
        "All four correct (%)",
        "87.1",
        100 * sum(named.values()) / len(named),
        per_case_path,
    )
    add(
        "Section IV-E",
        "All four correct, QC-complete (%)",
        "89.5",
        100 * statistics.fmean(complete),
        per_case_path,
    )
    add(
        "Section IV-E",
        "All four correct, flagged (%)",
        "51.0",
        100 * statistics.fmean(flagged),
        per_case_path,
    )
    add(
        "Table V",
        "All four correct, lumen M1 (%)",
        "83.1",
        100 * statistics.fmean(lumen),
        lumen_path,
    )
    add("Section IV-E", "Lowest side-branch edge precision (%)", "29", min(precision), edges_path)
    add("Section IV-E", "Highest side-branch edge precision (%)", "57", max(precision), edges_path)


def add_pcat(add):
    lumen_path = "imagecas-x/pcat_lumen_vs_expert_160.json"
    whole_path = "imagecas-x/pcat_auto_vs_expert_800.json"
    control_path = "imagecas-x/pcat_expert_mask_control_800.json"
    printed = dict(
        lumen=(
            (lumen_path, "Table V / Fig. 4", "M1"),
            dict(
                RCA=("158", "+0.7", "0.97", "-2.6", "+4.1"),
                LAD=("154", "+0.2", "0.93", "-5.3", "+5.7"),
                LCx=("154", "-0.4", "0.95", "-4.9", "+4.1"),
            ),
        ),
        whole=(
            (whole_path, "Section IV-F / Fig. 4", "whole-vessel"),
            dict(
                RCA=("784", "-2.6", "0.84", "-9.5", "+4.3"),
                LAD=("788", "-4.3", "0.76", "-10.6", "+2.0"),
                LCx=("788", "-4.3", "0.75", "-11.2", "+2.5"),
            ),
        ),
    )
    for (path, location, model), by_vessel in printed.values():
        data = load_json(path)
        for vessel in VESSELS:
            saved = data[vessel]
            low, high = limits(saved["diff_mean"], saved["diff_sd"])
            computed = saved["n_both"], saved["diff_mean"], saved["icc"], low, high
            labels = ("pairs", "bias (HU)", "ICC", "lower LoA (HU)", "upper LoA (HU)")
            for paper, label, value in zip(by_vessel[vessel], labels, computed):
                add(location, f"{model} {vessel} {label}", paper, value, path)
    vessels = load_json(control_path)["vessels"]
    differences = {name: vessels[name]["fai_difference_mean_hu"] for name in VESSELS}
    for vessel, paper in zip(VESSELS, ("3.04", "4.24", "3.78")):
        add(
            "Section IV-F",
            f"Whole-vessel minus lumen {vessel} (HU, magnitude)",
            paper,
            abs(differences[vessel]),
            control_path,
        )
    smallest, largest = min(map(abs, differences.values())), max(map(abs, differences.values()))
    add("Abstract", "Smallest paired PCAT difference (HU)", "3.0", smallest, control_path)
    add("Abstract", "Largest paired PCAT difference (HU)", "4.2", largest, control_path)


def add_eat(add):
    visual_path = "visual-lab/visuallab_default_per_patient.csv"
    learned_path = "imagecas-x/eat_test_result.json"
    test10 = sorted(load_csv(visual_path), key=lambda row: row["patient"])[-10:]
    learned = load_json(learned_path)
    add("Section IV-G", "Visual Lab rule Dice", "0.664", mean(test10, "dice_eat"), visual_path)
    add(
        "Section IV-G",
        "Visual Lab rule Dice, line ignored",
        "0.705",
        mean(test10, "dice_eat_ign_line"),
        visual_path,
    )
    add("Section IV-G", "Visual Lab sensitivity", "0.83", mean(test10, "eat_sens"), visual_path)
    add("Section IV-G", "Visual Lab PPV", "0.57", mean(test10, "eat_ppv"), visual_path)
    add(
        "Section IV-G",
        "Learned predictor Dice against rule",
        "0.857",
        learned["dice_vs_rule_mean"],
        learned_path,
    )
    add(
        "Section IV-G",
        "Learned predictor volume correlation",
        "0.997",
        learned["volume_r"],
        learned_path,
    )


def add_calcium(add):
    gated_path = "coca/cacseg_heldout_result.json"
    nongated_path = "coca/cacseg_nongated_c4_result.json"
    ablation_path = "coca/validation_ablations.csv"
    gated = load_json(gated_path)["test"]
    nongated = load_json(nongated_path)["nongated"]
    ablations = {row["configuration"]: float(row["dice"]) for row in load_csv(ablation_path)}
    crop = ablations["C2 heart crop"]
    add("Table V", "Gated 3D Dice", "0.881", gated["dice3d_mean"], gated_path)
    add("Table V", "Reference-positive gated scans", "65", gated["n_positive"], gated_path)
    add("Section IV-H", "Lesion sensitivity", "0.903", gated["lesion_sensitivity"], gated_path)
    add("Section IV-H", "Lesion PPV", "0.873", gated["lesion_ppv"], gated_path)
    add("Table V", "Gated weighted kappa", "0.854", gated["kappa_lin_4cat"], gated_path)
    add("Table V", "Gated score ICC", "0.946", gated["icc21"], gated_path)
    add("Section IV-H", "Gated score bias", "+34", gated["bland_altman_bias"], gated_path)
    add("Section IV-H", "Gated lower LoA", "-322", gated["bland_altman_loa"][0], gated_path)
    add("Section IV-H", "Gated upper LoA", "+391", gated["bland_altman_loa"][1], gated_path)
    add("Table V", "Non-gated scans", "210", nongated["n"], nongated_path)
    add("Table V", "Non-gated weighted kappa", "0.573", nongated["kappa_lin_4cat"], nongated_path)
    add("Table V", "Non-gated score ICC", "0.820", nongated["icc21"], nongated_path)
    add(
        "Section IV-I",
        "Calcium ResEnc-L gain (points)",
        "21.8",
        100 * (ablations["C4 heart crop with ResEnc-L"] - crop),
        ablation_path,
    )
    add(
        "Section IV-I",
        "Calcium mirroring loss (points)",
        "5.0",
        100 * (crop - ablations["C2 heart crop with mirroring"]),
        ablation_path,
    )
    note = "Earlier single-run validation result; no saved summary in this repository"
    add("Section IV-I", "Coronary ResEnc-L gain (points)", "2.2", None, "", note=note)
    add("Section IV-I", "Coronary mirroring loss (points)", "2.7", None, "", note=note)


def add_external(add):
    path = "multid4cad/multid4cad_summary.json"
    summary = load_json(path)
    eat, pat, tree = summary["eat"], summary["pat"], summary["tree"]
    rule, learned = eat["volume_rule_ml"], eat["volume_learned_ml"]
    plain, fat_window = pat["lad_vs_reference_hu"], pat["lad_vs_reference_fat_window_hu"]
    auc = {row["feature"]: row["auc"] for row in summary["cad_association"]}
    eat_auc = [value for name, value in auc.items() if name.startswith("EAT volume")]
    pcat_auc = [value for name, value in auc.items() if name.startswith("FAI")]
    local_path = "imagecas-x/lumen_tree_summary_160.csv"
    internal = load_csv(local_path)
    internal_complete = sum(row["status"] == "complete_candidate" for row in internal)

    add("Abstract", "External rule EAT volume correlation", "0.944", rule["pearson_r"], path)
    add(
        "Abstract", "External rule Dice, matched window", "0.800", eat["dice_rule_fat_window"], path
    )
    add("Table V", "MultiD4CAD rule Dice", "0.725", eat["dice_rule"], path)
    add("Table V", "MultiD4CAD learned Dice", "0.715", eat["dice_learned"], path)
    add("Table V", "MultiD4CAD learned volume correlation", "0.696", learned["pearson_r"], path)
    add("Table V", "MultiD4CAD rule volume bias (mL)", "-78", rule["bias"], path)
    add("Table V", "MultiD4CAD learned volume bias (mL)", "-46", learned["bias"], path)
    add("Section IV-G", "MultiD4CAD rule lower LoA (mL)", "-203", rule["loa"][0], path)
    add("Section IV-G", "MultiD4CAD rule upper LoA (mL)", "+47", rule["loa"][1], path)
    add("Section IV-G", "LM landmark found", "98", eat["lm_landmark_found"], path)
    add(
        "Section IV-G",
        "Reference voxels outside fat window (%)",
        "16",
        100 * eat["reference_fraction_outside_fat_window"],
        path,
    )
    add("Section IV-G", "Sac coverage (%)", "75", 100 * eat["sac_coverage_fraction"], path)
    add("Section IV-F", "External PCAT paired LAD", "90", plain["n"], path)
    add("Section IV-F", "External PCAT correlation", "0.82", plain["pearson_r"], path)
    add("Section IV-F", "External PCAT bias (HU)", "-2.6", plain["bias"], path)
    add("Section IV-F", "External PCAT lower LoA (HU)", "-14.0", plain["loa"][0], path)
    add("Section IV-F", "External PCAT upper LoA (HU)", "+8.9", plain["loa"][1], path)
    add("Section IV-F", "External PCAT bias, fat window (HU)", "+3.6", fat_window["bias"], path)
    add(
        "Section IV-F",
        "External PCAT lower LoA, fat window (HU)",
        "-6.7",
        fat_window["loa"][0],
        path,
    )
    add(
        "Section IV-F",
        "External PCAT upper LoA, fat window (HU)",
        "+13.8",
        fat_window["loa"][1],
        path,
    )
    add("Section IV-J", "Median components, MultiD4CAD", "13", tree["median_components"], path)
    add(
        "Section IV-J",
        "Median components, internal",
        "5",
        statistics.median(float(r["n_components_raw"]) for r in internal),
        local_path,
    )
    add("Section IV-J", "Both ostia found, of 118", "100", tree["both_ostia_found"], path)
    add("Table V", "QC-complete candidates, internal, of 160", "149", internal_complete, local_path)
    add(
        "Section IV-J",
        "QC-complete candidates, internal (%)",
        "93.1",
        100 * internal_complete / len(internal),
        local_path,
    )
    add(
        "Section IV-J",
        "QC-complete candidates, external, of 118",
        "37",
        tree["complete_candidates"],
        path,
    )
    add(
        "Section IV-J",
        "QC-complete candidates, external (%)",
        "31.4",
        100 * tree["complete_candidates"] / tree["n"],
        path,
    )
    for vessel, paper in zip(VESSELS, ("96", "90", "90")):
        add(
            "Section IV-J",
            f"External PCAT available, {vessel}",
            paper,
            pat["available"][vessel],
            path,
        )
    add("Section IV-J", "Lowest CAD AUC, EAT", "0.40", min(eat_auc), path)
    add("Section IV-J", "Highest CAD AUC, EAT", "0.44", max(eat_auc), path)
    add("Section IV-J", "Lowest CAD AUC, PCAT", "0.44", min(pcat_auc), path)
    add("Section IV-J", "Highest CAD AUC, PCAT", "0.57", max(pcat_auc), path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new CSV report")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing CSV report")
    args = parser.parse_args(argv)
    rows = build_rows()
    if args.output:
        if args.output.exists() and not args.overwrite:
            parser.error("Output already exists; use --overwrite to replace it")
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(Counter(row["status"] for row in rows), indent=1))
    for row in rows:
        if row["status"] != "match":
            print(
                f"{row['status']:12} {row['location']:14} {row['quantity']}: paper {row['paper_value']}, repository {row['repository_value'] or 'none'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
