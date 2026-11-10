"""Score complete EAT predictions against rule labels without overwriting results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coronary_tree.evaluate.export_contract import IMAGECASX_TEST_DIGEST, case_ids_digest
from coronary_tree.evaluate.segmentation import AFFINE_TOL, load_pair


def read_case_ids(path: Path, expected_count: int = 160) -> list[str]:
    ids = path.read_text(encoding="utf-8-sig").split()
    if (
        len(ids) != expected_count
        or len(set(ids)) != expected_count
        or any(re.fullmatch(r"[0-9]+", case) is None for case in ids)
    ):
        raise ValueError(f"Expected {expected_count} distinct official numeric case IDs")
    if expected_count == 160 and case_ids_digest(ids) != IMAGECASX_TEST_DIGEST:
        raise ValueError("Case IDs differ from the official ImageCAS-X test-160 set")
    return ids


def require_case_set(directory: Path, expected: list[str]) -> None:
    actual = {path.name[:-7] for path in directory.glob("*.nii.gz")}
    if actual != set(expected):
        raise ValueError(
            f"{directory}: missing or unexpected cases; no partial-scope summary written"
        )


def score_predictions(pred: Path, gt: Path, cases: Path, output: Path, *, expected_count=160):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Existing summary preserved: {output}")
    ids = read_case_ids(cases, expected_count)
    require_case_set(pred, ids)
    require_case_set(gt, ids)
    rows = []
    for case in sorted(ids):
        label, prediction, problems = load_pair(gt / f"{case}.nii.gz", pred / f"{case}.nii.gz")
        if len(prediction.shape) != 3 or len(label.shape) != 3 or prediction.shape != label.shape:
            raise ValueError(f"{case}: incompatible prediction/reference shape")
        if (
            problems
            or not np.isfinite(prediction.affine).all()
            or not np.isfinite(label.affine).all()
            or not np.allclose(prediction.affine, label.affine, atol=AFFINE_TOL, rtol=0)
        ):
            raise ValueError(f"{case}: prediction/reference geometry differs")
        p = np.asanyarray(prediction.dataobj)
        g = np.asanyarray(label.dataobj)
        if not np.issubdtype(p.dtype, np.number) or not np.issubdtype(g.dtype, np.number):
            raise ValueError(f"{case}: masks must contain numeric scalar values")
        if not np.isfinite(p).all() or not np.isfinite(g).all():
            raise ValueError(f"{case}: nonfinite mask values")
        p, g = p > 0, g > 0
        spacing = np.asarray(prediction.header.get_zooms()[:3])
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"{case}: invalid voxel spacing")
        voxel_ml = float(np.prod(spacing)) / 1000.0
        rows.append(
            dict(
                case=case,
                dice=float(2 * (p & g).sum() / max(1, p.sum() + g.sum())),
                pred_ml=float(p.sum() * voxel_ml),
                rule_ml=float(g.sum() * voxel_ml),
            )
        )
    dice = np.array([row["dice"] for row in rows])
    predicted = np.array([row["pred_ml"] for row in rows])
    reference = np.array([row["rule_ml"] for row in rows])
    correlation = (
        float(np.corrcoef(predicted, reference)[0, 1])
        if len(rows) > 1 and np.std(predicted) > 0 and np.std(reference) > 0
        else None
    )
    result = dict(
        n=len(rows),
        dice_vs_rule_mean=float(dice.mean()),
        dice_median=float(np.median(dice)),
        pred_ml_median=float(np.median(predicted)),
        rule_ml_median=float(np.median(reference)),
        volume_r=correlation,
        volume_bias_ml=float((predicted - reference).mean()),
        per_case=rows,
    )
    payload = json.dumps(result, indent=1, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = score_predictions(args.pred_dir, args.gt_dir, args.cases, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print("EAT-SCORE", {key: value for key, value in result.items() if key != "per_case"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
