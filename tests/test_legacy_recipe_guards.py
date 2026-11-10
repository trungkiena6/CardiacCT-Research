"""Legacy recipes must fail closed; all fixtures use disposable local files."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@pytest.mark.parametrize(
    "recipe,variable,existing",
    [
        ("predict_out_of_sample.sh", "COROSEG_OOS_WORK", "pred_1/old.nii.gz"),
        ("predict_out_of_sample.sh", "COROSEG_OOS_WORK", "pred/old.nii.gz"),
        ("eat_train.sh", "COROSEG_EAT_WORK", "test_pred/old.nii.gz"),
        ("eat_train.sh", "COROSEG_EAT_WORK", "eat_test_result.json"),
    ],
)
def test_recipes_refuse_existing_outputs_before_waiting_or_predicting(
    tmp_path, recipe, variable, existing
):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is unavailable")
    work = tmp_path / "work"
    original = work / existing
    original.parent.mkdir(parents=True)
    original.write_bytes(b"historical result")
    env = os.environ.copy()
    env[variable] = shell_path(work)
    result = subprocess.run(
        [bash, shell_path(REPO / "scripts/training" / recipe)],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 1
    assert "preserved" in result.stderr and "fresh" in result.stderr
    assert original.read_bytes() == b"historical result"


def merge_code():
    script = (REPO / "scripts/training/predict_out_of_sample.sh").read_text()
    lines = script.splitlines()
    end = lines.index("PY_OOS_MERGE")
    start = next(i for i, line in enumerate(lines[:end]) if "<<" in line and "PY_OOS_MERGE" in line)
    return compile("\n".join(lines[start + 1 : end]), "oos_merge_fixture", "exec")


def prediction_scope(tmp_path):
    scope = {str(split): [str(split)] for split in range(1, 5)}
    (tmp_path / "oos_case_ids.json").write_text(json.dumps(scope))
    for split, ids in scope.items():
        directory = tmp_path / f"pred_{split}"
        directory.mkdir()
        (directory / f"{ids[0]}.nii.gz").write_bytes(b"fixture mask")
    return scope


def test_oos_merge_requires_exact_ids_not_equal_counts(tmp_path, monkeypatch):
    prediction_scope(tmp_path)
    wrong = tmp_path / "pred_3"
    (wrong / "3.nii.gz").rename(wrong / "999.nii.gz")
    monkeypatch.setenv("COROSEG_OOS_WORK", str(tmp_path))
    with pytest.raises(ValueError, match="prediction cases differ"):
        exec(merge_code(), {})
    assert not (tmp_path / "pred").exists()


def test_oos_merge_links_complete_scope_and_preserves_existing_merged_masks(tmp_path, monkeypatch):
    prediction_scope(tmp_path)
    monkeypatch.setenv("COROSEG_OOS_WORK", str(tmp_path))
    exec(merge_code(), {})
    merged = tmp_path / "pred"
    assert {p.name for p in merged.iterdir()} == {f"{i}.nii.gz" for i in range(1, 5)}
    for split in range(1, 5):
        assert os.path.samefile(
            merged / f"{split}.nii.gz", tmp_path / f"pred_{split}/{split}.nii.gz"
        )
    original = (merged / "1.nii.gz").read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        exec(merge_code(), {})
    assert (merged / "1.nii.gz").read_bytes() == original


def test_oos_link_failure_propagates(tmp_path, monkeypatch):
    prediction_scope(tmp_path)
    monkeypatch.setenv("COROSEG_OOS_WORK", str(tmp_path))

    def link_failure(*args):
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", link_failure)
    with pytest.raises(OSError, match="simulated link failure"):
        exec(merge_code(), {})


def load_eat():
    spec = importlib.util.spec_from_file_location(
        "eat_recipe_scoring", REPO / "scripts/score_eat_predictions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eat_pair(tmp_path):
    module = load_eat()
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    cases = tmp_path / "cases.txt"
    cases.write_text("1\n2\n", encoding="utf-8")
    for case, count in [("1", 1), ("2", 2)]:
        data = np.zeros((2, 2, 2), dtype=np.uint8)
        data.ravel()[:count] = 1
        image = nib.Nifti1Image(data, np.eye(4))
        nib.save(image, pred / f"{case}.nii.gz")
        nib.save(image, gt / f"{case}.nii.gz")
    return module, pred, gt, cases, tmp_path / "eat_test_result.json"


def test_eat_complete_summary_preserves_schema_and_refuses_overwrite(tmp_path):
    module, pred, gt, cases, out = eat_pair(tmp_path)
    result = module.score_predictions(pred, gt, cases, out, expected_count=2)
    assert set(result) == {
        "n",
        "dice_vs_rule_mean",
        "dice_median",
        "pred_ml_median",
        "rule_ml_median",
        "volume_r",
        "volume_bias_ml",
        "per_case",
    }
    assert result["n"] == 2 and result["dice_vs_rule_mean"] == pytest.approx(1.0)
    original = out.read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        module.score_predictions(pred, gt, cases, out, expected_count=2)
    assert out.read_bytes() == original


def test_eat_missing_case_cannot_produce_partial_summary(tmp_path):
    module, pred, gt, cases, out = eat_pair(tmp_path)
    (pred / "2.nii.gz").unlink()
    with pytest.raises(ValueError, match="partial-scope"):
        module.score_predictions(pred, gt, cases, out, expected_count=2)
    assert not out.exists()


def test_eat_geometry_and_nonfinite_masks_stop_without_summary(tmp_path):
    module, pred, gt, cases, out = eat_pair(tmp_path)
    path = pred / "2.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] = 2.0
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)
    with pytest.raises(ValueError, match="geometry differs"):
        module.score_predictions(pred, gt, cases, out, expected_count=2)
    assert not out.exists()
    invalid = nib.Nifti1Image(np.full((2, 2, 2), np.nan, dtype=np.float32), np.eye(4))
    nib.save(invalid, path)
    with pytest.raises(ValueError, match="nonfinite"):
        module.score_predictions(pred, gt, cases, out, expected_count=2)
    assert not out.exists()


def test_eat_duplicate_and_wrong_160_case_lists_are_rejected(tmp_path):
    module = load_eat()
    cases = tmp_path / "cases.txt"
    cases.write_text("1\n1\n")
    with pytest.raises(ValueError, match="distinct"):
        module.read_case_ids(cases, expected_count=2)
    cases.write_text("\n".join(map(str, range(1, 161))))
    with pytest.raises(ValueError, match="official ImageCAS-X"):
        module.read_case_ids(cases)
