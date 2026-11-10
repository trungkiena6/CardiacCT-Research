"""Release selection must not publish local analysis outputs or execution sidecars."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_builder", ROOT / "scripts/build_release.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def put(root, relative, content="fixture\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def release_tree(tmp_path, monkeypatch):
    for name in builder.TOP_LEVEL:
        put(tmp_path, name, "")
    monkeypatch.setattr(builder, "REPO", tmp_path)
    return tmp_path


def selected(root):
    return {path.relative_to(root).as_posix() for path in builder.collect_files()}


def test_only_reviewed_public_results_and_canonical_updates_are_selected(release_tree):
    root = release_tree
    public = {
        "results/coca/cacseg_heldout_result.json",
        "results/multid4cad/audit_eat_pcat_provenance.json",
        "results/visual-lab/visuallab_default_per_patient.csv",
    }
    updates = {
        "results/" + spec.relative_csv for spec in builder.export_specs(source_set="updated")
    }
    private = {
        "results/multid4cad/multid4cad_eat_per_patient.csv",
        "results/multid4cad/multid4cad_pat_per_patient.csv",
        "results/coca/cacseg_heldout_per_patient.csv",
        "results/coca/cacseg_heldout_per_case.csv",
        "results/imagecas/new_run/run_info.json",
        "results/imagecas/predictions/predict_from_raw_data_args.json",
        "results/imagecas/.prediction_cache.json",
        "results/imagecas/checkpoint_avg.pth.cache.json",
        "results/imagecas/split1_resencl_single_updated_result.json",
        "results/imagecas-x/m1_updated_per_case.csv.cache.json",
    }
    for name in public | updates | private:
        put(root, name)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    actual = selected(root)
    assert public | updates <= actual
    assert len(updates) == 10 and not private & actual
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_hidden_and_private_source_metadata_are_not_selected(release_tree):
    root = release_tree
    safe = {"src/package/module.py", "scripts/task.py", "scripts/training/train.sh"}
    excluded = {
        "scripts/.prediction_cache.json",
        "scripts/prediction.cache.json",
        "scripts/generated/run_info.json",
        "scripts/.cache/hidden.py",
        "scripts/private/patient_export.py",
        "scripts/__pycache__/module.py",
        "configs/.private/clinical.csv",
        "tests/private/fixture.py",
    }
    for name in safe | excluded:
        put(root, name)
    actual = selected(root)
    assert safe <= actual and not excluded & actual


@pytest.mark.parametrize("directory", ["results/segmentation_dice", "data"])
def test_generated_run_specifications_or_duplicate_data_are_refused(release_tree, directory):
    put(release_tree, directory + "/run_info.json", "{}\n")
    with pytest.raises(ValueError, match="Generated run specifications/input duplicates"):
        builder.collect_files()
