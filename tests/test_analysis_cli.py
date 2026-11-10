"""Analysis entrypoints must import and show help without touching private data."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_analysis_script(name, monkeypatch):
    pytest.importorskip(
        "SimpleITK", reason="Analysis CLI checks require the optional analysis dependencies"
    )
    if name == "visuallab_eat_eval":
        pytest.importorskip("PIL")
    original = Path.iterdir

    def no_private_discovery(path):
        normalized = str(path).replace("\\", "/").lower()
        if "imagecas-work/multid4cad" in normalized or "imagecas-work/visuallab" in normalized:
            pytest.fail(f"Import/help attempted to discover private data: {path}")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", no_private_discovery)
    spec = importlib.util.spec_from_file_location(
        "analysis_cli_" + name, REPO / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["multid4cad_eval", "visuallab_eat_eval"])
def test_import_and_help_do_not_access_private_dataset(name, tmp_path, monkeypatch, capsys):
    private_root = tmp_path / "absent_private_dataset"
    monkeypatch.setenv("MULTID4CAD_WORK", str(private_root))
    monkeypatch.setenv("VISUALLAB_WORK", str(private_root))
    monkeypatch.setenv("TOTAL_SEGMENTATOR_EXE", str(tmp_path / "TotalSegmentator"))
    module = load_analysis_script(name, monkeypatch)
    assert (module.W if name == "multid4cad_eval" else module.VL) == private_root
    monkeypatch.setattr(
        module, "discover_patients", lambda: pytest.fail("--help discovered patients")
    )
    monkeypatch.setattr(sys, "argv", [name + ".py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert not private_root.exists()


@pytest.mark.parametrize("name", ["multid4cad_eval", "visuallab_eat_eval"])
def test_discovery_uses_only_explicit_temporary_directory(name, tmp_path, monkeypatch):
    module = load_analysis_script(name, monkeypatch)
    for patient in ("P003", "P001"):
        (tmp_path / patient).mkdir()
    (tmp_path / "not_a_patient_file").write_text("fixture", encoding="utf-8")
    assert module.discover_patients(tmp_path) == ["P001", "P003"]


def test_visual_lab_grid_worker_receives_its_tuning_scope_explicitly(monkeypatch):
    module = load_analysis_script("visuallab_eat_eval", monkeypatch)
    visited = []

    def load(patient):
        visited.append(patient)
        return object()

    monkeypatch.setattr(module, "load", load)
    monkeypatch.setattr(
        module, "score", lambda *args: {"dice_eat": 0.5, "eat_ppv": 0.4, "eat_sens": 0.6}
    )
    monkeypatch.setattr(
        module, "discover_patients", lambda: pytest.fail("Worker discovered a private population")
    )
    values = tuple(options[0] for options in module.GRID.values())
    result = module._grid_job((values, (0.35, 0.35, 3.0), ["TUNE_A", "TUNE_B"]))
    assert visited == ["TUNE_A", "TUNE_B"]
    assert result["dice_eat_tune"] == pytest.approx(0.5)
