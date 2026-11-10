"""Disposable inference/cache fixtures; no research artifacts or real training."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prediction_cache as cache
import score_segmentation as scorer


def load_script(name):
    pytest.importorskip(
        "SimpleITK",
        reason="Ensemble integration tests require the optional imaging/training dependencies",
    )
    spec = importlib.util.spec_from_file_location(
        "workflow_" + name, REPO / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metrics(path, *, bad_hd=False, status="ok", ids=("Case0001", "Case0002")):
    fields = ["case", "status", "reason_codes", *scorer.METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(ids):
            writer.writerow(
                dict(
                    case=case,
                    status=status,
                    reason_codes="",
                    dice=0.8,
                    cldice=0.9,
                    hd95_mm="" if bad_hd and index == 1 else 2.0,
                    n_components_gt=2,
                    n_components_pred=3,
                    gt_centerline_coverage_1mm=0.95,
                )
            )


def test_missing_endpoint_stops_for_review_without_bootstrap_crash(tmp_path):
    path = tmp_path / "cases.csv"
    write_metrics(path, bad_hd=True)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="undefined/invalid hd95_mm.*retained"):
        scorer.summarize_csv(path, n_boot=10)
    assert path.read_bytes() == original


def test_complete_metrics_have_full_scope_means_and_paired_component_errors(tmp_path):
    path = tmp_path / "cases.csv"
    write_metrics(path)
    result = scorer.summarize_csv(path, {"Case0001", "Case0002"}, n_boot=20)
    assert result["n"] == 2 and result["dice"]["mean"] == pytest.approx(0.8)
    assert result["betti0_err"]["mean"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="scope"):
        scorer.read_evaluation_rows(path, {"Case0001"})


@pytest.mark.parametrize("existing", ["summary", "csv"])
def test_score_refuses_existing_outputs_before_calling_evaluator(tmp_path, monkeypatch, existing):
    out = tmp_path / "score.json"
    destination = out if existing == "summary" else out.with_suffix(".per_case.csv")
    destination.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        scorer.subprocess, "run", lambda *a, **k: pytest.fail("Evaluator must not run")
    )
    assert scorer.main([str(tmp_path / "pred"), str(tmp_path / "gt"), str(out)]) == 1
    assert destination.read_text(encoding="utf-8") == "original"


def test_score_uses_repository_evaluator_and_retains_failed_csv(tmp_path, monkeypatch):
    gt = tmp_path / "gt"
    gt.mkdir()
    for case in ("Case0001", "Case0002"):
        (gt / f"{case}.nii.gz").touch()
    out = tmp_path / "score.json"

    def failed_evaluator(command, **kwargs):
        assert Path(command[1]) == REPO / "src/coronary_tree/evaluate/segmentation.py"
        write_metrics(Path(command[command.index("--output") + 1]), status="error")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(scorer.subprocess, "run", failed_evaluator)
    assert scorer.main([str(tmp_path / "pred"), str(gt), str(out)]) == 1
    assert out.with_suffix(".per_case.csv").is_file() and not out.exists()


def make_pairs(directory, ids=("Case0001",)):
    directory.mkdir(parents=True, exist_ok=True)
    for case in ids:
        (directory / f"{case}.nii.gz").write_bytes(b"mask")
        (directory / f"{case}.npz").write_bytes(b"probabilities")


def test_equal_counts_with_wrong_ids_are_not_a_verified_cache(tmp_path):
    out = tmp_path / "cached"
    make_pairs(out, ("Case9999",))
    original = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(cache.CacheError, match="Unverified.*new output directory"):
        cache.reuse_cache(out, {"checkpoint": "signature"}, ["Case0001"], probabilities=True)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == original


def test_verified_cache_requires_pairs_hashes_and_matching_request(tmp_path):
    out = tmp_path / "cached"
    make_pairs(out)
    request = {"checkpoint": "hash-a", "step": 0.25}
    cache.record_cache(out, request, ["Case0001"], probabilities=True)
    assert cache.reuse_cache(out, request, ["Case0001"], probabilities=True)
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        cache.reuse_cache(out, dict(request, step=0.5), ["Case0001"], probabilities=True)
    (out / "Case0001.npz").write_bytes(b"modified probabilities")
    with pytest.raises(cache.CacheError, match="hashes changed"):
        cache.reuse_cache(out, request, ["Case0001"], probabilities=True)


def test_incomplete_pairs_cannot_be_certified(tmp_path):
    out = tmp_path / "cached"
    out.mkdir()
    (out / "Case0001.nii.gz").touch()
    with pytest.raises(cache.CacheError, match="pairs"):
        cache.record_cache(out, {}, ["Case0001"], probabilities=True)
    assert not (out / cache.CACHE_FILE).exists()


def make_model(tmp_path, module, monkeypatch):
    res = tmp_path / "models"
    monkeypatch.setattr(module, "RES", res)
    tr, pl = module.MODELS["resenc_l"]
    fold = res / f"{tr}__{pl}__3d_fullres" / "fold_0"
    fold.mkdir(parents=True)
    for name in ("checkpoint_final.pth", "checkpoint_best.pth"):
        (fold / name).write_bytes(name.encode("ascii"))
    for name in ("plans.json", "dataset.json"):
        (fold.parent / name).write_text("{}", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / "Case0001_0000.nii.gz").write_bytes(b"input")
    return fold, images


def fake_nnunet_sources(tmp_path, monkeypatch, trainer="nnUNetTrainer_250epochs_NoMirroring"):
    package = tmp_path / "package/nnunetv2"
    inference = package / "inference"
    inference.mkdir(parents=True)
    (inference / "predict_from_raw_data.py").write_text("def predict_entry_point(): pass\n")
    trainers = package / "training/nnUNetTrainer"
    trainers.mkdir(parents=True)
    source = trainers / "fixture_trainer.py"
    source.write_text(f"class {trainer}: pass\n")
    monkeypatch.setattr(
        cache, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)])
    )
    monkeypatch.setattr(cache, "version", lambda name: "test-distribution")
    return source


def test_prediction_cache_binds_real_checkpoint_and_step(tmp_path, monkeypatch):
    module = load_script("imagecas_ensemble_access3")
    fold, images = make_model(tmp_path, module, monkeypatch)
    fake_nnunet_sources(tmp_path, monkeypatch)
    out = tmp_path / "predictions"
    calls = []

    def predict(command, **kwargs):
        calls.append(command)
        assert command[:2] == [sys.executable, "-c"]
        assert "predict_entry_point" in command[2]
        assert "--disable_tta" in command
        make_pairs(Path(command[command.index("-o") + 1]))

    monkeypatch.setattr(module.subprocess, "run", predict)
    module.predict("resenc_l", images, out, "checkpoint_best.pth", 0.25)
    module.predict("resenc_l", images, out, "checkpoint_best.pth", 0.25)
    assert len(calls) == 1
    data = json.loads((out / cache.CACHE_FILE).read_text())
    assert (
        data["request"]["checkpoint"]["sha256"]
        == hashlib.sha256((fold / "checkpoint_best.pth").read_bytes()).hexdigest()
    )
    assert data["request"]["step_size"] == 0.25
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        module.predict("resenc_l", images, out, "checkpoint_final.pth", 0.25)
    (fold / "checkpoint_best.pth").write_bytes(b"new checkpoint")
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        module.predict("resenc_l", images, out, "checkpoint_best.pth", 0.25)
    assert len(calls) == 1


def test_trainer_source_edit_invalidates_cache_and_missing_class_stops_inference(
    tmp_path, monkeypatch
):
    module = load_script("imagecas_ensemble_access3")
    _, images = make_model(tmp_path, module, monkeypatch)
    source = fake_nnunet_sources(tmp_path, monkeypatch)
    out = tmp_path / "predictions"
    calls = []

    def predict(command, **kwargs):
        calls.append(command)
        assert command[0] == sys.executable
        assert command[3] == str(source.parents[3])
        make_pairs(Path(command[command.index("-o") + 1]))

    monkeypatch.setattr(module.subprocess, "run", predict)
    module.predict("resenc_l", images, out, "checkpoint_final.pth", 0.25)
    original = (out / "Case0001.nii.gz").read_bytes()
    source.write_text(source.read_text() + "learning_rate = 0.001\n")
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        module.predict("resenc_l", images, out, "checkpoint_final.pth", 0.25)
    source.write_text("class OtherTrainer: pass\n")
    with pytest.raises(cache.CacheError, match="Source for trainer.*unavailable"):
        module.predict("resenc_l", images, tmp_path / "new", "checkpoint_final.pth", 0.25)
    assert len(calls) == 1 and (out / "Case0001.nii.gz").read_bytes() == original


def test_ensemble_build_preserves_unverified_output_and_tracks_inputs(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    module = load_script("imagecas_ensemble_access3")
    source = tmp_path / "source"
    source.mkdir()
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8)), str(source / "Case0001.nii.gz")
    )
    probs = np.stack([np.full((2, 2, 2), 0.2), np.full((2, 2, 2), 0.8)])
    np.savez_compressed(source / "Case0001.npz", probabilities=probs)
    out = tmp_path / "ensemble"
    module.build({"model": source}, {"model": 1.0}, out, 0)
    original = (out / "Case0001.nii.gz").read_bytes()
    module.build({"model": source}, {"model": 1.0}, out, 0)
    assert (out / "Case0001.nii.gz").read_bytes() == original
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        module.build({"model": source}, {"model": 2.0}, out, 0)
    (out / cache.CACHE_FILE).unlink()
    with pytest.raises(cache.CacheError, match="Unverified"):
        module.build({"model": source}, {"model": 1.0}, out, 0)
    assert (out / "Case0001.nii.gz").read_bytes() == original


def test_runtime_library_upgrade_invalidates_existing_prediction_cache(tmp_path, monkeypatch):
    module = load_script("imagecas_ensemble_access3")
    _, images = make_model(tmp_path, module, monkeypatch)
    fake_nnunet_sources(tmp_path, monkeypatch)
    runtime = {"torch": "2.8.0"}
    monkeypatch.setattr(cache, "version", lambda name: runtime.get(name, "fixture-version"))
    out = tmp_path / "predictions"
    calls = []

    def predict(command, **kwargs):
        calls.append(command)
        make_pairs(Path(command[command.index("-o") + 1]))

    monkeypatch.setattr(module.subprocess, "run", predict)
    module.predict("resenc_l", images, out, "checkpoint_final.pth", 0.25)
    original = (out / "Case0001.nii.gz").read_bytes()
    runtime["torch"] = "2.14.0"
    with pytest.raises(cache.CacheError, match="configuration.*changed"):
        module.predict("resenc_l", images, out, "checkpoint_final.pth", 0.25)
    assert len(calls) == 1 and (out / "Case0001.nii.gz").read_bytes() == original


def test_averaged_checkpoint_reuse_requires_unchanged_source_checkpoints(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    module = load_script("imagecas_ensemble_access3")
    fold, _ = make_model(tmp_path, module, monkeypatch)
    torch.save({"network_weights": {"weight": torch.tensor([2.0])}}, fold / "checkpoint_best.pth")
    torch.save({"network_weights": {"weight": torch.tensor([4.0])}}, fold / "checkpoint_final.pth")
    assert module.avg_checkpoint("resenc_l") == "checkpoint_avg.pth"
    out = fold / "checkpoint_avg.pth"
    original = out.read_bytes()
    averaged = torch.load(out, weights_only=False)
    assert averaged["network_weights"]["weight"].item() == pytest.approx(3.0)
    module.avg_checkpoint("resenc_l")
    assert out.read_bytes() == original
    torch.save({"network_weights": {"weight": torch.tensor([8.0])}}, fold / "checkpoint_final.pth")
    with pytest.raises(cache.CacheError, match="changed derived checkpoint"):
        module.avg_checkpoint("resenc_l")
    assert out.read_bytes() == original


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def run_shell(tmp_path, script_name, stubs):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is not installed")
    workspace = tmp_path / "workspace"
    (workspace / "coroseg-work/logs").mkdir(parents=True)
    plans = workspace / "coroseg-work/nnunet_preprocessed/Dataset901_ImageCAS_S1"
    plans.mkdir(parents=True)
    (plans / "nnUNetResEncUNetL40GPlans.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8", newline="\n")
        stub.chmod(0o755)
    content = (
        (REPO / "scripts/training" / script_name)
        .read_text()
        .replace("/workspace", shell_path(workspace))
    )
    content = content.replace(
        "#!/bin/bash\n",
        "#!/bin/bash\nexport PATH=" + shlex.quote(shell_path(bin_dir)) + ':"$PATH"\n',
        1,
    )
    script = tmp_path / "job.sh"
    script.write_text(content, encoding="utf-8", newline="\n")
    return subprocess.run([bash, shell_path(script)], text=True, capture_output=True), workspace


def test_training_shell_propagates_trainer_failure(tmp_path):
    result, _ = run_shell(
        tmp_path,
        "train_resencl.sh",
        {
            "ls": "for ((i=0;i<750;i++)); do echo file; done",
            "python3": "echo 1",
            "nnUNetv2_train": "exit 23",
        },
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert "failed rc=23" in result.stderr


@pytest.mark.parametrize("stage,code", [("A", 17), ("B", 29)])
def test_finetune_stops_and_propagates_stage_failure(tmp_path, stage, code):
    marker = tmp_path / "trainer_called"
    stubs = {
        "du": "printf '100 a\\n200 b\\n'",
        "df": "printf 'Avail\\n1000000\\n'",
        "python3": "exit 17" if stage == "A" else "echo 'TEST-2 RESULT ok'",
        "nnUNetv2_train": f"touch {shlex.quote(shell_path(marker))}\nexit 29",
    }
    result, workspace = run_shell(tmp_path, "finetune_resencl.sh", stubs)
    assert result.returncode == code, result.stdout + result.stderr
    assert marker.exists() is (stage == "B")
    assert f"{stage}-LOI rc={code}" in (workspace / "finetune_resencl.log").read_text()
