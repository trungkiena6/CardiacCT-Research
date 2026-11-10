"""Content-verified caches for inference and derived masks; never replace old data."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

CACHE_FILE = ".prediction_cache.json"


class CacheError(RuntimeError):
    pass


def file_identity(path: Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def signature(request: dict) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def input_identity(images: Path) -> tuple[list[str], dict]:
    images = Path(images)
    files = sorted(images.glob("*.nii.gz"))
    if not files:
        raise CacheError(f"No input images in {images}")
    if any(not path.name.endswith("_0000.nii.gz") for path in files):
        raise CacheError("These CT scripts require exactly one _0000.nii.gz channel per case")
    ids = [path.name[:-12] for path in files]
    if len(ids) != len(set(ids)):
        raise CacheError("Duplicate input case IDs")
    return ids, {path.name: file_identity(path) for path in files}


def artifact_identity(directory: Path) -> dict:
    directory = Path(directory)
    if any(path.is_dir() for path in directory.iterdir()):
        raise CacheError(f"Unexpected subdirectory in cache {directory}")
    return {
        path.name: file_identity(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != CACHE_FILE
    }


def _check_pairs(directory: Path, ids, probabilities: bool) -> None:
    expected = set(ids)
    if not expected or len(expected) != len(ids):
        raise CacheError("Expected case IDs must be nonempty and unique")
    masks = {path.name[:-7] for path in directory.glob("*.nii.gz")}
    probs = {path.stem for path in directory.glob("*.npz")}
    if masks != expected or probs != (expected if probabilities else set()):
        raise CacheError(
            f"Incomplete or unexpected case IDs / mask-probability pairs in {directory}"
        )


def _unsafe(directory: Path, reason: str):
    raise CacheError(
        f"{reason}: {directory}. Existing artifacts were preserved; use a new output directory."
    )


def reuse_cache(directory: Path, request: dict, ids, *, probabilities: bool) -> bool:
    directory = Path(directory)
    if not directory.exists():
        return False
    if not directory.is_dir():
        _unsafe(directory, "Output is not a directory")
    if not any(directory.iterdir()):
        return False
    manifest = directory / CACHE_FILE
    if not manifest.is_file():
        _unsafe(directory, "Unverified nonempty output")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        _check_pairs(directory, ids, probabilities)
        valid = (
            data["schema_version"] == 1
            and data["request"] == request
            and data["request_sha256"] == signature(request)
            and data["case_ids"] == sorted(ids)
            and data["probabilities"] is probabilities
            and data["artifacts"] == artifact_identity(directory)
        )
    except (OSError, ValueError, KeyError, TypeError, CacheError) as exc:
        _unsafe(directory, f"Invalid cache ({exc})")
    if not valid:
        _unsafe(directory, "Cache inputs, configuration or artifact hashes changed")
    return True


def record_cache(directory: Path, request: dict, ids, *, probabilities: bool) -> None:
    directory = Path(directory)
    _check_pairs(directory, ids, probabilities)
    data = dict(
        schema_version=1,
        request=request,
        request_sha256=signature(request),
        case_ids=sorted(ids),
        probabilities=probabilities,
        artifacts=artifact_identity(directory),
    )
    with (directory / CACHE_FILE).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def reuse_file(path: Path, request: dict) -> bool:
    path = Path(path)
    metadata = path.with_name(path.name + ".cache.json")
    if not path.exists() and not metadata.exists():
        return False
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
        valid = (
            data["request"] == request
            and data["request_sha256"] == signature(request)
            and data["artifact"] == file_identity(path)
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        _unsafe(path, "Unverified or changed derived checkpoint")
    return True


def record_file(path: Path, request: dict) -> None:
    path = Path(path)
    data = dict(request=request, request_sha256=signature(request), artifact=file_identity(path))
    with path.with_name(path.name + ".cache.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def predictor_identity(trainer: str) -> dict:
    """Bind inference to the installed sources, including custom trainer classes."""
    try:
        spec = find_spec("nnunetv2")
        locations = list(spec.submodule_search_locations) if spec else []
        if len(locations) != 1:
            raise CacheError("A single readable nnunetv2 source tree is required")
        package = Path(locations[0]).resolve()
        if not (package / "inference/predict_from_raw_data.py").is_file():
            raise CacheError("nnunetv2 prediction entrypoint source is unavailable")
        sources, trainer_sources = {}, []
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(package).as_posix()
            sources[relative] = file_identity(path)["sha256"]
            if relative.startswith("training/nnUNetTrainer/"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                if any(
                    isinstance(node, ast.ClassDef) and node.name == trainer
                    for node in ast.walk(tree)
                ):
                    trainer_sources.append(relative)
        if not trainer_sources:
            raise CacheError(
                f"Source for trainer {trainer!r} is unavailable; inference was not started"
            )
        return dict(
            package_parent=str(package.parent),
            distribution_version=version("nnunetv2"),
            runtime_versions={
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scipy",
                    "nibabel",
                    "SimpleITK",
                    "dynamic_network_architectures",
                )
            },
            python_sources=sources,
            source_tree_sha256=signature(sources),
            trainer_sources=trainer_sources,
            helper=file_identity(Path(__file__)),
            python_executable=file_identity(Path(sys.executable)),
            python_version=sys.version,
        )
    except CacheError:
        raise
    except (OSError, ValueError, SyntaxError, ImportError, TypeError, AttributeError) as exc:
        raise CacheError(f"Cannot fingerprint nnunetv2/trainer source: {exc}") from exc


def prediction_command(request: dict, arguments: list[str]) -> list[str]:
    # Pin the package import as well as the interpreter. PATH console scripts may
    # belong to a different environment than the one whose sources were hashed.
    entrypoint = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; "
        "predict_entry_point()"
    )
    return [sys.executable, "-c", entrypoint, request["predictor"]["package_parent"], *arguments]


def inference_request(
    images: Path, fold: Path, model: str, trainer: str, plans: str, checkpoint: str, step: float
) -> tuple[list[str], dict]:
    ids, inputs = input_identity(images)
    request = dict(
        operation="nnunet_prediction",
        model=model,
        trainer=trainer,
        plans_name=plans,
        checkpoint=file_identity(fold / checkpoint),
        plans=file_identity(fold.parent / "plans.json"),
        dataset=file_identity(fold.parent / "dataset.json"),
        inputs=inputs,
        predictor=predictor_identity(trainer),
        configuration="3d_fullres",
        fold=0,
        step_size=step,
        mirroring=False,
        save_probabilities=True,
    )
    return ids, request


def ensemble_request(
    directories: dict[str, Path], weights: dict[str, float], t_mm3: float, source: Path
) -> tuple[list[str], dict]:
    if not directories or set(directories) != set(weights):
        raise CacheError("Ensemble members and weights must match and be nonempty")
    if (
        not math.isfinite(t_mm3)
        or t_mm3 < 0
        or any(not math.isfinite(w) or w <= 0 for w in weights.values())
    ):
        raise CacheError("Ensemble weights must be positive and component threshold nonnegative")
    first = next(iter(directories.values()))
    ids = sorted(path.name[:-7] for path in first.glob("*.nii.gz"))
    sources = {}
    for name, directory in directories.items():
        _check_pairs(directory, ids, True)
        sources[name] = artifact_identity(directory)
        cache = directory / CACHE_FILE
        if cache.is_file():
            sources[name][CACHE_FILE] = file_identity(cache)
    return ids, dict(
        operation="probability_ensemble",
        sources=sources,
        weights=weights,
        threshold=0.5,
        threshold_comparison=">=",
        component_threshold_mm3=t_mm3,
        connectivity=26,
        builder=file_identity(source),
    )
