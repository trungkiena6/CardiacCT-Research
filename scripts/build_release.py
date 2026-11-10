"""Build a checked source snapshot with hashes; never invent experiment outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coronary_tree import __version__  # noqa: E402
from coronary_tree.evaluate.export_contract import export_specs  # noqa: E402

TOP_LEVEL = (
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
)
FILE_TYPES = {
    "src": {".py"},
    "scripts": {".py", ".sh"},
    "tests": {".py"},
    "configs": {".csv", ".txt"},
    "docs/figures": {".png", ".pdf"},
    ".github/workflows": {".yml", ".yaml"},
}
# Reviewed public artifacts only. New analysis outputs are private unless explicitly approved here.
PUBLIC_RESULT_FILES = frozenset(
    {
        "coca/cacseg_heldout_result.json",
        "coca/cacseg_nongated_c4_result.json",
        "coca/cacseg_split_heldout.json",
        "coca/validation_ablations.csv",
        "imagecas/foursplit_summary.json",
        "imagecas/split1_resencl_single_per_case.csv",
        "imagecas/split1_resencl_single_result.json",
        "imagecas/split2_resencl_single_per_case.csv",
        "imagecas/split2_resencl_single_result.json",
        "imagecas/split3_resencl_single_per_case.csv",
        "imagecas/split3_resencl_single_result.json",
        "imagecas/split4_resencl_single_per_case.csv",
        "imagecas/split4_resencl_single_result.json",
        "imagecas/test_access_3_choice.json",
        "imagecas/test_access_3_per_case.csv",
        "imagecas/test_access_3_result.json",
        "imagecas/test_access_6_result.json",
        "imagecas/paired_segmentation_tests.json",
        "imagecas/test_access_6_per_case.csv",
        "imagecas/test_access_6_choice.json",
        "imagecas/split1_validation_experiments/pilot1000_decision.json",
        "imagecas/split1_validation_experiments/pilot1000_val_per_case.csv",
        "imagecas/split1_validation_experiments/postproc_choice.json",
        "imagecas/split1_validation_experiments/postproc_val_sweep.json",
        "imagecas/split1_validation_experiments/resencl_250ep_val_per_case.csv",
        "imagecas/split1_validation_experiments/skelrec_decision.json",
        "imagecas/split1_validation_experiments/skelrec_val_per_case.csv",
        "imagecas/replication/rerun_split1_resencl_single_per_case.csv",
        "imagecas/replication/rerun_split1_resencl_single_result.json",
        "imagecas/replication/rerun_split2_resencl_single_per_case.csv",
        "imagecas/replication/rerun_split2_resencl_single_result.json",
        "imagecas/replication/rerun_split3_resencl_single_per_case.csv",
        "imagecas/replication/rerun_split3_resencl_single_result.json",
        "imagecas/replication/rerun_split4_resencl_single_per_case.csv",
        "imagecas/replication/rerun_split4_resencl_single_result.json",
        "imagecas/replication/rerun_test_access_3_per_case.csv",
        "imagecas/replication/rerun_test_access_3_result.json",
        "imagecas-x/replication/rerun_m1_per_case.csv",
        "imagecas-x/replication/rerun_m1_postprocess_per_case.csv",
        "imagecas-x/replication/rerun_m1_postprocess_result.json",
        "imagecas-x/replication/rerun_m1_result.json",
        "imagecas-x/branch_names_800_edges.csv",
        "imagecas-x/branch_names_800_per_case.csv",
        "imagecas-x/branch_names_800_segments.csv",
        "imagecas-x/branch_names_lumen_160_per_case.csv",
        "imagecas-x/eat_test_result.json",
        "imagecas-x/imagecasx_evaluate_py_results.json",
        "imagecas-x/imagecasx_evaluate_py_results_access2.json",
        "imagecas-x/label_convention_800.csv",
        "imagecas-x/lumen_access2_choice.json",
        "imagecas-x/lumen_access2_test_result.json",
        "imagecas-x/lumen_access2_test_result.per_case.csv",
        "imagecas-x/lumen_tree_summary_160.csv",
        "imagecas-x/oos_tree_summary_800.csv",
        "imagecas-x/pcat_auto_800_per_vessel.csv",
        "imagecas-x/pcat_auto_vs_expert_800.json",
        "imagecas-x/pcat_expert_800_per_vessel.csv",
        "imagecas-x/pcat_expert_mask_control_800.json",
        "imagecas-x/pcat_expert_wholevessel_800_per_vessel.csv",
        "imagecas-x/pcat_lumen_160_per_vessel.csv",
        "imagecas-x/pcat_lumen_vs_expert_160.json",
        "imagecas-x/shell_sensitivity_800/per_vessel.csv",
        "imagecas-x/shell_sensitivity_800/protocol.json",
        "imagecas-x/shell_sensitivity_800/summary.json",
        "imagecas-x/validation_selection.csv",
        "imagecas-x/xbench_ours_per_case.csv",
        "imagecas-x/xbench_ours_result.json",
        "multid4cad/audit_eat_pcat_provenance.json",
        "multid4cad/multid4cad_summary.json",
        "paper_values.csv",
        "release_segmentation_summary.csv",
        "release_segmentation_provenance.json",
        "visual-lab/visuallab_default_per_patient.csv",
    }
)
GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", "private", "release-reports", "venv", "build", "dist"}
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_files():
    paths = [REPO / name for name in TOP_LEVEL]
    if (REPO / "LICENSE").is_file():
        paths.append(REPO / "LICENSE")
    for directory, suffixes in FILE_TYPES.items():
        paths.extend(
            path
            for path in (REPO / directory).rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and not any(
                part.startswith(".") or part in GENERATED_DIRECTORIES
                for part in path.relative_to(REPO / directory).parts
            )
        )
    allowed_results = PUBLIC_RESULT_FILES | {
        spec.relative_csv for spec in export_specs(source_set="updated")
    }
    paths.extend(path for name in allowed_results if (path := REPO / "results" / name).is_file())
    for path in paths:
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(REPO.resolve())
        ):
            raise ValueError(f"Invalid release input: {path}")
    if (REPO / "results/segmentation_dice").exists() or (REPO / "data").exists():
        raise ValueError(
            "Generated run specifications/input duplicates belong outside the release tree"
        )
    markdown = [
        path
        for path in REPO.rglob("*.md")
        if not any(
            part
            in {
                ".git",
                ".pytest_cache",
                ".venv",
                "venv",
                "build",
                "dist",
                "private",
                "release-reports",
            }
            for part in path.relative_to(REPO).parts
        )
    ]
    if set(markdown) != {REPO / "README.md"}:
        raise ValueError(
            "Consolidate release documentation in README.md; remove redundant Markdown files"
        )
    return sorted(set(paths), key=lambda path: path.relative_to(REPO).as_posix())


def verify_sources(paths):
    for path in paths:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        elif path.suffix == ".json":

            def reject_constant(value):
                raise ValueError(f"Nonstandard JSON number {value} in {path}")

            json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", readme) + re.findall(r'src="([^"]+)"', readme)
    for link in links:
        if "://" not in link and not link.startswith("#") and not (REPO / link).exists():
            raise ValueError(f"Broken README link: {link}")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else None
    bash_command = str(bash) if bash and bash.is_file() else shutil.which("bash")
    if bash_command:
        for path in paths:
            if path.suffix == ".sh":
                subprocess.run(
                    [bash_command, "-n", str(path)], check=True, capture_output=True, text=True
                )


def git_state():
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if Path(top).resolve() != REPO.resolve():
            return {
                "repository_head_available": False,
                "identity_note": "This snapshot is not rooted in that Git checkout; file hashes identify it.",
            }
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        )
        return {
            "repository_head": head,
            "working_tree_clean": not bool(status.strip()),
            "identity_note": "File hashes identify this snapshot; HEAD alone does not identify uncommitted changes.",
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "repository_head_available": False,
            "identity_note": "File hashes identify this source snapshot.",
        }


def verify_style(ruff_path=None, shfmt_path=None):
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    required = next(
        value.split("==", 1)[1]
        for value in config["project"]["optional-dependencies"]["dev"]
        if value.startswith("ruff==")
    )
    local_ruff = Path(sys.executable).parent / ("ruff.exe" if os.name == "nt" else "ruff")
    ruff_path = ruff_path or (local_ruff if local_ruff.is_file() else shutil.which("ruff"))
    if not ruff_path:
        raise ValueError("Install .[dev] or supply --ruff with the pinned formatter executable")
    ruff_version = subprocess.check_output([str(ruff_path), "--version"], text=True).strip()
    if ruff_version != f"ruff {required}":
        raise ValueError(f"Expected Ruff {required}; found {ruff_version}")
    for command in (["check", "--no-cache"], ["format", "--check", "--no-cache"]):
        subprocess.run([str(ruff_path), *command, "src", "scripts", "tests"], cwd=REPO, check=True)
    result = {"python": {"tool": ruff_version, "lint_passed": True, "format_passed": True}}
    shfmt_path = shfmt_path or shutil.which("shfmt")
    if shfmt_path:
        shell_version = subprocess.check_output([str(shfmt_path), "--version"], text=True).strip()
        subprocess.run(
            [str(shfmt_path), "-ln", "bash", "-i", "2", "-ci", "-sr", "-d", "scripts"],
            cwd=REPO,
            check=True,
        )
        result["shell"] = {"tool": shell_version, "format_passed": True}
    else:
        result["shell"] = {
            "format_checked": False,
            "note": "Supply --shfmt to check Bash formatting locally; CI checks it on Linux.",
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "dist")
    parser.add_argument(
        "--ruff", type=Path, help="Pinned Ruff executable; defaults to the current environment"
    )
    parser.add_argument(
        "--shfmt", type=Path, help="Optional shfmt executable for a local Bash formatting check"
    )
    args = parser.parse_args(argv)
    try:
        paths = collect_files()
        verify_sources(paths)
        style = verify_style(args.ruff, args.shfmt)
        tested_hashes = {
            path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        archived = load_script("audit_dice_evidence").audit(source_set="archived")
        if not archived["valid_exports"]:
            raise ValueError("Archived CSV verification failed")
        updated = load_script("audit_dice_evidence").audit(source_set="updated")
        command = [sys.executable, "-B", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
        tests = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        print(tests.stdout, end="")
        if tests.returncode:
            print(tests.stderr, file=sys.stderr)
            raise ValueError("Release tests failed; no archive written")
        if collect_files() != paths:
            raise ValueError("Release inputs changed during verification; rerun after edits finish")
        payloads = {path.relative_to(REPO).as_posix(): path.read_bytes() for path in paths}
        checksums = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        if checksums != tested_hashes:
            raise ValueError(
                "Source/data bytes changed during tests; no untested snapshot will be written"
            )
        identity = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode("utf-8")).hexdigest()
        metadata = {
            "schema_version": 1,
            "version": __version__,
            "snapshot_type": "review_source_snapshot",
            "source_tree_sha256": identity,
            "git": git_state(),
            "files": checksums,
            "verification": {
                "tests_returncode": tests.returncode,
                "test_summary": tests.stdout.strip().splitlines()[-1],
                "python": sys.version,
                "archived_csvs_valid": True,
                "style": style,
            },
            "archived_segmentation": archived["summaries"],
            "updated_exports_complete": updated["valid_exports"],
            "execution_provenance_verified": False,
            "license_file_present": (REPO / "LICENSE").is_file(),
            "evidence_note": "CSV coverage/arithmetic are checked. Missing updated exports and execution provenance are not synthesized.",
        }
        payloads["RELEASE.json"] = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        payloads["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in sorted(payloads.items())
        ).encode("utf-8")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        archive = args.output_dir / f"CardiacCT-Research-{__version__}-{identity[:12]}.zip"
        prefix = f"CardiacCT-Research-{__version__}/"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                handle.writestr(info, data)
        with zipfile.ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise ValueError("Archive integrity check failed")
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "archive": str(archive),
                    "sha256": archive_digest,
                    "source_files": len(paths),
                    "archived_csvs_valid": True,
                    "updated_exports_complete": updated["valid_exports"],
                    "execution_provenance_verified": False,
                },
                indent=2,
            )
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
