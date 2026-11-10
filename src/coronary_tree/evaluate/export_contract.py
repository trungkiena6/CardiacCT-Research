"""Explicit export names and official test scopes, independent of run provenance."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

REQUIRED_COLUMNS = frozenset(
    {"case", "status", "dice", "cldice", "hd95_mm", "n_components_gt", "n_components_pred"}
)
# UTF-8, lexicographically sorted IDs, one per line including the final newline.
IMAGECAS_TEST_DIGESTS = {
    1: "13e431b7f92675f485185e5b8e4481a5eee6b3e33f592c903d8e6150cbe885e7",
    2: "432f48e4cb0c996814721e2773cc841802e546ab19a40e105d6c483297b5d41c",
    3: "b12fe521c379fca2b5fb90582cab8cedaff2ff20f2c4f059c90500004153fccd",
    4: "90c1cbe371cfd172b49fd4ee6b04674830840b6d8fb48dcb4374ec0012121761",
}
IMAGECASX_TEST_DIGEST = "c27da83417e908a1b1e73eebd65d973f68287febe55016007ed293b5ab75f61c"


@dataclass(frozen=True)
class ExportSpec:
    run_id: str
    claim_id: str
    relative_csv: str
    dataset: str
    split: int | None = None


def export_specs(source_set="updated", include_ensemble_split1=True):
    """Select one source set without falling back to another run's files."""
    if source_set not in {"archived", "updated"}:
        raise ValueError("source_set must be 'archived' or 'updated'")
    updated = source_set == "updated"
    suffix = "_updated" if updated else ""
    specs = [
        ExportSpec(
            f"split{s}_resencl_single{suffix}",
            "T3_SINGLE",
            f"imagecas/split{s}_resencl_single{suffix}_per_case.csv",
            "ImageCAS",
            s,
        )
        for s in range(1, 5)
    ]
    if updated:
        specs += [
            ExportSpec(
                f"split{s}_ensemble",
                "T3_ENSEMBLE",
                f"imagecas/split{s}_ensemble_per_case.csv",
                "ImageCAS",
                s,
            )
            for s in range(1 if include_ensemble_split1 else 2, 5)
        ]
    else:
        specs.append(
            ExportSpec(
                "test_access_3", "T3_ENSEMBLE", "imagecas/test_access_3_per_case.csv", "ImageCAS", 1
            )
        )
    return tuple(
        specs
        + [
            ExportSpec(
                "m1_updated" if updated else "xbench_ours",
                "T4_M1",
                "imagecas-x/m1_updated_per_case.csv"
                if updated
                else "imagecas-x/xbench_ours_per_case.csv",
                "ImageCAS-X",
            ),
            ExportSpec(
                "m1_postprocess_updated" if updated else "lumen_access2",
                "T4_M1_PP",
                "imagecas-x/m1_postprocess_updated_per_case.csv"
                if updated
                else "imagecas-x/lumen_access2_test_result.per_case.csv",
                "ImageCAS-X",
            ),
        ]
    )


def case_ids_digest(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def load_scopes(split_manifest: Path, imagecasx_test_list: Path):
    """Check the full ImageCAS 700/50/250 partition and official test identities."""
    with Path(split_manifest).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields, rows = reader.fieldnames or [], list(reader)
    if not {"case_id", "split_1", "split_2", "split_3", "split_4"}.issubset(fields) or len(
        fields
    ) != len(set(fields)):
        raise ValueError("ImageCAS manifest needs unique case_id and split_1..split_4 columns")
    if len(rows) != 1000:
        raise ValueError("Official ImageCAS manifest must contain exactly 1000 case rows")
    ids = []
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("ImageCAS manifest row width differs from its header")
        if re.fullmatch(r"[1-9][0-9]*", row["case_id"]) is None:
            raise ValueError("ImageCAS case_id must be a canonical positive integer")
        ids.append(int(row["case_id"]))
    if len(set(ids)) != 1000 or set(ids) != set(range(1, 1001)):
        raise ValueError("ImageCAS manifest must contain each case ID 1..1000 exactly once")
    named_ids = [f"Case{case:04d}" for case in ids]
    splits = {}
    for split in range(1, 5):
        column = f"split_{split}"
        if Counter(row[column] for row in rows) != {"train": 700, "val": 50, "test": 250}:
            raise ValueError(
                f"ImageCAS Split {split} must have 700 train, 50 val and 250 test cases"
            )
        splits[split] = {case for case, row in zip(named_ids, rows) if row[column] == "test"}
    if len(set.union(*splits.values())) != 1000:
        raise ValueError("ImageCAS test splits must be disjoint and cover all 1000 cases")
    for split, cases in splits.items():
        if case_ids_digest(cases) != IMAGECAS_TEST_DIGESTS[split]:
            raise ValueError(f"ImageCAS Split {split} differs from the official test population")
    lumen = Path(imagecasx_test_list).read_text(encoding="utf-8-sig").split()
    if (
        len(lumen) != 160
        or len(set(lumen)) != 160
        or any(re.fullmatch(r"[1-9][0-9]*", case) is None for case in lumen)
    ):
        raise ValueError(
            "ImageCAS-X test.txt must contain exactly 160 distinct canonical numeric IDs"
        )
    if case_ids_digest(lumen) != IMAGECASX_TEST_DIGEST:
        raise ValueError("ImageCAS-X test.txt differs from the official test-160 population")
    return splits, set(lumen)


def inspect_csv(path: Path, expected_ids):
    """Validate all rows before exposing means; does not verify execution provenance."""
    path, expected = Path(path), set(expected_ids)
    if not expected:
        raise ValueError("Expected case set must not be empty")
    result = dict(
        path=str(path),
        status="invalid",
        expected_n=len(expected),
        n_rows=None,
        mean_dice=None,
        mean_cldice=None,
        mean_hd95_mm=None,
        mean_component_error=None,
        errors=[],
    )
    errors = result["errors"]
    if not path.is_file():
        return dict(result, status="missing", errors=["CSV not found"])
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields, rows = reader.fieldnames or [], list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"Cannot read UTF-8 CSV: {type(exc).__name__}: {exc}")
        return result
    result["n_rows"] = len(rows)
    missing = sorted(REQUIRED_COLUMNS - set(fields))
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if len(fields) != len(set(fields)):
        errors.append("Duplicate column names")
    if errors:
        return result
    ids = [row.get("case") or "" for row in rows]
    duplicates = sorted(case for case, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate case IDs: {', '.join(duplicates)}")
    if len(rows) != len(expected):
        errors.append(f"Expected {len(expected)} case rows, found {len(rows)}")
    absent, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
    if absent:
        errors.append(f"Missing case IDs: {', '.join(absent)}")
    if extra:
        errors.append(f"Unexpected case IDs: {', '.join(extra)}")
    metrics = {key: [] for key in ("dice", "cldice", "hd95_mm")}
    component_errors = []
    for line, row in enumerate(rows, 2):
        label = f"line {line}, case {row.get('case')!r}"
        if None in row or any(value is None for value in row.values()):
            errors.append(f"{label}: row width differs from header")
        if row["status"] != "ok":
            errors.append(
                f"{label}: status={row['status']!r}; reason={row.get('reason_codes', '')!r}"
            )
        elif (row.get("reason_codes") or "").strip():
            errors.append(f"{label}: reason requires review: {row['reason_codes']}")
        for key in metrics:
            try:
                value = float(row[key])
                if not math.isfinite(value) or value < 0 or (key != "hd95_mm" and value > 1):
                    raise ValueError("out of range")
                metrics[key].append(value)
            except (TypeError, ValueError):
                bounds = "finite and >= 0" if key == "hd95_mm" else "finite and in [0, 1]"
                errors.append(f"{label}: {key} must be {bounds}, found {row[key]!r}")
        counts = []
        for key in ("n_components_gt", "n_components_pred"):
            if not isinstance(row[key], str) or re.fullmatch(r"[0-9]+", row[key]) is None:
                errors.append(f"{label}: {key} must be a nonnegative integer")
            else:
                counts.append(int(row[key]))
        if len(counts) == 2:
            component_errors.append(abs(counts[1] - counts[0]))
    if not errors:
        result.update(
            status="valid",
            mean_dice=fmean(metrics["dice"]),
            mean_cldice=fmean(metrics["cldice"]),
            mean_hd95_mm=fmean(metrics["hd95_mm"]),
            mean_component_error=fmean(component_errors),
            case_ids_sha256=case_ids_digest(ids),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result
