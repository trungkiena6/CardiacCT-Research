"""Dataset preparation must preserve geometry and existing source material."""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from coronary_tree.data.prepare import convert_label, copy_image, main


def pair(tmp_path, offset=0, label_shape=(4, 4, 4)):
    image = tmp_path / "image.nii.gz"
    label = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), image)
    affine = np.eye(4)
    affine[0, 3] = offset
    nib.save(nib.Nifti1Image(np.ones(label_shape, dtype=np.uint8), affine), label)
    return image, label


@pytest.mark.parametrize("offset,shape", [(5, (4, 4, 4)), (0, (3, 4, 4))])
def test_label_geometry_is_not_silently_replaced(tmp_path, offset, shape):
    image, label = pair(tmp_path, offset, shape)
    output = tmp_path / "converted.nii.gz"
    with pytest.raises(ValueError, match="geometry differs"):
        convert_label(label, image, output)
    assert not output.exists()


def test_existing_image_with_equal_size_but_different_content_is_preserved(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"aaaa")
    destination.write_bytes(b"bbbb")
    with pytest.raises(FileExistsError):
        copy_image(source, destination)
    assert source.read_bytes() == b"aaaa" and destination.read_bytes() == b"bbbb"


def test_identical_existing_image_can_be_reused(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"same bytes")
    copy_image(source, destination)
    assert copy_image(source, destination)["bytes"] == len(b"same bytes")


def test_existing_different_label_is_not_overwritten(tmp_path):
    image, label = pair(tmp_path)
    output = tmp_path / "converted.nii.gz"
    convert_label(label, image, output)
    original = output.read_bytes()
    changed = np.zeros((4, 4, 4), dtype=np.uint8)
    nib.save(nib.Nifti1Image(changed, np.eye(4)), label)
    with pytest.raises(FileExistsError):
        convert_label(label, image, output)
    assert output.read_bytes() == original


def test_identical_label_conversion_is_idempotent(tmp_path):
    image, label = pair(tmp_path)
    output = tmp_path / "converted.nii.gz"
    first = convert_label(label, image, output)
    original = output.read_bytes()
    assert convert_label(label, image, output) == first
    assert output.read_bytes() == original


def split_arguments(tmp_path):
    manifest = Path(__file__).resolve().parents[1] / "configs/imagecas_v2_split_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    selected = {
        part: next(int(row["case_id"]) for row in rows if row["split_1"] == part)
        for part in ("train", "val")
    }
    raw = tmp_path / "raw/Dataset901_ImageCAS_S1"
    for directory in (raw / "imagesTr", raw / "labelsTr"):
        directory.mkdir(parents=True)
    for cid in selected.values():
        (raw / "imagesTr" / f"Case{cid:04d}_0000.nii.gz").touch()
        (raw / "labelsTr" / f"Case{cid:04d}.nii.gz").touch()
    args = [
        "--manifest",
        str(manifest),
        "--cases-dir",
        str(tmp_path / "cases"),
        "--split",
        "1",
        "--dataset-id",
        "901",
        "--nnunet-raw",
        str(tmp_path / "raw"),
        "--nnunet-preprocessed",
        str(tmp_path / "pre"),
        "--work",
        str(tmp_path / "work"),
        "--stage",
        "splits",
    ]
    return args, selected


def test_incomplete_dataset_cannot_be_logged_as_official_split(tmp_path):
    args, _ = split_arguments(tmp_path)
    with pytest.raises(ValueError, match="750 prepared"):
        main(args)
    assert not list(tmp_path.rglob("splits_final.json"))
    assert not list(tmp_path.rglob("dataset_build_s1.json"))


def test_explicit_partial_split_records_actual_ids_and_counts(tmp_path, capsys):
    args, selected = split_arguments(tmp_path)
    main(args + ["--skip-missing-labels"])
    split = json.loads(next(tmp_path.rglob("splits_final.json")).read_text())[0]
    log = json.loads(next(tmp_path.rglob("dataset_build_s1.json")).read_text())["splits"]
    assert split == {part: [f"Case{cid:04d}"] for part, cid in selected.items()}
    assert log["train"] == [selected["train"]] and log["val"] == [selected["val"]]
    assert log["protocol_complete"] is False
    assert len(log["skipped"]["train"]) == 699 and len(log["skipped"]["val"]) == 49
    assert "train=1 val=1" in capsys.readouterr().out
