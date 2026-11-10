"""Prepare nnU-Net Dataset906_ImageCASX_lumen from the Kaggle ImageCAS volumes (/workspace/cases/<group>/<id>.img.nii.gz)
and the ImageCAS-X lumen labels (/workspace/xdata/ImageCAS-X_dataset/segmentations/<id>.coronary.nii.gz, binarised > 0).

stages: train-only (560 train)  |  add-val (+80 val, writes the 1-fold splits_final.json)  |  test (160 imagesTs + /workspace/test_labels_x)
        volumes (hard-links the 800 volumes into <xdata>/volumes for the ImageCAS-X evaluate.py)
Images are hard-linked (no copy); labels are written as uint8 with the segmentation's geometry (identical to the image grid).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

X = Path("/workspace/xdata/ImageCAS-X_dataset")
CASES = Path("/workspace/cases")
DS = "Dataset906_ImageCASX_lumen"
RAW = Path(os.environ["nnUNet_raw"]) / DS
PRE = Path(os.environ["nnUNet_preprocessed"]) / DS
TEST_LABELS = Path("/workspace/test_labels_x")


def ids(name):
    return [s for s in (X / "filelist" / f"{name}.txt").read_text().split() if s.strip()]


def grp(i):
    b = (int(i) - 1) // 200 * 200
    return f"{b + 1}-{b + 200}"


def img(i):
    return CASES / grp(i) / f"{i}.img.nii.gz"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def label(args):
    i, dst = args
    if dst.exists():
        return
    s = sitk.ReadImage(str(X / "segmentations" / f"{i}.coronary.nii.gz"))
    o = sitk.GetImageFromArray((sitk.GetArrayFromImage(s) > 0).astype(np.uint8))
    o.CopyInformation(s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(o, str(dst))


def add(set_name, lab_dir):
    cs = ids(set_name)
    for i in cs:
        link(img(i), RAW / "imagesTr" / f"{i}_0000.nii.gz")
    with ProcessPoolExecutor(32) as ex:
        list(ex.map(label, [(i, lab_dir / f"{i}.nii.gz") for i in cs]))
    return cs


stage = sys.argv[1]
if stage == "train-only":
    cs = add("train", RAW / "labelsTr")
    (RAW / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "lumen": 1},
                "numTraining": len(cs),
                "file_ending": ".nii.gz",
            },
            indent=1,
        )
    )
elif stage == "add-val":
    tr, va = ids("train"), add("val", RAW / "labelsTr")
    d = json.loads((RAW / "dataset.json").read_text())
    d["numTraining"] = len(tr) + len(va)
    (RAW / "dataset.json").write_text(json.dumps(d, indent=1))
    PRE.mkdir(parents=True, exist_ok=True)
    (PRE / "splits_final.json").write_text(json.dumps([{"train": tr, "val": va}]))
elif stage == "test":
    cs = ids("test")
    for i in cs:
        link(img(i), RAW / "imagesTs" / f"{i}_0000.nii.gz")
    with ProcessPoolExecutor(32) as ex:
        list(ex.map(label, [(i, TEST_LABELS / f"{i}.nii.gz") for i in cs]))
elif stage == "volumes":
    for i in ids("train") + ids("val") + ids("test"):
        link(img(i), X / "volumes" / f"{i}.img.nii.gz")
else:
    raise SystemExit(f"unknown stage {stage}")
print(stage, "done", flush=True)
