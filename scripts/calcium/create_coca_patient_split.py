# Chia tap theo benh nhan, co dinh, co seed, phan tang theo nhom Agatston tham chieu.
# Dung Dataset910 (full FOV, chi ca co voi cua train+val, splits_final 1 fold) va thu muc du doan val_all/test_all.
import collections
import csv
import hashlib
import json
import os
import random
from pathlib import Path

W = Path("/workspace/cacseg-work")
RAW = W / "raw_all"
rows = [r for r in csv.DictReader(open(RAW / "manifest.csv")) if r["status"] == "ok"]


def cat(a):
    a = float(a)
    return 0 if a <= 0 else 1 if a <= 10 else 2 if a <= 100 else 3 if a <= 400 else 4


rng = random.Random(20261007)
split = {"test": [], "val": [], "train": []}
for c in range(5):
    pids = sorted(r["pid"] for r in rows if cat(r["agatston_gt"]) == c)
    rng.shuffle(pids)
    nt = max(1, round(0.15 * len(pids)))
    nv = max(1, round(0.10 * len(pids)))
    split["test"] += pids[:nt]
    split["val"] += pids[nt : nt + nv]
    split["train"] += pids[nt + nv :]
pos = {r["pid"] for r in rows if int(r["mask_voxels"]) > 0}
info = {
    "seed": 20261007,
    "manifest_sha256": hashlib.sha256(open(RAW / "manifest.csv", "rb").read()).hexdigest(),
    "counts": {k: {"all": len(v), "positive": sum(p in pos for p in v)} for k, v in split.items()},
    "categories": {
        k: sorted(
            collections.Counter(
                cat(next(r["agatston_gt"] for r in rows if r["pid"] == p)) for p in v
            ).items()
        )
        for k, v in split.items()
    },
    **{k: sorted(v, key=int) for k, v in split.items()},
}
json.dump(info, open(W / "split_heldout.json", "w"), indent=1)
stem = lambda p: f"coca_{int(p):04d}"
D = W / "nnunet_raw" / "Dataset910_COCA_CAC_HO"
(D / "imagesTr").mkdir(parents=True, exist_ok=True)
(D / "labelsTr").mkdir(exist_ok=True)
for p in split["train"] + split["val"]:
    if p in pos:
        for src, dst in (
            (RAW / "images" / f"{stem(p)}_0000.nii.gz", D / "imagesTr" / f"{stem(p)}_0000.nii.gz"),
            (RAW / "labels" / f"{stem(p)}.nii.gz", D / "labelsTr" / f"{stem(p)}.nii.gz"),
        ):
            dst.exists() or os.link(src, dst)
n_tr = sum(1 for p in split["train"] + split["val"] if p in pos)
json.dump(
    {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "cac": 1},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "name": "Dataset910_COCA_CAC_HO",
        "description": "COCA gated, held-out protocol; positives of train+val; research-only.",
    },
    open(D / "dataset.json", "w"),
    indent=1,
)
(W / "nnunet_preprocessed" / "Dataset910_COCA_CAC_HO").mkdir(parents=True, exist_ok=True)
json.dump(
    [
        {
            "train": [stem(p) for p in split["train"] if p in pos],
            "val": [stem(p) for p in split["val"] if p in pos],
        }
    ],
    open(W / "nnunet_preprocessed" / "Dataset910_COCA_CAC_HO" / "splits_final.json", "w"),
)
for name in ("val", "test"):
    (W / f"{name}_all").mkdir(exist_ok=True)
    (W / f"{name}_labels").mkdir(exist_ok=True)
    for p in split[name]:
        for src, dst in (
            (
                RAW / "images" / f"{stem(p)}_0000.nii.gz",
                W / f"{name}_all" / f"{stem(p)}_0000.nii.gz",
            ),
            (RAW / "labels" / f"{stem(p)}.nii.gz", W / f"{name}_labels" / f"{stem(p)}.nii.gz"),
        ):
            dst.exists() or os.link(src, dst)
print("SPLIT-DONE", json.dumps(info["counts"]), info["categories"])
