# Chuyen toan bo 787 benh nhan COCA gated (co va khong co xml) sang NIfTI + nhan, song song 32 tien trinh,
# va ghi manifest.csv (has_xml, so ROI, trang thai, voxel vôi, Agatston tham chieu tu nhan) de chia tap theo benh nhan.
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/workspace/cacseg")
from coca_to_nnunet import load_patient, rasterize
from eval_cacseg_nongated import agatston

ROOT = Path("/workspace/coca/cocacoronarycalciumandchestcts-2/Gated_release_final")
OUT = Path("/workspace/cacseg-work/raw_all")
IMG = OUT / "images"
LAB = OUT / "labels"


def one(pid):
    rec = {
        "pid": pid,
        "has_xml": 0,
        "n_roi": 0,
        "status": "ok",
        "mask_voxels": 0,
        "agatston_gt": 0.0,
        "shape": "",
        "spacing": "",
    }
    try:
        image, hu, slice_z = load_patient(ROOT / "patient" / pid)
        xml = ROOT / "calcium_xml" / f"{pid}.xml"
        if xml.is_file():
            rec["has_xml"] = 1
            mask, checks = rasterize(image, hu, slice_z, xml)
            rec["n_roi"] = len(checks)
            bad = [
                c
                for c in checks
                if c["hu_max_plist"] is not None
                and abs(c["hu_max_raster"] - c["hu_max_plist"]) > 50
            ]
            if bad:
                rec["status"] = f"skip_hu_mismatch_{len(bad)}of{len(checks)}"
                return rec
        else:
            mask = np.zeros(hu.shape, dtype=np.uint8)
        rec["mask_voxels"] = int(mask.sum())
        rec["agatston_gt"] = round(agatston(hu, mask.astype(bool), image.GetSpacing()), 1)
        rec["shape"] = "x".join(map(str, hu.shape))
        rec["spacing"] = "x".join(f"{s:.3f}" for s in image.GetSpacing())
        stem = f"coca_{int(pid):04d}"
        sitk.WriteImage(image, str(IMG / f"{stem}_0000.nii.gz"))
        lab = sitk.GetImageFromArray(mask)
        lab.CopyInformation(image)
        sitk.WriteImage(lab, str(LAB / f"{stem}.nii.gz"))
    except Exception as e:
        rec["status"] = f"error_{type(e).__name__}"
    return rec


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    LAB.mkdir(parents=True, exist_ok=True)
    pids = sorted(
        (p.name for p in (ROOT / "patient").iterdir() if p.is_dir() and p.name.isdigit()), key=int
    )
    with ProcessPoolExecutor(32) as ex:
        rows = list(ex.map(one, pids))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(
        "CONVERT-DONE",
        len(rows),
        "patients;",
        len(ok),
        "ok;",
        sum(r["has_xml"] for r in ok),
        "with xml;",
        sum(1 for r in ok if r["mask_voxels"] > 0),
        "positive masks;",
        [
            f"{r[chr(112) + chr(105) + chr(100)]}:{r[chr(115) + chr(116) + chr(97) + chr(116) + chr(117) + chr(115)]}"
            for r in rows
            if r["status"] != "ok"
        ][:25],
        flush=True,
    )
