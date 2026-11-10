"""Chuyển COCA (Stanford AIMI) sang dataset nnU-Net cho model CAC ("CACSeg").

Nguồn: Gated_release_final/patient/<N>/ (DICOM) + calcium_xml/<N>.xml (plist OsiriX:
mỗi ổ vôi là đa giác toạ độ mm theo lát, kèm tên nhánh và thống kê HU Min/Mean/Max).
Ra: imagesTr/coca_NNN_0000.nii.gz + labelsTr/coca_NNN.nii.gz (nhãn nhị phân vôi hoá).

Tự kiểm: HU trong mask raster phải khớp thống kê plist (Min >= 130...). Chạy:
  python coca_to_nnunet.py --root <coca>/cocacoronarycalciumandchestcts-2 \
      --out ~/cacseg-work/nnunet_raw/Dataset902_COCA_CAC [--check 5] [--limit N]
Chỉ dùng nghiên cứu phi thương mại theo thoả thuận Stanford.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
from pathlib import Path

import numpy as np

POINT = re.compile(r"\(([-0-9.e]+),\s*([-0-9.e]+),\s*([-0-9.e]+)\)")


def parse_point(text: str):
    m = POINT.match(text.strip())
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


def load_patient(dicom_dir: Path):
    """Đọc series lớn nhất; layout COCA: patient/<N>/<tên series>/*.dcm."""
    import SimpleITK as sitk

    best = None
    for directory in [dicom_dir, *(d for d in sorted(dicom_dir.iterdir()) if d.is_dir())]:
        if any(f.name.startswith(".azDownload") for f in directory.glob("*.dcm")):
            raise RuntimeError(f"con file tam azcopy trong {directory.name}")
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        raise RuntimeError("khong doc duoc series")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best)
    image = reader.Execute()
    # Toạ độ z vật lý từng lát: một số ca thiếu lát nên không được giả định lưới đều
    import pydicom

    slice_z = np.array(
        [float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in best]
    )
    return image, sitk.GetArrayFromImage(image).astype(np.int16), slice_z


def rasterize(image, hu, slice_z, plist_path: Path):
    """Raster hoá mọi ROI của một bệnh nhân; trả (mask, báo cáo kiểm)."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    with plist_path.open("rb") as fh:
        doc = plistlib.load(fh)
    mask = np.zeros(hu.shape, dtype=np.uint8)
    checks = []
    gap = float(np.median(np.abs(np.diff(slice_z)))) if len(slice_z) > 1 else 3.0
    for entry in doc.get("Images", []):
        for roi in entry.get("ROIs", []):
            points = [parse_point(p) for p in roi.get("Point_mm", [])]
            points = [p for p in points if p]
            if len(points) < 3:
                continue
            idx = [image.TransformPhysicalPointToContinuousIndex(p) for p in points]
            # z theo lát vật lý gần nhất (chịu được series thiếu lát)
            z_mm = float(np.median([p[2] for p in points]))
            z = int(np.argmin(np.abs(slice_z - z_mm)))
            if abs(slice_z[z] - z_mm) > 0.75 * gap:
                checks.append(
                    {
                        "name": roi.get("Name", "?"),
                        "z": None,
                        "pixels": 0,
                        "hu_max_raster": -32768,
                        "hu_max_plist": roi.get("Max"),
                        "hu_mean_raster": 0.0,
                        "hu_mean_plist": 0.0,
                        "loi": "lat chua o vo nay bi thieu trong series",
                    }
                )
                continue
            poly = [(i[0], i[1]) for i in idx]  # (col, row) liên tục
            canvas = PILImage.new("L", (hu.shape[2], hu.shape[1]), 0)
            ImageDraw.Draw(canvas).polygon(poly, outline=1, fill=1)
            region = np.array(canvas, dtype=bool)
            if not region.any():
                continue
            # nhãn = đa giác ∩ (HU >= 130): đúng định nghĩa voxel vôi hoá Agatston
            mask[z] |= region & (hu[z] >= 130)
            vals = hu[z][region]
            checks.append(
                {
                    "name": roi.get("Name", "?"),
                    "z": z,
                    "pixels": int(region.sum()),
                    "hu_max_raster": int(vals.max()),
                    "hu_max_plist": roi.get("Max"),
                    "hu_mean_raster": round(float(vals.mean()), 1),
                    "hu_mean_plist": round(float(roi.get("Mean", 0)), 1),
                }
            )
    return mask, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, required=True, help=".../cocacoronarycalciumandchestcts-2"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check",
        type=int,
        default=None,
        help="chỉ kiểm một bệnh nhân, in đối chiếu HU, không ghi dataset",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    import SimpleITK as sitk

    gated = args.root / "Gated_release_final"
    xml_dir = gated / "calcium_xml"
    patient_root = gated / "patient"

    if args.check is not None:
        pid = str(args.check)
        image, hu, slice_z = load_patient(patient_root / pid)
        mask, checks = rasterize(image, hu, slice_z, xml_dir / f"{pid}.xml")
        print(f"benh nhan {pid}: {hu.shape}, mask voxels {int(mask.sum())}, ROI {len(checks)}")
        for c in checks[:10]:
            print("  ", c)
        return 0

    images_tr = args.out / "imagesTr"
    labels_tr = args.out / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    patients = sorted(
        (p for p in patient_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else 10**9,
    )
    if args.limit:
        patients = patients[: args.limit]
    for pdir in patients:
        pid = pdir.name
        stem = f"coca_{int(pid):04d}" if pid.isdigit() else f"coca_{pid}"
        img_out = images_tr / f"{stem}_0000.nii.gz"
        lab_out = labels_tr / f"{stem}.nii.gz"
        if img_out.exists() and lab_out.exists():
            done += 1
            continue
        xml = xml_dir / f"{pid}.xml"
        try:
            image, hu, slice_z = load_patient(pdir)
            if xml.is_file():
                mask, checks = rasterize(image, hu, slice_z, xml)
                # chỉ cần 1 ROI sai là loại: nhãn thiếu ổ sẽ dạy model bỏ sót vôi
                bad = [
                    c
                    for c in checks
                    if c["hu_max_plist"] is not None
                    and abs(c["hu_max_raster"] - c["hu_max_plist"]) > 50
                ]
                if bad:
                    print(f"[!] {pid}: {len(bad)}/{len(checks)} ROI lech HU — bo qua ca nay")
                    skipped += 1
                    continue
            else:
                mask = np.zeros(hu.shape, dtype=np.uint8)  # ca không vôi
            sitk.WriteImage(image, str(img_out))
            lab = sitk.GetImageFromArray(mask)
            lab.CopyInformation(image)
            sitk.WriteImage(lab, str(lab_out))
            done += 1
            if done % 25 == 0:
                print(f"da xong {done} ca")
        except Exception as exc:
            print(f"[!] {pid}: {type(exc).__name__}: {exc}")
            skipped += 1
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "cac": 1},
        "numTraining": done,
        "file_ending": ".nii.gz",
        "name": "Dataset902_COCA_CAC",
        "description": "COCA gated CT + calcium polygons (Stanford AIMI); research-only.",
    }
    (args.out / "dataset.json").write_text(json.dumps(dataset_json, indent=1), encoding="utf-8")
    print(f"HOAN TAT: {done} ca, bo qua {skipped}; dataset.json da ghi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
