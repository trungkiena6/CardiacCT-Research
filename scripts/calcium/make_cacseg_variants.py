"""Dựng các dataset biến thể CACSeg từ Dataset903 (thang thí nghiệm tăng Dice).

  904 = crop theo bbox tim + margin (giảm nền, tăng mật độ foreground)
  905 = crop + kênh 2 nhị phân HU>=130 (ứng viên vôi dẫn đường)
  906 = crop + nhãn đa lớp theo nhánh từ plist COCA (LAD/LCx/RCA/khác)

Cần mask tim (run_heart_masks.py --images-dir ... --out-dir coca_heart). Các ca
thiếu mask tim bị bỏ qua và ghi log. Mọi biến thể giữ nguyên case id để dùng chung
splits_final.json của Dataset903 (so sánh cùng 87 ca validation).
"""

from __future__ import annotations

import argparse
import json
import plistlib
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coca_to_nnunet import parse_point  # noqa: E402

BRANCH_LABELS = {
    "Left Anterior Descending Artery": 1,
    "Left Circumflex Artery": 2,
    "Right Coronary Artery": 3,
    # "Left Coronary Artery" (LM) và tên khác -> 4
}
OTHER_LABEL = 4


def heart_bbox(heart: sitk.Image, margin_mm: float):
    arr = sitk.GetArrayFromImage(heart) > 0
    if not arr.any():
        return None
    zs, ys, xs = np.nonzero(arr)
    sx, sy, sz = heart.GetSpacing()
    mx, my, mz = (int(round(margin_mm / s)) for s in (sx, sy, sz))
    shape = arr.shape  # z, y, x
    lo = [max(0, int(xs.min()) - mx), max(0, int(ys.min()) - my), max(0, int(zs.min()) - mz)]
    hi = [
        min(shape[2], int(xs.max()) + mx + 1),
        min(shape[1], int(ys.max()) + my + 1),
        min(shape[0], int(zs.max()) + mz + 1),
    ]
    size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    return lo, size  # thứ tự x, y, z của SimpleITK


def crop(image: sitk.Image, lo, size) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=size, index=lo)


def branch_label_volume(
    image: sitk.Image, hu: np.ndarray, slice_z: np.ndarray, xml_path: Path
) -> np.ndarray:
    """Nhãn đa lớp theo nhánh trên lưới gốc (trước khi crop)."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    label = np.zeros(hu.shape, dtype=np.uint8)
    if not xml_path.is_file():
        return label
    with xml_path.open("rb") as fh:
        doc = plistlib.load(fh)
    gap = float(np.median(np.abs(np.diff(slice_z)))) if len(slice_z) > 1 else 3.0
    for entry in doc.get("Images", []):
        for roi in entry.get("ROIs", []):
            points = [parse_point(p) for p in roi.get("Point_mm", [])]
            points = [p for p in points if p]
            if len(points) < 3:
                continue
            z_mm = float(np.median([p[2] for p in points]))
            z = int(np.argmin(np.abs(slice_z - z_mm)))
            if abs(slice_z[z] - z_mm) > 0.75 * gap:
                continue
            idx = [image.TransformPhysicalPointToContinuousIndex(p) for p in points]
            canvas = PILImage.new("L", (hu.shape[2], hu.shape[1]), 0)
            ImageDraw.Draw(canvas).polygon([(i[0], i[1]) for i in idx], outline=1, fill=1)
            region = np.array(canvas, dtype=bool) & (hu[z] >= 130)
            value = BRANCH_LABELS.get(str(roi.get("Name", "")), OTHER_LABEL)
            # nhánh ghi sau không đè nhánh ghi trước (ổ chồng lấn hiếm)
            label[z][region & (label[z] == 0)] = value
    return label


def slice_positions(case_dir: Path):
    import pydicom

    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(case_dir)) or ()
    best = None
    for sid in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(case_dir), sid)
        if best is None or len(files) > len(best):
            best = files
    for sub in sorted(d for d in case_dir.iterdir() if d.is_dir()):
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(sub)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(sub), sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        raise RuntimeError("khong doc duoc series")
    return np.array(
        [float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in best]
    )


def dataset_json(out: Path, name: str, channels: dict, labels: dict, n: int) -> None:
    (out / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": channels,
                "labels": labels,
                "numTraining": n,
                "file_ending": ".nii.gz",
                "name": name,
                "description": "Bien the thi nghiem tu Dataset903 (COCA, non-commercial research).",
            },
            indent=1,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="Dataset903 root")
    parser.add_argument("--heart-dir", type=Path, required=True)
    parser.add_argument(
        "--coca-root", type=Path, required=True, help=".../cocacoronarycalciumandchestcts-2"
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    xml_dir = args.coca_root / "Gated_release_final" / "calcium_xml"
    patient_root = args.coca_root / "Gated_release_final" / "patient"
    out904 = args.out_root / "Dataset904_COCA_CAC_CROP"
    out905 = args.out_root / "Dataset905_COCA_CAC_CAND"
    out906 = args.out_root / "Dataset906_COCA_CAC_BRANCH"
    for out in (out904, out905, out906):
        (out / "imagesTr").mkdir(parents=True, exist_ok=True)
        (out / "labelsTr").mkdir(parents=True, exist_ok=True)

    images = sorted((args.src / "imagesTr").glob("coca_*_0000.nii.gz"))
    if args.limit:
        images = images[: args.limit]
    done = skipped = 0
    for img_path in images:
        stem = img_path.name[: -len("_0000.nii.gz")]  # coca_0005
        pid = str(int(stem.split("_")[1]))  # 5
        final_905 = out905 / "imagesTr" / f"{stem}_0001.nii.gz"
        final_906 = out906 / "labelsTr" / f"{stem}.nii.gz"
        if final_905.exists() and final_906.exists():
            done += 1
            continue
        heart_path = args.heart_dir / f"{stem}.nii.gz"
        if not heart_path.is_file():
            skipped += 1
            print(f"[!] {stem}: chua co mask tim")
            continue
        try:
            image = sitk.ReadImage(str(img_path))
            label = sitk.ReadImage(str(args.src / "labelsTr" / f"{stem}.nii.gz"))
            heart = sitk.ReadImage(str(heart_path))
            box = heart_bbox(heart, args.margin_mm)
            if box is None:
                skipped += 1
                print(f"[!] {stem}: mask tim rong")
                continue
            lo, size = box
            c_img, c_lab = crop(image, lo, size), crop(label, lo, size)
            hu_c = sitk.GetArrayFromImage(c_img)
            # 904: crop
            sitk.WriteImage(c_img, str(out904 / "imagesTr" / f"{stem}_0000.nii.gz"))
            sitk.WriteImage(c_lab, str(out904 / "labelsTr" / f"{stem}.nii.gz"))
            # 905: crop + kênh ứng viên
            sitk.WriteImage(c_img, str(out905 / "imagesTr" / f"{stem}_0000.nii.gz"))
            cand = sitk.GetImageFromArray((hu_c >= 130).astype(np.uint8))
            cand.CopyInformation(c_img)
            sitk.WriteImage(cand, str(final_905))
            (out905 / "labelsTr" / f"{stem}.nii.gz").write_bytes(
                (out904 / "labelsTr" / f"{stem}.nii.gz").read_bytes()
            )
            # 906: crop + nhãn nhánh
            hu_full = sitk.GetArrayFromImage(image).astype(np.int16)
            slice_z = slice_positions(patient_root / pid)
            branch = branch_label_volume(image, hu_full, slice_z, xml_dir / f"{pid}.xml")
            branch_img = sitk.GetImageFromArray(branch)
            branch_img.CopyInformation(image)
            sitk.WriteImage(c_img, str(out906 / "imagesTr" / f"{stem}_0000.nii.gz"))
            sitk.WriteImage(crop(branch_img, lo, size), str(final_906))
            done += 1
            if done % 50 == 0:
                print(f"da xong {done} ca", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[!] {stem}: {type(exc).__name__}: {exc}")
    dataset_json(out904, "Dataset904_COCA_CAC_CROP", {"0": "CT"}, {"background": 0, "cac": 1}, done)
    dataset_json(
        out905,
        "Dataset905_COCA_CAC_CAND",
        {"0": "CT", "1": "noNorm"},
        {"background": 0, "cac": 1},
        done,
    )
    dataset_json(
        out906,
        "Dataset906_COCA_CAC_BRANCH",
        {"0": "CT"},
        {"background": 0, "lad": 1, "lcx": 2, "rca": 3, "other": 4},
        done,
    )
    print(f"HOAN TAT BIEN THE: {done} ca, bo qua {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
