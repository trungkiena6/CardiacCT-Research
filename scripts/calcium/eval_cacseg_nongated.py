"""Kiểm chuẩn ngoài CACSeg trên 213 ca CT ngực không gated của COCA.

Ba bước:
  prepare  — DICOM -> NIfTI cho nnUNetv2_predict (nong_<N>_0000.nii.gz)
  (ngoài script) nnUNetv2_predict -i <out>/images -o <out>/pred -d 902 -c 3d_fullres \
                 -f 0 -tr nnUNetTrainer_250epochs
  score    — Agatston từ mask dự đoán, đối chiếu scores.xlsx (điểm + nhóm nguy cơ)

Agatston chuẩn: theo lát 2D, thành phần liên thông >= 1 mm2, trọng số HU max
(1: 130-199, 2: 200-299, 3: 300-399, 4: >=400), nhân area mm2; hệ số lát = dz/3mm.
Nhóm nguy cơ: 0 / 1-100 / 101-400 / >400.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

RISK_EDGES = (0.0, 100.0, 400.0)


def risk_group(score: float) -> int:
    if score <= 0:
        return 0
    for i, edge in enumerate(RISK_EDGES[1:], start=1):
        if score <= edge:
            return i
    return len(RISK_EDGES)


def load_series(dicom_dir: Path):
    import SimpleITK as sitk

    best = None
    stack = [dicom_dir]
    while stack:
        d = stack.pop()
        stack.extend(x for x in d.iterdir() if x.is_dir())
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d)) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        raise RuntimeError("khong doc duoc series")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best)
    return reader.Execute()


def read_refs(xlsx: Path) -> dict[str, float]:
    from openpyxl import load_workbook

    ws = load_workbook(xlsx, read_only=True).active
    refs = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[0] is not None and row[5] is not None:
            refs[str(row[0]).strip().rstrip("Aa")] = float(row[5])
    return refs


def agatston(hu: np.ndarray, mask: np.ndarray, spacing) -> float:
    """spacing = (sx, sy, sz) mm theo thứ tự SimpleITK; hu/mask [z, y, x]."""
    from scipy import ndimage

    px_area = spacing[0] * spacing[1]
    factor = spacing[2] / 3.0
    total = 0.0
    for z in range(hu.shape[0]):
        m = mask[z] & (hu[z] >= 130)
        if not m.any():
            continue
        labels, n = ndimage.label(m)
        for comp in range(1, n + 1):
            sel = labels == comp
            area = float(sel.sum()) * px_area
            if area < 1.0:
                continue
            peak = int(hu[z][sel].max())
            weight = 4 if peak >= 400 else 3 if peak >= 300 else 2 if peak >= 200 else 1
            total += area * weight
    return total * factor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["prepare", "score"])
    parser.add_argument(
        "--root", type=Path, required=True, help=".../cocacoronarycalciumandchestcts-2"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--heart-dir",
        type=Path,
        default=None,
        help="mask tim (run_heart_masks.py); cham diem trong tim + dilate",
    )
    parser.add_argument("--heart-dilate-mm", type=float, default=10.0)
    args = parser.parse_args()
    import SimpleITK as sitk

    nongated = args.root / "deidentified_nongated"
    refs = read_refs(nongated / "scores.xlsx")
    ids = sorted(refs, key=lambda s: int(s) if s.isdigit() else 10**9)
    if args.limit:
        ids = ids[: args.limit]

    if args.mode == "prepare":
        images = args.out / "images"
        images.mkdir(parents=True, exist_ok=True)
        done = missing = 0
        for cid in ids:
            dst = images / f"nong_{cid}_0000.nii.gz"
            if dst.exists():
                done += 1
                continue
            src = nongated / cid
            if not src.is_dir():
                print(f"[!] {cid}: khong co thu muc")
                missing += 1
                continue
            try:
                sitk.WriteImage(load_series(src), str(dst))
                done += 1
            except Exception as exc:
                print(f"[!] {cid}: {type(exc).__name__}: {exc}")
                missing += 1
        print(f"PREPARE XONG: {done} ca, loi/thieu {missing}")
        return 0

    # score
    rows = []
    for cid in ids:
        img_p = args.out / "images" / f"nong_{cid}_0000.nii.gz"
        pred_p = args.out / "pred" / f"nong_{cid}.nii.gz"
        if not (img_p.exists() and pred_p.exists()):
            continue
        image = sitk.ReadImage(str(img_p))
        hu = sitk.GetArrayFromImage(image).astype(np.int16)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_p))).astype(bool)
        if args.heart_dir is not None:
            heart_p = args.heart_dir / f"nong_{cid}.nii.gz"
            if not heart_p.exists():
                continue
            from scipy import ndimage

            heart = sitk.GetArrayFromImage(sitk.ReadImage(str(heart_p))).astype(bool)
            if heart.shape != mask.shape:
                print(f"[!] {cid}: mask tim lech kich thuoc, bo qua")
                continue
            sx, sy, sz = image.GetSpacing()
            dist = ndimage.distance_transform_edt(~heart, sampling=(sz, sy, sx))
            mask = mask & (dist <= args.heart_dilate_mm)
        score = agatston(hu, mask, image.GetSpacing())
        rows.append(
            {
                "id": cid,
                "agatston_pred": round(score, 1),
                "agatston_ref": refs[cid],
                "group_pred": risk_group(score),
                "group_ref": risk_group(refs[cid]),
            }
        )
    if not rows:
        print("Chua co du lieu pred — chay nnUNetv2_predict truoc")
        return 1
    out_csv = args.out / "cacseg_nongated_scores.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pred = np.array([r["agatston_pred"] for r in rows])
    ref = np.array([r["agatston_ref"] for r in rows])
    gp = np.array([r["group_pred"] for r in rows])
    gr = np.array([r["group_ref"] for r in rows])
    agree = float((gp == gr).mean())
    # kappa trọng số tuyến tính 4 nhóm
    n_cat = len(RISK_EDGES) + 1
    obs = np.zeros((n_cat, n_cat))
    for a, b in zip(gr, gp):
        obs[a, b] += 1
    obs /= obs.sum()
    w = 1 - np.abs(np.subtract.outer(np.arange(n_cat), np.arange(n_cat))) / (n_cat - 1)
    exp = np.outer(obs.sum(1), obs.sum(0))
    kappa = float((1 - (1 - (w * obs).sum()) / (1 - (w * exp).sum())))
    r_pearson = float(np.corrcoef(pred, ref)[0, 1]) if len(rows) > 2 else float("nan")
    print(f"SCORE XONG: {len(rows)} ca -> {out_csv}")
    print(f"  Pearson r (diem):      {r_pearson:.3f}")
    print(f"  Khop nhom nguy co:     {agree * 100:.1f}%")
    print(f"  Kappa trong so:        {kappa:.3f}")
    print(
        f"  Lech diem trung vi:    {np.median(pred - ref):+.1f} "
        f"(IQR {np.percentile(pred - ref, 25):+.1f}..{np.percentile(pred - ref, 75):+.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
