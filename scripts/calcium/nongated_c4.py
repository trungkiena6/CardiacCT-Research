# Phan C: 211 CT nguc non-gated COCA. Crop theo bbox tim + 20 mm -> du doan C4 -> dan ve luoi goc -> hau xu ly da chon
# (nguong 0.5, snap 400, gate tim 10 mm va ngoai DMC 3 mm) -> Agatston -> so voi scores.xlsx (tham chieu gated cua Stanford).
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import stats

sys.path.insert(0, "/workspace/cacseg")
import cacseg_heldout_eval as e
from eval_cacseg_nongated import agatston, read_refs
from make_cacseg_variants import crop, heart_bbox

W = Path("/workspace/cacseg-work/nongated")
CROP = W / "crop"
PRED = W / "pred_c4"
CROP.mkdir(exist_ok=True)
ch = json.loads(e.CHOICE.read_text())
boxes = {}
for img_path in sorted((W / "images").glob("*_0000.nii.gz")):
    stem = img_path.name[:-12]
    hp = W / "masks" / stem / "heart.nii.gz"
    if not hp.exists():
        continue
    box = heart_bbox(sitk.ReadImage(str(hp)), 20.0)
    if box is None:
        continue
    lo, size = box
    boxes[stem] = (lo, size)
    out = CROP / img_path.name
    if not out.exists():
        sitk.WriteImage(crop(sitk.ReadImage(str(img_path)), lo, size), str(out))
print("crops", len(boxes), flush=True)
if not (PRED.is_dir() and len(list(PRED.glob("*.npz"))) == len(boxes)):
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i",
            str(CROP),
            "-o",
            str(PRED),
            "-d",
            "911",
            "-c",
            "3d_fullres",
            "-f",
            "0",
            "-tr",
            e.TRAINER,
            "-p",
            "nnUNetResEncUNetL40GPlans",
            "--save_probabilities",
            "-npp",
            "4",
            "-nps",
            "4",
        ],
        check=True,
    )
refs = read_refs(
    Path("/workspace/coca/cocacoronarycalciumandchestcts-2/deidentified_nongated/scores.xlsx")
)
rows = []
for stem, (lo, size) in sorted(boxes.items()):
    cid = stem.split("_")[1].lstrip("0") or "0"
    if cid not in refs:
        continue
    img = sitk.ReadImage(str(W / "images" / f"{stem}_0000.nii.gz"))
    hu = sitk.GetArrayFromImage(img).astype(np.int16)
    sp = img.GetSpacing()
    p = np.load(PRED / f"{stem}.npz")["probabilities"][1].astype(np.float32)
    prob = np.zeros(hu.shape, np.float32)
    (lx, ly, lz), (sx, sy, sz) = lo, size
    prob[lz : lz + sz, ly : ly + sy, lx : lx + sx] = p
    h = sitk.GetArrayFromImage(sitk.ReadImage(str(W / "masks" / stem / "heart.nii.gz"))) > 0
    heart = ndi.distance_transform_edt(~h, sampling=sp[::-1]) <= 10.0
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(W / "masks" / stem / "aorta.nii.gz"))) > 0
    if ch["heart_gate"] == 2 and a.any():
        heart &= ndi.distance_transform_edt(~a, sampling=sp[::-1]) > e.AORTA_DILATE_MM
    pred = e.postprocess(prob, hu, sp, ch["threshold"], ch["snap_cap_mm3"], ch["heart_gate"], heart)
    rows.append(
        {"id": cid, "agatston_pred": round(agatston(hu, pred, sp), 1), "agatston_ref": refs[cid]}
    )
    print(rows[-1], flush=True)
with open(W / "nongated_c4_per_case.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


def summ(rs):
    ap = np.array([r["agatston_pred"] for r in rs])
    ar = np.array([r["agatston_ref"] for r in rs])
    c5p = [e.category(v, e.EDGES5) for v in ap]
    c5r = [e.category(v, e.EDGES5) for v in ar]
    c4p = [e.category(v, e.EDGES4) for v in ap]
    c4r = [e.category(v, e.EDGES4) for v in ar]
    d = ap - ar
    return {
        "n": len(rs),
        "kappa_lin_5cat": e.weighted_kappa(c5r, c5p, 5),
        "kappa_quad_5cat": e.weighted_kappa(c5r, c5p, 5, True),
        "kappa_lin_4cat": e.weighted_kappa(c4r, c4p, 4),
        "accuracy_4cat": float(np.mean(np.array(c4p) == np.array(c4r))),
        "icc21": e.icc21(ap, ar),
        "pearson": float(stats.pearsonr(ap, ar)[0]),
        "spearman": float(stats.spearmanr(ap, ar)[0]),
        "bias": float(d.mean()),
        "loa": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
        "zero_ref_called_positive": int(
            sum(1 for r in rs if r["agatston_ref"] == 0 and r["agatston_pred"] > 0)
        ),
        "n_zero_ref": int(sum(1 for r in rs if r["agatston_ref"] == 0)),
    }


s = summ(rows)
rng = np.random.default_rng(0)
ci = {
    k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for k, v in {
        k: [summ([rows[i] for i in rng.integers(0, len(rows), len(rows))])[k] for _ in range(2000)]
        for k in ("kappa_lin_4cat", "icc21", "accuracy_4cat")
    }.items()
}
(W / "nongated_c4_result.json").write_text(
    json.dumps({"choice": ch, "nongated": s, "ci95": ci}, indent=2)
)
print("NONGATED-C4 RESULT", s, ci, flush=True)
