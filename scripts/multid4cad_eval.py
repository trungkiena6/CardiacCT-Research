"""MultiD4CAD (Prinzi, Militello et al., Sci Data 2025; 118 CCTA, Siemens, 120 kVp; radiologist EAT masks and pericoronary-fat masks
around the LAD; CAD label + risk factors) as an external check of the pipeline. Data under DUA: nothing from it is redistributed.

Facts established on the files (15 Oct 2026): the full volumes in the EAT folder are stored as HU + 1024 with a header that claims
RAS while the stored z axis runs superior -> inferior; D:/imagecas-work/multid4cad/convert_hu.py writes HU int16 copies reoriented to
the ImageCAS/nnU-Net LAS convention (and the radiologist masks identically, stage/eat_ref); the EAT mask = fat voxels inside a
radiologist ROI over the slices they chose; the
pericoronary (PAT) volume is a small reformatted volume around the LAD whose affine is a copy of the full volume's, so it cannot be
placed voxel-wise in the full volume: the PAT comparison is patient-level (mean HU and volume), not voxel-level.

stages:
  eat      : rule-based EAT (our protocol, TotalSegmentator + LM landmark) and the self-trained EAT model vs the radiologist mask:
             Dice inside the radiologist's slice range, Dice inside the overlap of both ranges, volume agreement (Bland-Altman).
  pat      : our LAD PCAT (FAI, volume) vs the radiologist PAT mask (mean HU, volume): r, bias, LoA.
  clinical : CAD label vs pipeline features (FAI per vessel, EAT volumes): AUC, logistic regression adjusted for age, sex, BMI.

  python scripts/multid4cad_eval.py eat|pat|clinical|all --out results/multid4cad

Set MULTID4CAD_WORK to the private work root containing MultiD4CAD_Dataset_v1,
stage, totalseg, tree and model outputs. The historical default is D:/imagecas-work/multid4cad.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eat_rule_based import FAT, compute_eat
from eat_rule_based import P as EAT_P  # noqa: E402
from multid4cad_pat_report import render_pat_report  # noqa: E402

W = Path(os.environ.get("MULTID4CAD_WORK", "D:/imagecas-work/multid4cad"))
M = W / "MultiD4CAD_Dataset_v1"
EATD = M / "Epicardial Adipose Tissue Segmentations (118 patients)"
PATD = M / "Pericoronaric Adipose Tissue Segmentations (118 patients)"
STAGE, TS, TREES, PCAT_CSV, EAT_MODEL = (
    W / "stage/images",
    W / "totalseg",
    W / "tree",
    W / "pcat/pcat_per_vessel.csv",
    W / "eat_model_pred",
)


def discover_patients(directory: Path | None = None) -> list[str]:
    """Discover private data only when an evaluation stage is requested."""
    directory = EATD if directory is None else directory
    patients = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not patients:
        raise ValueError(f"No patient directories found in {directory}")
    return patients


def case_of(pid):  # P001 -> Case0001 (CoroTree naming)
    return f"Case{int(pid[1:]):04d}"


def read_csv_semicolon(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [
        {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()} for r in rows
    ]


def load_zyx(path, flip):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return a[::-1] if flip else a


def lm_z(pid, img, flip):
    p = TREES / case_of(pid) / "tree.json"
    if not p.exists():
        return None
    left = json.loads(p.read_text(encoding="utf-8"))["trees"].get("left") or {}
    edges = left.get("edges") or []
    if not edges:
        return None
    x, y, z = edges[0]["points_xyz_mm"][0]
    k = img.TransformPhysicalPointToIndex([float(-x), float(-y), float(z)])[2]
    return img.GetSize()[2] - 1 - k if flip else k


def dice(a, b):
    return 2.0 * float((a & b).sum()) / max(1, int(a.sum()) + int(b.sum()))


def eat_one(pid):
    img = sitk.ReadImage(str(STAGE / f"{pid}_0000.nii.gz"))
    flip = img.GetDirection()[8] < 0
    hu = load_zyx(STAGE / f"{pid}_0000.nii.gz", flip).astype(np.int16)
    lab = load_zyx(TS / f"{pid}.nii.gz", flip)
    ref = (
        load_zyx(W / "stage/eat_ref" / f"{pid}.nii.gz", flip) > 0
    )  # radiologist mask, reoriented exactly like the image (convert_hu.py)
    if lab.shape != hu.shape or ref.shape != hu.shape:
        raise RuntimeError(f"{pid}: grid mismatch {lab.shape} {ref.shape} {hu.shape}")
    sp = img.GetSpacing()[::-1]
    vox_ml = float(np.prod(sp)) / 1000.0
    heart = lab == 51
    if not heart.any():
        raise RuntimeError(f"{pid}: no heart label")
    zs = np.nonzero(heart.any(axis=(1, 2)))[0]
    zl = lm_z(pid, img, flip)
    z_top = (
        int(min(zs.max(), zl + round(EAT_P["sup_mm"] / sp[0]))) if zl is not None else int(zs.max())
    )
    z_bot = int(zs.min())
    eat, sac, _ = compute_eat(hu, lab, sp, z_bot, z_top, EAT_P)
    rz = np.nonzero(ref.any(axis=(1, 2)))[0]
    rlo, rhi = int(rz.min()), int(rz.max())
    in_ref = np.zeros_like(ref)
    in_ref[rlo : rhi + 1] = True
    olo, ohi = max(rlo, z_bot), min(rhi, z_top)
    in_both = np.zeros_like(ref)
    in_both[olo : ohi + 1] = True
    ref_fat = ref & (hu >= FAT[0]) & (hu <= FAT[1])  # radiologist mask restricted to our fat window
    row = dict(
        pid=pid,
        flip=flip,
        ref_ml=ref.sum() * vox_ml,
        ref_slices_mm=(rhi - rlo + 1) * sp[0],
        ours_ml=eat.sum() * vox_ml,
        ours_in_ref_range_ml=(eat & in_ref).sum() * vox_ml,
        ref_in_our_range_ml=(ref[z_bot : z_top + 1]).sum() * vox_ml,
        our_range_mm=(z_top - z_bot + 1) * sp[0],
        ref_above_our_top_mm=max(0, rhi - z_top) * sp[0],
        ref_below_our_bottom_mm=max(0, z_bot - rlo) * sp[0],
        lm_found=zl is not None,
        dice_rule_refrange=dice(eat & in_ref, ref),
        dice_rule_overlap=dice(eat & in_both, ref & in_both),
        dice_rule_refrange_fatwin=dice(eat & in_ref, ref_fat),
        sac_covers_ref=float((sac & ref).sum() / max(1, ref.sum())),
        ref_hu_mean=float(hu[ref].mean()),
        ref_hu_p1=float(np.percentile(hu[ref], 1)),
        ref_hu_p99=float(np.percentile(hu[ref], 99)),
        ref_frac_in_fatwin=float(((hu[ref] >= FAT[0]) & (hu[ref] <= FAT[1])).mean()),
        ours_hu_mean=float(hu[eat].mean()) if eat.any() else None,
    )
    mp = EAT_MODEL / f"{pid}.nii.gz"
    if mp.exists():
        mdl = load_zyx(mp, flip) > 0
        row.update(
            model_ml=mdl.sum() * vox_ml,
            model_in_ref_range_ml=(mdl & in_ref).sum() * vox_ml,
            dice_model_refrange=dice(mdl & in_ref, ref),
            dice_model_overlap=dice(mdl & in_both, ref & in_both),
            dice_model_vs_rule=dice(mdl, eat),
        )
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def bland(x, y):
    d = np.asarray(y) - np.asarray(x)
    return dict(
        n=len(d),
        bias=float(d.mean()),
        sd=float(d.std(ddof=1)),
        loa=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
        r=float(np.corrcoef(x, y)[0, 1]),
    )


def stage_eat(out: Path, workers: int, patients=None):
    patients = discover_patients() if patients is None else patients
    rows, errs = [], []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(eat_one, p): p for p in patients}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:  # noqa: BLE001
                errs.append((futs[f], repr(e)[:160]))
            if i % 20 == 0 or i == len(futs):
                print(f"eat {i}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: r["pid"])
    write_csv(out / "multid4cad_eat_per_patient.csv", rows)
    m = lambda k: np.array([r[k] for r in rows if r.get(k) is not None])  # noqa: E731
    L = [
        f"# EAT: rule-based estimator and self-trained model vs the MultiD4CAD radiologist masks ({len(rows)} patients, {len(errs)} errors)",
        "",
        "Radiologist mask = fat voxels inside their ROI over the slices they chose (their superior limit is usually above ours). Dice is computed",
        "inside the radiologist's slice range (ours cut to their range) and inside the overlap of both ranges. Volumes in mL.",
        "",
        "| comparison | Dice mean | Dice median | IQR |",
        "|---|---:|---:|---|",
    ]
    for k, lab in (
        ("dice_rule_refrange", "rule, inside radiologist range"),
        ("dice_rule_overlap", "rule, inside overlap of ranges"),
        ("dice_rule_refrange_fatwin", "rule vs radiologist mask cut to -190…-30 HU"),
        ("dice_model_refrange", "self-trained model, inside radiologist range"),
        ("dice_model_overlap", "self-trained model, inside overlap"),
        ("dice_model_vs_rule", "model vs rule (consistency)"),
    ):
        v = m(k)
        if len(v):
            L.append(
                f"| {lab} | {v.mean():.3f} | {np.median(v):.3f} | {np.percentile(v, 25):.3f}–{np.percentile(v, 75):.3f} |"
            )
    L += [
        "",
        "## Volumes",
        "",
        "| pair | n | bias (ours − radiologist) | SD | 95 % LoA | r |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for a, b, lab in (
        ("ref_ml", "ours_in_ref_range_ml", "rule inside radiologist range vs radiologist"),
        ("ref_ml", "ours_ml", "rule, our protocol vs radiologist (ranges differ)"),
        (
            "ref_ml",
            "model_in_ref_range_ml",
            "self-trained model inside radiologist range vs radiologist",
        ),
    ):
        ok = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        if len(ok) >= 3:
            s = bland([r[a] for r in ok], [r[b] for r in ok])
            L.append(
                f"| {lab} | {s['n']} | {s['bias']:+.1f} mL | {s['sd']:.1f} | [{s['loa'][0]:+.1f}, {s['loa'][1]:+.1f}] | {s['r']:.3f} |"
            )
    L += [
        "",
        "## Ranges and windows",
        "",
        f"- Radiologist range: median {np.median(m('ref_slices_mm')):.0f} mm (IQR {np.percentile(m('ref_slices_mm'), 25):.0f}–{np.percentile(m('ref_slices_mm'), 75):.0f}); ours: median {np.median(m('our_range_mm')):.0f} mm.",
        f"- Radiologist range extends above our LM + 35 mm limit by median {np.median(m('ref_above_our_top_mm')):.0f} mm and below our heart-bottom limit by median {np.median(m('ref_below_our_bottom_mm')):.0f} mm.",
        f"- HU inside the radiologist masks: mean {m('ref_hu_mean').mean():.1f}, p1 {np.median(m('ref_hu_p1')):.0f}, p99 {np.median(m('ref_hu_p99')):.0f}; fraction inside our -190…-30 window {m('ref_frac_in_fatwin').mean():.3f}.",
        f"- Our sac covers {100 * m('sac_covers_ref').mean():.1f} % of the radiologist fat voxels (median {100 * np.median(m('sac_covers_ref')):.1f} %).",
        f"- Radiologist EAT volume median {np.median(m('ref_ml')):.0f} mL (IQR {np.percentile(m('ref_ml'), 25):.0f}–{np.percentile(m('ref_ml'), 75):.0f}); rule, our protocol: median {np.median(m('ours_ml')):.0f} mL.",
        f"- LM landmark found in {int(m('lm_found').sum())}/{len(rows)}.",
    ]
    if errs:
        L += ["", "Errors: " + "; ".join(f"{p}: {e}" for p, e in errs[:8])]
    (out / "multid4cad_eat_vs_radiologist.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("EAT-STAGE-DONE", flush=True)


def pat_one(pid):
    im = sitk.ReadImage(str(PATD / pid / "coronaricDatasetNIFTI.nii"))
    mk = sitk.ReadImage(str(PATD / pid / "coronaricDatasetNIFTI_mask.nii"))
    a = sitk.GetArrayFromImage(im).astype(np.float32)
    m = sitk.GetArrayFromImage(mk) > 0
    vox_ml = float(np.prod(im.GetSpacing())) / 1000.0
    hu = a[m]
    return dict(
        pid=pid,
        pat_ref_ml=round(m.sum() * vox_ml, 3),
        pat_ref_hu=round(float(hu.mean()), 2),
        pat_ref_hu_fatwin=round(float(hu[(hu >= FAT[0]) & (hu <= FAT[1])].mean()), 2)
        if ((hu >= FAT[0]) & (hu <= FAT[1])).any()
        else None,
        pat_ref_frac_fatwin=round(float(((hu >= FAT[0]) & (hu <= FAT[1])).mean()), 3),
        pat_ref_slices=int(m.any(axis=(1, 2)).sum()),
        pat_ref_p5=round(float(np.percentile(hu, 5)), 1),
        pat_ref_p95=round(float(np.percentile(hu, 95)), 1),
    )


def stage_pat(out: Path, workers: int, patients=None):
    patients = discover_patients() if patients is None else patients
    with ProcessPoolExecutor(workers) as ex:
        ref = {r["pid"]: r for r in ex.map(pat_one, patients)}
    auto = {}
    if PCAT_CSV.exists():
        for r in csv.DictReader(open(PCAT_CSV, encoding="utf-8")):
            auto[(r["case"], r["vessel"])] = r
    rows = []
    for pid in patients:
        r = dict(ref[pid])
        a = auto.get((case_of(pid), "LAD"), {})
        r.update(
            lad_fai_auto=float(a["fai_hu"]) if a.get("fai_hu") not in (None, "", "None") else None,
            lad_pcat_ml_auto=float(a["pcat_volume_ml"])
            if a.get("pcat_volume_ml") not in (None, "", "None")
            else None,
            lad_status=a.get("status", "no_pcat"),
        )
        for v in ("RCA", "LCx"):
            b = auto.get((case_of(pid), v), {})
            r[f"{v.lower()}_fai_auto"] = (
                float(b["fai_hu"]) if b.get("fai_hu") not in (None, "", "None") else None
            )
        rows.append(r)
    write_csv(out / "multid4cad_pat_per_patient.csv", rows)
    report = render_pat_report(rows)
    (out / "multid4cad_pat_vs_radiologist.md").write_text(report, encoding="utf-8")
    print(report)
    print("PAT-STAGE-DONE", flush=True)


def stage_clinical(out: Path, patients=None):
    patients = discover_patients() if patients is None else patients
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    clin = {r["PatientID"]: r for r in read_csv_semicolon(M / "clinicalDataWithLabels.CSV")}
    lab_key = [k for k in next(iter(clin.values())) if k.lower().startswith("label")][0]
    feats = {}
    for name, path in (
        ("eat", out / "multid4cad_eat_per_patient.csv"),
        ("pat", out / "multid4cad_pat_per_patient.csv"),
    ):
        if path.exists():
            for r in csv.DictReader(open(path, encoding="utf-8")):
                feats.setdefault(r["pid"], {}).update(
                    {
                        k: (float(v) if v not in ("", "None", "True", "False") and _isnum(v) else v)
                        for k, v in r.items()
                        if k != "pid"
                    }
                )
    cand = [
        ("ref_ml", "EAT volume, radiologist (mL)"),
        ("ours_ml", "EAT volume, rule (mL)"),
        ("model_ml", "EAT volume, self-trained model (mL)"),
        ("lad_fai_auto", "FAI LAD, automatic (HU)"),
        ("rca_fai_auto", "FAI RCA, automatic (HU)"),
        ("lcx_fai_auto", "FAI LCx, automatic (HU)"),
        ("pat_ref_hu", "PAT mean HU, radiologist mask"),
    ]
    L = [
        "# CAD label vs pipeline features on MultiD4CAD (78 CAD / 40 non-CAD)",
        "",
        "Sanity check of clinical plausibility, not a clinical result: single centre, 118 patients, CAD group younger than the non-CAD group in this dataset,",
        "so every association is shown crude and adjusted for age, sex and BMI. OR per 1 SD of the feature (logistic regression, statsmodels).",
        "",
        "| feature | n | AUC (crude) | OR per SD, adjusted | 95 % CI | p |",
        "|---|---:|---:|---:|---|---:|",
    ]
    y_all = {p: int(float(clin[p][lab_key])) for p in clin}
    base = {
        p: (
            float(clin[p]["Age"]),
            float(clin[p]["Gender (male=1 - female=0)"]),
            float(clin[p]["BMI"]),
        )
        for p in clin
    }
    for k, lab in cand:
        ids = [
            p
            for p in patients
            if p in feats and feats[p].get(k) not in (None, "", "None") and p in y_all
        ]
        if len(ids) < 20:
            continue
        x = np.array([feats[p][k] for p in ids], float)
        y = np.array([y_all[p] for p in ids])
        auc = roc_auc_score(y, x)
        xs = (x - x.mean()) / x.std(ddof=1)
        X = sm.add_constant(np.column_stack([xs, np.array([base[p] for p in ids])]))
        try:
            fit = sm.Logit(y, X).fit(disp=0)
            orr = float(np.exp(fit.params[1]))
            ci = np.exp(fit.conf_int()[1])
            pval = float(fit.pvalues[1])
            L.append(
                f"| {lab} | {len(ids)} | {auc:.3f} | {orr:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {pval:.3f} |"
            )
        except Exception as e:  # noqa: BLE001
            L.append(f"| {lab} | {len(ids)} | {auc:.3f} | – | – | fit failed: {type(e).__name__} |")
    ages = np.array([base[p][0] for p in patients])
    yy = np.array([y_all[p] for p in patients])
    L += [
        "",
        f"Age: CAD {ages[yy == 1].mean():.1f} ± {ages[yy == 1].std():.1f} vs non-CAD {ages[yy == 0].mean():.1f} ± {ages[yy == 0].std():.1f} years (AUC of age alone {roc_auc_score(yy, ages):.3f}).",
        "CAD definition is not stated in the data descriptor; treat the label as 'referred with suspected CAD and labelled by the providing centre'.",
    ]
    (out / "multid4cad_clinical.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("CLINICAL-STAGE-DONE", flush=True)


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("eat", "pat", "clinical", "all"))
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "multid4cad"
    )
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    patients = discover_patients()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("eat", "all"):
        stage_eat(a.out, a.workers, patients)
    if a.stage in ("pat", "all"):
        stage_pat(a.out, a.workers, patients)
    if a.stage in ("clinical", "all"):
        stage_clinical(a.out, patients)


if __name__ == "__main__":
    main()
