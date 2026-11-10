"""Run CoroTree-0.1 on a folder of predicted masks and (optionally) score against trees built from the GT masks.

Usage:
  run.py --images D:/imagecas-work/val_s1 --pred <pred dir> --heart-aorta-root <sibling work dir> --out <out dir>
         [--gt D:/imagecas-work/val_labels_s1] [--model-sha <sha256>] [--workers 6] [--cases CaseNNNN ...]
Per case: <out>/CaseNNNN/{mask.nii.gz, tree.json, centerlines.vtp, qc.json, preview.png} and, with --gt,
<out>/CaseNNNN/reference_tree.json built from the GT mask with the same pipeline.
Summary: <out>/summary.csv + summary.json (root detection per side, side agreement, tree length vs reference,
tree coverage of the reference skeleton within 1 mm, unknown fragments, status distribution).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coronary_tree.tree.bridge import BRIDGE_P, bridge_all, evidence_volume, load_probability  # noqa: E402
from coronary_tree.tree.build import (  # noqa: E402
    P,
    build_trees,
    coverage_of,
    tree_points,
    write_preview,
    write_qc_json,
    write_tree_json,
    write_vtp,
)
from coronary_tree.tree.ostia import ostium_candidates  # noqa: E402
from coronary_tree.tree.skeleton_graph import skeleton_graph  # noqa: E402

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_from_mask(
    mask,
    heart,
    aorta,
    spacing,
    affine,
    prob=None,
    ct=None,
    bridge_params=None,
    gt_edt=None,
    accept=True,
):
    ost = ostium_candidates(mask, heart, aorta, spacing, affine)
    G, sk, edt = skeleton_graph(mask, spacing, affine)
    res = build_trees(G, ost, mask, spacing, affine, P)
    _, n_raw = ndi.label(mask, structure=STRUCT26)
    records = []
    if bridge_params is not None and (prob is not None or ct is not None):
        ev = evidence_volume(prob, ct, bridge_params["hu_lo"], bridge_params["hu_hi"])
        pre = dict(
            status=res["status"],
            unknown_length_mm=res["unknown_length_mm"],
            n_unknown=len(res["unknown_fragments"]),
        )
        for it in range(bridge_params.get("max_iter", 2)):
            recs = bridge_all(
                G,
                res,
                ev,
                prob,
                ct,
                mask,
                edt,
                spacing,
                affine,
                bridge_params,
                gt_edt_mm=gt_edt,
                accept=accept,
            )
            for r in recs:
                r["iteration"] = it
            records.extend(recs)
            if not any(r.get("added") for r in recs):
                break
            res = build_trees(G, ost, mask, spacing, affine, P)
        res["bridging"] = dict(
            mode=("suggestions_only" if not accept else "auto"),
            before=pre,
            params={k: v for k, v in bridge_params.items()},
            candidates=records,
            n_accepted=sum(1 for r in records if r.get("added")),
            note="G2.1 calibration on Split-1 val found no acceptance rule with a false-bridge rate <= 10 %; "
            "automatic bridging is disabled, candidates are reviewer suggestions (FR-10)",
        )
    return res, ost, int(n_raw)


def process_case(
    case: str,
    image_p: str,
    pred_p: str,
    ha_p: str,
    gt_p: str | None,
    out_dir: str,
    model_sha: str,
    prob_p: str | None = None,
    bridge_params: dict | None = None,
    collect_only: bool = False,
) -> dict:
    out = Path(out_dir) / case
    out.mkdir(parents=True, exist_ok=True)
    row = dict(case=case)
    try:
        img = nib.load(image_p)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        affine = img.affine
        pred_img = nib.load(pred_p)
        mask = np.asanyarray(pred_img.dataobj) > 0.5
        ha = np.asanyarray(nib.load(ha_p).dataobj)
        heart, aorta = ha == 51, ha == 52
        prob = load_probability(prob_p) if prob_p and Path(prob_p).exists() else None
        ct = np.asanyarray(img.dataobj).astype(np.float32) if bridge_params is not None else None
        gt_edt = None
        if gt_p and bridge_params is not None:
            gt_edt = ndi.distance_transform_edt(
                ~(np.asanyarray(nib.load(gt_p).dataobj) > 0.5), sampling=spacing
            )
        res, ost, n_raw = tree_from_mask(
            mask,
            heart,
            aorta,
            spacing,
            affine,
            prob,
            ct,
            bridge_params,
            gt_edt,
            accept=not collect_only,
        )
        input_sha = sha256_file(Path(image_p))
        shutil.copyfile(pred_p, out / "mask.nii.gz")
        write_tree_json(out / "tree.json", res, model_sha, input_sha)
        write_qc_json(out / "qc.json", res, model_sha, input_sha, ost, n_raw)
        if res.get("bridging"):
            q = json.loads((out / "qc.json").read_text(encoding="utf-8"))
            q["bridging"] = res["bridging"]
            (out / "qc.json").write_text(json.dumps(q, indent=1, default=float), encoding="utf-8")
            b = res["bridging"]
            row.update(
                n_bridge_candidates=len([r for r in b["candidates"] if r.get("found")]),
                n_bridges_accepted=b["n_accepted"],
                n_bridges_true=sum(
                    1 for r in b["candidates"] if r.get("accepted") and r.get("gt_true")
                ),
                n_bridges_false=sum(
                    1 for r in b["candidates"] if r.get("accepted") and r.get("gt_true") is False
                ),
                status_before_bridging=b["before"]["status"],
                unknown_len_before_mm=b["before"]["unknown_length_mm"],
                bridge_records=json.dumps(b["candidates"], default=float),
            )
        write_vtp(out / "centerlines.vtp", res)
        write_preview(out / "preview.png", res, mask, affine)
        row.update(
            status=res["status"],
            reason_codes=";".join(res["reason_codes"]),
            n_components_raw=n_raw,
            left_found="left" in res["roots"],
            right_found="right" in res["roots"],
            left_len_mm=res["trees"]["left"]["length_mm"] if res["trees"]["left"] else 0.0,
            right_len_mm=res["trees"]["right"]["length_mm"] if res["trees"]["right"] else 0.0,
            unknown_len_mm=res["unknown_length_mm"],
            n_unknown=len(res["unknown_fragments"]),
            n_pruned=len(res["log"]["pruned_edges"]),
            n_cycles=len(res["log"]["cycles_removed"]),
            n_bridges_cut=len(res["log"]["bridges_cut"]),
            n_ostium_candidates=len(ost["candidates"]),
            n_violations=len(res["violations"]),
            root_conf_left=res["roots"].get("left", {}).get("confidence"),
            root_conf_right=res["roots"].get("right", {}).get("confidence"),
        )
        if gt_p:
            gt = np.asanyarray(nib.load(gt_p).dataobj) > 0.5
            ref, ost_ref, n_ref = tree_from_mask(gt, heart, aorta, spacing, affine)
            write_tree_json(out / "reference_tree.json", ref, "GT", input_sha)
            for side in ("left", "right"):
                row[f"ref_{side}_found"] = side in ref["roots"]
                row[f"ref_{side}_len_mm"] = (
                    ref["trees"][side]["length_mm"] if ref["trees"][side] else 0.0
                )
                if side in ref["roots"] and side in res["roots"]:
                    row[f"root_dist_{side}_mm"] = float(
                        np.linalg.norm(
                            np.asarray(ref["roots"][side]["xyz_mm"])
                            - np.asarray(res["roots"][side]["xyz_mm"])
                        )
                    )
                    row[f"cov_{side}"] = coverage_of(
                        tree_points(ref, (side,)), tree_points(res, (side,)), 1.0
                    )
                else:
                    row[f"root_dist_{side}_mm"] = None
                    row[f"cov_{side}"] = 0.0 if side in ref["roots"] else None
            row["ref_status"] = ref["status"]
            row["cov_all_assigned"] = coverage_of(tree_points(ref), tree_points(res), 1.0)
            row["cov_incl_unknown"] = coverage_of(
                tree_points(ref), tree_points(res, ("left", "right", "unknown")), 1.0
            )
            row["ref_n_components_raw"] = n_ref
    except Exception as e:  # noqa: BLE001
        import traceback

        row.update(
            status="error",
            reason_codes=f"exception:{type(e).__name__}:{e}",
            traceback=traceback.format_exc()[-600:],
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument(
        "--heart-aorta-root",
        type=Path,
        required=True,
        help="context-mask root containing <numeric-case-id>/heart-aorta.nii.gz",
    )
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-sha", default="unknown")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument(
        "--prob-dir",
        type=Path,
        default=None,
        help="folder with CaseNNNN.npz probability maps (nnU-Net --save_probabilities)",
    )
    ap.add_argument(
        "--bridge", action="store_true", help="enable evidence-based bridging (needs --prob-dir)"
    )
    ap.add_argument("--gap-max-mm", type=float, default=BRIDGE_P["gap_max_mm"])
    ap.add_argument("--search-mm", type=float, default=BRIDGE_P["search_mm"])
    ap.add_argument("--ev-min-mean", type=float, default=BRIDGE_P["ev_min_mean"])
    ap.add_argument("--align-min", type=float, default=BRIDGE_P["align_min"])
    ap.add_argument(
        "--collect-only",
        action="store_true",
        help="record bridge candidates without modifying the trees",
    )
    a = ap.parse_args()
    bridge_params = None
    if a.bridge:
        bridge_params = dict(
            BRIDGE_P,
            gap_max_mm=a.gap_max_mm,
            ev_min_mean=a.ev_min_mean,
            align_min=a.align_min,
            search_mm=a.search_mm,
        )
    names = a.cases or sorted(p.name[:-7] for p in a.pred.glob("Case*.nii.gz"))
    jobs = []
    for n in names:
        cid = int(n[4:])
        jobs.append(
            (
                n,
                str(a.images / f"{n}_0000.nii.gz"),
                str(a.pred / f"{n}.nii.gz"),
                str(a.heart_aorta_root / str(cid) / "heart-aorta.nii.gz"),
                str(a.gt / f"{n}.nii.gz") if a.gt else None,
                str(a.out),
                a.model_sha,
                str(a.prob_dir / f"{n}.npz") if a.prob_dir else None,
                bridge_params,
                a.collect_only,
            )
        )
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_case, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(
                f"{i}/{len(futs)} {r['case']} {r.get('status')} {r.get('reason_codes', '')}",
                flush=True,
            )
    rows.sort(key=lambda r: r["case"])
    a.out.mkdir(parents=True, exist_ok=True)
    if bridge_params:
        cands = []
        for r in rows:
            for c in json.loads(r.pop("bridge_records", "[]") or "[]"):
                cands.append(dict(case=r["case"], **c))
        if cands:
            cf = list({k: None for c in cands for k in c})
            with (a.out / "bridge_candidates.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cf)
                w.writeheader()
                w.writerows(cands)
    fields = list({k: None for r in rows for k in r})
    with (a.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r.get("status") != "error"]
    summ = dict(
        n=len(rows),
        n_error=len(rows) - len(ok),
        status={
            s: sum(1 for r in ok if r["status"] == s)
            for s in ("complete_candidate", "needs_review", "failed")
        },
        reason_codes={},
        left_found=float(np.mean([r["left_found"] for r in ok])) if ok else None,
        right_found=float(np.mean([r["right_found"] for r in ok])) if ok else None,
        both_found=float(np.mean([r["left_found"] and r["right_found"] for r in ok]))
        if ok
        else None,
        mean_unknown_len_mm=float(np.mean([r["unknown_len_mm"] for r in ok])) if ok else None,
        mean_n_unknown=float(np.mean([r["n_unknown"] for r in ok])) if ok else None,
    )
    for r in ok:
        for c in (r["reason_codes"] or "").split(";"):
            if c:
                summ["reason_codes"][c] = summ["reason_codes"].get(c, 0) + 1
    if bridge_params and ok:
        acc = sum(int(r.get("n_bridges_accepted") or 0) for r in ok)
        tr = sum(int(r.get("n_bridges_true") or 0) for r in ok)
        fa = sum(int(r.get("n_bridges_false") or 0) for r in ok)
        summ["bridging"] = dict(
            mode=("suggestions_only" if a.collect_only else "auto"),
            note=(
                "automatic bridging disabled after Split-1 val calibration"
                if a.collect_only
                else ""
            ),
            params=bridge_params,
            candidates=sum(int(r.get("n_bridge_candidates") or 0) for r in ok),
            accepted=acc,
            true=tr,
            false=fa,
            false_rate=(fa / (tr + fa)) if (tr + fa) else None,
            status_before={
                s_: sum(1 for r in ok if r.get("status_before_bridging") == s_)
                for s_ in ("complete_candidate", "needs_review", "failed")
            },
            mean_unknown_len_before_mm=float(
                np.mean([float(r.get("unknown_len_before_mm") or 0) for r in ok])
            ),
        )
    if a.gt and ok:
        for side in ("left", "right"):
            both = [r for r in ok if r.get(f"ref_{side}_found") and r.get(f"{side}_found")]
            summ[f"ref_{side}_found"] = float(
                np.mean([bool(r.get(f"ref_{side}_found")) for r in ok])
            )
            summ[f"{side}_found_when_ref_found"] = (
                float(np.mean([r[f"{side}_found"] for r in ok if r.get(f"ref_{side}_found")]))
                if any(r.get(f"ref_{side}_found") for r in ok)
                else None
            )
            summ[f"root_dist_{side}_mm_median"] = (
                float(np.median([r[f"root_dist_{side}_mm"] for r in both])) if both else None
            )
            summ[f"root_dist_{side}_le5mm"] = (
                float(np.mean([r[f"root_dist_{side}_mm"] <= 5 for r in both])) if both else None
            )
            summ[f"cov_{side}_mean"] = (
                float(np.mean([r[f"cov_{side}"] for r in ok if r.get(f"cov_{side}") is not None]))
                if both
                else None
            )
            summ[f"len_ratio_{side}_median"] = (
                float(
                    np.median(
                        [
                            r[f"{side}_len_mm"] / r[f"ref_{side}_len_mm"]
                            for r in both
                            if r[f"ref_{side}_len_mm"] > 0
                        ]
                    )
                )
                if both
                else None
            )
        summ["cov_all_assigned_mean"] = float(np.mean([r["cov_all_assigned"] for r in ok]))
        summ["cov_incl_unknown_mean"] = float(np.mean([r["cov_incl_unknown"] for r in ok]))
        summ["frac_cov_all_assigned_ge_0.80"] = float(
            np.mean([r["cov_all_assigned"] >= 0.8 for r in ok])
        )
        summ["ref_status"] = {
            s: sum(1 for r in ok if r.get("ref_status") == s)
            for s in ("complete_candidate", "needs_review", "failed")
        }
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
