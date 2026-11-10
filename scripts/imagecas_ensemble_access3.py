"""Third ImageCAS Split-1 test access.

select : score the four candidates on the 50 validation cases, then the cheap variants
         (a) checkpoint averaging, (b) member weights, (c) step 0.25 — greedily, each kept only
         if validation Dice rises; sweep T; apply the go/no-go rule; write test_access_3_choice.json.
test   : refuse unless the choice file says go and no test-3 output exists; predict the 250 test
         cases with the chosen configuration, score, bootstrap, paired difference to access 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagecas_ensemble_access2 import EVAL, MODELS, RES, T_GRID, W, score  # noqa: E402
from prediction_cache import (
    ensemble_request,
    file_identity,
    inference_request,
    prediction_command,
    record_cache,
    record_file,
    reuse_cache,
    reuse_file,
)
from score_segmentation import read_evaluation_rows, require_new_files

MODELS3 = dict(MODELS, resenc_l_ft=("nnUNetTrainer_ft250_NM", "nnUNetResEncUNetL40GPlans"))
CANDIDATES = {
    "resenc_l_ft": ["resenc_l_ft"],
    "unet_ft500+resenc_l_ft": ["unet_ft500", "resenc_l_ft"],
    "resenc_l+resenc_l_ft": ["resenc_l", "resenc_l_ft"],
    "unet_ft500+resenc_l+resenc_l_ft": ["unet_ft500", "resenc_l", "resenc_l_ft"],
}
FT = "resenc_l_ft"
FT_WEIGHTS = [0.5, 0.65, 0.8]
ACCESS2_VAL_DICE = 0.8416819578687148
GO_MARGIN = 0.002
CHOICE = W / "test_access_3_choice.json"
V3 = W / "testaccess3" / "val"


def fold_dir(model: str) -> Path:
    tr, pl = MODELS3[model]
    return RES / f"{tr}__{pl}__3d_fullres" / "fold_0"


def available(model: str) -> bool:
    return (fold_dir(model) / "checkpoint_final.pth").is_file()


def avg_checkpoint(model: str) -> str:
    """mean(checkpoint_best, checkpoint_final) saved next to them; returns the checkpoint name."""
    out = fold_dir(model) / "checkpoint_avg.pth"
    request = dict(
        operation="checkpoint_average",
        best=file_identity(fold_dir(model) / "checkpoint_best.pth"),
        final=file_identity(fold_dir(model) / "checkpoint_final.pth"),
        builder=file_identity(Path(__file__)),
    )
    if not reuse_file(out, request):
        import torch

        best = torch.load(
            fold_dir(model) / "checkpoint_best.pth", map_location="cpu", weights_only=False
        )
        final = torch.load(
            fold_dir(model) / "checkpoint_final.pth", map_location="cpu", weights_only=False
        )
        final["network_weights"] = {
            k: (best["network_weights"][k].float() + v.float()) / 2
            if torch.is_tensor(v) and v.is_floating_point()
            else v
            for k, v in final["network_weights"].items()
        }
        with out.open("xb") as handle:
            torch.save(final, handle)
        record_file(out, request)
    return "checkpoint_avg.pth"


def pred_dir(model: str, split: str, chk: str, step: float) -> Path:
    old = W / "testaccess2" / split / model
    if chk == "checkpoint_final.pth" and step == 0.5 and old.is_dir():
        return old  # access-2 predictions are reused unchanged
    return W / "testaccess3" / split / f"{model}__{chk[:-4]}__s{step}"


def predict(model: str, images: Path, out: Path, chk: str, step: float) -> None:
    tr, pl = MODELS3[model]
    ids, request = inference_request(images, fold_dir(model), model, tr, pl, chk, step)
    if reuse_cache(out, request, ids, probabilities=True):
        return
    subprocess.run(
        prediction_command(
            request,
            [
                "-i",
                str(images),
                "-o",
                str(out),
                "-d",
                "901",
                "-c",
                "3d_fullres",
                "-f",
                "0",
                "-tr",
                tr,
                "-p",
                pl,
                "-chk",
                chk,
                "-step_size",
                str(step),
                "--disable_tta",
                "--save_probabilities",
                "-npp",
                "4",
                "-nps",
                "4",
            ],
        ),
        check=True,
    )
    record_cache(out, request, ids, probabilities=True)


def build(dirs: dict[str, Path], weights: dict[str, float], out: Path, t_mm3: float) -> None:
    ids, request = ensemble_request(dirs, weights, t_mm3, Path(__file__))
    if reuse_cache(out, request, ids, probabilities=False):
        return
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(dirs.values()))
    wsum = sum(weights.values())
    for case in ids:
        ref = first / f"{case}.nii.gz"
        probabilities = []
        for member, directory in dirs.items():
            with np.load(directory / f"{case}.npz") as data:
                probabilities.append(weights[member] * data["probabilities"][1])
        prob = sum(probabilities) / wsum
        mask = prob >= 0.5
        img = sitk.ReadImage(str(ref))
        if t_mm3 > 0 and mask.any():
            vox = float(np.prod(img.GetSpacing()))
            lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
            sizes = ndi.sum(mask, lab, range(1, n + 1)) * vox
            mask = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= t_mm3])
        o = sitk.GetImageFromArray(mask.astype(np.uint8))
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out / ref.name))
    record_cache(out, request, ids, probabilities=False)


def evaluate(
    tag: str,
    members: list[str],
    chks: dict[str, str],
    step: float,
    weights: dict[str, float],
    t: float,
    images: Path,
    gt: Path,
    root: Path,
) -> dict:
    dirs = {}
    for m in members:
        dirs[m] = pred_dir(m, root.name, chks[m], step)
        predict(m, images, dirs[m], chks[m], step)
    build(dirs, weights, root / tag, t)
    res = score(root / tag, gt, root / f"{tag}.csv")
    print(tag, res, flush=True)
    return res


def select() -> None:
    require_new_files(CHOICE)
    val_in, val_gt = W / "val_in", W / "val_labels"
    if not available(FT):
        raise SystemExit("resenc_l_ft has no checkpoint_final.pth yet")
    trail = {}
    results = {}
    for name, members in CANDIDATES.items():
        if not all(available(m) for m in members):
            continue
        chks = {m: "checkpoint_final.pth" for m in members}
        results[name] = evaluate(
            f"cand_{name}", members, chks, 0.5, {m: 1.0 for m in members}, 0, val_in, val_gt, V3
        )
    if not results:
        raise RuntimeError("No candidate has all required checkpoints")
    best = max(results, key=lambda k: results[k]["dice"])
    members = CANDIDATES[best]
    chks = {m: "checkpoint_final.pth" for m in members}
    weights = {m: 1.0 for m in members}
    step = 0.5
    cur = results[best]
    trail["candidates"] = results
    # (a) checkpoint averaging for the fine-tuned member
    if FT in members:
        chk_a = dict(chks, **{FT: avg_checkpoint(FT)})
        r = evaluate(f"a_{best}_avg", members, chk_a, step, weights, 0, val_in, val_gt, V3)
        trail["a_checkpoint_avg"] = r
        if r["dice"] > cur["dice"]:
            chks, cur = chk_a, r
    # (b) member weights for a two-member ensemble containing the fine-tuned member
    if len(members) == 2 and FT in members:
        other = [m for m in members if m != FT][0]
        trail["b_weights"] = {}
        for wft in FT_WEIGHTS:
            if wft == 0.5:
                continue
            w = {FT: wft, other: 1 - wft}
            r = evaluate(f"b_{best}_w{wft}", members, chks, step, w, 0, val_in, val_gt, V3)
            trail["b_weights"][str(wft)] = r
            if r["dice"] > cur["dice"]:
                weights, cur = w, r
    # (c) sliding-window step 0.25
    r = evaluate(f"c_{best}_s0.25", members, chks, 0.25, weights, 0, val_in, val_gt, V3)
    trail["c_step_0.25"] = r
    if r["dice"] > cur["dice"]:
        step, cur = 0.25, r
    # T sweep, same rule as access 2
    base = cur
    pp = {0: base}
    for t in T_GRID[1:]:
        pp[t] = evaluate(f"pp_{best}_T{t}", members, chks, step, weights, t, val_in, val_gt, V3)
    ok = [
        t
        for t in T_GRID
        if base["dice"] - pp[t]["dice"] <= 0.002 and base["cldice"] - pp[t]["cldice"] <= 0.005
    ]
    t_star = min(ok, key=lambda t: pp[t]["hd95_mm"]) if ok else 0
    chosen = pp[t_star]
    go = chosen["dice"] >= ACCESS2_VAL_DICE + GO_MARGIN
    choice = {
        "configuration": best,
        "members": members,
        "checkpoints": chks,
        "weights": weights,
        "step": step,
        "T_mm3": t_star,
        "val_chosen": chosen,
        "go": go,
        "go_rule": f"val dice >= {ACCESS2_VAL_DICE:.4f} + {GO_MARGIN}",
        "trail": trail,
        "val_postprocessing": {str(k): v for k, v in pp.items()},
    }
    with CHOICE.open("x", encoding="utf-8") as handle:
        json.dump(choice, handle, indent=2, allow_nan=False)
    print(
        "CHOICE3",
        best,
        chks,
        weights,
        "step",
        step,
        "T",
        t_star,
        chosen,
        "GO" if go else "NO-GO",
        flush=True,
    )


def bootstrap(v: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    if not len(v) or not np.isfinite(v).all():
        raise ValueError("Bootstrap requires a complete, finite metric vector")
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def test() -> None:
    if not CHOICE.is_file():
        raise SystemExit("no choice file: run select first")
    choice = json.loads(CHOICE.read_text())
    if not choice["go"]:
        raise SystemExit("choice file says NO-GO: the test set is not accessed")
    root = W / "testaccess3" / "test"
    result_path = W / "test_access_3_result.json"
    require_new_files(result_path, root / "final_per_case.csv")
    if (root / "FINAL").exists():
        raise SystemExit("test-3 output already exists: the test set may be scored only once more")
    dirs = {}
    for m in choice["members"]:
        dirs[m] = pred_dir(m, "test", choice["checkpoints"][m], choice["step"])
        predict(m, W / "test_in", dirs[m], choice["checkpoints"][m], choice["step"])
    build(dirs, choice["weights"], root / "FINAL", choice["T_mm3"])
    res = score(root / "FINAL", W / "test_labels", root / "final_per_case.csv")
    rows = {r["case"]: r for r in read_evaluation_rows(root / "final_per_case.csv")}
    prev = {
        r["case"]: r
        for r in read_evaluation_rows(W / "testaccess2" / "test" / "final_per_case.csv", rows)
    }
    ci, paired = {}, {}
    for k in ("dice", "cldice", "hd95_mm", "gt_centerline_coverage_1mm"):
        v = np.array([float(rows[c][k]) for c in sorted(rows)])
        ci[k] = bootstrap(v)
        d = np.array([float(rows[c][k]) - float(prev[c][k]) for c in sorted(rows)])
        paired[k] = {"mean_diff_vs_access2": float(d.mean()), "ci95": bootstrap(d)}
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {"choice": choice, "test": res, "ci95": ci, "paired_vs_access2": paired},
            handle,
            indent=2,
            allow_nan=False,
        )
    print("TEST-3 RESULT", res, ci, paired, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "test"])
    {"select": select, "test": test}[ap.parse_args().stage]()
