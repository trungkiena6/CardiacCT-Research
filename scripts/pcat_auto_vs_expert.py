"""PCAT/FAI agreement: automatic pipeline (out-of-sample CoroSeg + CoroTree + rule names) vs expert anatomy (ImageCAS-X centrelines +
lumen) on the same 800 scans, same PCAT definition. Per vessel: n measured by both, FAI means, paired difference (mean, SD, 95 % limits of
agreement), ICC(2,1), share of |diff| > 5 HU, PCAT-volume ICC; plus availability of the automatic measurement and its failure reasons.

  python scripts/pcat_auto_vs_expert.py --expert results/imagecas-x/pcat_expert_800_per_vessel.csv --auto D:/imagecas-work/imagecas-x/oos_pcat/pcat_per_vessel.csv --out results/imagecas-x/pcat_auto_vs_expert_800.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coronary_tree.pcat.compare import icc21  # noqa: E402


def load(p, rename):
    out = {}
    for r in csv.DictReader(Path(p).open(encoding="utf-8")):
        c = f"Case{int(r['case']):04d}" if rename else r["case"]
        v = r.get("fai_hu")
        out[(c, r["vessel"])] = dict(
            fai=float(v) if v not in ("", None, "None") else None,
            vol=float(r["pcat_volume_ml"])
            if r.get("pcat_volume_ml") not in ("", None, "None")
            else None,
            status=r.get("status"),
            reason=r.get("reason", ""),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expert", required=True)
    ap.add_argument("--auto", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    E, A = load(a.expert, True), load(a.auto, False)
    cases = sorted({k[0] for k in E})
    L = [
        f"# PCAT/FAI: automatic pipeline vs expert anatomy on ImageCAS-X ({len(cases)} scans)",
        "",
        "Same scans, same PCAT definition (shell r_lumen + 0.5 mm to + 2 r_lumen, fat -190…-30 HU, RCA 10–50 mm, LAD/LCx 0–40 mm from the LM end).",
        "Expert = ImageCAS-X centrelines + lumen masks; automatic = out-of-sample CoroSeg (official 4-split ResEnc-L models) + CoroTree + rule-based names.",
        "",
        "| vessel | n both | FAI expert mean | FAI auto mean | mean diff (auto − expert) | SD diff | 95 % LoA | ICC(2,1) | abs diff > 5 HU | PCAT volume ICC | auto available |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    summary = {}
    for v in ("RCA", "LAD", "LCx"):
        keys = [
            k
            for k in E
            if k[1] == v and k in A and E[k]["fai"] is not None and A[k]["fai"] is not None
        ]
        n_auto = sum(1 for k in A if k[1] == v and A[k]["fai"] is not None)
        n_exp = sum(1 for k in E if k[1] == v and E[k]["fai"] is not None)
        if len(keys) < 3:
            L.append(
                f"| {v} | {len(keys)} | – | – | – | – | – | – | – | – | {n_auto}/{len(cases)} |"
            )
            continue
        x = np.array([E[k]["fai"] for k in keys])
        y = np.array([A[k]["fai"] for k in keys])
        d = y - x
        vx = np.array([E[k]["vol"] for k in keys])
        vy = np.array([A[k]["vol"] for k in keys])
        summary[v] = dict(
            n_both=len(keys),
            fai_expert=float(x.mean()),
            fai_auto=float(y.mean()),
            diff_mean=float(d.mean()),
            diff_sd=float(d.std(ddof=1)),
            icc=icc21(x, y),
            gt5=float((np.abs(d) > 5).mean()),
            vol_icc=icc21(vx, vy),
            n_auto=n_auto,
            n_expert=n_exp,
        )
        s = summary[v]
        L.append(
            f"| {v} | {len(keys)} | {x.mean():.1f} | {y.mean():.1f} | {d.mean():+.1f} | {d.std(ddof=1):.1f} | [{d.mean() - 1.96 * d.std(ddof=1):+.1f}, {d.mean() + 1.96 * d.std(ddof=1):+.1f}] | "
            f"{s['icc']:.2f} | {100 * s['gt5']:.0f} % | {s['vol_icc']:.2f} | {n_auto}/{len(cases)} ({n_exp} expert) |"
        )
    L += ["", "## Automatic measurement failures", ""]
    for v in ("RCA", "LAD", "LCx"):
        c = Counter(
            (A[k]["status"], A[k]["reason"]) for k in A if k[1] == v and A[k]["fai"] is None
        )
        L.append(
            f"- {v}: "
            + (
                ", ".join(f"{st}/{rs or '-'}: {n}" for (st, rs), n in c.most_common())
                if c
                else "none"
            )
        )
    L += [
        "",
        "Reading guide: ICC(2,1) is absolute agreement of single measurements; the limits of agreement are in HU. A |diff| > 5 HU share is the fraction of vessels",
        "where the automatic FAI would differ from the expert-anatomy FAI by more than 5 HU (about two thirds of the between-patient SD of 7 HU).",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))
    print("PCAT-COMPARE-DONE", flush=True)


if __name__ == "__main__":
    main()
