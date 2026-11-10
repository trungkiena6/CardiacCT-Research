"""Repeatability of the PCAT prototype between two segmentation models (same cases, same pipeline):
per vessel, paired FAI difference (mean, SD, 95 % limits of agreement), ICC(2,1) and the share of cases where both
runs produced a value. Also summarises each run alone. Usage: compare.py --a <run A csv> --b <run B csv> --out <md>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def icc21(x, y):
    """Two-way random, single measures, absolute agreement."""
    M = np.stack([x, y], 1)
    n, k = M.shape
    ms_r = k * np.var(M.mean(1), ddof=1)
    ms_c = n * np.var(M.mean(0), ddof=1)
    resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def load(p):
    rows = list(csv.DictReader(Path(p).open(encoding="utf-8")))
    out = {}
    for r in rows:
        v = r.get("fai_hu")
        out[(r["case"], r["vessel"])] = dict(
            fai=float(v) if v not in ("", None, "None") else None,
            vol=float(r["pcat_volume_ml"])
            if r.get("pcat_volume_ml") not in ("", None, "None")
            else None,
            status=r.get("status"),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="CoroSeg-S1-base-1.0")
    ap.add_argument("--label-b", default="CoroSeg-S1-cldice-1.1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    L = [
        f"# PCAT prototype: repeatability between {a.label_a} and {a.label_b}",
        "",
        "Same 50 val cases, same CoroTree-0.2 + rule-based names + PCAT definition; only the segmentation model differs.",
        "",
        "| vessel | n both | FAI A mean | FAI B mean | mean diff (B−A) | SD diff | 95 % LoA | ICC(2,1) | |diff| > 5 HU | volume ICC |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for v in ("RCA", "LAD", "LCx"):
        keys = [
            k
            for k in A
            if k[1] == v and k in B and A[k]["fai"] is not None and B[k]["fai"] is not None
        ]
        if len(keys) < 3:
            L.append(f"| {v} | {len(keys)} | – | – | – | – | – | – | – | – |")
            continue
        x = np.array([A[k]["fai"] for k in keys])
        y = np.array([B[k]["fai"] for k in keys])
        d = y - x
        vx = np.array([A[k]["vol"] for k in keys])
        vy = np.array([B[k]["vol"] for k in keys])
        L.append(
            f"| {v} | {len(keys)} | {x.mean():.1f} | {y.mean():.1f} | {d.mean():+.1f} | {d.std(ddof=1):.1f} | [{d.mean() - 1.96 * d.std(ddof=1):+.1f}, {d.mean() + 1.96 * d.std(ddof=1):+.1f}] | {icc21(x, y):.2f} | {int((np.abs(d) > 5).sum())} | {icc21(vx, vy):.2f} |"
        )
    for name, R in ((a.label_a, A), (a.label_b, B)):
        L += [
            "",
            f"## {name} alone",
            "",
            "| vessel | n ok | FAI mean (SD) | FAI range | PCAT volume mean mL | failed / needs_review |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for v in ("RCA", "LAD", "LCx"):
            ok = [r for k, r in R.items() if k[1] == v and r["fai"] is not None]
            allv = [r for k, r in R.items() if k[1] == v]
            if not ok:
                L.append(f"| {v} | 0 | – | – | – | {len(allv)} |")
                continue
            f = np.array([r["fai"] for r in ok])
            vol = np.array([r["vol"] for r in ok])
            L.append(
                f"| {v} | {len(ok)} | {f.mean():.1f} ({f.std():.1f}) | {f.min():.0f} … {f.max():.0f} | {vol.mean():.2f} | {sum(1 for r in allv if r['status'] != 'ok')} |"
            )
    L += [
        "",
        "Reference: CRISP-CT reported FAI around −70 to −80 HU in healthy vessels with an RCA cut-off of −70.1 HU at 120 kVp; ImageCAS NIfTI carries no kVp, "
        "so absolute values are not comparable to published thresholds. Agreement between two models is a lower bound on measurement noise, not accuracy; "
        "validation against reader measurements (G2c.5) is still required.",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
