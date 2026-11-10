"""Draw Fig. 4 (PCAT agreement) from the saved per-vessel results.

Output PDFs retain vector text and marks, with PNG previews at 400 dpi.

Run from any directory: python scripts/draw_fig4_pcat_agreement.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACTS = WORKSPACE / "results"
OUT = WORKSPACE / "figures"
INK, MUTED, GRID = "#17222D", "#516473", "#DCE3E9"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
VESSELS = ("RCA", "LAD", "LCx")
COLORS = dict(zip(VESSELS, (BLUE, ORANGE, GREEN)))
PROVENANCE: dict = {
    "revision_date": "2026-10-24",
    "scope": "Presentation-only redraw of saved scalar measurements; no inference.",
    "sources": {},
    "figures": {},
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def register(path: Path) -> Path:
    data = path.read_bytes()
    PROVENANCE["sources"][path.relative_to(WORKSPACE).as_posix()] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    return path


def rows(relative: str) -> list[dict]:
    with register(ARTIFACTS / relative).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def saved_json(relative: str) -> dict:
    return json.loads(register(ARTIFACTS / relative).read_text(encoding="utf-8"))


def number(row: dict, key: str) -> float | None:
    try:
        value = float(row[key])
        return value if np.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def agreement(x, y) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y) or not np.isfinite([x, y]).all():
        raise ValueError("Agreement plots require at least two finite pairs.")
    difference = y - x
    bias, sd = float(difference.mean()), float(difference.std(ddof=1))
    return {
        "n": len(x),
        "bias": bias,
        "difference_sd": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "mean_range": [float(((x + y) / 2).min()), float(((x + y) / 2).max())],
        "difference_range": [float(difference.min()), float(difference.max())],
    }


def bland_altman(
    ax, x, y, color: str, title: str, unit: str, xlim: tuple, ylim: tuple, icc: float | None = None
) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    result = agreement(x, y)
    ax.scatter(
        (x + y) / 2, y - x, s=6.5, color=color, alpha=0.55, edgecolors="none", rasterized=False
    )
    for value, style in (
        (result["bias"], "-"),
        (result["loa_lower"], "--"),
        (result["loa_upper"], "--"),
    ):
        ax.axhline(value, color=MUTED, lw=0.65, ls=style)
    ax.set(xlim=xlim, ylim=ylim)
    ax.grid(color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    stat = f"r = {result['pearson_r']:.2f}"
    if icc is not None:
        result["icc21_saved"] = icc
        stat += f"; ICC(2,1) = {icc:.2f}"
        ax.text(0, 1.20, title, transform=ax.transAxes, va="bottom", fontsize=8, fontweight="bold")
        ax.text(0, 1.105, stat, transform=ax.transAxes, va="bottom", fontsize=6.5)
    else:
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.text(0.02, 0.97, stat, transform=ax.transAxes, va="top", fontsize=6.5)
    annotation = (
        f"Bias {result['bias']:+.1f} {unit}; "
        f"LoA [{result['loa_lower']:+.1f}, {result['loa_upper']:+.1f}]"
    )
    # Keep numeric annotations outside the data area, including tail cases.
    ax.text(0, 1.015, annotation, transform=ax.transAxes, va="bottom", ha="left", fontsize=6.1)
    if not (
        xlim[0] <= result["mean_range"][0] <= result["mean_range"][1] <= xlim[1]
        and ylim[0] <= result["difference_range"][0] <= result["difference_range"][1] <= ylim[1]
    ):
        raise ValueError(f"Plot would hide saved observations: {title}")
    return result


def save(fig, stem: str, title: str, details: dict):
    name = stem
    pdf, png = OUT / f"{name}.pdf", OUT / f"{name}.png"
    fig.savefig(
        pdf,
        metadata={
            "Title": title,
            "Subject": "Redrawn from saved scalar measurements; no model inference",
            "Creator": "draw_revision_result_figures.py (Matplotlib)",
        },
    )
    fig.savefig(png, dpi=400)
    plt.close(fig)
    details["files"] = [pdf.name, png.name]
    PROVENANCE["figures"][name] = details
    print(pdf)
    print(png)


def draw_pcat():
    expert = {
        (int(r["case"]), r["vessel"]): r for r in rows("imagecas-x/pcat_expert_800_per_vessel.csv")
    }
    lumen = {
        (int(r["case"][4:]), r["vessel"]): r
        for r in rows("imagecas-x/pcat_lumen_160_per_vessel.csv")
    }
    whole = {
        (int(r["case"][4:]), r["vessel"]): r
        for r in rows("imagecas-x/pcat_auto_800_per_vessel.csv")
    }
    summaries = [
        saved_json("imagecas-x/pcat_lumen_vs_expert_160.json"),
        saved_json("imagecas-x/pcat_auto_vs_expert_800.json"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.115, top=0.84, wspace=0.15, hspace=0.55)
    panels = {}
    for i, auto in enumerate((lumen, whole)):
        for j, vessel in enumerate(VESSELS):
            keys = [
                k
                for k in auto
                if k[1] == vessel
                and k in expert
                and number(auto[k], "fai_hu") is not None
                and number(expert[k], "fai_hu") is not None
            ]
            x = [number(expert[k], "fai_hu") for k in keys]
            y = [number(auto[k], "fai_hu") for k in keys]
            saved = summaries[i][vessel]
            label = f"({chr(97 + 3 * i + j)}) {vessel} (n = {len(keys)})"
            result = bland_altman(
                axes[i, j], x, y, COLORS[vessel], label, "HU", (-112, -50), (-27, 27), saved["icc"]
            )
            if result["n"] != saved["n_both"]:
                raise ValueError(f"Saved PCAT pair count changed: {vessel}")
            for generated, stored in (("bias", "diff_mean"), ("difference_sd", "diff_sd")):
                if not np.isclose(result[generated], saved[stored], atol=1e-8):
                    raise ValueError(f"Saved PCAT statistic changed: {vessel}, {stored}")
            panels[f"{'lumen_test160' if i == 0 else 'whole_oos800'}_{vessel}"] = result
            if j == 0:
                axes[i, j].set_ylabel("Automatic − expert (HU)")
            if i == 1:
                axes[i, j].set_xlabel("Mean attenuation (HU)")
    fig.text(0.08, 0.972, "M1 lumen: test-160", fontweight="bold", fontsize=8.4, va="top")
    fig.text(
        0.08, 0.53, "Whole-vessel: out-of-sample 800", fontweight="bold", fontsize=8.4, va="top"
    )
    fig.text(
        0.08,
        0.015,
        "Solid: mean difference. Dashed: limits of agreement (mean ± 1.96 SD).",
        fontsize=6.7,
        color=MUTED,
    )
    save(
        fig,
        "fig4_pcat_agreement",
        "PCAT attenuation agreement by anatomy source",
        {
            "panels": panels,
            "note": "Same saved pairs and statistics as original; axes expanded to show all observations.",
        },
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_pcat()
    provenance = OUT / "fig4_pcat_agreement_provenance.json"
    provenance.write_text(
        json.dumps(PROVENANCE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(provenance)


if __name__ == "__main__":
    main()
