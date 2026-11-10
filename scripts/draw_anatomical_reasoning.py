"""Render the manuscript's anatomical-reasoning mechanism as a vector schematic.

All geometry is schematic and generated locally; it is not a patient image or a
performance result. Method constants and definitions are transcribed from
the manuscript (tree, PCAT, EAT and paired expert-mask methods).

Run from any directory: python draw_anatomical_reasoning.py
Outputs: figures/fig2_anatomical_reasoning.{pdf,png} and
         figures/fig1_framework_overview.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE = "#27649A"
DARK = "#142D43"
INK = "#17222D"
MID = "#516473"
LIGHT = "#EDF4FA"
EDGE = "#B5C7D7"
PALE = "#D1E3F2"
GRAY = "#E7EAED"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "text.color": INK,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": WHITE,
    }
)


def txt(ax, x, y, s, size=7.3, weight="normal", ha="left", va="center", color=INK, **kw):
    return ax.text(
        x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, linespacing=1.22, **kw
    )


def line(ax, xy, color=BLUE, lw=1.4, **kw):
    return ax.plot(*zip(*xy), color=color, lw=lw, solid_capstyle="round", **kw)[0]


def arrow(ax, start, end, color=BLUE, lw=1.0, style="-|>", **kw):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=7,
        color=color,
        lw=lw,
        shrinkA=0,
        shrinkB=0,
        **kw,
    )
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, text=None, fill=LIGHT, edge=EDGE, size=7.2, weight="normal", radius=1.5):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=0.7,
    )
    ax.add_patch(p)
    if text:
        txt(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, ha="center")
    return p


def panel(ax, letter, title):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    box(ax, 0.7, 0.7, 98.6, 98.6, fill=WHITE, edge=EDGE, radius=1.8)
    box(ax, 0.8, 87, 98.4, 12.1, fill=LIGHT, edge=LIGHT, radius=1.6)
    txt(ax, 4, 93.2, letter, size=9.0, weight="bold", color=BLUE)
    txt(ax, 10, 93.2, title, size=8.4, weight="bold")


def draw_tree(ax):
    panel(ax, "a", "From predicted anatomy to a named tree")
    box(ax, 4, 66.5, 37, 15, "nnU-Net\nlumen mask $L$", size=7.7)
    box(ax, 4, 43, 37, 15, "Anatomical frame\nheart $H$, aorta $A_o$", size=7.3)
    arrow(ax, (41, 74), (49, 74))
    line(ax, [(44, 50.5), (46.5, 50.5), (46.5, 67), (49, 67)], lw=0.9)
    arrow(ax, (46.5, 67), (49, 67))

    # Two rooted schematic coronary trees, with node markers.
    edges = [
        [(55, 77), (51, 65), (54, 53), (59, 40)],
        [(54, 53), (48.5, 45)],
        [(59, 40), (56, 32)],
        [(59, 40), (65, 34)],
        [(72, 77), (76, 64)],
        [(76, 64), (71, 51), (69, 38)],
        [(71, 51), (62.5, 46)],
        [(76, 64), (84, 57), (92, 51)],
        [(84, 57), (85.5, 46)],
    ]
    for pts in edges:
        line(ax, pts, lw=2.0)
    for x, y in [(55, 77), (72, 77)]:
        ax.add_patch(Circle((x, y), 1.8, facecolor=WHITE, edgecolor=BLUE, lw=1.1))
    for x, y in [(54, 53), (59, 40), (76, 64), (71, 51), (84, 57)]:
        ax.add_patch(Circle((x, y), 1.0, facecolor=BLUE, edgecolor=BLUE, lw=0.5))
    txt(ax, 53, 83, "ostia", size=6.8, color=MID)
    txt(ax, 76, 76.5, "LM", size=7.4, weight="bold")
    txt(ax, 47, 34, "RCA", size=7.4, weight="bold")
    txt(ax, 68, 31, "LAD", size=7.4, weight="bold")
    txt(ax, 88.5, 42, "LCx", size=7.4, weight="bold")
    txt(ax, 6, 34, r"$r(\mathbf{p})=D_L(\mathbf{p})$", size=8.2)
    line(ax, [(4, 25.5), (96, 25.5)], color=EDGE, lw=0.6)
    txt(ax, 5, 20.3, "3D thinning → graph edges + local radii", size=7.3)
    txt(ax, 5, 12.6, "Ostial roots → prune spurs; retain fragments", size=7.2)
    txt(ax, 5, 5.6, "RAS geometry rules → names + confidence", size=7.2)


def draw_pcat(ax):
    panel(ax, "b", "Branch identity defines PCAT sampling")
    # Round cross-section drawn in display-isotropic coordinates. The axes'
    # physical aspect ratio is compensated to keep the schematic circular.
    fig = ax.figure
    bbox = ax.get_position()
    ratio = (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())
    center = np.array([22, 63.5])
    scale = np.array([ratio, 1.0])
    a = np.linspace(0, 2 * np.pi, 150)

    def disk(radius, fc, ec, lw=0.7, ls="-", hatch=None):
        xy = center + np.column_stack((np.cos(a), np.sin(a))) * radius * scale
        ax.add_patch(
            Polygon(
                xy, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, hatch=hatch
            )
        )

    disk(19.25, PALE, BLUE, lw=0.9, hatch="////")
    disk(8.25, WHITE, BLUE, lw=0.8)
    disk(5.5, GRAY, DARK, lw=0.7)
    ax.plot(*center, "+", color=DARK, ms=5, mew=0.9)
    txt(ax, 22, 38.2, "cross-section (schematic)", size=6.6, color=MID, ha="center")
    txt(ax, 48, 79, "RCA: 10–50 mm from ostium", size=7.1)
    txt(ax, 48, 69.8, "LAD / LCx: first 40 mm", size=7.1)
    txt(ax, 48, 62.5, "after the LM", size=7.1)
    txt(ax, 48, 51.8, "Local radius floor: 0.6 mm", size=7.1)
    txt(ax, 48, 42.5, "Shell width: $2r$", size=7.4, weight="bold")
    # Place radial boundary labels in the lower part of the cross section.
    arrow(ax, tuple(center + np.array([0, -5.5])), (22, 49), color=MID, lw=0.6, style="-")
    txt(ax, 21, 47.5, "$r$", size=7.8, ha="right")
    arrow(
        ax, tuple(center + np.array([ratio * 8.25, 0])), (39.5, 58.7), color=MID, lw=0.6, style="-"
    )
    txt(ax, 35.3, 55.1, "$r+0.5$", size=6.7, ha="center")
    txt(ax, 6, 31.2, r"$r+0.5<d(\mathbf{x},C_v)\leq 3r+0.5$  (mm)", size=8.3)
    txt(ax, 6, 21.7, "Exclude coronary, heart and aorta masks", size=7.2)
    box(ax, 4, 4.5, 92, 11, "Retain fat: −190 to −30 HU → mean HU + volume", size=7.1)


def draw_eat(ax):
    panel(ax, "c", "Anatomical constraints generate EAT targets")
    # Illustrative cardiac mass and software-sac outlines, not a medical image.
    xy = np.array(
        [[6, 66], [11, 78], [24, 80], [34, 72], [38, 57], [32, 43], [23, 38], [12, 43], [6, 55]]
    )
    hull = np.array(
        [
            [3.7, 65],
            [8.7, 81],
            [24, 83],
            [37, 74],
            [41, 56],
            [34, 40],
            [23, 35],
            [9.3, 40],
            [3.7, 53],
        ]
    )
    ax.add_patch(Polygon(hull, facecolor=PALE, edgecolor=BLUE, lw=0.9))
    ax.add_patch(Polygon(xy, facecolor=GRAY, edgecolor=MID, lw=0.7))
    # A small fat component connected to the surface in the software sac.
    fat = np.array(
        [[9.8, 41.0], [12, 42.7], [23, 37.7], [31.5, 42.6], [33.4, 40.6], [23, 35.7], [10.5, 40.6]]
    )
    ax.add_patch(Polygon(fat, facecolor=BLUE, edgecolor=BLUE, hatch="////", lw=0.5))
    txt(ax, 23, 62, "$M$", size=10, ha="center", color=MID)
    txt(ax, 6, 31, "Software sac $P$", size=7.3, color=BLUE)
    txt(ax, 46, 79, "1. Heart + nearby great vessels", size=7.1, weight="bold")
    txt(ax, 49, 71.5, "Exclude descending aorta", size=7.0)
    txt(ax, 46, 60.7, "2. Slice hull + 4 mm dilation", size=7.1, weight="bold")
    txt(ax, 49, 53, "≤20 mm from mass; exclude organs", size=6.65)
    txt(ax, 46, 42.4, "3. Fat components near heart", size=7.1, weight="bold")
    txt(ax, 49, 34.7, "−190 to −30 HU; within 3 mm of $\\partial H$", size=6.65)
    line(ax, [(4, 27), (96, 27)], color=EDGE, lw=0.6)
    txt(ax, 5, 21.2, "Axial limit: LM ostium + 35 mm, capped at heart top", size=6.95)
    txt(ax, 5, 13.9, "Expert LM for training targets; automatic LM externally*", size=6.85)
    box(ax, 4, 2.8, 29, 7.6, "Rule targets", size=7.0)
    arrow(ax, (33, 6.6), (38, 6.6))
    box(ax, 38, 2.8, 29, 7.6, "Train EAT network", size=6.8)
    arrow(ax, (67, 6.6), (72, 6.6))
    box(ax, 72, 2.8, 24, 7.6, "CCTA → EAT", size=6.9)


def draw_review(ax):
    panel(ax, "d", "Paired anatomy-source analysis and review")
    box(
        ax,
        4,
        73.8,
        92,
        9.7,
        "Fixed scan + expert centrelines + PCAT routine",
        size=7.4,
        weight="bold",
    )
    line(ax, [(50, 73.5), (50, 68), (24, 68)], lw=0.9)
    line(ax, [(50, 68), (76, 68)], lw=0.9)
    arrow(ax, (24, 68), (24, 63.8))
    arrow(ax, (76, 68), (76, 63.8))
    box(ax, 4, 51.4, 40, 12, "Expert lumen\nmask source", size=7.2)
    box(ax, 56, 51.4, 40, 12, "Expert whole-vessel\nmask source", size=7.2)
    arrow(ax, (24, 51), (24, 46.8))
    arrow(ax, (76, 51), (76, 46.8))
    box(ax, 4, 37, 40, 9.5, "Mask-derived shell", size=7.0, fill=WHITE)
    box(ax, 56, 37, 40, 9.5, "Mask-derived shell", size=7.0, fill=WHITE)
    arrow(ax, (44, 41.8), (56, 41.8), style="<->", lw=0.9)
    txt(
        ax,
        50,
        31,
        "Paired difference retains annotation-source differences",
        size=6.95,
        ha="center",
    )
    line(ax, [(4, 25), (96, 25)], color=EDGE, lw=0.6)
    txt(ax, 5, 19.5, "Review record", size=7.6, weight="bold", color=BLUE)
    txt(ax, 5, 12.5, "Masks + names • PCAT mean HU / volume • EAT volume", size=6.95)
    txt(ax, 5, 5.5, "Source / definition provenance • tree / segment / EAT flags", size=6.7)


def draw_overview():
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    fig.subplots_adjust(left=0.003, right=0.997, top=0.99, bottom=0.015)
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 72)
    ax.set_axis_off()
    box(ax, 0.5, 0.5, 219, 71, fill=WHITE, edge=EDGE, radius=2)

    # CCTA routing. Solid arrows denote inference; dashed arrows denote the
    # pseudo-label training route. All outputs remain first-pass review labels.
    box(ax, 3, 32, 23, 24, "Single\nCCTA", size=8.0, weight="bold")
    box(ax, 36, 49, 40, 17, "Coronary lumen\nResEnc-L nnU-Net", size=7.5)
    box(ax, 87, 49, 40, 17, "Graph reconstruction\n+ branch names", size=7.5)
    box(ax, 139, 49, 36, 17, "Vessel-specific\nPCAT sampling", size=7.5)
    box(ax, 36, 24, 40, 17, "Anatomical frame\nTotalSegmentator", size=7.5)
    box(ax, 87, 24, 40, 17, "Anatomical EAT rule\n+ LM / heart-top limit", size=7.15)
    box(ax, 139, 24, 36, 17, "EAT network\nCCTA-only inference", size=7.2)
    box(
        ax,
        186,
        24,
        30,
        42,
        "Named tree\nPCAT + EAT\nmeasurements\nQuality flags\nExpert review",
        size=7.45,
        weight="bold",
    )
    line(ax, [(26, 44), (30, 44), (30, 57.5), (35, 57.5)], lw=0.9)
    arrow(ax, (30, 57.5), (36, 57.5))
    line(ax, [(30, 44), (30, 32.5), (35, 32.5)], lw=0.9)
    arrow(ax, (30, 32.5), (36, 32.5))
    arrow(ax, (76, 57.5), (87, 57.5))
    arrow(ax, (127, 57.5), (139, 57.5))
    arrow(ax, (175, 57.5), (186, 57.5))
    arrow(ax, (76, 32.5), (87, 32.5))
    arrow(ax, (107, 49), (107, 41))
    txt(ax, 110, 45, "LM", size=6.6, color=MID)
    # The frame also constrains ostium detection and downstream PCAT exclusion.
    arrow(ax, (66, 41), (89, 49), color=MID, lw=0.7)
    txt(ax, 62.5, 45.3, "context", size=6.3, color=MID, ha="center")
    arrow(ax, (127, 32.5), (139, 32.5), lw=0.9, linestyle=(0, (2, 2)))
    txt(ax, 133, 44, "train", size=6.3, color=MID, ha="center")
    arrow(ax, (175, 32.5), (186, 32.5))
    # Rule EAT is also an output, independent of the network alternative.
    line(ax, [(117, 24), (117, 20), (181, 20), (181, 41)], lw=0.8)
    arrow(ax, (181, 41), (186, 41))
    txt(ax, 149, 20.5, "rule EAT", size=6.2, color=MID, va="bottom", ha="center")

    # Companion scan is a separate modality-specific route.
    box(ax, 3, 3.5, 38, 10.5, "Non-contrast CT", size=7.35)
    arrow(ax, (41, 8.75), (50, 8.75))
    box(ax, 50, 3.5, 89, 10.5, "Companion calcium segmentation + Agatston score", size=7.3)
    txt(ax, 147, 8.7, "– –  Training     →  Inference", size=6.85, color=MID)
    pdf = OUT / "fig1_framework_overview.pdf"
    png = OUT / "fig1_framework_overview.png"
    fig.savefig(
        pdf,
        metadata={
            "Title": "Automated cardiac phenotyping framework overview",
            "Subject": "Local code-native schematic without dataset names or performance metrics",
            "Creator": "draw_anatomical_reasoning.py (Matplotlib)",
        },
    )
    fig.savefig(png, dpi=400)
    plt.close(fig)
    print(pdf)
    print(png)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.55))
    fig.subplots_adjust(
        left=0.004, right=0.996, top=0.996, bottom=0.038, hspace=0.042, wspace=0.024
    )
    draw_tree(axs[0, 0])
    draw_pcat(axs[0, 1])
    draw_eat(axs[1, 0])
    draw_review(axs[1, 1])
    fig.text(
        0.012,
        0.014,
        "* If no LM ostium is available, the EAT rule uses the heart top. Schematics are not patient images.",
        fontsize=6.6,
        color=MID,
        ha="left",
        va="center",
    )
    pdf = OUT / "fig2_anatomical_reasoning.pdf"
    png = OUT / "fig2_anatomical_reasoning.png"
    fig.savefig(
        pdf,
        metadata={
            "Title": "Anatomical reasoning and paired anatomy-source evaluation",
            "Subject": "Code-native method schematic; no patient images or numeric performance results",
            "Creator": "draw_anatomical_reasoning.py (Matplotlib)",
        },
    )
    fig.savefig(png, dpi=400)
    plt.close(fig)
    print(pdf)
    print(png)
    draw_overview()


if __name__ == "__main__":
    main()
