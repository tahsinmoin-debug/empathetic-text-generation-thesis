"""Column-width architecture figure for the ICCIT paper (no title: IEEE captions go below the figure).

Usage:
    python "ICCIT format writing/_build/make_figure.py"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_architecture.png"
BLUE, BLUE_BG = "#2a78d6", "#e3eefb"
ORANGE, ORANGE_BG = "#eb6834", "#fce6dd"
GREY, GREY_BG = "#9a988f", "#f3f2ef"


def box(ax, x, y, w, h, text, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label=None, lx=0.0, ly=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.5, color="#444444",
                ha="center", va="center")


def main():
    fig, ax = plt.subplots(figsize=(3.5, 4.3), dpi=300)
    ax.set_xlim(-0.25, 10.25)
    ax.set_ylim(0, 12.6)
    ax.axis("off")

    box(ax, 2.6, 11.0, 4.8, 1.2, "User message", GREY, GREY_BG)
    box(ax, 0.0, 8.4, 4.3, 1.8, "d1: RoBERTa-base\n(125M) emotion\ndetector, 32 classes", BLUE, BLUE_BG)
    box(ax, 5.7, 8.4, 4.3, 1.8, "Attention cues\n[CLS] attention,\ntop 40% of words", BLUE, BLUE_BG)
    box(ax, 0.0, 5.6, 4.0, 1.9, "Stage 1: Phi-4-mini\nchooses one of\nd1's top-3 emotions", ORANGE, ORANGE_BG)
    box(ax, 6.0, 5.6, 4.0, 1.9, "Prompt builder\nemotion-specific\nguidance + 3 rules", GREY, GREY_BG)
    box(ax, 1.0, 2.6, 8.0, 1.9, "Stage 2: Phi-4-mini (3.8B, 4-bit)\nwrites the reply\nT = 0.7, top-p = 0.9", ORANGE, ORANGE_BG)
    box(ax, 2.6, 0.2, 4.8, 1.3, "Reply + visible\nemotion decision", GREY, GREY_BG)

    arrow(ax, 4.0, 11.0, 2.15, 10.2)
    arrow(ax, 6.0, 11.0, 7.85, 10.2)
    arrow(ax, 2.15, 8.4, 2.15, 7.5, "top-3", lx=0.75)
    arrow(ax, 7.85, 8.4, 7.85, 7.5, "cue words", lx=1.05)
    arrow(ax, 4.05, 6.25, 5.95, 6.25)
    ax.text(5.0, 6.62, "chosen", fontsize=7.0, color="#444444", ha="center", va="bottom")
    arrow(ax, 8.0, 5.6, 6.6, 4.5)
    arrow(ax, 5.0, 2.6, 5.0, 1.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
