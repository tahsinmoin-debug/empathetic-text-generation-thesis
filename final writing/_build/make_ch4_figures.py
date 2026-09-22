"""Figures for Chapter 4 of the final draft.

Writes five figures. Two replace the writing-kit versions (same file name, placed in
figures_override/ so the build picks them up); three are new:

    fig_design_process.png      the evidence-gated design loop        (4.1)
    fig_final_architecture.png  the complete system pipeline          (4.1)
    fig_data_pipeline.png       raw corpus to the four datasets used  (4.3)
    fig_attention_cues.png      how cue words are taken from d1       (4.2)
    fig_prompt_construction.png how the stage-2 prompt is assembled   (4.2)

Usage:
    python "final writing/_build/make_ch4_figures.py"
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_OVERRIDE = Path(__file__).resolve().parents[1] / "figures_override"
OUT_NEW = Path(__file__).resolve().parents[1] / "figures"

BLUE, BLUE_BG = "#1f6fb2", "#dbeafe"          # detector / data
ORANGE, ORANGE_BG = "#d2601a", "#fdead9"      # generator
GREY, GREY_BG = "#8a8a84", "#f1f1ee"          # input, output, glue
GREEN, GREEN_BG = "#2f7d4f", "#dff2e6"        # datasets produced


def box(ax, x, y, w, h, text, edge=GREY, face=GREY_BG, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.5, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            linespacing=1.35, zorder=3, fontweight=weight)


def arrow(ax, p1, p2, label=None, lx=0.0, ly=0.0, color="#4a4a4a", style="-|>", rad=0.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, shrinkA=1, shrinkB=1,
                                connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if label:
        ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=8.5,
                color="#3a3a3a", ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", (folder / name).name)


# ---------------------------------------------------------------- 1. design process
def design_process():
    fig, ax = canvas(9.2, 2.6, (0, 100), (0, 30))
    steps = ["Problem and\nconstraints", "Data\npreparation", "Alternative\ndesigns",
             "Simulation on the\nsame 300\nconversations", "Evaluation with\nfive signals",
             "Paired statistical\ntesting"]
    w, gap, y = 13.0, 2.2, 13.5
    for i, text in enumerate(steps):
        x = 1.0 + i * (w + gap)
        box(ax, x, y, w, 10.5, text, BLUE, BLUE_BG)
        if i:
            arrow(ax, (x - gap, y + 5.25), (x, y + 5.25))
    x_last = 1.0 + len(steps) * (w + gap)
    box(ax, x_last, y, w, 10.5, "Adopt only what\nthe evidence\nsupports", ORANGE, ORANGE_BG)
    arrow(ax, (x_last - gap, y + 5.25), (x_last, y + 5.25))
    arrow(ax, (x_last + w / 2, y), (1.0 + 2 * (w + gap) + w / 2, y), color=ORANGE, rad=0.28)
    ax.text(52, 2.0, "a rejected design returns to the design stage; nothing is kept on intuition",
            fontsize=9, color="#3a3a3a", ha="center")
    save(fig, OUT_OVERRIDE, "fig_design_process.png")


# ---------------------------------------------------------------- 2. system pipeline
def system_pipeline():
    fig, ax = canvas(9.4, 4.6, (0, 100), (0, 52))
    box(ax, 1, 20, 15, 12, "User message\n+ last 4\nuser turns", GREY, GREY_BG)

    box(ax, 21, 33, 20, 13, "d1: emotion detector\nRoBERTa-base (125M)\n32-way softmax",
        BLUE, BLUE_BG)
    box(ax, 21, 6, 20, 13, "Attention cues\nlast-layer [CLS] attention\ntop 40% of words",
        BLUE, BLUE_BG)

    box(ax, 46, 33, 20, 13, "Stage 1: emotion choice\nPhi-4-mini picks one\nof d1's top 3 (greedy)",
        ORANGE, ORANGE_BG)
    box(ax, 46, 6, 20, 13, "Prompt builder\nper-emotion guidance\n+ 3 response rules",
        GREY, GREY_BG)

    box(ax, 71, 20, 24, 14, "Stage 2: reply generation\nPhi-4-mini-instruct (3.8B, 4-bit)\n"
                            "T = 0.7, top-p 0.9\n+ last 10 exchanges", ORANGE, ORANGE_BG)
    box(ax, 71, 1, 24, 11, "Reply shown to the user\n+ audit trail: candidates,\n"
                           "chosen emotion, cue words", GREY, GREY_BG)

    arrow(ax, (16, 28), (21, 39), )
    arrow(ax, (16, 24), (21, 13))
    arrow(ax, (41, 39.5), (46, 39.5), "top 3")
    arrow(ax, (41, 12.5), (46, 12.5), "cue words")
    arrow(ax, (56, 33), (56, 19), "chosen emotion", lx=11.5)
    arrow(ax, (66, 12.5), (71, 24))
    arrow(ax, (83, 20), (83, 12))
    save(fig, OUT_OVERRIDE, "fig_final_architecture.png")


# ---------------------------------------------------------------- 3. data pipeline
def data_pipeline():
    fig, ax = canvas(9.6, 4.2, (0, 132), (0, 50))
    box(ax, 1, 20, 22, 12, "EmpatheticDialogues\ntrain / valid / test\n88,687 utterances",
        GREY, GREY_BG, size=8.8)
    box(ax, 27, 20, 22, 12, "Cleaning\nrestore commas,\ndrop malformed rows",
        BLUE, BLUE_BG, size=8.8)
    box(ax, 53, 20, 22, 12, "Reduction\nkeep the first message\nof each conversation",
        BLUE, BLUE_BG, size=8.8)

    arrow(ax, (23, 26), (27, 26))
    arrow(ax, (49, 26), (53, 26))

    box(ax, 83, 38, 48, 10, "Detector data: 17,839 / 2,763 / 2,542\nfirst messages, one of 32 emotion labels",
        GREEN, GREEN_BG, size=8.6)
    box(ax, 83, 26, 48, 10, "Generator training pairs: 17,749\nmessage with the human reply (LoRA baseline)",
        GREEN, GREEN_BG, size=8.6)
    box(ax, 83, 14, 48, 10, "Evaluation set: 300 test conversations\ndrawn with seed 42, identical for every system",
        GREEN, GREEN_BG, size=8.6)
    box(ax, 83, 2, 48, 10, "Judge sample: 64 of those conversations\nshortest and longest message per emotion",
        GREEN, GREEN_BG, size=8.6)

    for y in (43, 31, 19, 7):
        arrow(ax, (75, 26), (83, y), rad=-0.10 if y > 26 else 0.10)
    save(fig, OUT_NEW, "fig_data_pipeline.png")


# ---------------------------------------------------------------- 4. attention cues
def attention_cues():
    fig, ax = canvas(9.4, 3.6, (0, 100), (0, 40))
    box(ax, 1, 24, 20, 10, "User message\n(word pieces)", GREY, GREY_BG)
    box(ax, 26, 24, 22, 10, "RoBERTa-base\n12 layers, eager\nattention enabled", BLUE, BLUE_BG)
    box(ax, 53, 24, 22, 10, "Last layer:\n[CLS] attention row,\naveraged over 12 heads", BLUE, BLUE_BG)
    box(ax, 80, 24, 19, 10, "Word scores\n(pieces merged\ninto whole words)", BLUE, BLUE_BG)
    arrow(ax, (21, 29), (26, 29))
    arrow(ax, (48, 29), (53, 29))
    arrow(ax, (75, 29), (80, 29))

    tokens = [("my", 0.18), ("dog", 0.81), ("disappeared", 0.93), ("while", 0.21),
              ("i", 0.12), ("was", 0.10), ("away", 0.62), ("at", 0.14), ("college", 0.74)]
    x = 6.0
    for word, score in tokens:
        w = 2.0 + 1.15 * len(word)
        keep = score >= 0.6
        box(ax, x, 8, w, 6, word, BLUE if keep else "#c9c9c4",
            BLUE_BG if keep else "#f6f6f4", size=9.5,
            weight="bold" if keep else "normal")
        ax.text(x + w / 2, 5.6, f"{score:.2f}", ha="center", va="center", fontsize=7.5,
                color="#1f6fb2" if keep else "#9a9a95")
        x += w + 1.4
    arrow(ax, (50, 24), (50, 15.5))
    ax.text(50, 19.8, "keep the highest-scoring 40% of words (3 to 14 spans)",
            fontsize=8.8, color="#3a3a3a", ha="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax.text(50, 1.6, 'cue words passed to the prompt builder:  "dog", "disappeared", "away", "college"',
            fontsize=9, color="#1f6fb2", ha="center")
    save(fig, OUT_NEW, "fig_attention_cues.png")


# ---------------------------------------------------------------- 5. prompt construction
def prompt_construction():
    fig, ax = canvas(9.4, 4.8, (0, 100), (0, 56))
    inputs = [
        ("Chosen emotion\n(stage 1)\n\nlonely", BLUE, BLUE_BG),
        ("Cue words\n(d1 attention)\n\n\"moved away\", \"parents\"", BLUE, BLUE_BG),
        ("Per-emotion guidance\n(1 of 32)\n\nfocus on the isolation\nitself; avoid a\nsurface-level fix", GREY, GREY_BG),
        ("Three response rules\n\nacknowledge the feeling,\nask exactly one question,\nstay within 100 words", GREY, GREY_BG),
    ]
    w, gap = 22.0, 3.0
    for i, (text, edge, face) in enumerate(inputs):
        x = 1.0 + i * (w + gap)
        box(ax, x, 40, w, 15, text, edge, face, size=8.6)
        arrow(ax, (x + w / 2, 40), (50, 33))

    box(ax, 12, 18, 76, 14,
        "System prompt sent to Phi-4-mini\n\n"
        "\"The person is feeling LONELY. They mentioned: moved away, parents.\n"
        "Acknowledge the isolation itself, gently. Do not offer a surface-level fix.\n"
        "Reply in under 100 words and end with exactly one short question.\"",
        ORANGE, ORANGE_BG, size=8.8)

    box(ax, 20, 0, 60, 13, "Stage 2 reply\n\n\"That sounds really isolating - moving away from your parents\n"
                           "changes everything at once. How are you finding the quiet?\"",
        GREY, GREY_BG, size=8.6)
    arrow(ax, (50, 18), (50, 13))
    save(fig, OUT_NEW, "fig_prompt_construction.png")


if __name__ == "__main__":
    design_process()
    system_pipeline()
    data_pipeline()
    attention_cues()
    prompt_construction()
