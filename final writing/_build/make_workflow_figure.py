"""The Chapter 1 workflow: the whole project on one page.

Drawn in the shape the sample reports use for their Figure 1.1 - the work as it was carried
out, top to bottom, with the two strands side by side, the outcome of each step written into
its own box, and a legend for what the colours mean. It reuses the measured-box canvas from
make_ch4_figures.py, so every box is sized from its own text and nothing can overlap.

Usage:
    python "final writing/_build/make_workflow_figure.py"
"""

import sys
import textwrap
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_ch4_figures import Canvas, OUT_NEW  # noqa: E402

TITLE = ("#1a237e", "#c5cae9")      # the thesis itself
DATA = ("#5d4037", "#efebe9")       # the corpus
BUILD = ("#1565c0", "#e3f2fd")      # a design step and what it selected
TEST = ("#6a1b9a", "#f3e5f5")       # an evaluation step
REJECT = ("#c62828", "#ffebee")     # a design measured and rejected
FINDING = ("#ef6c00", "#fff3e0")    # a finding that shaped the design
FINAL = ("#2e7d32", "#e8f5e9")      # the system that was adopted

SIZE = 6.5


def block(head, body, width=46):
    """A bold first line and body text, each wrapped to fit its column."""
    lines = textwrap.wrap(head, width)
    for para in body:
        lines += textwrap.wrap(para, width)
    return "\n".join(lines)


def column(c, x, rows, feed):
    """Draw one strand, arrowing from the box above into the box below."""
    prev = None
    for y, text, colour in rows:
        node = c.node(x, y, text, colour, size=SIZE, bold_first=True)
        c.arrow(prev["bottom"] if prev else feed["bottom"], node["top"],
                rad=0 if prev else (0.10 if x < 50 else -0.10))
        prev = node
    return prev


def legend(c, entries, y, x0=4, per_row=3):
    for i, (label, (edge, face)) in enumerate(entries):
        row, col = divmod(i, per_row)
        x = x0 + col * 32
        yy = y - row * 4.2
        c.ax.add_patch(FancyBboxPatch((x, yy), 2.6, 1.9,
                                      boxstyle="round,pad=0.1,rounding_size=0.3",
                                      linewidth=1.1, edgecolor=edge, facecolor=face, zorder=5))
        c.ax.text(x + 3.8, yy + 0.95, label, fontsize=6.2, va="center", color="#37474f", zorder=5)


def workflow():
    c = Canvas(6.6, 10.0)

    title = c.node(50, 97,
                   block("A Decoupled Empathetic Response System",
                         ["RoBERTa-base emotion detection and Phi-4-mini generation, "
                          "inside one 8 GB consumer GPU"], 78),
                   TITLE, size=SIZE + 0.8, bold_first=True)

    data = c.node(50, 88,
                  block("Data: EmpatheticDialogues",
                        ["23,144 conversations, 32 emotions, one label each; the first message of "
                         "every conversation is what the detector reads, and the same 300 test "
                         "conversations are answered by every system"], 76),
                  DATA, size=SIZE, bold_first=True)
    c.arrow(title["bottom"], data["top"])

    left, right = 24, 76
    c.caption(left, 81.8, "Building the system", size=7.4, colour=BUILD[0])
    c.caption(right, 81.8, "Testing against the alternatives", size=7.4, colour=TEST[0])

    build = [
        (73, block("Detector candidates (d1)",
                   ["GPT-2 Medium, BERT, ELECTRA, RoBERTa-base and RoBERTa-large, "
                    "trained identically",
                    "Selected: RoBERTa-base, 50.3% top-1, 84.3% top-5"]), BUILD),
        (59, block("Generator candidates (d2)",
                   ["Qwen2.5-1.5B, Gemma-3-4B and Phi-4-mini, with the detector and "
                    "the prompt held fixed",
                    "Selected: Phi-4-mini, loaded in 4-bit"]), BUILD),
        (45, block("Linking the two models",
                   ["argmax, confidence-gated, distribution, candidate-pick and two-stage",
                    "Selected: two-stage, with per-emotion guidance and cue words"]), BUILD),
        (31, block("Extended into a conversational agent",
                   ["The detector reads the last 4 user turns, the generator keeps 10 "
                    "exchanges, and the rules relax after the first turn"]), BUILD),
    ]
    test = [
        (73, block("Designs built without a detector",
                   ["Zero-shot prompting, a LoRA fine-tuned model, and single models of "
                    "1.5B, 3.8B and 7.6B parameters"]), TEST),
        (60.5, block("Retrieval augmentation: rejected",
                     ["Rule adherence fell from 63.7% to 43.7% and no other measure improved"]),
         REJECT),
        (48, block("Five signals, same 300 conversations",
                   ["Rule adherence; an empathy panel of two judges and a classifier; "
                    "emotional fidelity; repetition; and a human rating study"]), TEST),
        (36, block("Paired statistics",
                   ["McNemar, Wilcoxon, a 10,000-sample bootstrap and Holm correction "
                    "across the baselines"]), TEST),
        (25, block("Two findings that settled the design",
                   ["Three points more detector accuracy changed no measure of reply "
                    "quality, and the same generator alone scores 4.75 on empathy "
                    "against our 5.05"]), FINDING),
    ]
    last_build = column(c, left, build, data)
    last_test = column(c, right, test, data)

    final = c.node(50, 14,
                   block("Final system: two-stage RoBERTa-base + Phi-4-mini",
                         ["3.96B parameters, 3.40 GB peak memory, 4.17 s per reply",
                          "Rule adherence 60.0%  |  empathy panel 2nd of 8  |  emotional "
                          "fidelity 36.7%, the best of all twelve systems",
                          "Every reply carries its audit trail: the three candidate emotions, "
                          "the one chosen, and the words behind it"], 80),
                   FINAL, size=SIZE + 0.3, bold_first=True)
    c.arrow(last_build["bottom"], final["top"], rad=-0.12)
    c.arrow(last_test["bottom"], final["top"], rad=0.12)

    legend(c, [("Design step and what it selected", BUILD), ("Evaluation", TEST),
               ("Rejected after measurement", REJECT), ("Finding that shaped the design", FINDING),
               ("Adopted system", FINAL), ("Data", DATA)], y=5.0)
    c.save(OUT_NEW, "fig_workflow.png")


if __name__ == "__main__":
    workflow()
