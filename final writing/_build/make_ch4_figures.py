"""Figures for Chapter 4 of the final draft.

Nothing in these figures can overlap, because no size is guessed: node() draws its text
first, measures the drawn text with the renderer, and only then draws a box around it.
Arrows attach to box edges through the returned anchors, and a feedback line that would
otherwise cross the flow is routed down a clear lane with route().

Writes five figures. Two replace the writing-kit versions (same file name, written to
figures_override/ so the build picks them up); three are new:

    fig_design_process.png      the evidence-gated design loop        (4.1)
    fig_final_architecture.png  the complete system pipeline          (4.1)
    fig_attention_cues.png      how cue words are taken from d1       (4.2)
    fig_prompt_construction.png how the stage-2 prompt is assembled   (4.2)
    fig_data_pipeline.png       raw corpus to the four datasets used  (4.3)

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

# one colour per role, so a reader can follow a part through every figure
INPUT = ("#455a64", "#eceff1")      # slate  - what the user supplies
DETECT = ("#1565c0", "#e3f2fd")     # blue   - the detector d1
CUES = ("#6a1b9a", "#f3e5f5")       # purple - attention cues
GEN = ("#e65100", "#fff3e0")        # orange - the generator d2
PROMPT = ("#00695c", "#e0f2f1")     # teal   - prompt building
OUTPUT = ("#2e7d32", "#e8f5e9")     # green  - what leaves the system
STEP = ("#3949ab", "#e8eaf6")       # indigo - a process step
CORPUS = ("#5d4037", "#efebe9")     # brown  - raw data
DECIDE = ("#c62828", "#ffebee")     # red    - the decision point
MUTED = ("#b0bec5", "#f5f5f5")      # grey   - something set aside


class Canvas:
    """A drawing surface whose boxes are measured, not guessed."""

    def __init__(self, width_in, height_in, xlim=100.0, ylim=100.0):
        self.fig, self.ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.axis("off")
        self.ux = xlim / (width_in * 72.0)      # data units per point, horizontally
        self.uy = ylim / (height_in * 72.0)     # data units per point, vertically
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()

    def _text(self, cx, cy, s, size, colour=None, weight="normal"):
        return self.ax.text(cx, cy, s, ha="center", va="center", fontsize=size,
                            linespacing=1.5, zorder=3, color=colour or "#212121",
                            fontweight=weight)

    def _extent(self, txt):
        """Width and height of a drawn text, in data units."""
        bb = txt.get_window_extent(renderer=self.renderer)
        inv = self.ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return abs(x1 - x0), abs(y1 - y0)

    def node(self, cx, cy, text, colour=INPUT, size=8.5, pad_x=10.0, pad_y=8.0,
             bold_first=False, min_w=0.0):
        """A box sized from the text it holds, so the text can never spill out."""
        edge, face = colour
        lines = text.split("\n")
        if bold_first and len(lines) > 1:
            head = self._text(cx, cy, lines[0], size, edge, "bold")
            body = self._text(cx, cy, "\n".join(lines[1:]), size)
            hw, hh = self._extent(head)
            bw, bh = self._extent(body)
            gap = size * 0.5 * self.uy
            tw, th = max(hw, bw), hh + gap + bh
            head.set_position((cx, cy + th / 2 - hh / 2))
            body.set_position((cx, cy - th / 2 + bh / 2))
        else:
            body = self._text(cx, cy, text, size, edge if bold_first else None,
                              "bold" if bold_first else "normal")
            tw, th = self._extent(body)
        w = max(min_w, tw + 2 * pad_x * self.ux)
        h = th + 2 * pad_y * self.uy
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                         boxstyle="round,pad=0.01,rounding_size=0.02",
                                         linewidth=1.5, edgecolor=edge, facecolor=face, zorder=2))
        return dict(cx=cx, cy=cy, w=w, h=h,
                    left=(cx - w / 2, cy), right=(cx + w / 2, cy),
                    top=(cx, cy + h / 2), bottom=(cx, cy - h / 2))

    def width_of(self, text, size=8.5, pad_x=10.0, bold_first=False):
        """Width the box for this text will need, measured off-canvas."""
        lines = text.split("\n")
        parts = ([(lines[0], "bold"), ("\n".join(lines[1:]), "normal")]
                 if bold_first and len(lines) > 1 else [(text, "normal")])
        widest = 0.0
        for s, weight in parts:
            probe = self._text(-200, -200, s, size, weight=weight)
            widest = max(widest, self._extent(probe)[0])
            probe.remove()
        return widest + 2 * pad_x * self.ux

    def pack_row(self, texts, size, pad_x=10.0, bold_first=False,
                 left=2.0, right=98.0, min_gap=2.0):
        """Centres for a row of boxes, spaced so that no two can touch."""
        widths = [self.width_of(s, size, pad_x, bold_first) for s in texts]
        gap = (right - left - sum(widths)) / max(1, len(widths) - 1)
        if gap < min_gap:
            print(f"  warning: row too wide by "
                  f"{sum(widths) + min_gap * (len(widths) - 1) - (right - left):.1f} units")
            gap = min_gap
        centres, x = [], left
        for w in widths:
            centres.append(x + w / 2)
            x += w + gap
        return centres, widths

    def arrow(self, p1, p2, label=None, lx=0.0, ly=0.0, colour="#37474f", rad=0.0, size=7.6):
        self.ax.annotate("", xy=p2, xytext=p1,
                         arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3,
                                         shrinkA=2, shrinkB=3,
                                         connectionstyle=f"arc3,rad={rad}"), zorder=1)
        if label:
            self.ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label,
                         fontsize=size, color=colour, ha="center", va="center", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    def route(self, points, colour="#37474f"):
        """An arrow that follows a corner path, to keep a line clear of every box."""
        self.ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                     color=colour, lw=1.3, solid_capstyle="round", zorder=1)
        self.arrow(points[-2], points[-1], colour=colour)

    def caption(self, x, y, text, size=7.8, colour="#546e7a", ha="center"):
        self.ax.text(x, y, text, fontsize=size, color=colour, ha=ha, va="center",
                     linespacing=1.5, zorder=3)

    def save(self, folder, name):
        folder.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(folder / name, bbox_inches="tight", pad_inches=0.08)
        plt.close(self.fig)
        print("wrote", name)


# ---------------------------------------------------------------- 1. design process
def design_process():
    c = Canvas(5.6, 6.6)
    steps = [("Problem and constraints\n32 emotions, 8 GB, a visible decision", STEP),
             ("Data preparation\nfirst message of each conversation", STEP),
             ("Alternative designs\ndetectors, generators, links, baselines", STEP),
             ("Simulation\nthe same 300 conversations, same decoding", STEP),
             ("Evaluation with five signals\nrules, WASSA, validation, judges, people", STEP),
             ("Paired statistical testing\nMcNemar, Wilcoxon, bootstrap, Holm", STEP),
             ("Does the evidence support the change?", DECIDE),
             ("Adopt it, and record what it cost", OUTPUT)]
    ys = [95, 83, 71, 59, 46, 33, 20, 6]
    nodes = []
    for (text, colour), y in zip(steps, ys):
        nodes.append(c.node(62, y, text, colour, size=8.2, bold_first=True))
    for a, b in zip(nodes, nodes[1:]):
        c.arrow(a["bottom"], b["top"])

    # feedback path, routed down a clear lane so it crosses no box
    lane = min(n["cx"] - n["w"] / 2 for n in nodes) - 5
    dec, des = nodes[6], nodes[2]
    c.route([dec["left"], (lane, dec["cy"]), (lane, des["cy"]), des["left"]], colour=DECIDE[0])
    c.caption(lane - 2, 60, "no:\nbuild and test\nanother design", colour=DECIDE[0], ha="right")
    c.caption(65, 13, "yes", size=7.6, ha="left")
    c.save(OUT_OVERRIDE, "fig_design_process.png")


# ---------------------------------------------------------------- 2. system pipeline
def system_pipeline():
    c = Canvas(6.2, 6.9)
    user = c.node(50, 96, "User message\n+ the last four user turns", INPUT, bold_first=True)

    d1 = c.node(25, 79, "d1: emotion detector\nRoBERTa-base (125M)\n32-way softmax", DETECT,
                bold_first=True)
    cues = c.node(76, 79, "Attention cues\nlast-layer [CLS] attention,\ntop 40% of words", CUES,
                  bold_first=True)
    c.arrow(user["bottom"], d1["top"], rad=0.12)
    c.arrow(user["bottom"], cues["top"], rad=-0.12)

    stage1 = c.node(25, 59, "Stage 1: emotion choice\nPhi-4-mini picks one of\nd1's three candidates",
                    GEN, bold_first=True)
    c.arrow(d1["bottom"], stage1["top"], "top 3", lx=6.5, colour=DETECT[0])

    prompt = c.node(50, 36, "Prompt builder\nguidance for that one emotion,\ncue words, three rules",
                    PROMPT, bold_first=True)
    c.arrow(stage1["bottom"], (prompt["cx"] - prompt["w"] / 4, prompt["top"][1]),
            "chosen emotion", lx=-5, colour=GEN[0])
    c.arrow(cues["bottom"], (prompt["cx"] + prompt["w"] / 4, prompt["top"][1]),
            "cue words", lx=7, colour=CUES[0])

    stage2 = c.node(50, 19, "Stage 2: reply generation\nPhi-4-mini-instruct (3.8B, 4-bit),\n"
                            "T = 0.7, top-p 0.9, last 10 exchanges", GEN, bold_first=True)
    c.arrow(prompt["bottom"], stage2["top"])

    out = c.node(50, 6, "Reply to the user, with an audit trail:\n"
                        "candidates, chosen emotion, cue words", OUTPUT, bold_first=True)
    c.arrow(stage2["bottom"], out["top"])
    c.save(OUT_OVERRIDE, "fig_final_architecture.png")


# ---------------------------------------------------------------- 3. attention cues
def attention_cues():
    c = Canvas(6.4, 3.8)
    heads = ["User message\n(word pieces)",
             "RoBERTa-base\n12 layers,\neager attention",
             "Final layer\n[CLS] row, mean\nover 12 heads",
             "Word scores\npieces merged\ninto words"]
    colours = [INPUT, DETECT, DETECT, CUES]
    centres, widths = c.pack_row(heads, 7.4, left=3, right=97, min_gap=4)
    row = [c.node(x, 87, s, col, size=7.4, min_w=w)
           for x, s, col, w in zip(centres, heads, colours, widths)]
    for p, q in zip(row, row[1:]):
        c.arrow(p["right"], q["left"])

    tokens = [("my", 0.18), ("dog", 0.81), ("disappeared", 0.93), ("while", 0.21),
              ("i", 0.12), ("was", 0.10), ("away", 0.62), ("at", 0.14), ("college", 0.74)]
    words = [w for w, _ in tokens]
    widths = [c.width_of(w, 8.2, pad_x=5) for w in words]
    total = sum(widths) + 2.0 * (len(words) - 1)
    x = 50 - total / 2
    for (word, score), w in zip(tokens, widths):
        kept = score >= 0.6
        n = c.node(x + w / 2, 32, word, CUES if kept else MUTED, size=8.2, pad_x=5, pad_y=4)
        c.caption(n["cx"], 21, f"{score:.2f}", size=7.2, colour=CUES[0] if kept else "#90a4ae")
        x += w + 2.0

    c.arrow(row[3]["bottom"], (row[3]["cx"], 43), colour=CUES[0])
    c.caption(row[3]["cx"] - 4, 58, "keep the\nhighest-scoring 40%\nof words\n(at least 3, at most 14)",
              colour=CUES[0], ha="right")
    c.caption(50, 7, 'cue words sent to the prompt builder:  "dog", "disappeared", "away", "college"',
              size=8.2, colour=CUES[0])
    c.save(OUT_NEW, "fig_attention_cues.png")


# ---------------------------------------------------------------- 4. prompt construction
def prompt_construction():
    c = Canvas(6.4, 4.6)
    inputs = ["Chosen emotion\nstage 1:\nlonely",
              "Cue words\nd1 attention:\nmoved away,\nparents",
              "Guidance for lonely\n1 of 32:\nname the isolation,\nno quick fix",
              "Three rules\nacknowledge,\nask one question,\nunder 100 words"]
    colours = [GEN, CUES, PROMPT, PROMPT]
    centres, widths = c.pack_row(inputs, 7.0, bold_first=True, left=2, right=98, min_gap=2.5)
    tops = [c.node(x, 88, s, col, size=7.0, bold_first=True, min_w=w)
            for x, s, col, w in zip(centres, inputs, colours, widths)]

    prompt = c.node(50, 45,
                    "System prompt sent to Phi-4-mini\n"
                    "\"The person is feeling LONELY. They mentioned: moved away, parents.\n"
                    "Acknowledge the isolation itself, gently; do not offer a surface-level fix.\n"
                    "Reply in under 100 words and end with exactly one short question.\"",
                    GEN, size=7.4, bold_first=True)
    for n, col in zip(tops, colours):
        # aim each arrow at a point inside the prompt box, so the four lines fan in tidily
        target_x = prompt["cx"] + (n["cx"] - prompt["cx"]) * 0.4
        c.arrow(n["bottom"], (target_x, prompt["top"][1]), colour=col[0])

    reply = c.node(50, 10,
                   "Stage 2 reply\n\"That sounds really isolating - moving away from your parents\n"
                   "changes everything at once. How are you finding the quiet?\"",
                   OUTPUT, size=7.4, bold_first=True)
    c.arrow(prompt["bottom"], reply["top"])
    c.save(OUT_NEW, "fig_prompt_construction.png")


# ---------------------------------------------------------------- 5. data pipeline
def data_pipeline():
    c = Canvas(6.4, 4.2)
    steps = ["EmpatheticDialogues\n23,144 conversations,\n88,687 utterances",
             "Cleaning\nrestore commas, drop\nbad and empty rows",
             "Reduction\nkeep the first message\nof each conversation"]
    colours = [CORPUS, STEP, STEP]
    sw = max(c.width_of(s, 8.0, bold_first=True) for s in steps)
    left_cx = 2 + sw / 2
    column = [c.node(left_cx, y, s, col, size=8.0, bold_first=True, min_w=sw)
              for y, s, col in zip((88, 55, 20), steps, colours)]
    for p, q in zip(column, column[1:]):
        c.arrow(p["bottom"], q["top"])

    outs = [(90, "Detector data\n17,839 / 2,763 / 2,542 first messages"),
            (66, "Generator pairs\n17,749 messages with the human reply"),
            (42, "Evaluation set\n300 test conversations, seed 42"),
            (16, "Judge sample\n64 of those, two per emotion")]
    ow = max(c.width_of(s, 7.8, bold_first=True) for _, s in outs)
    bus = left_cx + sw / 2 + 5            # a lane clear of both columns
    out_cx = 98 - ow / 2
    for y, text in outs:
        n = c.node(out_cx, y, text, OUTPUT, size=7.8, bold_first=True, min_w=ow)
        c.route([column[2]["right"], (bus, column[2]["cy"]), (bus, y), n["left"]],
                colour=OUTPUT[0])
    c.save(OUT_NEW, "fig_data_pipeline.png")


if __name__ == "__main__":
    design_process()
    system_pipeline()
    attention_cues()
    prompt_construction()
    data_pipeline()
