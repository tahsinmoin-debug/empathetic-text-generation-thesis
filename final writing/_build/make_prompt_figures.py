"""The three "show me exactly what was sent" figures.

These print the real text rather than a picture of a screen, so a reader can check the wording
and a printer can render it cleanly. Every line comes from the code or the data files, not
retyped. The canvas is measured in points (one data unit = one point), so font sizes, line
spacing and box heights are all in the same unit and nothing can be mis-scaled.

    fig_pipeline_prompt.png  the system prompt the generator actually receives    (4.2.6)
    fig_preprocessing.png    one conversation before and after preparation        (4.3.3)
    fig_judge_prompt.png     the prompt each judge receives, and what is hidden   (5.1.8)

Usage:
    python "final writing/_build/make_prompt_figures.py"
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
FIGS = OUT / "figures"
sys.path.insert(0, str(ROOT / "emotion_prompts"))

INK, MUTED = "#1a1a1a", "#6b7280"
GEN = ("#e65100", "#fff3e0")      # the generator's prompt
JUDGE = ("#6a1b9a", "#f3e5f5")    # the judges' prompt
RAW = ("#5d4037", "#efebe9")      # data as released
CLEAN = ("#2e7d32", "#e8f5e9")    # data after preparation
MARK = "#c62828"                  # what the figure is pointing at

WIDTH_IN = 6.4
SIZE = 6.3                        # monospace body size, in points
LEAD = SIZE * 1.62                # baseline to baseline
PAD = 7.0


class Sheet:
    """A page measured in points, filled from the top down."""

    def __init__(self, height_pt):
        self.w = WIDTH_IN * 72
        self.fig, self.ax = plt.subplots(figsize=(WIDTH_IN, height_pt / 72), dpi=200)
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, height_pt)
        self.ax.axis("off")
        self.y = height_pt - 2

    def panel(self, lines, colour, title):
        edge, face = colour
        h = len(lines) * LEAD + SIZE * 2.4 + 2 * PAD
        self.ax.add_patch(FancyBboxPatch((2, self.y - h), self.w - 4, h,
                                         boxstyle="round,pad=1.5,rounding_size=4",
                                         linewidth=1.2, edgecolor=edge, facecolor=face, zorder=1))
        self.ax.text(PAD + 2, self.y - PAD - SIZE, title, fontsize=SIZE + 1.5,
                     fontweight="bold", color=edge, va="center", zorder=2)
        for i, line in enumerate(lines):
            text, colour_of = line if isinstance(line, tuple) else (line, INK)
            self.ax.text(PAD + 2, self.y - PAD - SIZE * 2.4 - (i + 0.5) * LEAD, text,
                         fontsize=SIZE, color=colour_of, family="monospace", va="center", zorder=2)
        self.y -= h
        return self.y

    def gap(self, points):
        self.y -= points

    def heading(self, text, colour):
        self.ax.text(PAD + 2, self.y - SIZE, text, fontsize=SIZE + 1.5, fontweight="bold",
                     color=colour, va="center")
        self.y -= SIZE * 2.6

    def lines(self, rows, colour=INK, mono=True, size=None):
        size = size or SIZE
        for row in rows:
            text, c = row if isinstance(row, tuple) else (row, colour)
            self.ax.text(PAD + 2, self.y - size, text, fontsize=size, color=c, va="center",
                         family="monospace" if mono else "sans-serif")
            self.y -= size * 1.62

    def save(self, name):
        FIGS.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(FIGS / name, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        plt.close(self.fig)
        print(f"wrote {name}  (unused space: {self.y:.0f} pt)")


def wrap(text, width, indent="    "):
    """Wrap a real line without changing its words."""
    if len(text) <= width:
        return [text]
    return textwrap.wrap(text, width=width, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


# ---------------------------------------------------------------- 1. the generator's prompt
def pipeline_prompt():
    from emotion_prompts import build_emotion_prompt  # noqa: E402

    built = build_emotion_prompt("lonely", "moved away, parents, quiet")
    lines = [ln for raw in built.split("\n") for ln in wrap(raw, 92)]
    note = ("Line 1 names the emotion chosen in stage 1; lines 2 and 3 carry the cue words the detector's attention "
            "selected; lines 4 and 5 are the guidance written for this one emotion, one of 32. The three rules and the "
            "word limit are identical for every emotion, and for every system compared in Chapter 5.")
    note_lines = wrap(note, 108, indent="")

    s = Sheet(len(lines) * LEAD + len(note_lines) * SIZE * 1.6 + 78)
    s.panel(lines, GEN, "System prompt sent to the generator, for a message read as LONELY")
    s.gap(10)
    s.lines(note_lines, colour=MUTED, mono=False, size=SIZE + 0.2)
    s.save("fig_pipeline_prompt.png")


# ---------------------------------------------------------------- 2. before and after preparation
def preprocessing():
    raw = pd.read_csv(ROOT / "ed_data" / "empatheticdialogues" / "train.csv", on_bad_lines="skip")
    conv = raw[raw["conv_id"] == "hit:0_conv:1"].sort_values("utterance_idx")

    before = [("idx  speaker  utterance", MUTED)]
    for _, r in conv.head(4).iterrows():
        body = str(r["utterance"])
        colour = MARK if "_comma_" in body else INK
        for j, piece in enumerate(wrap(body, 72, indent="")):
            head = f"{r['utterance_idx']:<5}{r['speaker_idx']:<9}" if j == 0 else " " * 14
            before.append((head + piece, colour))
    before.append(("...  two further turns, and the columns 'prompt' (the situation description),", MUTED))
    before.append(("     'context' (the emotion label, the same on every row), 'selfeval' and 'tags'", MUTED))

    first = conv.iloc[0]
    clean = str(first["utterance"]).replace("_comma_", ",")
    after = [("text", MUTED)]
    after += [(p, INK) for p in wrap(clean, 88, indent="")]
    after += [("", INK), ("label", MUTED), (str(first["context"]), INK)]

    steps = [
        ("1.  restore the literal string _comma_ to a comma   (red above)", INK),
        ("2.  keep the first utterance of each conversation, drop the later turns", INK),
        ("3.  keep the message and the emotion label, drop the other six columns", INK),
        ("4.  drop rows that do not parse as CSV, and rows with an empty message or label", INK),
        ("23,144 conversations, 88,687 utterances  ->  17,839 / 2,763 / 2,542 labelled messages",
         CLEAN[0]),
    ]

    s = Sheet((len(before) + len(after)) * LEAD + len(steps) * SIZE * 1.7 + 130)
    s.panel(before, RAW, "Before: the file as released, one row per utterance (conversation hit:0_conv:1)")
    s.gap(12)
    s.heading("Preparation", MUTED)
    s.lines(steps)
    s.gap(10)
    s.panel(after, CLEAN, "After: one row per conversation, which is what the detector is trained on")
    s.save("fig_preprocessing.png")


# ---------------------------------------------------------------- 3. the judges' prompt
def judge_prompt():
    src = (ROOT / "llm_judge_evaluation" / "run_llm_judge.py").read_text(encoding="utf-8")
    template = src.split('JUDGE_PROMPT_TEMPLATE = """', 1)[1].split('"""', 1)[0]
    lines = []
    for raw in template.split("\n"):
        colour = MARK if "{user_text}" in raw or "{generated_response}" in raw else INK
        lines += [(piece, colour) for piece in wrap(raw, 92)]

    controls = [
        "Blind          the two red lines are everything the judge sees: no system name, no model",
        "               name, no gold reply, nothing saying which design wrote the response.",
        "Identical      both judges scored all 512 replies, so no system met a different judge.",
        "Independent    the judges come from different developers, neither related to our models.",
        "Deterministic  temperature 0, JSON-only answer, and every parse recorded.",
        "Checked        the judges agree at rho = 0.69 on empathy and rank the systems alike.",
    ]

    s = Sheet(len(lines) * LEAD + len(controls) * SIZE * 1.7 + 92)
    s.panel(lines, JUDGE, "The prompt each judge receives, once per reply")
    s.gap(12)
    s.heading("How the scoring was kept fair", JUDGE[0])
    s.lines(controls)
    s.save("fig_judge_prompt.png")


if __name__ == "__main__":
    pipeline_prompt()
    preprocessing()
    judge_prompt()
