"""Build the thesis figures from the CSV tables made by build_writing_tables.py.

Print-first static PNGs (300 dpi) for a Word document, so one light look, no interactivity.
Palette: validated categorical slots (blue, orange, aqua) - aqua sits below 3:1 contrast on white,
so every aqua mark also carries a direct label and a distinct marker shape. Non-highlighted systems
are neutral grey so the final system is the only coloured bar in comparison charts.

Run build_writing_tables.py first.

Usage:
    python writing/_scripts/build_writing_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#c9c7c1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT = {
    "zero_shot": "Zero-shot Qwen-1.5B", "fine_tuned_fixed": "Monolithic fine-tuned Qwen-1.5B",
    "single_model_same_size": "Single model Qwen-1.5B", "single_model_bigger": "Single model Qwen-7B",
    "prompt_engineered_pipeline": "Initial pipeline (GPT-2 + Qwen)", "rag_pipeline": "Initial pipeline + RAG",
    "bakeoff_qwen": "RoBERTa + Qwen (top-1)", "bakeoff_gemma": "RoBERTa + Gemma-4B (top-1)",
    "bakeoff_phi": "RoBERTa + Phi-4-mini (top-1)", "two_stage_qwen": "Two-stage Qwen",
    "two_stage_phi": "FINAL: two-stage Phi-4-mini",
}


def section(name):
    return W / name / "figures"


def save(fig, sections, filename):
    for s in sections:
        out = section(s)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_d1_backbones():
    df = pd.read_csv(W / "Chapter 4 - 4.2 Preliminary Design or Design (Model) Specification" / "tables" / "d1_backbone_comparison.csv")
    df = df.sort_values("Top-5 acc (%)")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, df["Top-5 acc (%)"], height=h, color=ORANGE, label="Top-5")
    ax.barh(y - h / 2 - 0.02, df["Top-1 acc (%)"], height=h, color=BLUE, label="Top-1")
    for i, (t1, t5) in enumerate(zip(df["Top-1 acc (%)"], df["Top-5 acc (%)"])):
        ax.text(t5 + 0.8, i + h / 2 + 0.02, f"{t5:.1f}", va="center", color=INK, fontsize=8)
        ax.text(t1 + 0.8, i - h / 2 - 0.02, f"{t1:.1f}", va="center", color=INK, fontsize=8)
    labels = [f"{m}\n{p}M params, {t} min" for m, p, t in zip(df["Backbone"], df["Parameters (M)"], df["Train time (min)"])]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy on official ED test split (%), N = 2,542")
    ax.set_title("d1 backbone comparison - same data and training settings", loc="left")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, reverse=True)
    grid_x(ax)
    save(fig, ["Chapter 4 - 4.2 Preliminary Design or Design (Model) Specification",
               "Chapter 5 - 5.2 Analysis of Design Solutions"], "fig_d1_backbone_comparison.png")


def highlight_bars(ax, labels, values, final_key, fmt="{:.1f}", xmax=100):
    colors = [BLUE if k == final_key else GREY for k in labels.index]
    ax.barh(np.arange(len(values)), values, color=colors, height=0.7)
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.01, i, fmt.format(v), va="center", fontsize=8,
                color=INK, fontweight="bold" if labels.index[i] == final_key else "normal")
    ax.set_yticks(np.arange(len(values)), labels.values, fontsize=8)
    grid_x(ax)


def fig_rule_adherence():
    df = pd.read_csv(W / "Chapter 5 - 5.1 Performance Evaluation" / "tables" / "all_systems_master_comparison.csv")
    df = df.set_index("Setting ID").sort_values("All rules (%)")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    highlight_bars(ax, df.index.to_series().map(SHORT), df["All rules (%)"].to_numpy(), "two_stage_phi")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Replies meeting ALL three rules (%), same 300 conversations")
    ax.set_title("Rule adherence: acknowledge + exactly one question + length limit", loc="left")
    save(fig, ["Chapter 5 - 5.1 Performance Evaluation", "Chapter 5 - 5.2 Analysis of Design Solutions"],
         "fig_rule_adherence_all_systems.png")


def fig_wassa_ci():
    root = W.parent
    per = pd.read_csv(root / "llm_judge_evaluation" / "wassa_empathy_scores.csv")
    rng = np.random.default_rng(42)
    rows = []
    for key, g in per.groupby("setting"):
        v = g["wassa_empathy_0_10"].to_numpy(float)
        boots = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
        rows.append((key, v.mean(), *np.percentile(boots, [2.5, 97.5])))
    df = pd.DataFrame(rows, columns=["key", "mean", "lo", "hi"]).sort_values("mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, r in df.iterrows():
        c = BLUE if r["key"] == "two_stage_phi" else INK2
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r["mean"], i, "o", color=c, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(r["hi"] + 0.05, i, f"{r['mean']:.2f}", va="center", fontsize=8, color=INK,
                fontweight="bold" if r["key"] == "two_stage_phi" else "normal")
    ax.set_yticks(range(len(df)), df["key"].map(SHORT), fontsize=8)
    ax.set_xlim(3.8, 6.3)
    ax.set_xlabel("WASSA 2024 empathy (0-10), mean and 95% bootstrap CI, N = 300 each")
    ax.set_title("Empathy classifier score by system", loc="left")
    grid_x(ax)
    save(fig, ["Chapter 5 - 5.1 Performance Evaluation", "Chapter 5 - 5.3 Statistical Analysis"],
         "fig_wassa_empathy_ci.png")


def fig_length_bands():
    df = pd.read_csv(W / "Chapter 5 - 5.4 Comparisons and Relationships" / "tables" / "gemma_length_band_analysis.csv")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    series = [("Gemma WASSA", "Gemma n", "RoBERTa + Gemma-4B", ORANGE, "s"),
              ("Phi (top-1) WASSA", "Phi (top-1) n", "RoBERTa + Phi-4-mini (top-1)", AQUA, "^"),
              ("Final two-stage Phi WASSA", "Final two-stage Phi n", "FINAL: two-stage Phi-4-mini", BLUE, "o")]
    for col, ncol, name, color, marker in series:
        mask = df[ncol] >= 5
        ax.plot(x[mask], df.loc[mask, col], color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        last = np.where(mask)[0][-1]
        ax.text(x[last] + 0.12, df.loc[last, col], name.split(":")[0].replace("RoBERTa + ", ""),
                va="center", fontsize=7.5, color=INK)
    ax.axvspan(3.5, 5.6, color="#f1f0ec", zorder=0)
    ax.text(4.55, 4.76, "above the 120-word limit", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, df["Length band (words)"])
    ax.set_xlim(-0.3, 5.9)
    ax.set_xlabel("Reply length band (words); points shown only where n >= 5")
    ax.set_ylabel("Mean WASSA (0-10)")
    ax.set_title("At matched length, Phi matches or beats Gemma", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, ["Chapter 5 - 5.4 Comparisons and Relationships", "Chapter 5 - 5.2 Analysis of Design Solutions"],
         "fig_wassa_by_length_band.png")


def fig_gantt():
    df = pd.read_csv(W / "Chapter 3 - 3.6 Project Management Plan" / "tables" / "project_timeline_from_repository.csv")
    months = ["Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026", "Jun 2026", "Jul 2026", "Aug 2026", "Sep 2026", "Oct 2026"]
    idx = {m: i for i, m in enumerate(months)}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, r in df.iloc[::-1].reset_index(drop=True).iterrows():
        start = idx[r["Start"]]
        end = idx.get(r["End"], idx["Oct 2026"] - 0.01)
        open_ended = r["End"] not in idx
        ax.barh(i, end - start + 1 if not open_ended else end - start, left=start, height=0.62,
                color=GREY if open_ended else BLUE, edgecolor="white", linewidth=2)
        ax.text(start + 0.08, i, r["Phase"], va="center", fontsize=7.5, color="white" if not open_ended else INK,
                fontweight="bold")
    ax.set_yticks(range(len(df)), df["Work package"].iloc[::-1], fontsize=7.5)
    ax.set_xticks(range(len(months) + 1), [""] * (len(months) + 1))
    ax.set_xticks([i + 0.5 for i in range(len(months))], [m[:3] for m in months], minor=True, fontsize=8)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(0, len(months))
    ax.set_title("Project timeline (completed = blue, in progress = grey)", loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, ["Chapter 3 - 3.6 Project Management Plan"], "fig_project_gantt.png")


def box(ax, xy, w, h, text, fc, ec, tc=INK, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.2))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=7, color=INK2)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    light_blue, light_orange = "#e3eefb", "#fde8df"
    fs = 7.2
    box(ax, (0.1, 2.1), 1.8, 1.1, "User message\n+ last 4\nuser turns", "#f4f3ef", GRID, fs=fs)
    box(ax, (2.6, 3.2), 2.6, 1.4, "d1: emotion detector\nRoBERTa-base (125M)\n32-way softmax", light_blue, BLUE, fs=fs)
    box(ax, (2.6, 0.6), 2.6, 1.4, "Attention cues\n[CLS] attention, top 40%\nof words (eager mode)", light_blue, BLUE, fs=fs)
    box(ax, (6.0, 3.2), 2.6, 1.4, "Stage 1: emotion pick\nPhi-4-mini chooses one\nof d1's top-3 (greedy)", light_orange, ORANGE, fs=fs)
    box(ax, (6.0, 0.6), 2.6, 1.4, "Prompt builder\n32 per-emotion guides\n+ 3 rules + cue words", "#f4f3ef", GRID, fs=fs)
    box(ax, (9.3, 2.0), 2.6, 1.6, "Stage 2: d2 generation\nPhi-4-mini-instruct\n4-bit NF4, T 0.7, top-p 0.9\n+ last 10 exchanges", light_orange, ORANGE, fs=fs)
    box(ax, (9.3, 0.2), 2.6, 1.0, "Empathetic reply\n+ audit trail", "#f4f3ef", GRID, fs=fs)
    arrow(ax, (1.9, 2.95), (2.6, 3.7))
    arrow(ax, (1.9, 2.35), (2.6, 1.5))
    arrow(ax, (5.2, 3.9), (6.0, 3.9))
    ax.text(5.6, 4.05, "top-3", ha="center", fontsize=6.8, color=INK2)
    arrow(ax, (5.2, 1.3), (6.0, 1.3))
    ax.text(5.6, 1.45, "cues", ha="center", fontsize=6.8, color=INK2)
    arrow(ax, (7.3, 3.2), (7.3, 2.0))
    ax.text(7.4, 2.6, "chosen\nemotion", ha="left", va="center", fontsize=6.8, color=INK2)
    arrow(ax, (8.6, 1.3), (9.3, 2.3))
    arrow(ax, (10.6, 2.0), (10.6, 1.2))
    ax.text(0.1, 5.1, "Final system architecture (decoupled two-stage pipeline)", fontsize=10, fontweight="bold", color=INK)
    save(fig, ["Chapter 4 - 4.1 Design Process or Methodology Overview",
               "Chapter 4 - 4.2 Preliminary Design or Design (Model) Specification"], "fig_final_architecture.png")


def fig_design_process():
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.4, 2.7)
    ax.axis("off")
    steps = ["Problem and\nconstraints", "Data prep\n(ED, first\nutterance)", "Alternative\ndesigns\n(d1, d2, links)",
             "Simulation\non 300 test\nconversations", "Multi-signal\nevaluation", "Statistical\ntesting", "Design\nadjustment"]
    w, gap = 1.36, 0.32
    for i, st in enumerate(steps):
        x = 0.1 + i * (w + gap)
        final = i == len(steps) - 1
        box(ax, (x, 0.75), w, 1.25, st, "#fde8df" if final else "#e3eefb", ORANGE if final else BLUE, fs=7)
        if not final:
            arrow(ax, (x + w, 1.37), (x + w + gap, 1.37))
    x_last = 0.1 + 6 * (w + gap) + w / 2
    x_alt = 0.1 + 2 * (w + gap) + w / 2
    ax.add_patch(FancyArrowPatch((x_last, 0.75), (x_alt, 0.75), connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                                 mutation_scale=11, color=ORANGE, lw=1.3))
    ax.text((x_last + x_alt) / 2, -0.32, "iterate until the evidence stops improving", ha="center", fontsize=7.5, color=INK2)
    ax.text(0.1, 2.45, "Design process", fontsize=10, fontweight="bold", color=INK)
    save(fig, ["Chapter 4 - 4.1 Design Process or Methodology Overview"], "fig_design_process.png")


def main():
    fig_d1_backbones()
    fig_rule_adherence()
    fig_wassa_ci()
    fig_length_bands()
    fig_gantt()
    fig_architecture()
    fig_design_process()
    print("Figures written.")


if __name__ == "__main__":
    main()
