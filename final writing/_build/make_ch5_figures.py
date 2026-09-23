"""Figures for Chapter 5 of the final draft.

Every value is read from the result CSVs in `final writing/tables/` or from the scored
output files, so no number is typed into this script. Two figures replace the writing-kit
versions of the same name (they are written to figures_override/, which the build prefers):
the kit's short-name map was missing three systems, which printed as "nan" bars, and its
empathy chart still included the superseded fine-tuned run.

    fig_rule_adherence_all_systems.png  rule adherence of every system        (5.1.3)
    fig_wassa_empathy_ci.png            empathy score with 95% CIs            (5.1.4)
    fig_emotion_fidelity.png            emotion validation of every system    (5.1.5)
    fig_confusion_pairs.png             the detector's most frequent errors   (5.1.2)
    fig_judge_dimensions.png            judge scores by dimension             (5.1.8)
    fig_radar_composite.png             five measures at once, four systems   (5.2.6)
    fig_forest_effects.png              final system vs baselines, with CIs   (5.4.2)
    fig_quality_vs_cost.png             rated quality against size and memory (5.5.7)
    fig_decoupled_vs_single.png         decoupled design vs its own generator (5.5.9)

Usage:
    python "final writing/_build/make_ch5_figures.py"
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
TABLES = OUT / "tables"
NEW = OUT / "figures"
OVERRIDE = OUT / "figures_override"

FINAL = "#1565c0"      # the final system
PIPE = "#7aa9d6"       # the other decoupled pipelines
SINGLE = "#e65100"     # a single model with no detector
SINGLE_L = "#f0a868"   # a smaller single model
PLAIN = "#9aa5ab"      # zero-shot and fine-tuned baselines
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e0dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})

SHORT = {
    "zero_shot": "Zero-shot Qwen-1.5B",
    "fine_tuned_fixed": "Fine-tuned Qwen-1.5B",
    "single_model_same_size": "Single model Qwen-1.5B",
    "single_model_phi": "Single model Phi-4-mini",
    "single_model_bigger": "Single model Qwen-7B",
    "prompt_engineered_pipeline": "Initial pipeline (GPT-2 + Qwen)",
    "rag_pipeline": "Initial pipeline + retrieval",
    "bakeoff_qwen": "RoBERTa + Qwen (top-1)",
    "bakeoff_gemma": "RoBERTa + Gemma-4B (top-1)",
    "bakeoff_phi": "RoBERTa + Phi-4-mini (top-1)",
    "two_stage_qwen": "Two-stage Qwen-1.5B",
    "two_stage_phi": "FINAL: two-stage Phi-4-mini",
    "two_stage_phi_robertalarge": "Final with RoBERTa-large",
}
FAMILY = {
    "two_stage_phi": FINAL,
    "single_model_phi": SINGLE, "single_model_bigger": SINGLE, "single_model_same_size": SINGLE_L,
    "zero_shot": PLAIN, "fine_tuned_fixed": PLAIN,
}
LEGEND = [("Final system", FINAL), ("Other decoupled pipelines", PIPE),
          ("Single model, no detector", SINGLE), ("Zero-shot or fine-tuned", PLAIN)]


def table_root(rel):
    """A results file from the repository root."""
    return pd.read_csv(ROOT / rel)


def table(name):
    """The full table, from the writing kit; `tables/` holds the trimmed print copies."""
    hits = sorted((ROOT / "writing").glob(f"*/tables/{name}"))
    return pd.read_csv(hits[0] if hits else TABLES / name)


def master():
    return table("all_systems_master_comparison.csv").set_index("Setting ID")


def colours(keys):
    return [FAMILY.get(k, PIPE) for k in keys]


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend_below(ax, entries=LEGEND, ncol=4, y=-0.30):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
    ax.legend(handles, [t for t, _ in entries], frameon=False, fontsize=7.5, ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, y), handlelength=1.1, columnspacing=1.2)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def bar_chart(col, xlabel, title, filename, folder, fmt="{:.1f}", xmax=100, height=3.9):
    df = master().sort_values(col)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, height))
    ax.barh(y, df[col], color=colours(df.index), height=0.72)
    for i, (k, v) in enumerate(zip(df.index, df[col])):
        ax.text(v + xmax * 0.012, i, fmt.format(v), va="center", fontsize=8, color=INK,
                fontweight="bold" if k == "two_stage_phi" else "normal")
    ax.set_yticks(y, [SHORT[k] for k in df.index], fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    grid_x(ax)
    legend_below(ax, y=-0.22)
    save(fig, folder, filename)


# ---------------------------------------------------------------- 5.1.3 and 5.1.5
def rule_adherence():
    bar_chart("All rules (%)",
              "Replies meeting all three rules (%), same 300 conversations",
              "Rule adherence: acknowledge, ask exactly one question, stay under 120 words",
              "fig_rule_adherence_all_systems.png", OVERRIDE)


def emotion_fidelity():
    bar_chart("Emotion validation (%)",
              "Replies read back as the conversation's true emotion (%), N = 300",
              "Emotion fidelity: does the reply express the feeling the person had?",
              "fig_emotion_fidelity.png", NEW, xmax=45)


# ---------------------------------------------------------------- 5.1.4
def wassa_ci():
    per = pd.read_csv(ROOT / "llm_judge_evaluation" / "wassa_empathy_scores.csv")
    per = per[per["setting"].isin(master().index)]   # the superseded fine-tuned run is excluded
    rng = np.random.default_rng(42)
    rows = []
    for key, g in per.groupby("setting"):
        v = g["wassa_empathy_0_10"].to_numpy(float)
        boots = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        rows.append((key, v.mean(), *np.percentile(boots, [2.5, 97.5])))
    df = pd.DataFrame(rows, columns=["key", "mean", "lo", "hi"]).sort_values("mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for i, r in df.iterrows():
        c = FAMILY.get(r["key"], PIPE)
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, linewidth=2.4, solid_capstyle="round")
        ax.plot(r["mean"], i, "o", color=c, markersize=6.5, markeredgecolor="white", markeredgewidth=1.4)
        ax.text(r["hi"] + 0.04, i, f"{r['mean']:.2f}", va="center", fontsize=8, color=INK,
                fontweight="bold" if r["key"] == "two_stage_phi" else "normal")
    ax.set_yticks(range(len(df)), [SHORT[k] for k in df["key"]], fontsize=8)
    ax.set_xlim(3.1, 6.4)
    ax.set_xlabel("WASSA 2024 empathy (0-10), mean and 95% bootstrap CI, N = 300 each")
    ax.set_title("Empathy classifier score, with the uncertainty of each mean", loc="left")
    grid_x(ax)
    legend_below(ax, y=-0.22)
    save(fig, OVERRIDE, "fig_wassa_empathy_ci.png")


# ---------------------------------------------------------------- 5.1.3 to 5.1.5
def scoreboard():
    """The three headline measures for every system, on one shared list of names."""
    df = master().sort_values("All rules (%)")
    per = pd.read_csv(ROOT / "llm_judge_evaluation" / "wassa_empathy_scores.csv")
    per = per[per["setting"].isin(df.index)]
    rng = np.random.default_rng(42)
    ci = {}
    for key, g in per.groupby("setting"):
        v = g["wassa_empathy_0_10"].to_numpy(float)
        boots = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        ci[key] = np.percentile(boots, [2.5, 97.5])

    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.12})
    cols = colours(df.index)
    bold = ["bold" if k == "two_stage_phi" else "normal" for k in df.index]

    ax = axes[0]
    ax.barh(y, df["All rules (%)"], color=cols, height=0.7)
    for i, v in enumerate(df["All rules (%)"]):
        ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=6.8, color=INK, fontweight=bold[i])
    ax.set_xlim(0, 104)
    ax.set_xlabel("All three rules (%)", fontsize=7.6)

    ax = axes[1]
    for i, k in enumerate(df.index):
        lo, hi = ci[k]
        ax.plot([lo, hi], [i, i], color=cols[i], linewidth=2.2, solid_capstyle="round")
        ax.plot(df.loc[k, "WASSA empathy (0-10)"], i, "o", color=cols[i], markersize=4.6,
                markeredgecolor="white", markeredgewidth=1.1)
        ax.text(hi + 0.07, i, f"{df.loc[k, 'WASSA empathy (0-10)']:.2f}", va="center",
                fontsize=6.8, color=INK, fontweight=bold[i])
    ax.set_xlim(3.1, 6.7)
    ax.set_xlabel("Empathy classifier (0-10)\nwith 95% CI", fontsize=7.6)

    ax = axes[2]
    ax.barh(y, df["Emotion validation (%)"], color=cols, height=0.7)
    for i, v in enumerate(df["Emotion validation (%)"]):
        ax.text(v + 0.7, i, f"{v:.1f}", va="center", fontsize=6.8, color=INK, fontweight=bold[i])
    ax.set_xlim(0, 45)
    ax.set_xlabel("Emotion fidelity (%)", fontsize=7.6)

    for i, ax in enumerate(axes):
        ax.set_yticks(y, [SHORT[k] for k in df.index] if i == 0 else [""] * len(df), fontsize=7.4)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in LEGEND]
    axes[1].legend(handles, [s for s, _ in LEGEND], frameon=False, fontsize=7.2, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.5, -0.26), handlelength=1.1)
    axes[0].set_title("The three headline measures, same 300 conversations for every system",
                      loc="left", fontsize=9.5)
    save(fig, NEW, "fig_headline_scoreboard.png")


# ---------------------------------------------------------------- 5.1.2
def confusion_pairs():
    df = table("d1_most_confused_pairs.csv").sort_values("Count")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(y, df["Count"], color="#6a1b9a", height=0.7, alpha=0.85)
    for i, (c, s) in enumerate(zip(df["Count"], df["Share of all errors (%)"])):
        ax.text(c + 0.4, i, f"{c}  ({s:.1f}%)", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{a}  read as  {b}" for a, b in zip(df["True emotion"], df["Predicted as"])],
                  fontsize=8)
    ax.set_xlim(0, df["Count"].max() * 1.35)
    ax.set_xlabel("Number of test conversations (share of all 1,263 detector errors)")
    ax.set_title("The detector's most frequent mistakes are near neighbours", loc="left")
    grid_x(ax)
    save(fig, NEW, "fig_confusion_pairs.png")


# ---------------------------------------------------------------- 5.1.8
def judge_dimensions():
    df = table("ai_judge_scores.csv").sort_values("Panel overall")
    short = []
    for name in df["System"]:
        key = next(k for k, v in LONG_TO_KEY.items() if k in name)
        short.append(SHORT[LONG_TO_KEY[key]])
    y = np.arange(len(df))
    h = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for off, col, colour, label in ((h, "Panel safety", "#2e7d32", "Safety"),
                                    (0, "Panel relevance", "#00695c", "Relevance"),
                                    (-h, "Panel empathy", "#6a1b9a", "Empathy")):
        ax.barh(y + off, df[col], height=h * 0.92, color=colour, label=label)
        for i, v in enumerate(df[col]):
            ax.text(v + 0.08, i + off, f"{v:.2f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(y, short, fontsize=8)
    ax.set_xlim(0, 11.6)
    ax.set_xlabel("Mean score of the two judges (1-10), 64 conversations")
    ax.set_title("What the judges reward: safety is easy, empathy is the hard one", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              handlelength=1.1, reverse=True)
    grid_x(ax)
    save(fig, NEW, "fig_judge_dimensions.png")


LONG_TO_KEY = {
    "Qwen2.5-7B": "single_model_bigger",
    "same generator: Phi-4-mini": "single_model_phi",
    "FINAL SYSTEM": "two_stage_phi",
    "top-1 emotion": "bakeoff_phi",
    "same size: Qwen2.5-1.5B": "single_model_same_size",
    "Initial pipeline: GPT-2": "prompt_engineered_pipeline",
    "Zero-shot": "zero_shot",
    "LoRA fine-tuned": "fine_tuned_fixed",
}


# ---------------------------------------------------------------- 5.1.8 judge coverage
def judge_emotion_coverage():
    """Every one of the 32 emotions reached the judges, and this is how each one scored."""
    n = table_root("llm_judge_evaluation/nemotron_openrouter_scores.csv")
    g = table_root("llm_judge_evaluation/gptoss_groq_scores.csv")
    test = pd.read_csv(ROOT / "ed_data" / "empatheticdialogues" / "test.csv", on_bad_lines="skip")
    emo = test.drop_duplicates("conv_id").set_index("conv_id")["context"]
    first = (test.sort_values(["conv_id", "utterance_idx"]).drop_duplicates("conv_id")
             .set_index("conv_id")["utterance"])

    j = n.merge(g, on=["conv_id", "setting"], suffixes=("_n", "_g"))
    j = j[j["setting"] == "two_stage_phi"].copy()
    j["emotion"] = j["conv_id"].map(emo)
    j["words"] = j["conv_id"].map(first).str.split().str.len()
    j["panel"] = (j["empathy_n"] + j["empathy_g"]) / 2
    per = j.groupby("emotion").agg(panel=("panel", "mean"), n=("panel", "size"),
                                   short=("words", "min"), long=("words", "max"))
    per = per.sort_values("panel")

    y = np.arange(len(per))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    mean = j["panel"].mean()
    ax.axvline(mean, color="#b0bec5", linewidth=1, linestyle="--", zorder=0)
    for i, (name, r) in enumerate(per.iterrows()):
        colour = FINAL if r["panel"] >= mean else "#7aa9d6"
        ax.plot([mean, r["panel"]], [i, i], color=colour, linewidth=1.2, alpha=0.5, zorder=1)
        ax.plot(r["panel"], i, "o", color=colour, markersize=5.5, zorder=2,
                markeredgecolor="white", markeredgewidth=1)
        ax.text(10.15, i, f"{int(r['short'])} and {int(r['long'])} words", fontsize=6.4,
                color=INK2, va="center")
    ax.set_yticks(y, per.index, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(4.6, 10.0)
    ax.set_xlabel("Judge empathy score of the final system (1-10), two conversations per emotion")
    ax.set_title(f"All 32 emotions reached the judges; the panel mean is {mean:.2f}", loc="left")
    ax.spines["left"].set_visible(False)
    grid_x(ax)
    save(fig, NEW, "fig_judge_emotion_coverage.png")


# ---------------------------------------------------------------- 5.2.6
def radar_composite():
    df = table("composite_ranking.csv")
    keys = [LONG_TO_KEY[next(k for k in LONG_TO_KEY if k in s)] for s in df["System"]]
    df = df.assign(key=keys).set_index("key")
    cols = ["All rules (%)", "WASSA empathy", "Emotion validation (%)", "AI-judge empathy",
            "Non-repetition (%)"]
    labels = ["Rule\nadherence", "WASSA\nempathy", "Emotion\nfidelity", "Judge\nempathy",
              "Freedom from\nrepetition"]
    scaled = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    show = ["two_stage_phi", "single_model_bigger", "single_model_phi", "single_model_same_size"]
    tone = {"two_stage_phi": FINAL, "single_model_bigger": SINGLE,
            "single_model_phi": "#b0651a", "single_model_same_size": SINGLE_L}

    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.0, 4.6), subplot_kw={"polar": True})
    for k in show:
        v = scaled.loc[k].tolist()
        v += v[:1]
        lw = 2.2 if k == "two_stage_phi" else 1.3
        ax.plot(ang, v, color=tone[k], linewidth=lw, label=SHORT[k])
        if k == "two_stage_phi":
            ax.fill(ang, v, color=tone[k], alpha=0.14)
    ax.set_xticks(ang[:-1], labels, fontsize=7.8)
    ax.set_yticks([0, 0.5, 1.0], ["", "", ""])
    ax.set_ylim(0, 1.08)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.set_title("Five measures at once: each axis runs from the weakest to the best system",
                 fontsize=9, pad=18)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    save(fig, NEW, "fig_radar_composite.png")


# ---------------------------------------------------------------- 5.4.2
def forest_effects():
    df = table("final_system_vs_baselines_significance.csv")
    ci = df["95% CI"].str.extract(r"\[([-+0-9.]+), *([-+0-9.]+)\]").astype(float)
    df = df.assign(lo=ci[0], hi=ci[1])
    df["key"] = [LONG_TO_KEY[next(k for k in LONG_TO_KEY if k in s)]
                 if any(k in s for k in LONG_TO_KEY) else "rag_pipeline"
                 for s in df["Final system vs"]]

    panels = [("All rules", "Rule adherence\n(percentage points)"),
              ("WASSA empathy", "Empathy classifier\n(0-10 scale)"),
              ("Emotion validation", "Emotion fidelity\n(percentage points)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.22})
    order = None
    for ax, (metric, xlabel) in zip(axes, panels):
        d = df[df["Metric"] == metric].copy()
        if order is None:
            order = d.sort_values("Difference")["key"].tolist()
        d = d.set_index("key").loc[[k for k in order if k in set(d["key"])]]
        y = np.arange(len(d))
        for i, (k, r) in enumerate(d.iterrows()):
            sig = str(r["Significant?"]).strip().upper() == "YES"
            c = FINAL if r["Difference"] > 0 else SINGLE
            c = c if sig else "#b6bec3"
            ax.plot([r["lo"], r["hi"]], [i, i], color=c, linewidth=2.2, solid_capstyle="round")
            ax.plot(r["Difference"], i, "o", color=c, markersize=5.5,
                    markeredgecolor="white", markeredgewidth=1.2)
        span = d["hi"].max() - d["lo"].min()
        ax.set_xlim(d["lo"].min() - span * 0.10, d["hi"].max() + span * 0.10)
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.axvline(0, color=INK2, linewidth=0.9, linestyle="--")
        ax.set_yticks(y, [SHORT[k] for k in d.index] if ax is axes[0] else [""] * len(d), fontsize=7.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel, fontsize=7.6)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    entries = [("Final system better (significant)", FINAL),
               ("Baseline better (significant)", SINGLE),
               ("No reliable difference", "#b6bec3")]
    handles = [plt.Line2D([0], [0], color=c, lw=2.4) for _, c in entries]
    axes[1].legend(handles, [t for t, _ in entries], frameon=False, fontsize=7.4, ncol=3,
                   loc="upper center", bbox_to_anchor=(0.5, -0.24))
    axes[0].set_title("The final system against each baseline, with 95% confidence intervals",
                      loc="left", fontsize=9.5)
    save(fig, NEW, "fig_forest_effects.png")


# ---------------------------------------------------------------- 5.5.7
def quality_vs_cost():
    par = table("parameter_efficiency_systems.csv")
    par = par[~par["System"].str.startswith("Final system as")]
    res = table("resource_efficiency.csv").set_index("System")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for _, r in par.iterrows():
        key = LONG_TO_KEY[next(k for k in LONG_TO_KEY if k in r["System"])]
        mem = float(res.loc[r["System"], "Peak GPU memory (GB)"])
        x, y = float(r["Parameters (B)"]), float(r["AI-judge overall"])
        c = FAMILY.get(key, PIPE)
        ax.scatter(x, y, s=mem * 260, color=c, alpha=0.25, edgecolors="none")
        ax.scatter(x, y, s=34, color=c, zorder=3)
        ax.annotate(f"{SHORT[key]}\n{mem:.2f} GB peak, {res.loc[r['System'], 'Mean s per reply']:.2f} s",
                    (x, y), textcoords="offset points", xytext=(0, -34), ha="center",
                    fontsize=7.4, color=INK2)
    ax.set_xlim(0.2, 9.2)
    ax.set_ylim(7.9, 9.2)
    ax.set_xlabel("Total parameters (billions)")
    ax.set_ylabel("AI-judge overall score (1-10)")
    ax.set_title("Rated quality against what it costs to run (bubble = peak GPU memory)", loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, NEW, "fig_quality_vs_cost.png")


# ---------------------------------------------------------------- 5.5.9
def decoupled_vs_single():
    df = table("decoupled_vs_single_generator.csv")
    df = df[df["95% CI"].str.startswith("[", na=False)].copy()
    ci = df["95% CI"].str.extract(r"\[([-+0-9.]+), *([-+0-9.]+)\]").astype(float)
    base = df["Decoupled (ours)"].astype(float)
    df = df.assign(lo=ci[0], hi=ci[1], Difference=df["Difference"].astype(float))
    # each measure sits on its own scale, so plot the difference as a share of the decoupled value
    df["rel"] = df["Difference"] / base * 100
    df["rel_lo"] = df["lo"] / base * 100
    df["rel_hi"] = df["hi"] / base * 100
    df = df.sort_values("rel")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    label_x = df["rel_hi"].max() + 5          # one tidy column of values, clear of every bar
    for i, (_, r) in enumerate(df.iterrows()):
        better = "decoupled" in str(r["Verdict"])
        worse = "single model better" in str(r["Verdict"])
        c = FINAL if better else (SINGLE if worse else "#b6bec3")
        ax.plot([r["rel_lo"], r["rel_hi"]], [i, i], color=c, linewidth=2.4, solid_capstyle="round")
        ax.plot(r["rel"], i, "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=1.3)
        ax.text(label_x, i, f"{r['Decoupled (ours)']}  vs  {r['Same generator alone']}",
                va="center", ha="left", fontsize=7.4, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.9, linestyle="--")
    ax.set_yticks(y, df["Measure"], fontsize=8)
    ax.set_xlim(df["rel_lo"].min() - 6, label_x + 34)
    ax.set_xlabel("Difference from the same generator used alone (% of the decoupled value)")
    ax.set_title("Same generator, with and without the detector", loc="left")
    entries = [("Decoupled design better", FINAL), ("Single model better", SINGLE),
               ("No reliable difference", "#b6bec3")]
    handles = [plt.Line2D([0], [0], color=c, lw=2.4) for _, c in entries]
    ax.legend(handles, [t for t, _ in entries], frameon=False, fontsize=7.4, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.24))
    grid_x(ax)
    save(fig, NEW, "fig_decoupled_vs_single.png")


if __name__ == "__main__":
    scoreboard()
    judge_emotion_coverage()
    rule_adherence()
    emotion_fidelity()
    wassa_ci()
    confusion_pairs()
    judge_dimensions()
    radar_composite()
    forest_effects()
    quality_vs_cost()
    decoupled_vs_single()
