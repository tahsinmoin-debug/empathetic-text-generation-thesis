"""Measure how repetitive each setting's replies are.

Motivation: hands-on testing of the conversational agent showed the model reusing phrasing -
"Remember, it's okay to feel this way" appeared in three of four consecutive replies, and two
closing questions were near-identical. That was an anecdotal observation. This turns it into a
number, so it can be reported as a measured limitation rather than an impression.

It also produces the diversity figures (Dist-1/2/3), which are a standard metric in empathetic
dialogue papers.

IMPORTANT - what these numbers can and cannot be compared against:
  * VALID: comparing settings in THIS project against each other. Every setting is scored on the
    same 300 conversations, so sample size is held constant.
  * NOT VALID: comparing these to Dist-n figures published in papers. Dist-n falls as the sample
    grows - measured directly on our own output, Dist-1 goes 32.3% (n=50) -> 15.6% (n=300). The
    published papers evaluate on the full ~2,500-conversation test set, so placing our numbers
    beside theirs would manufacture an advantage out of a measurement artifact.

What "boilerplate rate" means: the share of replies that contain at least one of the setting's
own 10 most-repeated 5-word phrases. A high value means the system leans on a small stock of
canned phrases rather than responding to each person specifically.

Usage:
    python measure_repetition.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
D2_DIR = ROOT_DIR / "evaluation_results_d2_settings_comparison"
OUTPUT_DIR = ROOT_DIR / "evaluation_results_diversity"

SETTINGS = [
    "two_stage_phi",
    "single_model_phi",
    "fine_tuned_fixed",
    "bakeoff_phi",
    "prompt_engineered_pipeline",
    "single_model_same_size",
    "single_model_bigger",
    "rag_pipeline",
    "zero_shot",
]

TOP_PHRASES = 10       # how many frequent phrases define "boilerplate"
PHRASE_LEN = 5         # a 5-word phrase is long enough to be a real stock sentence opener


def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(list_of_tokens, n):
    total, seen = 0, set()
    for toks in list_of_tokens:
        grams = ngrams(toks, n)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def analyse(setting):
    path = D2_DIR / f"{setting}_scored_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    replies = df["generated_response"].fillna("").astype(str).tolist()
    token_lists = [tokenize(r) for r in replies]

    phrase_counts = Counter()
    for toks in token_lists:
        # count each phrase once per reply, so one long rambling reply can't dominate
        phrase_counts.update(set(ngrams(toks, PHRASE_LEN)))

    top = phrase_counts.most_common(TOP_PHRASES)
    top_set = {p for p, _ in top}
    boilerplate_hits = sum(1 for toks in token_lists if top_set & set(ngrams(toks, PHRASE_LEN)))

    return {
        "setting": setting,
        "n_replies": len(replies),
        "dist_1": round(distinct_n(token_lists, 1) * 100, 2),
        "dist_2": round(distinct_n(token_lists, 2) * 100, 2),
        "dist_3": round(distinct_n(token_lists, 3) * 100, 2),
        "boilerplate_rate": round(boilerplate_hits / len(replies) * 100, 1) if replies else 0.0,
        "most_repeated_phrase": " ".join(top[0][0]) if top else "",
        "most_repeated_count": top[0][1] if top else 0,
        "_top": top,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (analyse(s) for s in SETTINGS) if r]
    if not rows:
        raise SystemExit("No scored prediction files found.")

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    table = table.sort_values("dist_2", ascending=False)
    table.to_csv(OUTPUT_DIR / "diversity_results.csv", index=False)

    print("=" * 92)
    print("  DIVERSITY AND REPETITION  (higher Dist = more varied; higher boilerplate = worse)")
    print("  Comparable BETWEEN these settings only - all use the same 300 conversations.")
    print("=" * 92)
    print(f"\n{'setting':<30}{'Dist-1':>8}{'Dist-2':>8}{'Dist-3':>8}{'boilerplate':>13}")
    print("-" * 92)
    for r in table.itertuples():
        print(f"{r.setting:<30}{r.dist_1:>7.2f}%{r.dist_2:>7.2f}%{r.dist_3:>7.2f}%{r.boilerplate_rate:>12.1f}%")

    print("\n\n" + "=" * 92)
    print("  WHAT EACH SYSTEM REPEATS MOST  (phrase, and how many of its 300 replies contain it)")
    print("=" * 92)
    for r in rows:
        print(f"\n### {r['setting']}")
        for phrase, count in r["_top"][:5]:
            pct = count / r["n_replies"] * 100
            print(f'   {count:>3} replies ({pct:>4.1f}%)  "{" ".join(phrase)}"')

    lines = [
        "# Diversity and Repetition\n",
        "Measured on the same 300 held-out conversations for every setting, so these are "
        "comparable to each other. They are NOT comparable to Dist-n values published in "
        "papers, which use the full ~2,500-conversation test set - Dist-n falls as sample size "
        "grows (measured on our own output: 32.3% at n=50 vs 15.6% at n=300).\n",
        "**Boilerplate rate** = share of replies containing at least one of that setting's own "
        "10 most-repeated 5-word phrases. Higher means more reliance on stock phrasing.\n",
        "| Setting | Dist-1 | Dist-2 | Dist-3 | Boilerplate |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| {r.setting} | {r.dist_1:.2f}% | {r.dist_2:.2f}% | {r.dist_3:.2f}% | {r.boilerplate_rate:.1f}% |"
        )
    lines.append("\n## Most repeated phrase per setting\n")
    lines.append("| Setting | Most repeated 5-word phrase | Replies containing it |")
    lines.append("|---|---|---:|")
    for r in rows:
        pct = r["most_repeated_count"] / r["n_replies"] * 100
        lines.append(
            f"| {r['setting']} | \"{r['most_repeated_phrase']}\" | {r['most_repeated_count']} ({pct:.1f}%) |"
        )
    (OUTPUT_DIR / "diversity_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n\nSaved: {OUTPUT_DIR / 'diversity_table.md'}")


if __name__ == "__main__":
    main()
