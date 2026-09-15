"""Combine the per-system benchmark results into one table, with paired tests on time per reply.

CPU only - loads no model. Run after benchmark_resources.py has been run for each system.

Usage:
    python summarize_resources.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINAL = "two_stage_phi"
ORDER = ["two_stage_phi", "single_model_same_size", "single_model_bigger"]


def boot_ci(diff, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    summaries = {p.stem.replace("_summary", ""): json.loads(p.read_text(encoding="utf-8"))
                 for p in RESULTS_DIR.glob("*_summary.json")}
    if not summaries:
        raise SystemExit("No results yet. Run benchmark_resources.py for each system first.")
    missing = [s for s in ORDER if s not in summaries]
    if missing:
        print(f"Note: not yet measured: {', '.join(missing)}")

    rows = []
    for s in [s for s in ORDER if s in summaries]:
        m = summaries[s]
        rows.append({"System": m["label"], "Params (B)": m["params_b"],
                     "Model memory (GB)": round(m["model_memory_mb"] / 1024, 2),
                     "Peak GPU memory (GB)": round(m["peak_memory_allocated_mb"] / 1024, 2),
                     "Mean s / reply": m["mean_s_per_reply"], "Median s / reply": m["median_s_per_reply"],
                     "Mean reply tokens": m["mean_reply_tokens"], "Tokens / s": m["mean_tokens_per_s_reply"],
                     "Load time (s)": m["load_time_s"], "N": m["timed_messages"]})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "resource_comparison.csv", index=False)
    print("\n" + table.to_string(index=False))

    if FINAL in summaries:
        f = summaries[FINAL]
        parts = {k: f[k] for k in ("mean_d1_and_cues_s", "mean_stage1_pick_s", "mean_stage2_reply_s") if k in f}
        print("\nFinal system time breakdown (mean seconds):", parts)

    tests = []
    per = {s: pd.read_csv(RESULTS_DIR / f"{s}_per_message.csv") for s in summaries}
    for base in [s for s in ORDER if s in per and s != FINAL and FINAL in per]:
        x = per[FINAL][["conv_id", "total_s"]].merge(per[base][["conv_id", "total_s"]], on="conv_id",
                                                       suffixes=("_final", "_base"))
        d = x["total_s_final"] - x["total_s_base"]
        lo, hi = boot_ci(d)
        tests.append({"Final system vs": summaries[base]["label"], "N": len(x),
                      "Mean difference (s / reply)": round(d.mean(), 3), "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                      "Wilcoxon p": round(wilcoxon(x["total_s_final"], x["total_s_base"]).pvalue, 5),
                      "Final is": "faster" if hi < 0 else ("slower" if lo > 0 else "not clearly different")})
        mem = f["peak_memory_allocated_mb"] - summaries[base]["peak_memory_allocated_mb"]
        tests[-1]["Peak memory difference (GB)"] = round(mem / 1024, 2)
    if tests:
        t = pd.DataFrame(tests)
        t.to_csv(RESULTS_DIR / "resource_paired_tests.csv", index=False)
        print("\nPaired comparison on the same messages (negative = final system faster / lighter):")
        print(t.to_string(index=False))
    print(f"\nSaved resource_comparison.csv and resource_paired_tests.csv in {RESULTS_DIR}")
    print("Note: time includes each system's full reply generation, and replies differ in length; "
          "compare 'Tokens / s' as well.")


if __name__ == "__main__":
    main()
