"""Merge teammate score files into llm_judge_evaluation/nemotron_openrouter_scores.csv.

Checks before writing: the rows must belong to the current judging target, must not already be
scored, and must not duplicate each other. A timestamped backup of the main file is kept.

Usage (from this folder):
    python merge_teammate_scores.py tasks_teammate1_scores.csv tasks_teammate2_scores.csv
    python merge_teammate_scores.py --dry-run tasks_teammate1_scores.csv
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "llm_judge_evaluation" / "nemotron_openrouter_scores.csv"
COLUMNS = ["conv_id", "setting", "parse_ok", "empathy", "relevance", "safety", "raw_response", "error"]


def main():
    ap = argparse.ArgumentParser(description="Merge teammate Nemotron scores into the main file.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen and write nothing.")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "llm_judge_evaluation"))
    from run_llm_judge import load_stratified_sample
    target = set(zip(*load_stratified_sample(2, 42)[["conv_id", "setting"]].values.T.tolist()))

    main_df = pd.read_csv(MAIN)
    have = set(zip(main_df.loc[main_df["parse_ok"] == True, "conv_id"],
                   main_df.loc[main_df["parse_ok"] == True, "setting"]))
    print(f"Main file: {len(main_df)} rows, {len(have)} scored.")

    incoming = []
    for f in args.files:
        d = pd.read_csv(f)
        missing = [c for c in COLUMNS if c not in d.columns]
        if missing:
            raise SystemExit(f"{f}: missing columns {missing}")
        ok = d[d["parse_ok"] == True].copy()
        in_target = ok[[(c, s) in target for c, s in zip(ok["conv_id"], ok["setting"])]]
        already = in_target[[(c, s) in have for c, s in zip(in_target["conv_id"], in_target["setting"])]]
        new = in_target[[(c, s) not in have for c, s in zip(in_target["conv_id"], in_target["setting"])]]
        print(f"{Path(f).name}: {len(d)} rows | {len(ok)} scored | {len(in_target)} in target | "
              f"{len(already)} already present | {len(new)} new")
        for col in ("empathy", "relevance", "safety"):
            bad = new[(new[col] < 1) | (new[col] > 10)]
            if len(bad):
                raise SystemExit(f"{f}: {len(bad)} rows have {col} outside 1-10")
        incoming.append(new[COLUMNS])

    add = pd.concat(incoming, ignore_index=True).drop_duplicates(["conv_id", "setting"])
    print(f"\nTotal new rows to add: {len(add)}")
    if add.empty or args.dry_run:
        print("Nothing written." if add.empty else "Dry run: nothing written.")
        return

    backup = MAIN.with_name(f"nemotron_openrouter_scores_BACKUP_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(MAIN, backup)
    out = pd.concat([main_df, add], ignore_index=True)
    out.to_csv(MAIN, index=False)
    print(f"Backup: {backup.name}\nWrote {MAIN} ({len(out)} rows).")
    print(out[out["parse_ok"] == True].groupby("setting").size().to_string())


if __name__ == "__main__":
    main()
