"""Experiment 1: how well does the generator label emotions on its own, compared with the detector?

This answers the question "why do you need RoBERTa at all, when the LLM could name the emotion itself?".
Phi-4-mini is asked to pick one of the same 32 emotions for the same conversations the detector is
scored on, so the two are compared on identical inputs with a paired test.

The labels are produced with greedy decoding (a classification decision, as in the pipeline's stage 1),
and answers that do not name one of the 32 emotions are counted as wrong but reported separately.

Usage:
    python measure_phi_emotion_labeling.py --num-samples 300     # the evaluation sample, about 5 minutes
    python measure_phi_emotion_labeling.py --num-samples 0       # all 2,542 test conversations, about 40 minutes
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "emotion_prompts"))
import generate_two_stage_pipeline as tsp  # noqa: E402
from emotion_prompts import EMOTION_GUIDANCE  # noqa: E402

MODEL_NAME = "microsoft/Phi-4-mini-instruct"
EMOTIONS = sorted(EMOTION_GUIDANCE)
OUT_DIR = ROOT_DIR / "evaluation_results_single_model_baseline"


def parse_args():
    p = argparse.ArgumentParser(description="Measure the generator's own emotion-labelling accuracy.")
    p.add_argument("--num-samples", type=int, default=300, help="0 = all test conversations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_prompt():
    return ("You are an expert at reading emotions in text. Decide which ONE of these 32 emotions the person is "
            "feeling:\n" + ", ".join(EMOTIONS) + "\n"
            "Reply with ONLY that single word, exactly as written in the list. No explanation.")


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phi_emotion_labeling.csv"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. Pass --overwrite to replace it.")

    pairs = tsp.extract_test_pairs(ROOT_DIR / "ed_data" / "empatheticdialogues" / "test.csv")
    if args.num_samples and args.num_samples > 0:
        pairs = pairs.sample(n=min(args.num_samples, len(pairs)), random_state=args.seed)
    pairs = pairs.reset_index(drop=True)
    print(f"Labelling {len(pairs)} conversations with {MODEL_NAME}.")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    model.eval()
    terms = tsp.build_terminators(tok)
    system_prompt = build_prompt()

    rows = []
    for i, row in enumerate(pairs.itertuples(), start=1):
        raw = tsp.generate(system_prompt, row.user_text, tok, model, terms, max_new_tokens=10, do_sample=False)
        low = raw.strip().lower()
        label = next((e for e in EMOTIONS if low == e), None) or next((e for e in EMOTIONS if e in low), None)
        rows.append({"conv_id": row.conv_id, "gold_emotion": row.gold_emotion, "phi_label": label,
                     "parse_ok": label is not None, "raw_output": raw})
        if i % 25 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    df["phi_correct"] = df["phi_label"].fillna("") == df["gold_emotion"]
    print(f"Phi labelled a valid emotion in {df['parse_ok'].mean():.1%} of cases.")
    print(f"Phi top-1 accuracy: {df['phi_correct'].mean():.1%}")

    # paired comparison with the detector on the same conversations
    d1_file = ROOT_DIR / "evaluation_results_two_stage" / "two_stage_phi_generation_output.csv"
    if d1_file.exists():
        d1 = pd.read_csv(d1_file)[["conv_id", "gold_emotion", "d1_top1"]]
        m = df.merge(d1, on="conv_id", suffixes=("", "_d1"))
        if len(m):
            m["d1_correct"] = m["d1_top1"] == m["gold_emotion"]
            sys.path.insert(0, str(ROOT_DIR / "writing" / "_scripts"))
            from build_writing_tables import mcnemar, boot_ci
            p, only_a, only_b = mcnemar(m["d1_correct"], m["phi_correct"])
            lo, hi = boot_ci(m["d1_correct"].astype(float) - m["phi_correct"].astype(float))
            print(f"\nPaired on {len(m)} conversations:")
            print(f"  detector (RoBERTa-base) {m['d1_correct'].mean():.1%} vs generator (Phi) {m['phi_correct'].mean():.1%}")
            print(f"  difference {(m['d1_correct'].mean() - m['phi_correct'].mean()) * 100:+.1f} pp "
                  f"[{lo * 100:+.1f}, {hi * 100:+.1f}], McNemar p = {p:.2g} "
                  f"(only detector right: {only_a}, only generator right: {only_b})")


if __name__ == "__main__":
    main()
