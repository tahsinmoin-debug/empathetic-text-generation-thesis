"""Experiment 2: the generator labels the emotion itself, then uses the same per-emotion guidance.

This is the strongest challenge to the decoupled design: if an LLM can name the emotion on its own and
then apply our emotion-specific prompt, the separate detector may be unnecessary. The only difference
from the final system is where the emotion comes from:

    final system      d1 (RoBERTa) top-3  ->  Phi picks one  ->  per-emotion prompt + attention cues
    this baseline     Phi names the emotion itself           ->  per-emotion prompt (no cues, no detector)

Everything else is identical: same 300 conversations, same per-emotion guidance text, same decoding
(temperature 0.7, top-p 0.9, at most 250 new tokens, seed 42). Attention cues cannot be produced without
a detector, so they are absent here; that difference is reported with the results.

Usage:
    python generate_single_model_phi_two_pass.py                 # about 30 minutes
    python generate_single_model_phi_two_pass.py --num-samples 5 # quick check
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
from emotion_prompts import EMOTION_GUIDANCE, build_emotion_prompt  # noqa: E402

MODEL_NAME = "microsoft/Phi-4-mini-instruct"
EMOTIONS = sorted(EMOTION_GUIDANCE)
VARIANT = "single_model_phi_two_pass"
OUT_DIR = ROOT_DIR / "evaluation_results_single_model_baseline"

LABEL_PROMPT = ("You are an expert at reading emotions in text. Decide which ONE of these 32 emotions the person is "
                "feeling:\n" + ", ".join(EMOTIONS) + "\n"
                "Reply with ONLY that single word, exactly as written in the list. No explanation.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate the two-pass single-model baseline (no detector).")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{VARIANT}_generation_output.csv"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. Pass --overwrite to replace it.")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    model.eval()
    terms = tsp.build_terminators(tok)

    pairs = tsp.extract_test_pairs(ROOT_DIR / "ed_data" / "empatheticdialogues" / "test.csv")
    if args.num_samples and args.num_samples > 0:
        pairs = pairs.sample(n=min(args.num_samples, len(pairs)), random_state=args.seed)
    pairs = pairs.reset_index(drop=True)
    print(f"Using {len(pairs)} conversations (seed={args.seed}) - the same sample as every other setting.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows, unparsed = [], 0
    for i, row in enumerate(pairs.itertuples(), start=1):
        raw = tsp.generate(LABEL_PROMPT, row.user_text, tok, model, terms, max_new_tokens=10, do_sample=False)
        low = raw.strip().lower()
        picked = next((e for e in EMOTIONS if low == e), None) or next((e for e in EMOTIONS if e in low), None)
        if picked is None:
            picked, unparsed = "sad", unparsed + 1  # same fallback policy as the pipeline: never fail a reply
        reply = tsp.generate(build_emotion_prompt(picked, context_cues=""), row.user_text, tok, model, terms,
                             max_new_tokens=args.max_new_tokens, do_sample=True)
        rows.append({"conv_id": row.conv_id, "variant": VARIANT, "user_text": row.user_text,
                     "gold_response": row.gold_response, "gold_emotion": row.gold_emotion,
                     "predicted_emotion": picked, "confidence": None, "self_label_raw": raw,
                     "generated_response": reply})
        if i % 10 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)} conversations processed")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Self-labels that did not name one of the 32 emotions: {unparsed}/{len(df)}")
    print(f"Self-label matches the gold emotion in {(df.predicted_emotion == df.gold_emotion).mean():.1%} of cases "
          f"(detector top-1 is 50.3% on the same test partition).")


if __name__ == "__main__":
    main()
