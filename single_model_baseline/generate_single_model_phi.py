"""Single-model Phi-4-mini baseline: the same generator as the final system, with no detector.

Why this baseline exists
------------------------
The existing single-model baselines use Qwen2.5-1.5B and Qwen2.5-7B, so the comparison with our
final system (RoBERTa-base + Phi-4-mini) mixes two things: removing the detector and changing the
generator. This baseline gives Phi-4-mini the detect-then-respond prompt with no detector, so the
ONLY difference from the final system is the decoupled design itself.

Everything else is held fixed: the same 300 test conversations (seed 42), the same system prompt as
the other single-model baselines, the same 4-bit NF4 quantisation as the final system's generator,
and the same decoding settings (temperature 0.7, top-p 0.9, at most 250 new tokens).

Usage:
    python generate_single_model_phi.py                    # 300 conversations, about 20-25 minutes
    python generate_single_model_phi.py --num-samples 5    # quick check first
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "emotion_prompts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_two_stage_pipeline as tsp  # noqa: E402  (same generation code as the final system)
from generate_bigger_single_model import SYSTEM_PROMPT, extract_test_pairs  # noqa: E402  (identical prompt)

MODEL_NAME = "microsoft/Phi-4-mini-instruct"
VARIANT = "single_model_phi"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluation_results_single_model_baseline"


def parse_args():
    p = argparse.ArgumentParser(description="Generate single-model Phi-4-mini (no detector) baseline responses.")
    p.add_argument("--num-samples", type=int, default=300, help="0 = all conversations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{VARIANT}_generation_output.csv"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists. Refusing to overwrite it (pass --overwrite if you are sure).")

    print(f"Loading {MODEL_NAME} in 4-bit (no detector, single model)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    model.eval()
    terminators = tsp.build_terminators(tokenizer)

    pairs_df = extract_test_pairs(ROOT_DIR / "ed_data" / "empatheticdialogues" / "test.csv")
    if args.num_samples and args.num_samples > 0:
        pairs_df = pairs_df.sample(n=min(args.num_samples, len(pairs_df)), random_state=args.seed)
    pairs_df = pairs_df.reset_index(drop=True)
    print(f"Using {len(pairs_df)} conversations (seed={args.seed}) - the same sample as every other setting.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    total = len(pairs_df)
    for i, row in enumerate(pairs_df.itertuples()):
        reply = tsp.generate(SYSTEM_PROMPT, row.user_text, tokenizer, model, terminators,
                             max_new_tokens=args.max_new_tokens, do_sample=True)
        rows.append({"conv_id": row.conv_id, "variant": VARIANT, "user_text": row.user_text,
                     "gold_response": row.gold_response, "gold_emotion": row.gold_emotion,
                     "predicted_emotion": None, "confidence": None, "generated_response": reply})
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"{i + 1}/{total} conversations processed")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
