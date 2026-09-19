"""P0 baseline (same-size variant): tests whether ONE model at the SAME size as the official
pipeline's d2 (Qwen2.5-1.5B-Instruct) can do "detect the emotion, then respond" in a single pass,
with no separate d1 detector at all - instead of splitting that work across two models.

This is deliberately the same model, same size, same 3 rules as the official pipeline - the ONLY
thing that changes is whether emotion detection is a separate model (d1) or folded into d2's own
prompt/reasoning. This isolates whether DECOUPLING itself helps, independent of model capability -
the companion experiment (a genuinely bigger single model) tests a different question (whether more
capacity alone removes the need for the two-model design).

Usage:
    python generate_same_size_single_model.py --num-samples 300 --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
D2_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_CSV = ROOT_DIR / "ed_data" / "empatheticdialogues" / "test.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluation_results_single_model_baseline"

SYSTEM_PROMPT = (
    "You are a highly empathetic counselor. Someone will share something with you.\n"
    "First, silently identify what emotion they are feeling. Then respond to them following "
    "these three rules strictly:\n"
    "1. Acknowledge and validate their feelings deeply using 2 to 3 comforting sentences.\n"
    "2. Share a deeply reassuring thought to help them feel safe and understood.\n"
    "3. Ask EXACTLY ONE short follow-up question at the very end to show you care. "
    "Do not ask any other questions.\n"
    "Do not mention your emotion analysis in your reply - just respond naturally, as if you "
    "already understood how they feel."
)


def clean_text(text):
    return str(text).replace("_comma_", ",").strip()


def extract_test_pairs(csv_path):
    """Same convention as the other generation scripts' extract_test_pairs()."""
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.sort_values(["conv_id", "utterance_idx"])
    pairs = []
    for conv_id, group in df.groupby("conv_id"):
        group = group.reset_index(drop=True)
        if len(group) >= 2:
            row1 = group.iloc[0]
            row2 = group.iloc[1]
            pairs.append(
                {
                    "conv_id": conv_id,
                    "gold_emotion": clean_text(row1["context"]),
                    "user_text": clean_text(row1["utterance"]),
                    "gold_response": clean_text(row2["utterance"]),
                }
            )
    return pd.DataFrame(pairs)


def generate_response(system_prompt, user_text, tokenizer, model, max_new_tokens=250, temperature=0.7, top_p=0.9):
    """Same generation block as test_full_pipeline.py's d2 step, for direct comparability."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate same-size single-model (no d1) baseline responses.")
    parser.add_argument("--num-samples", type=int, default=300, help="0 = use all available conversations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-1.5B-Instruct (same model/size as the official pipeline's d2, no d1)...")
    tokenizer = AutoTokenizer.from_pretrained(D2_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(D2_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    print(f"Loading test conversations from {TEST_CSV}...")
    pairs_df = extract_test_pairs(TEST_CSV)
    print(f"Found {len(pairs_df)} usable conversations.")

    if args.num_samples and args.num_samples > 0:
        pairs_df = pairs_df.sample(n=min(args.num_samples, len(pairs_df)), random_state=args.seed)
    pairs_df = pairs_df.reset_index(drop=True)
    print(f"Using {len(pairs_df)} sampled conversations (seed={args.seed}).")

    print(f"Generating responses (generation seed={args.seed}, reproducible run)...")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    total = len(pairs_df)
    for i, row in enumerate(pairs_df.itertuples()):
        response = generate_response(
            SYSTEM_PROMPT, row.user_text, tokenizer, model, max_new_tokens=args.max_new_tokens
        )
        rows.append(
            {
                "conv_id": row.conv_id,
                "variant": "single_model_same_size",
                "user_text": row.user_text,
                "gold_response": row.gold_response,
                "gold_emotion": row.gold_emotion,
                "predicted_emotion": None,
                "confidence": None,
                "generated_response": response,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"{i + 1}/{total} conversations processed")

    output_df = pd.DataFrame(rows)
    output_path = args.output_dir / "single_model_same_size_generation_output.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
