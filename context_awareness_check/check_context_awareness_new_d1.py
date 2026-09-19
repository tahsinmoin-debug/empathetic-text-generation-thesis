"""Context-Awareness & Validation check, re-run with the NEW, more accurate d1 (RoBERTa,
fixed text + architecture, 50.3%/84.3% vs the original GPT-2 d1's 48.2%/82.7%) as the judge.

This directly answers the actual question that matters for d1: not "does a more accurate d1
make d2's OUTPUT FORMAT score higher on rule-adherence" (structure), but "does a more accurate
d1 make the final reply's own emotional content more genuinely correct" (substance) - since
d1's predicted emotion is the ONLY signal d2 ever receives, and rule-adherence can be perfect
while validating a completely wrong feeling.

Uses the more accurate new d1 as the checker for EVERY setting (not just the new ones), so the
comparison is apples-to-apples against a single, more trustworthy judge - includes the two
bake-off settings (bakeoff_qwen, bakeoff_phi) which share the SAME new-d1 signal but differ in
d2, plus the original prompt_engineered_pipeline (old d1 + Qwen) for the key controlled
comparison: same d2 (Qwen), only d1 differs, isolating exactly what a more accurate d1 buys us.

Usage:
    python check_context_awareness_new_d1.py
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
D1_DIR = ROOT_DIR / "train_roberta_ed_d1" / "checkpoints" / "final"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evaluation_results_context_awareness_new_d1"
D2_COMPARISON_DIR = ROOT_DIR / "evaluation_results_d2_settings_comparison"
BAKEOFF_DIR = ROOT_DIR / "evaluation_results_d1_d2_bakeoff"
TWO_STAGE_DIR = ROOT_DIR / "evaluation_results_two_stage"

def _complete(path, expected=300):
    """A generation output counts as registered only when it holds the full evaluation sample.
    A short smoke-test file would otherwise shrink the conversations shared by all settings."""
    try:
        return path.exists() and len(pd.read_csv(path)) >= expected
    except Exception:
        return False


SOURCES = {
    "zero_shot": (D2_COMPARISON_DIR / "zero_shot_generation_output.csv", None),
    "fine_tuned": (D2_COMPARISON_DIR / "fine_tuned_generation_output.csv", None),
    "prompt_engineered_pipeline": (
        ROOT_DIR / "evaluation_results_ablation_confidence_gating" / "generation_output.csv", "argmax",
    ),
    "rag_pipeline": (
        ROOT_DIR / "evaluation_results_rag_pipeline" / "rag_pipeline_generation_output.csv", "rag_pipeline",
    ),
    "single_model_same_size": (
        ROOT_DIR / "evaluation_results_single_model_baseline" / "single_model_same_size_generation_output.csv",
        "single_model_same_size",
    ),
    "single_model_bigger": (
        ROOT_DIR / "evaluation_results_single_model_baseline" / "single_model_bigger_generation_output.csv",
        "single_model_bigger",
    ),
    # New d1 (RoBERTa) + candidate d2s from the bake-off - same new-d1 signal, different d2.
    "bakeoff_qwen": (BAKEOFF_DIR / "bakeoff_qwen_generation_output.csv", None),
    "bakeoff_phi": (BAKEOFF_DIR / "bakeoff_phi_generation_output.csv", None),
    "bakeoff_gemma": (BAKEOFF_DIR / "bakeoff_gemma_generation_output.csv", None),
    # Two-stage top-3 + per-emotion prompts. 'predicted_emotion' here is the emotion d2 actually
    # picked and generated from, so context-awareness measures faithfulness to that choice.
    "two_stage_qwen": (TWO_STAGE_DIR / "two_stage_qwen_generation_output.csv", None),
    "two_stage_phi": (TWO_STAGE_DIR / "two_stage_phi_generation_output.csv", None),
    # Detector sensitivity run (roberta-large d1). Still checked by the SAME roberta-base reader as
    # every other setting, so the larger detector never grades its own pipeline's replies.
    "two_stage_phi_robertalarge": (TWO_STAGE_DIR / "two_stage_phi_robertalarge_generation_output.csv", None),
}

# Corrected monolithic LoRA baseline (train_qwen2_5_monolithic_d2_fixed/). Registered only once its outputs
# exist, so every other run keeps working before it has been generated.
_FINE_TUNED_FIXED = D2_COMPARISON_DIR / "fine_tuned_fixed_generation_output.csv"
if _complete(_FINE_TUNED_FIXED):
    SOURCES["fine_tuned_fixed"] = (_FINE_TUNED_FIXED, None)

# Single-model Phi-4-mini baseline (same generator as the final system, no detector). Registered only
# once its outputs exist.
_SINGLE_PHI = ROOT_DIR / "evaluation_results_single_model_baseline" / "single_model_phi_generation_output.csv"
if _complete(_SINGLE_PHI):
    SOURCES["single_model_phi"] = (_SINGLE_PHI, "single_model_phi")

_TWO_PASS = ROOT_DIR / "evaluation_results_single_model_baseline" / "single_model_phi_two_pass_generation_output.csv"
if _complete(_TWO_PASS):
    SOURCES["single_model_phi_two_pass"] = (_TWO_PASS, "single_model_phi_two_pass")

# Settings where d1 actually ran and produced a real predicted_emotion.
SETTINGS_WITH_D1_SIGNAL = {
    "prompt_engineered_pipeline", "rag_pipeline", "bakeoff_qwen", "bakeoff_phi", "bakeoff_gemma",
    "two_stage_qwen", "two_stage_phi", "two_stage_phi_robertalarge",
}


def clean_text(text):
    return str(text).replace("_comma_", ",").strip()


def get_id2label(model):
    return {int(key): value for key, value in model.config.id2label.items()}


def classify_batch(texts, tokenizer, model, id2label, device, max_length=128):
    texts = [clean_text(t) for t in texts]
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [id2label[p] for p in preds]


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading the NEW d1 (RoBERTa, fixed text + architecture) as the 'what emotion does this reply sound like' checker...")
    tokenizer = AutoTokenizer.from_pretrained(D1_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(D1_DIR)
    model.to(device)
    model.eval()
    id2label = get_id2label(model)

    all_rows = []
    summary_rows = []

    for setting, (path, variant_filter) in SOURCES.items():
        if not path.exists():
            print(f"Skipping '{setting}': {path} not found")
            continue
        df = pd.read_csv(path)
        if variant_filter is not None:
            df = df[df["variant"] == variant_filter].copy()
        df = df.reset_index(drop=True)

        print(f"Scoring '{setting}' ({len(df)} replies)...")
        replies = df["generated_response"].fillna("").tolist()

        reflected_emotions = []
        batch_size = 16
        for start in range(0, len(replies), batch_size):
            batch = replies[start : start + batch_size]
            reflected_emotions.extend(classify_batch(batch, tokenizer, model, id2label, device))

        df["reply_reflected_emotion"] = reflected_emotions
        df["validation_match"] = df["reply_reflected_emotion"] == df["gold_emotion"]

        has_d1_signal = setting in SETTINGS_WITH_D1_SIGNAL
        if has_d1_signal:
            df["context_match"] = df["reply_reflected_emotion"] == df["predicted_emotion"]
        else:
            df["context_match"] = None

        df["setting"] = setting
        keep_cols = [
            "conv_id", "setting", "gold_emotion", "predicted_emotion",
            "reply_reflected_emotion", "validation_match", "context_match",
        ]
        all_rows.append(df[keep_cols])

        validation_accuracy = float(df["validation_match"].mean())
        context_accuracy = float(df["context_match"].mean()) if has_d1_signal else None

        summary_rows.append(
            {
                "setting": setting,
                "validation_accuracy_vs_gold": round(validation_accuracy, 4),
                "context_awareness_accuracy_vs_d1_signal": (
                    round(context_accuracy, 4) if context_accuracy is not None else "n/a (no d1 signal)"
                ),
                "n": len(df),
            }
        )
        ctx_str = f"{context_accuracy:.1%}" if context_accuracy is not None else "n/a"
        print(f"  -> Validation (vs gold): {validation_accuracy:.1%} | Context-awareness (vs d1): {ctx_str}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_dir / "context_awareness_per_reply.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "context_awareness_summary.csv", index=False)

    lines = ["| Setting | Validation Accuracy (vs true emotion) | Context-Awareness Accuracy (vs d1 signal) | N |",
             "|---|---:|---:|---:|"]
    for row in summary_rows:
        ctx_val = row["context_awareness_accuracy_vs_d1_signal"]
        ctx_cell = ctx_val if isinstance(ctx_val, str) else f"{ctx_val*100:.1f}%"
        lines.append(
            f"| {row['setting']} | {row['validation_accuracy_vs_gold']*100:.1f}% | {ctx_cell} | {row['n']} |"
        )
    (output_dir / "context_awareness_table.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {output_dir / 'context_awareness_table.md'}")


if __name__ == "__main__":
    main()
