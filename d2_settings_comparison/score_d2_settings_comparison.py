import argparse
import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
D2_COMPARISON_DIR = ROOT_DIR / "evaluation_results_d2_settings_comparison"
ABLATION_DIR = ROOT_DIR / "evaluation_results_ablation_confidence_gating"
RAG_DIR = ROOT_DIR / "evaluation_results_rag_pipeline"
SINGLE_MODEL_DIR = ROOT_DIR / "evaluation_results_single_model_baseline"
CURATED_FINETUNE_DIR = ROOT_DIR / "evaluation_results_curated_finetune"
BAKEOFF_DIR = ROOT_DIR / "evaluation_results_d1_d2_bakeoff"
TWO_STAGE_DIR = ROOT_DIR / "evaluation_results_two_stage"

# Each entry: output variant name -> (source csv, source 'variant' value to filter on, or None to use all rows)
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
    # Time-boxed fine-tune trial (d2_curated_finetune/) - d2 QLoRA-tuned on rule-following
    # targets distilled from the bigger model, still behind d1. Cheap local gate: if this
    # doesn't clearly beat prompt_engineered_pipeline here, stop - don't touch the judge panel.
    "curated_finetuned_pipeline": (
        CURATED_FINETUNE_DIR / "curated_finetuned_pipeline_generation_output.csv", None,
    ),
    # "Your pipeline" already has full 300-sample results saved from the confidence-gating
    # ablation - the "argmax" variant there is byte-identical to test_full_pipeline.py's prompt.
    "prompt_engineered_pipeline": (ABLATION_DIR / "generation_output.csv", "argmax"),
    # Same official pipeline, plus retrieved real examples shown to d2 before it responds.
    "rag_pipeline": (RAG_DIR / "rag_pipeline_generation_output.csv", "rag_pipeline"),
    # P0 baselines - one model doing detect-then-respond in one pass, no separate d1 at all.
    "single_model_same_size": (
        SINGLE_MODEL_DIR / "single_model_same_size_generation_output.csv",
        "single_model_same_size",
    ),
    "single_model_bigger": (
        SINGLE_MODEL_DIR / "single_model_bigger_generation_output.csv",
        "single_model_bigger",
    ),
    # d1 swap + d2 bake-off (d1_d2_bakeoff/) - NEW fixed d1 (RoBERTa) paired with 3 candidate
    # d2 models, all zero-shot prompted the same way. Cheap local gate before the judge panel.
    "bakeoff_qwen": (BAKEOFF_DIR / "bakeoff_qwen_generation_output.csv", None),
    "bakeoff_gemma": (BAKEOFF_DIR / "bakeoff_gemma_generation_output.csv", None),
    "bakeoff_phi": (BAKEOFF_DIR / "bakeoff_phi_generation_output.csv", None),
    # Two-stage top-3 + per-emotion prompts (emotion_prompts/) - new d1, d2 picks one of d1's
    # top-3, then generates with that emotion's specific guidance.
    "two_stage_qwen": (TWO_STAGE_DIR / "two_stage_qwen_generation_output.csv", None),
    "two_stage_phi": (TWO_STAGE_DIR / "two_stage_phi_generation_output.csv", None),
    "two_stage_phi_robertalarge": (TWO_STAGE_DIR / "two_stage_phi_robertalarge_generation_output.csv", None),
}

# Corrected monolithic LoRA baseline (train_qwen2_5_monolithic_d2_fixed/). Registered only once its outputs
# exist, so every other run keeps working before it has been generated.
_FINE_TUNED_FIXED = D2_COMPARISON_DIR / "fine_tuned_fixed_generation_output.csv"
if _complete(_FINE_TUNED_FIXED):
    SOURCES["fine_tuned_fixed"] = (_FINE_TUNED_FIXED, None)

# Single-model Phi-4-mini baseline (same generator as the final system, no detector). Registered only
# once its outputs exist.
_SINGLE_PHI = SINGLE_MODEL_DIR / "single_model_phi_generation_output.csv"
if _complete(_SINGLE_PHI):
    SOURCES["single_model_phi"] = (_SINGLE_PHI, "single_model_phi")

_TWO_PASS = SINGLE_MODEL_DIR / "single_model_phi_two_pass_generation_output.csv"
if _complete(_TWO_PASS):
    SOURCES["single_model_phi_two_pass"] = (_TWO_PASS, "single_model_phi_two_pass")

# PROVISIONAL rule-adherence heuristics - same as ablation_confidence_gating/score_ablation_responses.py,
# duplicated here (not imported) to keep each results folder self-contained, matching the rest of the repo.
ACK_KEYWORDS = [
    "understand",
    "sounds",
    "hear you",
    "makes sense",
    "must be",
    "must have been",
    "i can imagine",
    "i'm sorry",
    "im sorry",
    "that sounds",
    "i can see",
    "totally understandable",
    "valid",
]


def check_rule_adherence(response_text, max_words=120):
    lowered = response_text.lower()
    ack_present = any(keyword in lowered for keyword in ACK_KEYWORDS)
    question_count = response_text.count("?")
    exactly_one_question = question_count == 1
    word_count = len(response_text.split())
    length_ok = word_count <= max_words
    return {
        "ack_present": ack_present,
        "question_count": question_count,
        "exactly_one_question": exactly_one_question,
        "word_count": word_count,
        "length_ok": length_ok,
    }


def compute_bertscore(hypotheses, references, model_type):
    from bert_score import score as bertscore_score

    precision, recall, f1 = bertscore_score(hypotheses, references, model_type=model_type, verbose=False)
    summary = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    return summary, [float(v) for v in f1]


def load_variant_rows(csv_path, filter_variant):
    df = pd.read_csv(csv_path)
    if filter_variant is not None:
        df = df[df["variant"] == filter_variant]
    return df.reset_index(drop=True)


def score_variant(df_variant, bertscore_model):
    """Rule-adherence is the PRIMARY signal - it's the one thing our decoupled design is
    actually built to guarantee, already proven (via significance testing) to be our real,
    defensible edge over an undifferentiated single model. BERTScore-F1 is kept only as a
    rough "did this collapse into garbage" sanity check, not a ranking signal - BLEU/ROUGE-L
    were dropped: they score how closely a reply matches the ONE specific human reply in the
    dataset, which is a poor fit for empathetic response generation (many different replies
    can all be genuinely good - this is a known weak spot for these metrics in open-ended
    dialogue tasks, which is exactly why the AI-judge panel and WASSA exist)."""
    hypotheses = df_variant["generated_response"].fillna("").tolist()
    references = df_variant["gold_response"].fillna("").tolist()

    rule_rows = [check_rule_adherence(text) for text in hypotheses]
    rule_df = pd.DataFrame(rule_rows)

    bertscore_summary, bertscore_f1_per_row = compute_bertscore(hypotheses, references, bertscore_model)

    per_row = df_variant.copy()
    per_row = pd.concat([per_row, rule_df], axis=1)
    per_row["bertscore_f1"] = bertscore_f1_per_row

    # QR / IB - Deliverable I.3 (interrogative-bias statistics): mean questions per reply,
    # and the fraction of replies that ask MORE than one question (the actual annoying-bot problem).
    qr_mean_questions_per_reply = float(rule_df["question_count"].mean())
    ib_fraction_more_than_one_question = float((rule_df["question_count"] > 1).mean())

    metrics = {
        "total": len(df_variant),
        "bertscore_precision": bertscore_summary["precision"],
        "bertscore_recall": bertscore_summary["recall"],
        "bertscore_f1": bertscore_summary["f1"],
        "qr_mean_questions_per_reply": qr_mean_questions_per_reply,
        "ib_fraction_more_than_one_question": ib_fraction_more_than_one_question,
        "rule_adherence": {
            "ack_rate": float(rule_df["ack_present"].mean()),
            "one_question_rate": float(rule_df["exactly_one_question"].mean()),
            "length_ok_rate": float(rule_df["length_ok"].mean()),
            "all_rules_rate": float(
                (rule_df["ack_present"] & rule_df["exactly_one_question"] & rule_df["length_ok"]).mean()
            ),
        },
    }
    return metrics, per_row


def write_comparison_table(all_metrics, path):
    ranking = sorted(all_metrics.items(), key=lambda item: item[1]["rule_adherence"]["all_rules_rate"], reverse=True)
    lines = [
        "| Rank | Setting | BERTScore-F1 (sanity check) | QR (avg Q/reply) | IB (%replies >1 Q) | Ack% | OneQ% | LenOK% | AllRules% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (variant, metrics) in enumerate(ranking, start=1):
        ra = metrics["rule_adherence"]
        lines.append(
            f"| {rank} | {variant} | {metrics['bertscore_f1']:.4f} | {metrics['qr_mean_questions_per_reply']:.2f} | "
            f"{metrics['ib_fraction_more_than_one_question'] * 100:.1f} | {ra['ack_rate'] * 100:.1f} | "
            f"{ra['one_question_rate'] * 100:.1f} | {ra['length_ok_rate'] * 100:.1f} | {ra['all_rules_rate'] * 100:.1f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score and compare zero-shot d2, monolithic fine-tuned d2, and the prompt-engineered pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=D2_COMPARISON_DIR)
    parser.add_argument("--bertscore-model", default="roberta-large")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for variant_name, (csv_path, filter_variant) in SOURCES.items():
        if not csv_path.exists():
            print(f"Skipping '{variant_name}': {csv_path} not found yet. Run its generation script first.")
            continue
        df_variant = load_variant_rows(csv_path, filter_variant)
        if df_variant.empty:
            print(f"Skipping '{variant_name}': no matching rows in {csv_path}")
            continue

        print(f"Scoring '{variant_name}' ({len(df_variant)} rows)...")
        metrics, per_row = score_variant(df_variant, args.bertscore_model)
        all_metrics[variant_name] = metrics
        per_row.to_csv(args.output_dir / f"{variant_name}_scored_predictions.csv", index=False)
        with (args.output_dir / f"{variant_name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"{variant_name}: BERTScore-F1={metrics['bertscore_f1']:.4f} QR={metrics['qr_mean_questions_per_reply']:.2f} "
            f"IB={metrics['ib_fraction_more_than_one_question'] * 100:.1f}% "
            f"AllRules%={metrics['rule_adherence']['all_rules_rate'] * 100:.1f}"
        )

    if not all_metrics:
        raise SystemExit("No settings scored - none of the expected generation CSVs were found yet.")

    with (args.output_dir / "d2_settings_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    write_comparison_table(all_metrics, args.output_dir / "d2_settings_comparison_table.md")

    missing = [name for name in SOURCES if name not in all_metrics]
    if missing:
        print(f"\nNote: table is incomplete - still missing: {', '.join(missing)}")
    print(f"\nWrote d2_settings_summary.json and d2_settings_comparison_table.md to {args.output_dir}")


if __name__ == "__main__":
    main()
