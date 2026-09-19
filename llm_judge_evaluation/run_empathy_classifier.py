"""Scores every generated reply (all settings in SOURCES) with the WASSA 2024
Track 2 empathy classifier (DebertaForSequenceClassification trained on third-party empathy
ratings of conversation turns; 11 ordinal output classes, of which our replies receive 1-7).

Usage:
    python run_empathy_classifier.py
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

THESIS_ROOT = Path(__file__).resolve().parent.parent

def _complete(path, expected=300):
    """A generation output counts as registered only when it holds the full evaluation sample.
    A short smoke-test file would otherwise shrink the conversations shared by all settings."""
    try:
        return path.exists() and len(pd.read_csv(path)) >= expected
    except Exception:
        return False


SOURCES = {
    "zero_shot": (
        THESIS_ROOT / "evaluation_results_d2_settings_comparison" / "zero_shot_generation_output.csv",
        None,
    ),
    "fine_tuned": (
        THESIS_ROOT / "evaluation_results_d2_settings_comparison" / "fine_tuned_generation_output.csv",
        None,
    ),
    "prompt_engineered_pipeline": (
        THESIS_ROOT / "evaluation_results_ablation_confidence_gating" / "generation_output.csv",
        "argmax",
    ),
    "rag_pipeline": (
        THESIS_ROOT / "evaluation_results_rag_pipeline" / "rag_pipeline_generation_output.csv",
        "rag_pipeline",
    ),
    "single_model_same_size": (
        THESIS_ROOT / "evaluation_results_single_model_baseline" / "single_model_same_size_generation_output.csv",
        "single_model_same_size",
    ),
    "single_model_bigger": (
        THESIS_ROOT / "evaluation_results_single_model_baseline" / "single_model_bigger_generation_output.csv",
        "single_model_bigger",
    ),
    # d1 swap + d2 bake-off (new RoBERTa d1, old generic prompt)
    "bakeoff_qwen": (
        THESIS_ROOT / "evaluation_results_d1_d2_bakeoff" / "bakeoff_qwen_generation_output.csv", None,
    ),
    "bakeoff_gemma": (
        THESIS_ROOT / "evaluation_results_d1_d2_bakeoff" / "bakeoff_gemma_generation_output.csv", None,
    ),
    "bakeoff_phi": (
        THESIS_ROOT / "evaluation_results_d1_d2_bakeoff" / "bakeoff_phi_generation_output.csv", None,
    ),
    # Two-stage top-3 + per-emotion prompts + attention context cues
    "two_stage_qwen": (
        THESIS_ROOT / "evaluation_results_two_stage" / "two_stage_qwen_generation_output.csv", None,
    ),
    "two_stage_phi": (
        THESIS_ROOT / "evaluation_results_two_stage" / "two_stage_phi_generation_output.csv", None,
    ),
    # Detector sensitivity run: identical final pipeline, roberta-large as d1
    "two_stage_phi_robertalarge": (
        THESIS_ROOT / "evaluation_results_two_stage" / "two_stage_phi_robertalarge_generation_output.csv", None,
    ),
}

# Corrected monolithic LoRA baseline (train_qwen2_5_monolithic_d2_fixed/). Registered only once its outputs
# exist, so every other run keeps working before it has been generated.
_FINE_TUNED_FIXED = THESIS_ROOT / "evaluation_results_d2_settings_comparison" / "fine_tuned_fixed_generation_output.csv"
if _complete(_FINE_TUNED_FIXED):
    SOURCES["fine_tuned_fixed"] = (_FINE_TUNED_FIXED, None)

# Single-model Phi-4-mini baseline (same generator as the final system, no detector). Registered only
# once its outputs exist.
_SINGLE_PHI = THESIS_ROOT / "evaluation_results_single_model_baseline" / "single_model_phi_generation_output.csv"
if _complete(_SINGLE_PHI):
    SOURCES["single_model_phi"] = (_SINGLE_PHI, "single_model_phi")

WASSA_REPO_ID = "hyy-33/hyy33-WASSA-2024-Track-2"
WASSA_FILENAME = "empathy-class.pth"
BASE_TOKENIZER = "microsoft/deberta-base"


def load_all_rows(only=None):
    frames = []
    for setting, (path, variant_filter) in SOURCES.items():
        if only and setting not in only:
            continue
        df = pd.read_csv(path)
        if variant_filter is not None:
            df = df[df["variant"] == variant_filter].copy()
        subset = df[["conv_id", "generated_response"]].copy()
        subset["setting"] = setting
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def load_model(device):
    # transformers 5.x removed DeBERTa internals (StableDropout/XDropout/XSoftmax) that this
    # fully-pickled checkpoint references by name - without re-injecting them, torch.load fails
    # outright. See deberta_compat.py for why the shims are faithful, not arbitrary stubs.
    import deberta_compat

    injected = deberta_compat.install()
    if injected:
        print(f"Re-injected removed DeBERTa classes for checkpoint compatibility: {', '.join(injected)}")

    checkpoint_path = hf_hub_download(repo_id=WASSA_REPO_ID, filename=WASSA_FILENAME)
    pickled = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Don't use the unpickled object directly: its CONFIG is also stale (transformers 5.x expects
    # private attrs like _output_attentions that the old pickled DebertaConfig lacks, so the
    # forward pass crashes). Transplanting the weights into a freshly-built model avoids every
    # stale-object issue at once, rather than patching cascading incompatibilities one by one.
    # Verified: reproduces the pre-upgrade scores with 100% exact match across all 1,800 replies from
    # 6 settings (vs wassa_empathy_scores_BACKUP_pre_upgrade.csv).
    from transformers import DebertaConfig, DebertaForSequenceClassification

    state_dict = pickled.state_dict()
    num_labels = state_dict["classifier.weight"].shape[0]
    config = DebertaConfig.from_pretrained(BASE_TOKENIZER, num_labels=num_labels)
    model = DebertaForSequenceClassification(config)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"WASSA checkpoint failed to load cleanly - missing weights: {list(missing)[:5]}")

    model.to(device)
    model.eval()
    return model


def score_batch(model, tokenizer, texts, device):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="For smoke-testing: only score the first N rows")
    parser.add_argument("--settings", type=str, default=None,
                        help="Comma-separated settings to score. Their rows are merged into the existing "
                             "wassa_empathy_scores.csv; every other setting's scores are kept exactly as they are.")
    args = parser.parse_args()
    only = None
    if args.settings:
        only = {x.strip() for x in args.settings.split(",") if x.strip()}
        unknown = only - set(SOURCES)
        if unknown:
            raise SystemExit(f"Unknown setting(s): {sorted(unknown)}. Known: {sorted(SOURCES)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_all_rows(only)
    if args.limit is not None:
        df = df.groupby("setting", group_keys=False).head(args.limit).reset_index(drop=True)
    print(f"Scoring {len(df)} replies with the WASSA 2024 empathy classifier")

    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = load_model(device)

    scores = []
    texts = df["generated_response"].fillna("").tolist()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        scores.extend(score_batch(model, tokenizer, batch, device))
        done = min(start + args.batch_size, len(texts))
        if done % (args.batch_size * 5) == 0 or done == len(texts):
            print(f"  {done}/{len(texts)} scored")

    df["wassa_empathy_0_10"] = scores
    out_path = output_dir / "wassa_empathy_scores.csv"
    result = df[["conv_id", "setting", "wassa_empathy_0_10"]]
    if only and out_path.exists():
        kept = pd.read_csv(out_path)
        kept = kept[~kept["setting"].isin(only)]
        result = pd.concat([kept, result], ignore_index=True)
        print(f"Merged: kept {len(kept)} existing scores, added {len(df)} new ones")
    result.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
