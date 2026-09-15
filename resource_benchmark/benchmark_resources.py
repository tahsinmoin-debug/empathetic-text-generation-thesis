"""Resource benchmark: GPU memory and time per reply for the final system and the single-model baselines.

Supports the "resource efficient" claim with measured numbers instead of parameter counts alone.
The final system runs two models and calls Phi-4-mini twice per reply (stage 1 pick + stage 2 reply),
so fewer parameters does not automatically mean faster - this script measures it.

Fairness rules built in:
  * Each system reuses the EXACT generation code and prompts of its evaluated run (imported, not copied).
  * Same conversations for every system: the first N of the evaluation sample (seed 42).
  * Same decoding settings as the evaluation (temperature 0.7, top-p 0.9, max 250 new tokens).
  * ONE system per run (separate Python process), so peak-memory readings are not mixed between models.
  * Warm-up messages (default 3) are run first and excluded from timing.
  * Writes only to resource_benchmark/results/ - never touches the judged evaluation outputs.

Usage (run each line separately, one after another):
    python benchmark_resources.py --system two_stage_phi
    python benchmark_resources.py --system single_model_bigger
    python benchmark_resources.py --system single_model_same_size
    python summarize_resources.py
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(ROOT_DIR / "emotion_prompts"))
sys.path.insert(0, str(ROOT_DIR / "single_model_baseline"))

import generate_two_stage_pipeline as tsp  # noqa: E402
from attention_cues import assert_attention_available, extract_context_cues, load_d1_with_attention  # noqa: E402
from emotion_prompts import build_emotion_prompt  # noqa: E402

# Nominal parameter counts from the model cards (4-bit loading packs weights, so numel() would under-count).
SYSTEMS = {
    "two_stage_phi": {"label": "Final system: RoBERTa-base + Phi-4-mini (4-bit), two-stage",
                      "params_b": 0.125 + 3.84},
    "single_model_bigger": {"label": "Single model: Qwen2.5-7B-Instruct (4-bit)", "params_b": 7.62},
    "single_model_same_size": {"label": "Single model: Qwen2.5-1.5B-Instruct (fp16)", "params_b": 1.54},
}


def parse_args():
    p = argparse.ArgumentParser(description="Measure GPU memory and time per reply for one system.")
    p.add_argument("--system", choices=list(SYSTEMS), required=True)
    p.add_argument("--num-messages", type=int, default=50, help="Timed messages (after warm-up).")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warm-up messages.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=250)
    return p.parse_args()


def gpu_used_mb():
    """Whole-GPU memory in use as reported by the driver (includes other programs), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mb(x):
    return round(x / 1024 ** 2, 1)


def load_messages(n, seed):
    pairs = tsp.extract_test_pairs(tsp.TEST_CSV)
    pairs = pairs.sample(n=min(300, len(pairs)), random_state=seed).reset_index(drop=True)  # evaluation sample
    return pairs.head(n)


def load_4bit(name):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")


def build_system(name, device):
    """Load exactly what the evaluated run loaded. Returns a respond(text) -> (reply, timings, tokenizer)."""
    if name == "two_stage_phi":
        d1_tok = AutoTokenizer.from_pretrained(tsp.D1_DIR)
        d1 = load_d1_with_attention(tsp.D1_DIR).to(device)
        assert_attention_available(d1, d1_tok, device)
        id2label = tsp.get_id2label(d1)
        cand = tsp.D2_CANDIDATES["phi"]
        tok = AutoTokenizer.from_pretrained(cand["model_name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_4bit(cand["model_name"])
        model.eval()
        terms = tsp.build_terminators(tok)

        def respond(text, max_new_tokens):
            t = {}
            sync(); t0 = time.perf_counter()
            candidates = tsp.predict_d1_topk(text, d1_tok, d1, id2label, device)
            cues = extract_context_cues(text, d1_tok, d1, device)
            sync(); t1 = time.perf_counter()
            picked, _, _ = tsp.stage1_pick_emotion(text, candidates, tok, model, terms)
            sync(); t2 = time.perf_counter()
            reply = tsp.generate(build_emotion_prompt(picked, context_cues=cues), text, tok, model, terms,
                                 max_new_tokens=max_new_tokens, do_sample=True)
            sync(); t3 = time.perf_counter()
            t.update(d1_and_cues_s=t1 - t0, stage1_pick_s=t2 - t1, stage2_reply_s=t3 - t2)
            return reply, t, tok

        return respond

    if name == "single_model_bigger":
        import generate_bigger_single_model as base
        tok = AutoTokenizer.from_pretrained(base.BIG_MODEL_NAME)
        model = load_4bit(base.BIG_MODEL_NAME)
    else:
        import generate_same_size_single_model as base
        tok = AutoTokenizer.from_pretrained(base.D2_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(base.D2_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    def respond(text, max_new_tokens):
        sync(); t0 = time.perf_counter()
        reply = base.generate_response(base.SYSTEM_PROMPT, text, tok, model, max_new_tokens=max_new_tokens)
        sync()
        return reply, {"stage2_reply_s": time.perf_counter() - t0}, tok

    return respond


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. This benchmark measures GPU memory and must run on the GPU.")
    device = "cuda"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = SYSTEMS[args.system]

    gpu_before = gpu_used_mb()
    print(f"GPU: {torch.cuda.get_device_name(0)} | memory already in use by other programs: {gpu_before} MB")
    if gpu_before and gpu_before > 1500:
        print("WARNING: other programs are using more than 1.5 GB of GPU memory. Close browsers/games for "
              "clean numbers (the run will continue).")

    msgs = load_messages(args.warmup + args.num_messages, args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    print(f"Loading {info['label']} ...")
    t_load = time.perf_counter()
    respond = build_system(args.system, device)
    sync()
    load_s = time.perf_counter() - t_load
    after_load_alloc = torch.cuda.memory_allocated()
    after_load_reserved = torch.cuda.memory_reserved()
    print(f"Loaded in {load_s:.1f} s | model memory: {mb(after_load_alloc)} MB allocated")

    rows = []
    for i, row in enumerate(msgs.itertuples()):
        warm = i < args.warmup
        sync(); t0 = time.perf_counter()
        reply, parts, tok = respond(row.user_text, args.max_new_tokens)
        sync(); total = time.perf_counter() - t0
        new_tokens = len(tok(reply, add_special_tokens=False).input_ids)
        if warm:
            print(f"warm-up {i + 1}/{args.warmup} done ({total:.1f} s)")
            continue
        rows.append({"system": args.system, "conv_id": row.conv_id, "total_s": round(total, 4),
                     **{k: round(v, 4) for k, v in parts.items()},
                     "reply_tokens": new_tokens, "reply_words": len(reply.split()),
                     "tokens_per_s_reply": round(new_tokens / parts["stage2_reply_s"], 2)
                     if parts["stage2_reply_s"] > 0 else None})
        n = len(rows)
        if n % 10 == 0 or n == args.num_messages:
            print(f"{n}/{args.num_messages} timed messages | last {total:.1f} s")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{args.system}_per_message.csv", index=False)
    summary = {
        "system": args.system, "label": info["label"], "params_b": round(info["params_b"], 2),
        "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "os": platform.platform(),
        "timed_messages": len(df), "warmup_messages": args.warmup, "seed": args.seed,
        "load_time_s": round(load_s, 1),
        "model_memory_mb": mb(after_load_alloc),
        "model_memory_reserved_mb": mb(after_load_reserved),
        "peak_memory_allocated_mb": mb(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_mb": mb(torch.cuda.max_memory_reserved()),
        "gpu_used_by_others_before_mb": gpu_before, "gpu_used_total_at_end_mb": gpu_used_mb(),
        "mean_s_per_reply": round(df["total_s"].mean(), 3),
        "median_s_per_reply": round(df["total_s"].median(), 3),
        "p90_s_per_reply": round(df["total_s"].quantile(0.9), 3),
        "mean_reply_tokens": round(df["reply_tokens"].mean(), 1),
        "mean_tokens_per_s_reply": round(df["tokens_per_s_reply"].mean(), 2),
    }
    for part in ("d1_and_cues_s", "stage1_pick_s", "stage2_reply_s"):
        if part in df:
            summary[f"mean_{part}"] = round(df[part].mean(), 3)
    (RESULTS_DIR / f"{args.system}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for k in ("label", "model_memory_mb", "peak_memory_allocated_mb", "mean_s_per_reply",
              "median_s_per_reply", "mean_reply_tokens", "mean_tokens_per_s_reply"):
        print(f"{k:28s} {summary[k]}")
    print("=" * 60)
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
