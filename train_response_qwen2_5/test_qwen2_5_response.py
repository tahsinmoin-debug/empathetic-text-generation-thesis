import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
D1_MODEL_PATH = PROJECT_ROOT / "GPT 2 medium training" / "checkpoints" / "gpt2_medium" / "final"
D2_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "qwen2_5_1_5b_emotional_response_lora" / "final"

LABELS = ["Loss", "Anger", "Fear", "Self-Conscious", "Disgust"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full D1 -> D2 pipeline: user text -> GPT-2-medium emotion/context "
            "detection -> Qwen2.5 empathetic response."
        )
    )
    parser.add_argument("--d1-model-path", default=str(D1_MODEL_PATH), help="Path to GPT-2-medium d1 detector.")
    parser.add_argument("--d2-model-path", default=str(D2_MODEL_PATH), help="Path to trained Qwen2.5 d2 LoRA adapter.")
    parser.add_argument("--text", default="", help="User text. If omitted, interactive mode is used.")
    parser.add_argument("--detector-max-length", type=int, default=192)
    parser.add_argument("--cue-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-new-tokens", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--hide-signals",
        action="store_true",
        help="Print only the response, without detected emotion/context cues.",
    )
    return parser.parse_args()


def detector_sequence(label, text):
    return f"<emotion={label}> {text}"


def build_d2_prompt(text, emotion, context_cues):
    return (
        f"User: {text}\n"
        f"Detected emotion: {emotion}\n"
        f"Context cues: {context_cues}\n"
        "Assistant:"
    )


def clean_phrase(phrase):
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase.strip(" \t\r\n\"'.,;:!?()[]{}")


def fallback_cues(text, max_cues):
    candidates = re.split(r"[.!?;]+|\s+(?:and|but|because|when|while|so)\s+", text)
    cues = []
    for candidate in candidates:
        phrase = clean_phrase(candidate)
        if 3 <= len(phrase) <= 90:
            cues.append(phrase)
        if len(cues) >= max_cues:
            break
    return "; ".join(cues) if cues else clean_phrase(text[:90])


def sequence_loss(tokenizer, model, device, text, label, max_length):
    sequence = detector_sequence(label, text)
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()

    with torch.no_grad():
        output = model(**encoded, labels=labels)

    loss = output.loss.detach().float().item()
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return loss


def predict_emotion(tokenizer, model, device, text, max_length):
    scores = {
        label: sequence_loss(tokenizer, model, device, text, label, max_length)
        for label in LABELS
    }
    return min(scores, key=scores.get), scores


def extract_context_cues(tokenizer, model, device, text, predicted_emotion, max_length, max_cues):
    prompt = detector_sequence(predicted_emotion, text)
    text_start = prompt.find(text)
    if text_start < 0:
        return fallback_cues(text, max_cues)
    text_end = text_start + len(text)

    try:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
    except TypeError:
        return fallback_cues(text, max_cues)

    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded, output_attentions=True, use_cache=False)

    if not output.attentions:
        return fallback_cues(text, max_cues)

    final_attention = output.attentions[-1][0]
    token_scores = final_attention[:, -1, :].mean(dim=0).detach().float().cpu().tolist()

    candidate_spans = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if start < text_start or end > text_end:
            continue
        phrase_piece = prompt[start:end]
        if not phrase_piece.strip() or re.fullmatch(r"\W+", phrase_piece.strip()):
            continue
        candidate_spans.append(
            {
                "start": start - text_start,
                "end": end - text_start,
                "score": token_scores[idx],
            }
        )

    if not candidate_spans:
        return fallback_cues(text, max_cues)

    candidate_spans = sorted(candidate_spans, key=lambda item: item["score"], reverse=True)
    selected = sorted(candidate_spans[: min(14, len(candidate_spans))], key=lambda item: item["start"])

    merged = []
    for span in selected:
        if not merged:
            merged.append(span.copy())
            continue
        previous = merged[-1]
        if span["start"] <= previous["end"] + 2:
            previous["end"] = max(previous["end"], span["end"])
            previous["score"] = max(previous["score"], span["score"])
        else:
            merged.append(span.copy())

    phrase_rows = []
    for span in merged:
        phrase = clean_phrase(text[span["start"] : span["end"]])
        if 3 <= len(phrase) <= 90 and not re.fullmatch(r"\W+", phrase):
            phrase_rows.append({"phrase": phrase, "score": span["score"], "start": span["start"]})

    if not phrase_rows:
        return fallback_cues(text, max_cues)

    phrase_rows = sorted(phrase_rows, key=lambda item: item["score"], reverse=True)
    chosen = []
    seen = set()
    for row in phrase_rows:
        normalized = row["phrase"].lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        chosen.append(row)
        if len(chosen) >= max_cues:
            break

    chosen = sorted(chosen, key=lambda item: item["start"])
    return "; ".join(row["phrase"] for row in chosen)


def load_d1(d1_path):
    d1_path = Path(d1_path)
    if not d1_path.exists():
        raise FileNotFoundError(f"Missing d1 GPT-2-medium detector: {d1_path}")

    tokenizer = AutoTokenizer.from_pretrained(d1_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            d1_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(d1_path, torch_dtype=dtype)

    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_d2(adapter_path):
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing trained d2 adapter: {adapter_path}. Run Qwen2.5 training first.")

    adapter_config_path = adapter_path / "adapter_config.json"
    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, text, emotion, context_cues, args):
    prompt = build_d2_prompt(text, emotion, context_cues)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_pipeline(text, d1_tokenizer, d1_model, d1_device, d2_tokenizer, d2_model, args):
    emotion, _scores = predict_emotion(
        d1_tokenizer,
        d1_model,
        d1_device,
        text,
        args.detector_max_length,
    )
    context_cues = extract_context_cues(
        d1_tokenizer,
        d1_model,
        d1_device,
        text,
        emotion,
        args.detector_max_length,
        args.cue_count,
    )
    response = generate_response(d2_tokenizer, d2_model, text, emotion, context_cues, args)
    return emotion, context_cues, response


def print_result(emotion, context_cues, response, hide_signals):
    if hide_signals:
        print(response)
        return
    print(f"Detected emotion: {emotion}")
    print(f"Context cues: {context_cues}")
    print(f"Response: {response}")


def main():
    args = parse_args()
    d1_tokenizer, d1_model, d1_device = load_d1(args.d1_model_path)
    d2_tokenizer, d2_model = load_d2(args.d2_model_path)

    if args.text:
        emotion, context_cues, response = run_pipeline(
            args.text,
            d1_tokenizer,
            d1_model,
            d1_device,
            d2_tokenizer,
            d2_model,
            args,
        )
        print_result(emotion, context_cues, response, args.hide_signals)
        return

    print("Qwen2.5 D1 -> D2 emotional response testing. Type 'quit' to exit.")
    while True:
        text = input("User text: ").strip()
        if text.lower() == "quit":
            break
        emotion, context_cues, response = run_pipeline(
            text,
            d1_tokenizer,
            d1_model,
            d1_device,
            d2_tokenizer,
            d2_model,
            args,
        )
        print_result(emotion, context_cues, response, args.hide_signals)
        print("-" * 50)


if __name__ == "__main__":
    main()
