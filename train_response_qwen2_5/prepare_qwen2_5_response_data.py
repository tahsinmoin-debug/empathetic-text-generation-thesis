import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "GoEmotions" / "goemotions_multilevel_negative_dialogue.csv"
D1_MODEL_PATH = PROJECT_ROOT / "GPT 2 medium training" / "checkpoints" / "gpt2_medium" / "final"
OUTPUT_DIR = PROJECT_ROOT / "GoEmotions" / "response_generation_qwen2_5"

LABELS = ["Loss", "Anger", "Fear", "Self-Conscious", "Disgust"]
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare context-aware response-generation data for Qwen2.5 D2 training."
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="Input dialogue CSV.")
    parser.add_argument("--d1-model", default=str(D1_MODEL_PATH), help="Trained GPT-2-medium detector path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory for prepared CSVs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke testing. 0 means all rows.")
    parser.add_argument("--max-length", type=int, default=192, help="Maximum detector prompt length.")
    parser.add_argument("--cue-count", type=int, default=4, help="Maximum context cue phrases per row.")
    parser.add_argument("--flush-every", type=int, default=250, help="Progress print interval.")
    return parser.parse_args()


def detector_sequence(label, text):
    return f"<emotion={label}> {text}"


def extract_assistant_response(formatted_text):
    if not isinstance(formatted_text, str):
        return ""
    marker = "Assistant:"
    if marker not in formatted_text:
        return ""
    return formatted_text.split(marker, 1)[1].strip()


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


def clean_phrase(phrase):
    phrase = re.sub(r"\s+", " ", phrase).strip()
    phrase = phrase.strip(" \t\r\n\"'.,;:!?()[]{}")
    return phrase


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
                "idx": idx,
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


def build_training_text(text, predicted_emotion, context_cues, assistant_response):
    return (
        f"User: {text}\n"
        f"Detected emotion: {predicted_emotion}\n"
        f"Context cues: {context_cues}\n"
        f"Assistant: {assistant_response}"
    )


def save_splits(df, output_dir):
    train_parts = []
    dev_parts = []
    test_parts = []

    for label in LABELS:
        label_df = df[df["gold_emotion"] == label].sample(frac=1.0, random_state=SEED)
        train_end = int(len(label_df) * TRAIN_RATIO)
        dev_end = train_end + int(len(label_df) * DEV_RATIO)
        train_parts.append(label_df.iloc[:train_end])
        dev_parts.append(label_df.iloc[train_end:dev_end])
        test_parts.append(label_df.iloc[dev_end:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    dev_df = pd.concat(dev_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    dev_df.to_csv(output_dir / "dev.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    test_df.to_csv(output_dir / "test.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved train/dev/test: {len(train_df)}/{len(dev_df)}/{len(test_df)}")


def main():
    args = parse_args()
    data_path = Path(args.data)
    d1_model_path = Path(args.d1_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing input data: {data_path}")
    if not d1_model_path.exists():
        raise FileNotFoundError(f"Missing d1 model: {d1_model_path}")

    df = pd.read_csv(data_path)
    required = {"text", "level_2", "level_3", "formatted_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing columns: {sorted(missing)}")

    df = df[df["level_2"].isin(LABELS)].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["gold_emotion"] = df["level_2"].astype(str).str.strip()
    df["assistant_response"] = df["formatted_text"].apply(extract_assistant_response)
    df = df[(df["text"] != "") & (df["assistant_response"] != "")].reset_index(drop=True)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    print(f"Loaded {len(df)} usable examples from {data_path}")
    print(f"Loading d1 detector from {d1_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(d1_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            d1_model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            d1_model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
    model.to(device)
    model.eval()

    prepared_rows = []
    for idx, row in df.iterrows():
        text = row["text"]
        predicted_emotion, scores = predict_emotion(tokenizer, model, device, text, args.max_length)
        context_cues = extract_context_cues(
            tokenizer,
            model,
            device,
            text,
            predicted_emotion,
            args.max_length,
            args.cue_count,
        )
        training_text = build_training_text(
            text,
            predicted_emotion,
            context_cues,
            row["assistant_response"],
        )

        prepared_rows.append(
            {
                "text": text,
                "gold_emotion": row["gold_emotion"],
                "level_3": row["level_3"],
                "predicted_emotion": predicted_emotion,
                "context_cues": context_cues,
                "assistant_response": row["assistant_response"],
                "training_text": training_text,
                **{f"d1_loss_{label}": scores[label] for label in LABELS},
            }
        )

        if args.flush_every and (idx + 1) % args.flush_every == 0:
            print(f"Prepared {idx + 1}/{len(df)} examples")

    prepared_df = pd.DataFrame(prepared_rows)
    prepared_path = output_dir / "prepared_response_data.csv"
    prepared_df.to_csv(prepared_path, index=False, quoting=csv.QUOTE_MINIMAL)
    save_splits(prepared_df, output_dir)

    print(f"Prepared data saved to: {prepared_path}")
    print(f"Split files saved to: {output_dir}")


if __name__ == "__main__":
    main()
