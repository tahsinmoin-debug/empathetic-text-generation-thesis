import argparse
import csv
import gc
import json
import math
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


LABELS = ["Loss", "Anger", "Fear", "Self-Conscious", "Disgust"]


MODEL_CONFIGS = {
    "qwen2": {
        "path": Path("checkpoints/qwen_lora/final"),
        "loader": "peft",
        "template": "dialogue",
    },
    "gpt2_medium": {
        "path": Path("GPT 2 medium training/checkpoints/gpt2_medium/final"),
        "loader": "gpt2",
        "template": "tag",
    },
    "tinyllama": {
        "path": Path("checkpoints/tinyllama_lora/final"),
        "loader": "peft",
        "template": "tag",
    },
    "qwen2_5_detector": {
        "path": Path("checkpoints/qwen2_5_0_5b_emotion_detector_lora/final"),
        "loader": "peft",
        "template": "detector",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained emotion models by choosing the label with the lowest LM loss."
    )
    parser.add_argument(
        "--data",
        default="GoEmotions/goemotions_multilevel_negative.csv",
        help="CSV containing text and level_2 columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory for metrics and prediction CSV files.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=100,
        help="Balanced examples per emotion label. Use 0 for all rows.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS),
        choices=list(MODEL_CONFIGS),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU evaluation. This is much slower.",
    )
    return parser.parse_args()


def load_eval_rows(data_path, max_per_label):
    df = pd.read_csv(data_path)
    required = {"text", "level_2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing columns: {sorted(missing)}")

    df = df[df["level_2"].isin(LABELS)].copy()
    df["text"] = df["text"].astype(str)
    df["level_2"] = df["level_2"].astype(str)

    if max_per_label and max_per_label > 0:
        parts = []
        for label in LABELS:
            label_df = df[df["level_2"] == label]
            parts.append(label_df.sample(n=min(max_per_label, len(label_df)), random_state=42))
        df = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df[["text", "level_2"]]


def make_prompt(template, label, text):
    if template == "detector":
        return (
            "Classify the emotion of the text into one of these labels: "
            "Loss, Anger, Fear, Self-Conscious, Disgust.\n"
            f"Text: {text}\n"
            f"Emotion: {label}"
        )
    if template == "dialogue":
        return f"Emotion: {label}\nUser: {text}"
    return f"<emotion={label}> {text}"


def make_detector_prompt(text):
    return (
        "Classify the emotion of the text into one of these labels: "
        "Loss, Anger, Fear, Self-Conscious, Disgust.\n"
        f"Text: {text}\n"
        "Emotion:"
    )


def load_model(name, config, force_cpu):
    path = str(config["path"])
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing model path for {name}: {path}")

    if name == "gpt2_medium":
        tokenizer = GPT2Tokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(path)
        device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        model.to(device)
    else:
        adapter_config_path = Path(path) / "adapter_config.json"
        with adapter_config_path.open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        tokenizer_source = adapter_config.get("base_model_name_or_path", path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"trust_remote_code": True}
        if force_cpu or not torch.cuda.is_available():
            kwargs["device_map"] = {"": "cpu"}
        else:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(tokenizer_source, **kwargs)
        model = PeftModel.from_pretrained(base_model, path)
        device = next(model.parameters()).device

    model.eval()
    return tokenizer, model, device


def sequence_loss(tokenizer, model, device, text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()

    with torch.no_grad():
        output = model(**encoded, labels=labels)

    loss = output.loss.detach().float().item()
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return loss


def detector_label_loss(tokenizer, model, device, text, label):
    prompt = make_detector_prompt(text)
    label_text = f" {label}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    label_ids = tokenizer(label_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    input_ids = torch.cat([prompt_ids, label_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[0]] = -100

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = output.loss.detach().float().item()
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    return loss


def predict_label(tokenizer, model, device, template, text):
    scores = {}
    for label in LABELS:
        if template == "detector":
            scores[label] = detector_label_loss(tokenizer, model, device, text, label)
        else:
            prompt = make_prompt(template, label, text)
            scores[label] = sequence_loss(tokenizer, model, device, prompt)
    prediction = min(scores, key=scores.get)
    return prediction, scores


def empty_confusion():
    return {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}


def summarize(predictions):
    total = len(predictions)
    correct = sum(row["actual"] == row["predicted"] for row in predictions)
    confusion = empty_confusion()
    per_label = {}

    for row in predictions:
        confusion[row["actual"]][row["predicted"]] += 1

    for label in LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in LABELS)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = sum(stats["f1"] for stats in per_label.values()) / len(LABELS)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def write_predictions(path, predictions):
    score_fields = [f"score_{label}" for label in LABELS]
    fields = ["text", "actual", "predicted", "correct", *score_fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    rows = load_eval_rows(args.data, args.max_per_label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    print(f"Evaluating {len(rows)} rows from {args.data}")

    for model_name in args.models:
        config = MODEL_CONFIGS[model_name]
        print(f"\nLoading {model_name} from {config['path']}")
        tokenizer, model, device = load_model(model_name, config, args.cpu)

        predictions = []
        for index, row in rows.iterrows():
            predicted, scores = predict_label(
                tokenizer, model, device, config["template"], row["text"]
            )
            prediction_row = {
                "text": row["text"],
                "actual": row["level_2"],
                "predicted": predicted,
                "correct": row["level_2"] == predicted,
            }
            for label in LABELS:
                prediction_row[f"score_{label}"] = scores[label]
            predictions.append(prediction_row)

            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"{model_name}: {index + 1}/{len(rows)}")

        metrics = summarize(predictions)
        all_metrics[model_name] = metrics
        write_predictions(output_dir / f"{model_name}_predictions.csv", predictions)
        with (output_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        unload(model)

    ranking = sorted(all_metrics.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    print("\nRanking by accuracy")
    for rank, (model_name, metrics) in enumerate(ranking, start=1):
        print(
            f"{rank}. {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, correct={metrics['correct']}/{metrics['total']}"
        )


if __name__ == "__main__":
    main()
