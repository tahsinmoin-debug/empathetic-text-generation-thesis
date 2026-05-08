import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# -----------------------------
# 1. CONFIG
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "qwen2_5_0_5b_emotion_detector_lora" / "final"
LABELS = ["Loss", "Anger", "Fear", "Self-Conscious", "Disgust"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 2. PROMPT FORMAT
# -----------------------------

def build_prompt(text):
    return (
        "Classify the emotion of the text into one of these labels: "
        "Loss, Anger, Fear, Self-Conscious, Disgust.\n"
        f"Text: {text}\n"
        "Emotion:"
    )


def normalize_prediction(output_text):
    match = re.search(
        r"\b(Loss|Anger|Fear|Self-Conscious|Disgust)\b",
        output_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return "Unknown"

    raw_label = match.group(1).lower()
    for label in LABELS:
        if raw_label == label.lower():
            return label
    return "Unknown"


# -----------------------------
# 3. LOAD TOKENIZER AND MODEL
# -----------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

with (MODEL_PATH / "adapter_config.json").open("r", encoding="utf-8") as handle:
    adapter_config = json.load(handle)

base_model_name = adapter_config["base_model_name_or_path"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {"trust_remote_code": True}
if torch.cuda.is_available():
    model_kwargs["device_map"] = "auto"
    model_kwargs["torch_dtype"] = torch.float16
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
else:
    model_kwargs["device_map"] = {"": "cpu"}

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("Qwen2.5-0.5B emotion detector loaded successfully")


# -----------------------------
# 4. PREDICTION
# -----------------------------

def predict_emotion(text, max_new_tokens=8):
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    predicted_label = normalize_prediction(generated_text)
    return predicted_label, generated_text


# -----------------------------
# 5. INTERACTIVE TESTING LOOP
# -----------------------------

print("\nQwen2.5-0.5B Emotion Detection Testing")
print("Labels: Loss, Anger, Fear, Self-Conscious, Disgust")
print("Type 'quit' to exit\n")

while True:
    text = input("User text: ").strip()
    if text.lower() == "quit":
        break

    label, raw_output = predict_emotion(text)
    print(f"\nPredicted emotion: {label}")
    print(f"Raw model output: {raw_output}")
    print("-" * 60)
