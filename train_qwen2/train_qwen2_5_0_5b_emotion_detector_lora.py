import inspect
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# -----------------------------
# 1. CONFIG
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = PROJECT_ROOT / "GoEmotions" / "goemotions_multilevel_negative.csv"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "qwen2_5_0_5b_emotion_detector_lora"
SPLIT_DIR = PROJECT_ROOT / "GoEmotions" / "splits_qwen2_5_detector"

LABELS = ["Loss", "Anger", "Fear", "Self-Conscious", "Disgust"]

MAX_LENGTH = 192
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
BALANCE_TRAIN = True
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


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


def build_training_text(text, label):
    return f"{build_prompt(text)} {label}"


# -----------------------------
# 3. LOAD AND SPLIT DATASET
# -----------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required_columns = {"text", "level_2"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in {DATA_PATH}: {sorted(missing_columns)}")

df["level_2"] = df["level_2"].astype(str).str.strip()
df = df[df["level_2"].isin(LABELS)].copy()
if df.empty:
    raise ValueError(
        "No matching Level-2 labels found. Expected one of: "
        + ", ".join(LABELS)
    )

df["text"] = df["text"].astype(str)

train_parts = []
dev_parts = []
test_parts = []

for label in LABELS:
    label_df = df[df["level_2"] == label].sample(frac=1.0, random_state=SEED)
    train_end = int(len(label_df) * TRAIN_RATIO)
    dev_end = train_end + int(len(label_df) * DEV_RATIO)

    train_parts.append(label_df.iloc[:train_end])
    dev_parts.append(label_df.iloc[train_end:dev_end])
    test_parts.append(label_df.iloc[dev_end:])

train_df = pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
dev_df = pd.concat(dev_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

if BALANCE_TRAIN:
    min_count = train_df["level_2"].value_counts().min()
    balanced_parts = []
    for label in LABELS:
        label_train_df = train_df[train_df["level_2"] == label]
        balanced_parts.append(label_train_df.sample(n=min_count, random_state=SEED))
    train_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

train_df.to_csv(Path(SPLIT_DIR) / "train.csv", index=False)
dev_df.to_csv(Path(SPLIT_DIR) / "dev.csv", index=False)
test_df.to_csv(Path(SPLIT_DIR) / "test.csv", index=False)

print("Train label counts:")
print(train_df["level_2"].value_counts().to_string())
print("\nDev label counts:")
print(dev_df["level_2"].value_counts().to_string())
print("\nTest label counts:")
print(test_df["level_2"].value_counts().to_string())


# -----------------------------
# 4. TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    prompt = build_prompt(example["text"])
    label_text = f" {example['level_2']}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]

    max_prompt_length = MAX_LENGTH - len(label_ids)
    if max_prompt_length < 1:
        raise ValueError("MAX_LENGTH is too small to fit the emotion label.")

    prompt_ids = prompt_ids[:max_prompt_length]
    input_ids = prompt_ids + label_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + label_ids.copy()

    pad_length = MAX_LENGTH - len(input_ids)
    if pad_length > 0:
        input_ids += [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length
        labels += [-100] * pad_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = Dataset.from_pandas(train_df[["text", "level_2"]])
dev_dataset = Dataset.from_pandas(dev_df[["text", "level_2"]])

train_dataset = train_dataset.map(tokenize, remove_columns=["text", "level_2"])
dev_dataset = dev_dataset.map(tokenize, remove_columns=["text", "level_2"])


# -----------------------------
# 5. LOAD MODEL WITH QLORA
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()


# -----------------------------
# 6. TRAINING ARGS
# -----------------------------

training_args_kwargs = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate": LR,
    "fp16": True,
    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 3,
    "report_to": "none",
    "do_eval": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

training_args_signature = inspect.signature(TrainingArguments.__init__)
if "eval_strategy" in training_args_signature.parameters:
    training_args_kwargs["eval_strategy"] = "steps"
else:
    training_args_kwargs["evaluation_strategy"] = "steps"

training_args = TrainingArguments(**training_args_kwargs)


# -----------------------------
# 7. TRAIN
# -----------------------------

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": dev_dataset,
}

trainer_signature = inspect.signature(Trainer.__init__)
if "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)

print("Starting Qwen2.5-0.5B emotion detector training...")
trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Training complete!")
print(f"Final model saved to: {OUTPUT_DIR}/final")
print(f"Reusable split files saved to: {SPLIT_DIR}")
