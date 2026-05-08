import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = PROJECT_ROOT / "GoEmotions" / "response_generation_qwen2_5"
TRAIN_PATH = DATA_DIR / "train.csv"
DEV_PATH = DATA_DIR / "dev.csv"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "qwen2_5_1_5b_emotional_response_lora"

MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or resume Qwen2.5-1.5B D2 response QLoRA.")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint path to resume from, for example checkpoints\\qwen2_5_1_5b_emotional_response_lora\\checkpoint-750.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default the script uses cached local files only.",
    )
    return parser.parse_args()


def build_prompt(text, predicted_emotion, context_cues):
    return (
        f"User: {text}\n"
        f"Detected emotion: {predicted_emotion}\n"
        f"Context cues: {context_cues}\n"
        "Assistant:"
    )


def load_split(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python train_response_qwen2_5\\prepare_qwen2_5_response_data.py"
        )

    df = pd.read_csv(path)
    required = {"text", "predicted_emotion", "context_cues", "assistant_response"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for column in required:
        df[column] = df[column].fillna("").astype(str)

    df = df[(df["text"].str.strip() != "") & (df["assistant_response"].str.strip() != "")]
    return df.reset_index(drop=True)


args = parse_args()
local_files_only = not args.allow_download

print(f"Loading training data from {TRAIN_PATH}")
train_df = load_split(TRAIN_PATH)
dev_df = load_split(DEV_PATH)
print(f"Train examples: {len(train_df)}")
print(f"Dev examples: {len(dev_df)}")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
    local_files_only=local_files_only,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def tokenize(example):
    prompt = build_prompt(
        example["text"],
        example["predicted_emotion"],
        example["context_cues"],
    )
    response = f" {example['assistant_response']}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    max_prompt_length = MAX_LENGTH - len(response_ids)
    if max_prompt_length < 1:
        response_ids = response_ids[: MAX_LENGTH - 1]
        max_prompt_length = 1

    prompt_ids = prompt_ids[:max_prompt_length]
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids.copy()

    pad_length = MAX_LENGTH - len(input_ids)
    if pad_length > 0:
        input_ids += [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length
        labels += [-100] * pad_length
    else:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = Dataset.from_pandas(
    train_df[["text", "predicted_emotion", "context_cues", "assistant_response"]]
)
dev_dataset = Dataset.from_pandas(
    dev_df[["text", "predicted_emotion", "context_cues", "assistant_response"]]
)

train_dataset = train_dataset.map(
    tokenize,
    remove_columns=["text", "predicted_emotion", "context_cues", "assistant_response"],
)
dev_dataset = dev_dataset.map(
    tokenize,
    remove_columns=["text", "predicted_emotion", "context_cues", "assistant_response"],
)


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
    local_files_only=local_files_only,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()


training_args_kwargs = {
    "output_dir": str(OUTPUT_DIR),
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate": LR,
    "fp16": True,
    "logging_steps": 25,
    "eval_steps": 250,
    "save_steps": 250,
    "save_total_limit": 10,
    "report_to": "none",
    "do_eval": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "seed": SEED,
    "gradient_checkpointing": True,
}

training_args_signature = inspect.signature(TrainingArguments.__init__)
if "eval_strategy" in training_args_signature.parameters:
    training_args_kwargs["eval_strategy"] = "steps"
else:
    training_args_kwargs["evaluation_strategy"] = "steps"

training_args = TrainingArguments(**training_args_kwargs)

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

print("Starting Qwen2.5-1.5B emotional response QLoRA training...")
if args.resume_from_checkpoint:
    resume_path = Path(args.resume_from_checkpoint)
    if not resume_path.is_absolute():
        resume_path = PROJECT_ROOT / resume_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    print(f"Resuming from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=str(resume_path))
else:
    trainer.train()

final_dir = OUTPUT_DIR / "final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print("Training complete.")
print(f"Final adapter saved to: {final_dir}")
