import os
import inspect
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# 1. CONFIG
# -----------------------------

MODEL_NAME = "Qwen/Qwen2-1.5B"
DATA_PATH = "../GoEmotions/goemotions_multilevel_negative_dialogue.csv"
OUTPUT_DIR = "../checkpoints/qwen_lora"

MAX_LENGTH = 192
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv(DATA_PATH)

dataset = Dataset.from_pandas(df[["formatted_text"]])

# -----------------------------
# 3. TRAIN / VALID SPLIT
# -----------------------------

dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# -----------------------------
# 4. TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["formatted_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["formatted_text"])
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["formatted_text"])

# -----------------------------
# 5. LOAD MODEL (QLoRA)
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------------
# 6. LORA CONFIG
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 7. MEMORY OPTIMIZATION
# -----------------------------

model.gradient_checkpointing_enable()
model.config.use_cache = False

# -----------------------------
# 8. TRAINING ARGS
# -----------------------------

training_args_kwargs = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate": LR,
    "fp16": True,
    "logging_steps": 100,
    "eval_steps": 500,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "report_to": "none",
    "do_eval": True,
}

training_args_signature = inspect.signature(TrainingArguments.__init__)
if "eval_strategy" in training_args_signature.parameters:
    training_args_kwargs["eval_strategy"] = "steps"
else:
    training_args_kwargs["evaluation_strategy"] = "steps"

training_args = TrainingArguments(**training_args_kwargs)

# -----------------------------
# 9. TRAINER
# -----------------------------

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "data_collator": DataCollatorForLanguageModeling(tokenizer, mlm=False),
}

trainer_signature = inspect.signature(Trainer.__init__)
if "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)

# -----------------------------
# 10. TRAIN
# -----------------------------

print("🚀 Starting Qwen LoRA training...")
trainer.train()

# -----------------------------
# 11. SAVE MODEL
# -----------------------------

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("✅ Training complete!")
