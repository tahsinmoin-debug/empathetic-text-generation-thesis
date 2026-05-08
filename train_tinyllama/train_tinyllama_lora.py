import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


# -----------------------------
# 1. CONFIG
# -----------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATA_PATH = "../GoEmotions/goemotions_multilevel_negative.csv"
import os
print("Dataset path:", os.path.abspath(DATA_PATH))
print("File exists:", os.path.exists(DATA_PATH))
OUTPUT_DIR = "../checkpoints/tinyllama_lora"

MAX_LENGTH = 128
EPOCHS = 8
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv(DATA_PATH)

def format_row(row):
    return f"<emotion={row['level_2']}> {row['text']}"

df["input_text"] = df.apply(format_row, axis=1)

dataset = Dataset.from_pandas(df[["input_text"]])

# -----------------------------
# 3. TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["input_text"])

# -----------------------------
# 4. LOAD MODEL (4-bit)
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

# -----------------------------
# 5. LORA CONFIG
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
# 6. TRAINING ARGS
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=3,
    report_to="none",
)

# -----------------------------
# 7. TRAIN
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🚀 Starting TinyLLaMA LoRA training...")
trainer.train()

# -----------------------------
# 8. SAVE FINAL MODEL
# -----------------------------

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("✅ TinyLLaMA training complete!")
