import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# -----------------------------
# 1. CONFIGURATION
# -----------------------------

MODEL_NAME = "gpt2-medium"
DATA_PATH = "../data/goemotions_multilevel_negative.csv"
OUTPUT_DIR = "../checkpoints/gpt2_medium"

MAX_LENGTH = 128
EPOCHS = 30                 # high epochs, but safe
BATCH_SIZE = 2              # fits RTX 4060
GRAD_ACCUM = 8              # effective batch = 16
LEARNING_RATE = 5e-5
SAVE_EVERY_EPOCHS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} samples")

# -----------------------------
# 3. EMOPREPEND FORMATTING
# -----------------------------

def format_example(row):
    return f"<emotion={row['level_2']}> {row['text']}"

df["input_text"] = df.apply(format_example, axis=1)

dataset = Dataset.from_pandas(df[["input_text"]])

# -----------------------------
# 4. TOKENIZER
# -----------------------------

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["input_text"])

# -----------------------------
# 5. MODEL
# -----------------------------

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

model.train()

# -----------------------------
# 6. DATA COLLATOR
# -----------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 7. TRAINING ARGUMENTS
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,
    fp16=True,

    logging_steps=100,
    save_strategy="epoch",
    save_steps=None,
    save_total_limit=5,

    report_to="none",

    warmup_steps=500,
    weight_decay=0.01,

    dataloader_pin_memory=True,
)

# -----------------------------
# 8. TRAINER
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -----------------------------
# 9. TRAIN
# -----------------------------

print("🚀 Starting GPT-2 Medium training...")
trainer.train(resume_from_checkpoint="../checkpoints/gpt2_medium/checkpoint-16800")

print("✅ Training complete!")

# -----------------------------
# 10. SAVE FINAL MODEL
# -----------------------------

trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("🎉 Final model saved!")
