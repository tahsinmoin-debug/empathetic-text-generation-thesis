"""Monolithic LoRA baseline, corrected: Qwen2.5-1.5B-Instruct fine-tuned end to end on EmpatheticDialogues.

Why this file exists
--------------------
The original baseline (train_qwen2_5_monolithic_d2/train_qwen2_5_monolithic_d2.py, kept unchanged for the
record) set `tokenizer.pad_token = tokenizer.eos_token`. For Qwen2.5-Instruct the eos token is <|im_end|>,
the end-of-turn marker, and DataCollatorForLanguageModeling sets the label of every pad-token id to -100.
Every <|im_end|> was therefore ignored by the loss, the model never learned to end a reply, and 99% of its
test replies ran to the 250-token limit (0% rule adherence, 7.07 questions per reply). Real ED listener
replies average 10 words and only 4.7% ask more than one question, so that failure was a training bug,
not imitation of the data.

What changed (everything else is identical to the original, so it remains the same baseline)
------------------------------------------------------------------------------------------------
1. Padding uses Qwen's own pad token <|endoftext|>, which is distinct from <|im_end|>, so end-of-turn is
   learned. A start-up check aborts training if <|im_end|> would still be masked.
2. Hyper-parameters follow published LoRA/QLoRA settings instead of full fine-tuning values:
   learning rate 2e-4 and LoRA dropout 0.1 (Hu et al., 2022, GPT-2 Medium/Large; Dettmers et al., 2023,
   models up to 13B); weight decay 0.01 (Hu et al., 2022); warm-up 6% of steps, as for our detector
   (Liu et al., 2019).
3. The best epoch is kept by validation loss, with early stopping (patience 1) - the same selection rule
   as the detector - and the seed is set explicitly.

Unchanged: same system prompt, same data pairs (first user turn -> first listener reply), rank 16,
alpha 32, attention projections q/k/v/o, effective batch 16 (2 x 8), 3 epochs maximum, max length 256
(longest training pair is 177 tokens, so nothing is truncated), paged AdamW, fp16 base model.

Usage:
    python train_qwen2_5_monolithic_d2_fixed.py --max-train-samples 64 --num-train-epochs 1 --output-dir smoke   # smoke test
    python train_qwen2_5_monolithic_d2_fixed.py                                                                   # full run
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "train_qwen2_5_monolithic_d2"))
from train_qwen2_5_monolithic_d2 import SYSTEM_PROMPT, extract_dialogue_pairs  # noqa: E402  (identical prompt and data)

SEED = 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO = 2e-4, 0.01, 0.06
BATCH_SIZE, GRAD_ACCUM, MAX_LENGTH = 2, 8, 256


def parse_args():
    parser = argparse.ArgumentParser(description="Train the corrected monolithic (no emotion label) Qwen2.5-1.5B LoRA baseline.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "ed_data" / "empatheticdialogues"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all training pairs (use a small number for a smoke test)")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    return parser.parse_args()


def format_texts(user_texts, assistant_texts, tokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": u},
             {"role": "assistant", "content": a}],
            tokenize=False, add_generation_prompt=False,
        )
        for u, a in zip(user_texts, assistant_texts)
    ]


def check_end_of_turn_is_learned(tokenizer, collator):
    """Abort if the collator would ignore <|im_end|> - the exact bug of the original baseline."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text = format_texts(["I got a bonus at work."], ["That's great news!"], tokenizer)[0]
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    batch = collator([{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    end_labels = [lab for tok, lab in zip(ids, labels) if tok == end_id]
    if not end_labels or any(lab == -100 for lab in end_labels):
        raise SystemExit(f"ABORT: <|im_end|> labels are {end_labels}; the model would not learn to stop.")
    print(f"Check passed: pad token = {tokenizer.pad_token!r}, <|im_end|> is trained ({len(end_labels)} positions).")


def main():
    args = parse_args()
    set_seed(SEED)

    train_df = extract_dialogue_pairs(os.path.join(args.data_dir, "train.csv"))
    valid_df = extract_dialogue_pairs(os.path.join(args.data_dir, "valid.csv"))
    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=SEED).reset_index(drop=True)
        valid_df = valid_df.sample(n=min(max(args.max_train_samples // 5, 1), len(valid_df)), random_state=SEED).reset_index(drop=True)
    print(f"{len(train_df)} training pairs, {len(valid_df)} validation pairs.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids("<|im_end|>"):
        tokenizer.pad_token = "<|endoftext|>"
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    check_end_of_turn_is_learned(tokenizer, collator)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
                                             lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"))
    for param in model.parameters():  # trainable adapter weights in fp32 so fp16 mixed precision can unscale them
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    def tokenize(examples):
        return tokenizer(format_texts(examples["user_text"], examples["assistant_text"], tokenizer),
                         truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True, remove_columns=train_df.columns.tolist())
    valid_ds = Dataset.from_pandas(valid_df).map(tokenize, batched=True, remove_columns=valid_df.columns.tolist())

    # Transformers 5.x removed warmup_ratio; convert the 6% ratio to a step count (same as the detector script).
    steps_per_epoch = -(-len(train_ds) // (BATCH_SIZE * GRAD_ACCUM))
    warmup_steps = max(1, int(steps_per_epoch * args.num_train_epochs * WARMUP_RATIO))
    print(f"{steps_per_epoch} optimizer steps per epoch, {warmup_steps} warm-up steps.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=args.num_train_epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        fp16=True,
        optim="paged_adamw_32bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved best checkpoint (by validation loss) to {final_dir}")
    print("Validation loss per epoch:",
          [round(x["eval_loss"], 4) for x in trainer.state.log_history if "eval_loss" in x])


if __name__ == "__main__":
    main()
