# Corrected LoRA baseline: how to run it (Option A)

**Why:** the original LoRA baseline never learned to end its reply (the end-of-turn token was masked as padding),
so 99% of its replies ran to the 250-token limit. This folder retrains the same baseline with that bug fixed and
published LoRA settings. The old model and its judged outputs are **not** touched; the new one is saved as
`fine_tuned_fixed`.

Close Chrome, games and other GPU programs before each GPU step. Run the steps **one at a time, in order**.
Use PowerShell.

| Step | What it does | GPU? | Expected time |
|---|---|---|---|
| 1 | Smoke test (64 examples, 1 epoch) | yes, ~4 GB | 3-5 min |
| 2 | Full training (17,780 pairs, up to 3 epochs) | yes, ~5-6 GB | 1.5-3 h |
| 3 | Generate replies for the same 300 test conversations | yes, ~3.5 GB | 10-15 min |
| 4 | Rule-adherence scoring (all systems, incl. BERTScore) | yes | ~10 min |
| 5 | Emotion validation / context check | light | ~3 min |
| 6 | WASSA empathy score for the new system only | light | ~2 min |
| 7 | AI judges (64 conversations x 2 judges) | no (API) | Groq: minutes; Nemotron: 50 requests/day |

## 1. Smoke test (must print "Check passed")

```
cd F:\Thesis\train_qwen2_5_monolithic_d2_fixed
F:\Thesis\thesis_env\Scripts\python.exe train_qwen2_5_monolithic_d2_fixed.py --max-train-samples 64 --num-train-epochs 1 --output-dir smoke_test
```

Look for `Check passed: pad token = '<|endoftext|>', <|im_end|> is trained`. If it says `ABORT`, stop and send me the output.
If the smoke test finishes without errors, delete the `smoke_test` folder.

## 2. Full training

```
F:\Thesis\thesis_env\Scripts\python.exe train_qwen2_5_monolithic_d2_fixed.py
```

At the end it prints the validation loss per epoch and saves the best epoch to `checkpoints\final`. Copy that
last line for me.

## 3. Generate the 300 replies

```
cd F:\Thesis\d2_settings_comparison
F:\Thesis\thesis_env\Scripts\python.exe generate_monolithic_finetuned_responses.py --lora-dir F:\Thesis\train_qwen2_5_monolithic_d2_fixed\checkpoints\final --variant-name fine_tuned_fixed
```

Output: `evaluation_results_d2_settings_comparison\fine_tuned_generation_output.csv` is left unchanged; the new file is
`fine_tuned_fixed_generation_output.csv`.

## 4-6. Local scoring

```
cd F:\Thesis\d2_settings_comparison
F:\Thesis\thesis_env\Scripts\python.exe score_d2_settings_comparison.py
cd F:\Thesis\context_awareness_check
F:\Thesis\thesis_env\Scripts\python.exe check_context_awareness_new_d1.py
cd F:\Thesis\llm_judge_evaluation
F:\Thesis\thesis_env\Scripts\python.exe run_empathy_classifier.py --settings fine_tuned_fixed
```

## 7. AI judges (you or teammates)

```
cd F:\Thesis\llm_judge_evaluation
F:\Thesis\thesis_env\Scripts\python.exe run_llm_judge.py --judge gptoss_groq --settings fine_tuned_fixed
F:\Thesis\thesis_env\Scripts\python.exe run_llm_judge.py --judge nemotron_openrouter --settings fine_tuned_fixed --half short
F:\Thesis\thesis_env\Scripts\python.exe run_llm_judge.py --judge nemotron_openrouter --settings fine_tuned_fixed --half long
```

Nemotron allows 50 requests per day, so run `--half short` and `--half long` on different days or different accounts
(32 requests each). Re-running the same command retries only failed rows.

When all steps are done, tell Claude: the thesis tables, statistics and text will be updated from the new files.
