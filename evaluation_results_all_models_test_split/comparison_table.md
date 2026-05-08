# Emotion Detection Accuracy Comparison

Evaluation data: `GoEmotions/splits_qwen2_5_detector/test.csv`

Total test samples: 2,243

| Rank | Model | Accuracy | Macro F1 | Correct / Total |
|---:|---|---:|---:|---:|
| 1 | GPT-2 Medium | 97.73% | 96.22% | 2192 / 2243 |
| 2 | Qwen2.5-0.5B Emotion Detector LoRA | 35.93% | 33.43% | 806 / 2243 |
| 3 | TinyLLaMA LoRA | 26.13% | 23.85% | 586 / 2243 |
| 4 | Qwen2-1.5B LoRA | 14.22% | 8.78% | 319 / 2243 |

Highest detection accuracy: **GPT-2 Medium**.

Note: GPT-2 Medium, TinyLLaMA, and the older Qwen2 model were originally trained on the old emotion-conditioned format and may have seen examples from this split during training. The Qwen2.5-0.5B detector was trained with this split held out and uses label-only scoring.
