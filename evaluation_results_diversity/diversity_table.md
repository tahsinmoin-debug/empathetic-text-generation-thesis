# Diversity and Repetition

Measured on the same 300 held-out conversations for every setting, so these are comparable to each other. They are NOT comparable to Dist-n values published in papers, which use the full ~2,500-conversation test set - Dist-n falls as sample size grows (measured on our own output: 32.3% at n=50 vs 15.6% at n=300).

**Boilerplate rate** = share of replies containing at least one of that setting's own 10 most-repeated 5-word phrases. Higher means more reliance on stock phrasing.

| Setting | Dist-1 | Dist-2 | Dist-3 | Boilerplate |
|---|---:|---:|---:|---:|
| zero_shot | 12.48% | 57.15% | 81.96% | 34.7% |
| fine_tuned_fixed | 17.35% | 56.09% | 78.16% | 9.7% |
| rag_pipeline | 12.60% | 53.62% | 76.91% | 26.7% |
| single_model_same_size | 11.35% | 50.50% | 73.68% | 35.0% |
| prompt_engineered_pipeline | 11.91% | 49.49% | 72.43% | 23.7% |
| two_stage_phi | 11.50% | 47.84% | 71.54% | 58.7% |
| single_model_bigger | 9.76% | 44.63% | 69.33% | 58.7% |
| single_model_phi | 8.62% | 39.12% | 62.40% | 72.3% |
| bakeoff_phi | 7.57% | 35.97% | 59.63% | 77.0% |

## Most repeated phrase per setting

| Setting | Most repeated 5-word phrase | Replies containing it |
|---|---|---:|
| two_stage_phi | "how are you feeling right" | 49 (16.3%) |
| single_model_phi | "it's completely normal to feel" | 65 (21.7%) |
| fine_tuned_fixed | "how long have you been" | 4 (1.3%) |
| bakeoff_phi | "it's completely natural to feel" | 55 (18.3%) |
| prompt_engineered_pipeline | "what would you like to" | 21 (7.0%) |
| single_model_same_size | "is there anything specific about" | 23 (7.7%) |
| single_model_bigger | "i can really understand how" | 67 (22.3%) |
| rag_pipeline | "is there anything specific about" | 25 (8.3%) |
| zero_shot | "i'm sorry to hear that" | 30 (10.0%) |