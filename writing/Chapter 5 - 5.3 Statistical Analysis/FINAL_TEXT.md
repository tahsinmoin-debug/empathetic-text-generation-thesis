# 5.3 Statistical Analysis

This section determines which of the differences reported in Sections 5.1 and 5.2 are statistically reliable and which could have arisen by chance.

## 5.3.1 Statistical Methodology

Every generation system responds to the same 300 conversations and every detector is tested on the same 2,542 conversations, so all comparisons are **paired**. For binary outcomes per reply, namely whether all rules were followed, whether an acknowledgement was present, whether exactly one question was asked, whether the reply expressed the true emotion, and whether a classification was correct, we use **McNemar's exact test** [16], which considers only the conversations on which the two systems disagree. For continuous scores, namely WASSA empathy and AI-judge ratings, we use the **Wilcoxon signed-rank test** [17], which does not assume normally distributed differences. For each comparison we report the mean paired difference with a **95% bootstrap confidence interval** [18], computed from 10,000 resamples of conversations with seed 42. Because the final system is compared with seven baselines on each measure (six for the AI-judge scores), the p-values within each measure are adjusted with the **Holm-Bonferroni procedure** [19]. **A difference is considered significant only if its adjusted p-value is below 0.05 and its confidence interval excludes zero.** Comparisons between independent groups, as in the error-propagation analysis, use Fisher's exact test for binary outcomes and the Mann-Whitney U test for scores.

## 5.3.2 Final System versus Baselines

Table 5.13 reports all comparisons between the final system and the baselines. Three patterns emerge.

**Empathy and emotion fidelity improve significantly over comparable designs.** The final system's WASSA score exceeds that of every baseline except the larger single model: it is higher than the same-size single model by 0.21 points (95% CI [+0.08, +0.35], adjusted p = 0.012), the initial pipeline by 0.25 points ([+0.10, +0.39], p = 0.003), and the same generator with top-1 prompting by 0.16 points ([+0.05, +0.26], p = 0.008) and **the same generator used alone as a single model by 0.30 points** ([+0.19, +0.41], p < 0.001). Emotion validation is also higher than for every baseline except the larger single model, by 8.3 to 18.3 percentage points (all adjusted p < 0.013), including 9.7 points over the same generator used alone (p = 0.007). The difference from the larger single model, 4.3 points, is not significant (p = 0.19). Because the single-model baseline on the same generator scores no better than the 1.5-billion-parameter one on these two measures, the gains are attributable to the decoupled design rather than to generator size (Section 5.4.9).

**Rule adherence does not significantly exceed that of the same-size single model after correction.** The final system follows all rules in 60.0% of replies, compared with 51.3% for the same-size single model, a difference of 8.7 points (CI [+1.0, +16.3]). The unadjusted p-value is 0.032, but after Holm correction it is 0.095, so this difference is not claimed. The final system is significantly better than zero-shot prompting (+41.0 points), the fine-tuned model (+37.7) and the retrieval-augmented pipeline (+16.3), and does not differ significantly from the initial pipeline (-3.7) or from top-1 prompting with Phi-4-mini (-5.3). It is significantly **lower** than the same generator used alone as a single model (-13.7 points, adjusted p = 0.002) and than the larger single model (-22.3), the trade-off discussed in Section 5.4.9. On the one-question rule alone, it is significantly better than the same-size single model (+19.0 points, p < 0.001).

**The larger single model is significantly better on rules and empathy.** Qwen2.5-7B exceeds the final system in overall rule adherence (-22.3 points), acknowledgement (-16.7), the one-question rule (-6.3) and WASSA empathy (-0.71), and all of these differences are significant after correction.


| Metric | Final system vs | N | Final | Baseline | Difference | 95% CI | Test | p | p (Holm-adjusted) | Significant? |
|---|---|---|---|---|---|---|---|---|---|---|
| All rules | Single model, same generator: Phi-4-mini (4-bit) detects + responds | 300 | 60 | 73.67 | -13.67 | [-21.00, -6.33] | McNemar exact | 0 | 0 | YES |
| All rules | Single model, same size: Qwen2.5-1.5B detects + responds | 300 | 60 | 51.33 | 8.67 | [+1.00, +16.33] | McNemar exact | 0.03 | 0.1 | no |
| All rules | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 300 | 60 | 82.33 | -22.33 | [-29.33, -15.33] | McNemar exact | 0 | 0 | YES |
| All rules | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 300 | 60 | 19 | 41 | [+34.00, +47.67] | McNemar exact | 0 | 0 | YES |
| All rules | Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 300 | 60 | 22.33 | 37.67 | [+30.33, +45.00] | McNemar exact | 0 | 0 | YES |
| All rules | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 300 | 60 | 63.67 | -3.67 | [-11.00, +3.67] | McNemar exact | 0.37 | 0.39 | no |
| All rules | Initial pipeline + retrieval (RAG) | 300 | 60 | 43.67 | 16.33 | [+8.33, +24.00] | McNemar exact | 0 | 0 | YES |
| All rules | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 300 | 60 | 65.33 | -5.33 | [-13.00, +2.00] | McNemar exact | 0.19 | 0.39 | no |
| Acknowledgement | Single model, same generator: Phi-4-mini (4-bit) detects + responds | 300 | 66.33 | 77.33 | -11 | [-18.00, -4.00] | McNemar exact | 0 | 0.01 | YES |
| Acknowledgement | Single model, same size: Qwen2.5-1.5B detects + responds | 300 | 66.33 | 70.67 | -4.33 | [-11.33, +2.67] | McNemar exact | 0.24 | 0.48 | no |
| Acknowledgement | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 300 | 66.33 | 83 | -16.67 | [-23.33, -10.00] | McNemar exact | 0 | 0 | YES |
| Acknowledgement | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 300 | 66.33 | 81.67 | -15.33 | [-21.67, -8.67] | McNemar exact | 0 | 0 | YES |
| Acknowledgement | Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 300 | 66.33 | 30 | 36.33 | [+29.00, +43.67] | McNemar exact | 0 | 0 | YES |
| Acknowledgement | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 300 | 66.33 | 70.33 | -4 | [-10.67, +2.67] | McNemar exact | 0.28 | 0.48 | no |
| Acknowledgement | Initial pipeline + retrieval (RAG) | 300 | 66.33 | 57 | 9.33 | [+2.33, +16.33] | McNemar exact | 0.01 | 0.05 | no |
| Acknowledgement | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 300 | 66.33 | 72 | -5.67 | [-12.67, +1.33] | McNemar exact | 0.13 | 0.4 | no |
| One question | Single model, same generator: Phi-4-mini (4-bit) detects + responds | 300 | 92.67 | 96 | -3.33 | [-7.00, +0.33] | McNemar exact | 0.11 | 0.22 | no |
| One question | Single model, same size: Qwen2.5-1.5B detects + responds | 300 | 92.67 | 73.67 | 19 | [+13.33, +25.00] | McNemar exact | 0 | 0 | YES |
| One question | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 300 | 92.67 | 99 | -6.33 | [-9.67, -3.33] | McNemar exact | 0 | 0 | YES |
| One question | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 300 | 92.67 | 22.67 | 70 | [+64.67, +75.33] | McNemar exact | 0 | 0 | YES |
| One question | Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 300 | 92.67 | 74 | 18.67 | [+12.67, +24.67] | McNemar exact | 0 | 0 | YES |
| One question | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 300 | 92.67 | 87.33 | 5.33 | [+0.67, +10.00] | McNemar exact | 0.04 | 0.12 | no |
| One question | Initial pipeline + retrieval (RAG) | 300 | 92.67 | 78 | 14.67 | [+8.67, +20.33] | McNemar exact | 0 | 0 | YES |
| One question | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 300 | 92.67 | 94.67 | -2 | [-6.00, +1.67] | McNemar exact | 0.39 | 0.39 | no |
| Emotion validation | Single model, same generator: Phi-4-mini (4-bit) detects + responds | 300 | 36.67 | 27 | 9.67 | [+3.67, +15.33] | McNemar exact | 0 | 0.01 | YES |
| Emotion validation | Single model, same size: Qwen2.5-1.5B detects + responds | 300 | 36.67 | 25.67 | 11 | [+5.00, +17.00] | McNemar exact | 0 | 0 | YES |
| Emotion validation | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 300 | 36.67 | 32.33 | 4.33 | [-1.67, +10.33] | McNemar exact | 0.19 | 0.19 | no |
| Emotion validation | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 300 | 36.67 | 24.67 | 12 | [+6.00, +17.67] | McNemar exact | 0 | 0 | YES |
| Emotion validation | Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 300 | 36.67 | 18.33 | 18.33 | [+12.33, +24.33] | McNemar exact | 0 | 0 | YES |
| Emotion validation | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 300 | 36.67 | 26 | 10.67 | [+5.00, +16.33] | McNemar exact | 0 | 0 | YES |
| Emotion validation | Initial pipeline + retrieval (RAG) | 300 | 36.67 | 27.67 | 9 | [+3.00, +15.00] | McNemar exact | 0.01 | 0.01 | YES |
| Emotion validation | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 300 | 36.67 | 28.33 | 8.33 | [+3.00, +13.67] | McNemar exact | 0 | 0.01 | YES |
| WASSA empathy | Single model, same generator: Phi-4-mini (4-bit) detects + responds | 300 | 5.05 | 4.75 | 0.3 | [+0.19, +0.41] | Wilcoxon signed-rank | 0 | 0 | YES |
| WASSA empathy | Single model, same size: Qwen2.5-1.5B detects + responds | 300 | 5.05 | 4.84 | 0.21 | [+0.08, +0.35] | Wilcoxon signed-rank | 0.01 | 0.01 | YES |
| WASSA empathy | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 300 | 5.05 | 5.76 | -0.71 | [-0.84, -0.58] | Wilcoxon signed-rank | 0 | 0 | YES |
| WASSA empathy | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 300 | 5.05 | 4.25 | 0.8 | [+0.66, +0.94] | Wilcoxon signed-rank | 0 | 0 | YES |
| WASSA empathy | Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 300 | 5.05 | 3.36 | 1.69 | [+1.48, +1.90] | Wilcoxon signed-rank | 0 | 0 | YES |
| WASSA empathy | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 300 | 5.05 | 4.8 | 0.25 | [+0.10, +0.39] | Wilcoxon signed-rank | 0 | 0 | YES |
| WASSA empathy | Initial pipeline + retrieval (RAG) | 300 | 5.05 | 4.85 | 0.2 | [+0.05, +0.36] | Wilcoxon signed-rank | 0.01 | 0.01 | YES |
| WASSA empathy | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 300 | 5.05 | 4.89 | 0.16 | [+0.05, +0.26] | Wilcoxon signed-rank | 0 | 0.01 | YES |

*Table 5.13: Paired significance tests comparing the final system with each baseline*


## 5.3.3 Is the Detector the Bottleneck?

Three analyses test whether detection errors limit the quality of replies. First, in an **error-propagation analysis**, each pipeline's replies are divided according to whether the detector's emotion matched the gold label, and the two groups are compared (Table 5.14). In no pipeline does an incorrect label significantly reduce rule adherence; for the initial pipeline, for example, adherence is 61.2% when the label is correct and 66.0% when it is wrong (p = 0.40), and for the final system the difference is also not significant (p = 0.096). For the initial pipeline, where WASSA scores were divided in the same way, an incorrect label does not reduce empathy either (4.83 against 4.78, p = 0.72). Second, in a **controlled detector swap** with the generator fixed as Qwen2.5-1.5B, replacing the GPT-2 detector with the more accurate RoBERTa detector changes neither emotion validation (25.3% against 26.0%, p = 0.91) nor rule adherence (57.7% against 63.7%, p = 0.09). In contrast, the detector models themselves differ significantly: RoBERTa-large outperforms RoBERTa-base on top-1 accuracy (+2.9 points, p = 0.0002), and RoBERTa-base outperforms BERT and ELECTRA on top-5 accuracy (p ≤ 0.0001), as shown in Table 5.15.

Third, and most directly, we **substituted RoBERTa-large into the final system** and regenerated replies for the same 300 conversations, changing nothing else (Table 5.16). The larger detector was 3.0 points more accurate on these conversations (53.3% against 50.3%, p = 0.21), but no measure of reply quality improved: rule adherence was 60.0% against 54.7% (adjusted p = 0.60), WASSA empathy 5.05 against 5.01 (p = 0.85), emotion validation 36.7% against 38.0% (p = 0.85), and the one-question rule 92.7% against 95.7% (p = 0.60). Acknowledgement was nominally lower with the larger detector (66.3% against 57.7%, unadjusted p = 0.012) but not significantly so after correction (p = 0.06). Together, these results show that **detection accuracy is not the factor limiting reply quality**, which justifies the smaller detector.


| setting | metric | n_d1_correct | n_d1_wrong | mean_when_d1_correct | mean_when_d1_wrong | diff | ci_95_low | ci_95_high | test | p_value | significant_damage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt_engineered_pipeline | rule_adherence_all_rules | 147 | 153 | 0.61 | 0.66 | -0.05 | -0.15 | 0.06 | Fisher exact | 0.4 | False |
| prompt_engineered_pipeline | wassa_empathy_0_10 | 147 | 153 | 4.83 | 4.78 | 0.05 | -0.24 | 0.35 | Mann-Whitney U | 0.72 | False |
| prompt_engineered_pipeline | validation_match_vs_gold | 147 | 153 | 0.38 | 0.14 | 0.24 | 0.14 | 0.33 | Fisher exact | 0 | True |
| bakeoff_qwen | rule_adherence_all_rules | 151 | 149 | 0.6 | 0.55 | 0.05 | -0.06 | 0.17 | Fisher exact | 0.41 | False |
| bakeoff_qwen | validation_match_vs_gold | 151 | 149 | 0.38 | 0.12 | 0.26 | 0.17 | 0.36 | Fisher exact | 0 | True |
| bakeoff_phi | rule_adherence_all_rules | 151 | 149 | 0.64 | 0.67 | -0.04 | -0.14 | 0.07 | Fisher exact | 0.55 | False |
| bakeoff_phi | validation_match_vs_gold | 151 | 149 | 0.46 | 0.1 | 0.36 | 0.27 | 0.45 | Fisher exact | 0 | True |
| two_stage_qwen | rule_adherence_all_rules | 110 | 190 | 0.34 | 0.38 | -0.05 | -0.16 | 0.06 | Fisher exact | 0.46 | False |
| two_stage_qwen | validation_match_vs_gold | 110 | 190 | 0.45 | 0.16 | 0.29 | 0.18 | 0.4 | Fisher exact | 0 | True |
| two_stage_phi | rule_adherence_all_rules | 127 | 173 | 0.54 | 0.64 | -0.1 | -0.21 | 0.01 | Fisher exact | 0.1 | False |
| two_stage_phi | validation_match_vs_gold | 127 | 173 | 0.65 | 0.16 | 0.5 | 0.4 | 0.59 | Fisher exact | 0 | True |

*Table 5.14: Error-propagation analysis: reply quality when the detector's label is correct and when it is wrong*



| comparison | n | pipeline_mean | baseline_mean | diff | ci_95_low | ci_95_high | test | p_value | significant | metric | question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| roberta-base vs google__electra-base-discriminator | 2542 | 0.5 | 0.5 | -0 | -0.02 | 0.01 | McNemar exact | 0.96 | False | top1 | Why RoBERTa over ELECTRA? (same size class) |
| roberta-base vs google__electra-base-discriminator | 2542 | 0.85 | 0.83 | 0.02 | 0.01 | 0.04 | McNemar exact | 0 | True | top5 | Why RoBERTa over ELECTRA? (same size class) |
| roberta-base vs bert-base-uncased | 2542 | 0.5 | 0.49 | 0.01 | -0 | 0.03 | McNemar exact | 0.11 | False | top1 | Why RoBERTa over BERT? (same size class) |
| roberta-base vs bert-base-uncased | 2542 | 0.85 | 0.82 | 0.03 | 0.02 | 0.05 | McNemar exact | 0 | True | top5 | Why RoBERTa over BERT? (same size class) |
| roberta-large vs roberta-base | 2542 | 0.53 | 0.5 | 0.03 | 0.01 | 0.04 | McNemar exact | 0 | True | top1 | Is the larger model's advantage real? (why not bigger) |
| roberta-large vs roberta-base | 2542 | 0.87 | 0.85 | 0.02 | 0.01 | 0.03 | McNemar exact | 0 | True | top5 | Is the larger model's advantage real? (why not bigger) |

*Table 5.15: Paired significance tests between detector models*



| Group | Measure | RoBERTa-base d1 (%) | RoBERTa-large d1 (%) | Difference | 95% CI | Test | p | p (Holm over reply-quality measures) | Significant? |
|---|---|---|---|---|---|---|---|---|---|
| Detection | d1 top-1 correct | 50.3 | 53.3 | 3 | [-1.0, +7.3] | McNemar exact | 0.21 |  | no |
| Detection | True emotion in d1 top-3 | 78 | 79.3 | 1.3 | [-2.0, +4.7] | McNemar exact | 0.54 |  | no |
| Detection | Stage-1 pick correct | 42.3 | 44.7 | 2.3 | [-2.3, +7.0] | McNemar exact | 0.39 |  | no |
| Reply quality | All rules | 60 | 54.7 | -5.3 | [-12.3, +1.7] | McNemar exact | 0.16 | 0.6 | no |
| Reply quality | Acknowledgement | 66.3 | 57.7 | -8.7 | [-15.0, -2.0] | McNemar exact | 0.01 | 0.06 | no |
| Reply quality | One question | 92.7 | 95.7 | 3 | [-0.7, +6.7] | McNemar exact | 0.15 | 0.6 | no |
| Reply quality | Emotion validation | 36.7 | 38 | 1.3 | [-3.3, +6.0] | McNemar exact | 0.68 | 0.85 | no |
| Reply quality | WASSA empathy (0-10 points, not %) | 5.05 | 5.01 | -0.04 | [-0.14, +0.06] | Wilcoxon signed-rank | 0.43 | 0.85 | no |

*Table 5.16: Final system with RoBERTa-base and with RoBERTa-large as the detector*


## 5.3.4 The Stage-1 Emotion Selection

The generator's stage-1 choice matches the gold label in 42.3% of conversations, compared with 50.3% for the detector's top-1 prediction on the same conversations (Table 5.5). The difference of -8.0 points (95% CI [-13.3, -2.7]; McNemar p = 0.005), arising from 46 conversations on which only the detector's top-1 prediction was correct and 22 on which only the generator's choice was correct, is significant. The improvements of the two-stage design reported in Section 5.3.2 therefore cannot be explained by more accurate emotion selection, a point discussed in Section 5.6.

## 5.3.5 Confounding by Reply Length

WASSA scores correlate positively with reply length within every system, with Spearman correlations ranging from 0.14 to 0.54 (all p < 0.02; Table 5.17). A system that writes longer replies can therefore obtain higher WASSA scores without being more empathetic, consistent with the observation that lengthy LLM responses may be mistaken for greater empathy [29] and with the verbosity bias of LLM judges [13]. This affects two comparisons. It explains Gemma-3-4B's higher WASSA score (Section 5.4). It also strengthens, rather than weakens, the final system's advantage over top-1 Phi-4-mini prompting, because the final system obtains the higher score with *shorter* replies (57.1 against 80.4 words). The larger single model's advantage cannot be explained by length either, since its replies average 52.9 words.


| System | Mean words | Length OK (%) | WASSA mean | Spearman rho (length vs WASSA) | p |
|---|---|---|---|---|---|
| Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 75.9 | 83.7 | 4.25 | 0.54 | 0 |
| Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 10 | 100 | 3.36 | 0.57 | 0 |
| Single model, same size: Qwen2.5-1.5B detects + responds | 42 | 100 | 4.84 | 0.2 | 0 |
| Single model, same generator: Phi-4-mini (4-bit) detects + responds | 73.5 | 99.3 | 4.75 | 0.1 | 0.09 |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 52.9 | 100 | 5.76 | 0.19 | 0 |
| Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 35.2 | 100 | 4.8 | 0.33 | 0 |
| Initial pipeline + retrieval (RAG) | 34 | 100 | 4.85 | 0.24 | 0 |
| RoBERTa d1 + Qwen2.5-1.5B d2 (top-1 emotion) | 34.7 | 100 | 4.78 | 0.29 | 0 |
| RoBERTa d1 + Gemma-3-4B d2 (top-1 emotion) | 133.3 | 27.7 | 5.3 | 0.31 | 0 |
| RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 80.4 | 96 | 4.89 | 0.14 | 0.02 |
| Two-stage: RoBERTa top-3 -> Qwen2.5-1.5B picks + per-emotion prompt | 44.1 | 99.7 | 4.76 | 0.21 | 0 |
| FINAL SYSTEM - Two-stage: RoBERTa top-3 -> Phi-4-mini picks + per-emotion prompt | 57.1 | 100 | 5.05 | 0.26 | 0 |

*Table 5.17: Correlation between reply length and WASSA empathy score within each system*


## 5.3.6 Validity of the WASSA Classifier

Because several conclusions rely on WASSA scores, we examined how far they can be trusted, using the 384 replies of the six systems that were also scored by both AI judges (Table 5.18). At the **system level**, WASSA ranks the six systems in nearly the same order as the judges' mean empathy score (Spearman rho = 0.94); the only difference is that the same-size single model and top-1 Phi-4-mini prompting exchange places. At the **reply level**, agreement is weak but significant (rho = 0.25, p < 0.001) and persists after controlling for reply length (partial rho = 0.24), showing that it is not merely a shared preference for longer replies. **Within a single system**, agreement is weak (rho = 0.05 to 0.26, significant in only two of the six systems), compared with 0.60 between the two judges. [[PENDING: recompute with the corrected fine-tuned baseline once both judges have scored it.]] We therefore use WASSA only to compare system averages over 300 conversations, the purpose for which this evidence supports it, and never to judge individual replies. The provenance of the model, a peer-reviewed ACL 2024 shared-task system trained on human ratings of conversation turns and correlating with human ratings at r = 0.544 [14], [15], and its limitations are listed in the same table.


| Aspect | Check | Result | Source |
|---|---|---|---|
| Recognition | Where it comes from | WASSA 2024 shared task at ACL 2024 (14 teams); system paper hyy33, ACL Anthology 2024.wassa-1.39 | Findings paper 2024.wassa-1.30 |
| Recognition | What it was trained on | Third-party human empathy ratings (ordinal 1-5) of individual conversation turns (Track 2, CONV-Turn) | Findings paper |
| Recognition | Accuracy against humans on the task test set | Pearson r = .544 for empathy; 5th of 9 teams on the track average in the findings table (the authors report 4th) | Findings paper, Table 3 |
| Recognition | Caveat | The organisers' RoBERTa-base baseline reached r = .694 on empathy; the best team reached .582 | Findings paper, Table 3 |
| Recognition | Output format | 11 ordinal classes (argmax); our replies receive classes 1-7; the authors do not document how ratings map to classes | Checkpoint metadata; our scores |
| Our data | System-level ranking vs AI-judge panel (6 systems) | Spearman rho = 0.94 | computed |
| Our data | Per-reply agreement with the judges' mean empathy | rho = 0.25 (p = 4.9e-07, N = 384) | computed |
| Our data | Same, controlling for reply length | partial rho = 0.24 (p = 1.4e-06) | computed |
| Our data | Agreement within a single system | rho = 0.05 to 0.26; 2 of 6 significant | computed |
| Our data | For reference: judge vs judge on empathy | rho = 0.60 | computed |
| Reproducibility | Re-implementation check after a library upgrade | 1,800 of 1,800 pre-upgrade scores reproduced exactly | `llm_judge_evaluation/wassa_empathy_scores_BACKUP_pre_upgrade.csv` |

*Table 5.18: Evidence on the validity of the WASSA empathy classifier for this project*


## 5.3.7 AI-Judge Results and Agreement

Table 5.19 compares the final system's AI-judge scores with those of each judged baseline on the same 64 conversations. **Against the same-size single model, the final system is rated significantly higher** for empathy (+0.70, 95% CI [+0.39, +1.01], adjusted p < 0.001), relevance (+0.44, [+0.01, +0.86], p = 0.023) and overall (+0.44, [+0.17, +0.72], p = 0.001), while safety does not differ significantly. It is also rated significantly higher overall than the initial pipeline (+0.65), and zero-shot prompting (+1.18), with all adjusted p < 0.001. [[PENDING: comparison with the corrected fine-tuned baseline.]] **Against the larger single model, the final system's empathy score is significantly lower** (-0.30, [-0.55, -0.07], p = 0.048). Its overall score is also lower (-0.21, [-0.40, -0.03]), but this difference is not significant after correction (p = 0.073). **The judges do not distinguish the final system from top-1 Phi-4-mini prompting** on any dimension (overall +0.04, p = 0.71; empathy +0.01, p = 0.96). The advantage of the two-stage design over top-1 prompting reported in Section 5.3.2 therefore rests on WASSA empathy and emotion validation, not on the judges' ratings.

The two judges agree well with each other (Table 5.20). Across the 384 replies of the six judged systems, the Spearman correlation between Nemotron 3 Ultra and gpt-oss-120b is 0.60 for both empathy and the overall score. The judges rank the six systems in the same order, except that the final system and top-1 Phi-4-mini, whose panel scores differ by only 0.03, exchange second and third place (rank correlation 0.94). Mistral Large, originally the third judge, scored the five systems built first but became unavailable before the final systems could be judged. Rather than compare systems scored by different panels, we compare every system on the two judges that scored all of them. On the 256 replies of the four systems scored by all three judges (excluding the original, flawed fine-tuned run), the two-judge and three-judge averages correlate at 0.97 for both empathy and the overall score and rank the four systems identically, and the three judges agree with one another at 0.57 to 0.63 on empathy. Removing the third judge therefore does not change the conclusions. This agreement is consistent with the reduced bias expected from a panel of judges drawn from different model families [35].


| Dimension | Final system vs | N | Final | Baseline | Difference | 95% CI | p | p (Holm-adjusted) | Significant? |
|---|---|---|---|---|---|---|---|---|---|
| empathy | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 64 | 7.84 | 8.15 | -0.3 | [-0.55, -0.07] | 0.02 | 0.05 | YES |
| empathy | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 64 | 7.84 | 7.84 | 0.01 | [-0.23, +0.25] | 0.96 | 0.96 | no |
| empathy | Single model, same size: Qwen2.5-1.5B detects + responds | 64 | 7.84 | 7.15 | 0.7 | [+0.39, +1.01] | 0 | 0 | YES |
| empathy | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 64 | 7.84 | 6.98 | 0.87 | [+0.52, +1.24] | 0 | 0 | YES |
| empathy | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 64 | 7.84 | 5.98 | 1.86 | [+1.48, +2.23] | 0 | 0 | YES |
| relevance | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 64 | 8.45 | 8.68 | -0.23 | [-0.47, +0.02] | 0.04 | 0.08 | no |
| relevance | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 64 | 8.45 | 8.29 | 0.16 | [-0.13, +0.45] | 0.31 | 0.31 | no |
| relevance | Single model, same size: Qwen2.5-1.5B detects + responds | 64 | 8.45 | 8.02 | 0.44 | [+0.01, +0.86] | 0.01 | 0.02 | YES |
| relevance | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 64 | 8.45 | 7.63 | 0.82 | [+0.45, +1.20] | 0 | 0 | YES |
| relevance | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 64 | 8.45 | 7.23 | 1.23 | [+0.68, +1.73] | 0 | 0 | YES |
| safety | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 64 | 9.76 | 9.86 | -0.1 | [-0.23, +0.02] | 0.13 | 0.39 | no |
| safety | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 64 | 9.76 | 9.81 | -0.05 | [-0.17, +0.06] | 0.36 | 0.39 | no |
| safety | Single model, same size: Qwen2.5-1.5B detects + responds | 64 | 9.76 | 9.56 | 0.2 | [-0.01, +0.41] | 0.08 | 0.32 | no |
| safety | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 64 | 9.76 | 9.5 | 0.26 | [+0.01, +0.55] | 0.18 | 0.39 | no |
| safety | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 64 | 9.76 | 9.31 | 0.45 | [+0.23, +0.66] | 0 | 0 | YES |
| overall | Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 64 | 8.68 | 8.9 | -0.21 | [-0.40, -0.03] | 0.04 | 0.07 | no |
| overall | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 64 | 8.68 | 8.65 | 0.04 | [-0.16, +0.24] | 0.71 | 0.71 | no |
| overall | Single model, same size: Qwen2.5-1.5B detects + responds | 64 | 8.68 | 8.24 | 0.44 | [+0.17, +0.72] | 0 | 0 | YES |
| overall | Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 64 | 8.68 | 8.04 | 0.65 | [+0.35, +0.96] | 0 | 0 | YES |
| overall | Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 64 | 8.68 | 7.51 | 1.18 | [+0.83, +1.51] | 0 | 0 | YES |

*Table 5.19: Paired significance tests comparing the final system's AI-judge scores with each judged baseline*



| Check | N | Result | Without fine-tuned system |
|---|---|---|---|
| Per-reply agreement on empathy: Nemotron 3 Ultra vs gpt-oss-120b | 384 | rho = 0.596 (p = 3.0e-38) | rho = 0.596 |
| Per-reply agreement on overall: Nemotron 3 Ultra vs gpt-oss-120b | 384 | rho = 0.604 (p = 1.6e-39) | rho = 0.604 |
| Do both judges rank the 6 systems in the same order (overall)? | 6 | All but 2 systems; these swap places: RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) / FINAL SYSTEM - Two-stage: RoBERTa top-3 -> Phi-4-mini picks + per-emotion prompt | rank rho = 0.94 |
| Two-judge vs three-judge panel mean (empathy), 4 systems judged by all three | 256 | per-reply rho = 0.970 | identical system ranking |
| Two-judge vs three-judge panel mean (overall), 4 systems judged by all three | 256 | per-reply rho = 0.973 | identical system ranking |
| Three-judge agreement on empathy (all pairs), 4 systems | 256 | rho = 0.57 to 0.63 |  |

*Table 5.20: Agreement between the AI judges and comparison of the two-judge and three-judge panels*


## 5.3.8 Human Evaluation Statistics

[[PENDING: (1) number of participants excluded by the attention checks; (2) selection rate of each system, with a two-sided binomial test against a chance rate of one third; (3) a per-participant paired comparison of how often the final system was chosen over each baseline, using the Wilcoxon signed-rank test; (4) Fleiss' kappa for agreement between participants; and (5) the proportion of items on which the human majority choice matches the system scored highest by the AI judges.]]

## 5.3.9 Limitations of the Analysis

The generation sample of 300 conversations can detect differences of a few percentage points on binary measures but may miss smaller effects, and the AI-judge comparisons, based on 64 conversations, have correspondingly less statistical power. Emotion validation relies on our own detector reading replies of a kind it was not trained on. Holm correction was applied within each measure rather than across all measures, and treating the seven baselines as one family of comparisons is a design decision that we report explicitly.

## References

[13] L. Zheng, W.-L. Chiang, Y. Sheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, vol. 36, 2023; arXiv:2306.05685.  
[14] S. Giorgi, J. Sedoc, V. Barriere, and S. Tafreshi, "Findings of WASSA 2024 shared task on empathy and personality detection in interactions," in Proc. 14th Workshop Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA), 2024, pp. 369-379, doi: 10.18653/v1/2024.wassa-1.30.  
[15] H. Yang, L. Huang, T. Li, N. Rusnachenko, and H. Liang, "hyy33 at WASSA 2024 empathy and personality shared task: Using the CombinedLoss and FGM for enhancing BERT-based models in emotion and empathy prediction from conversation turns," in Proc. 14th Workshop Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA), 2024, pp. 430-434, doi: 10.18653/v1/2024.wassa-1.39.  
[16] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," Psychometrika, vol. 12, no. 2, pp. 153-157, 1947.  
[17] F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bulletin, vol. 1, no. 6, pp. 80-83, 1945.  
[18] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.  
[19] S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian J. Statistics, vol. 6, no. 2, pp. 65-70, 1979.  
[29] V. Sorin, D. Brin, Y. Barash, E. Konen, A. Charney, G. Nadkarni, and E. Klang, "Large language models and empathy: Systematic review," J. Med. Internet Res., vol. 26, p. e52597, 2024, doi: 10.2196/52597.  
[35] P. Verga, S. Hofstatter, S. Althammer, Y. Su, A. Piktus, A. Arkhangorodsky, M. Xu, N. White, and P. Lewis, "Replacing judges with juries: Evaluating LLM generations with a panel of diverse models," arXiv:2404.18796, 2024.  
