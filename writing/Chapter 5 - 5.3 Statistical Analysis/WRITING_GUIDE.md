# Chapter 5 - 5.3 Statistical Analysis

**Rubric:** CO6 (Chapter 5: 5.3 Statistical Analysis). Assessed by the defence panel and the supervisor.

## What this section must do

Establish which differences reported in 5.1 and 5.2 are **real** and which could be chance. Name each test and justify it, report effect size with a confidence interval, and correct for multiple comparisons. The *AI* (MDPI) 2025 reference paper reports a table of test statistics and p-values followed by a paragraph that interprets them honestly, including non-significant results. Do the same, adding confidence intervals and correction.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/final_system_vs_baselines_significance.csv` | **Main table**: final system vs 7 baselines on 5 metrics, Holm-corrected |
| `tables/two_stage_selection_accuracy.csv` | Stage-1 pick vs d1 top-1 (McNemar) |
| `tables/error_propagation.csv` | Does a wrong detector label damage the reply? |
| `tables/d1_backbone_significance.csv` | Detector backbone differences |
| `tables/length_vs_wassa_correlation.csv` | Spearman correlation of reply length with WASSA |
| `tables/initial_pipeline_significance_incl_judges.csv` | Earlier tests, including the AI-judge overall score |
| `tables/ai_judge_scores.csv` | Judge averages (final rows pending) |
| `tables/ai_judge_agreement_PENDING.csv` | Inter-judge agreement, to compute |
| `figures/fig_wassa_empathy_ci.png` | Confidence intervals, visual |

## Checklist

- [ ] 5.3.1 Statistical methodology (tests, CIs, correction, decision rule)
- [ ] 5.3.2 Final system vs baselines
- [ ] 5.3.3 Is the detector the bottleneck? (error propagation + backbone tests)
- [ ] 5.3.4 Stage-1 emotion pick
- [ ] 5.3.5 Confounding by reply length
- [ ] 5.3.6 AI-judge reliability and agreement `[[PENDING]]`
- [ ] 5.3.7 Human evaluation statistics `[[PENDING]]`
- [ ] 5.3.8 Limitations of the analysis

---

## Draft

### 5.3.1 Statistical Methodology

Every generation system answers the same 300 conversations and every detector is tested on the same 2,542, so comparisons are **paired**. For binary per-reply outcomes (all rules followed, acknowledgement, exactly one question, emotion validation, correct classification) we use **McNemar's exact test**, which considers only the conversations where the two systems disagree. For the continuous WASSA score we use the **Wilcoxon signed-rank test**, which does not assume normally distributed differences. For each comparison we report the mean paired difference with a **95% bootstrap confidence interval** (10,000 resamples of conversations, seed 42). Comparing the final system against seven baselines on each metric creates seven tests per metric, so p-values are adjusted with the **Holm-Bonferroni** procedure within each metric. **We call a difference significant only if the Holm-adjusted p-value is below 0.05 and the confidence interval excludes zero.** Independent-group comparisons (the error-propagation analysis) use Fisher's exact test for binary outcomes and the Mann-Whitney U test for scores.

### 5.3.2 Final System versus Baselines

Table 5.x reports all comparisons. Three patterns emerge.

**Empathy and emotion fidelity improve significantly over comparable designs.** The final system's WASSA score exceeds every baseline except the larger single model: the same-size single model by +0.21 points (95% CI [+0.08, +0.35], adjusted p = 0.012), the initial pipeline by +0.25 ([+0.10, +0.39], p = 0.003), and the same generator with top-1 prompting by +0.16 ([+0.05, +0.26], p = 0.008). Emotion validation is higher than all baselines except the larger single model, by +8.3 to +16.7 percentage points (all adjusted p < 0.013). Against the larger single model the difference, +4.3 points, is not significant (p = 0.19).

**Rule adherence does not significantly exceed the size-matched single model after correction.** The final system follows all rules in 60.0% of replies against 51.3% for the same-size single model (+8.7 points, CI [+1.0, +16.3]). The unadjusted p-value is 0.032, but after Holm correction it is 0.095, so we do not claim this difference. The final system is significantly better than zero-shot (+41.0), the fine-tuned model (+60.0) and RAG (+16.3), and not significantly different from the initial pipeline (-3.7) or top-1 Phi prompting (-5.3). On the exactly-one-question rule it is significantly better than the same-size single model (+19.0 points, p < 0.001).

**The larger single model is significantly better on rules and empathy.** Qwen2.5-7B exceeds the final system on all rules (-22.3 points), acknowledgement (-16.7), one question (-6.3) and WASSA (-0.71), all significant after correction.

[Table: final_system_vs_baselines_significance.csv]

### 5.3.3 Is the Detector the Bottleneck?

Two analyses test whether detection errors limit response quality. First, **error propagation**: we split each pipeline's replies by whether the detector's emotion matched the gold label and compare the groups (Table 5.x). In no pipeline does a wrong label significantly reduce rule adherence (for example, initial pipeline 61.2% when right vs 66.0% when wrong, p = 0.40; final system p = 0.096). For the initial pipeline, the one pipeline where WASSA was split this way, it does not reduce WASSA empathy either (4.83 vs 4.78, p = 0.72). Second, a **controlled detector swap**: keeping the generator fixed (Qwen2.5-1.5B) and replacing the GPT-2 detector with the more accurate RoBERTa detector changes neither emotion validation (25.3% vs 26.0%, p = 0.91) nor rule adherence (57.7% vs 63.7%, p = 0.09). By contrast, the detector backbones themselves do differ significantly: RoBERTa-large beats RoBERTa-base on top-1 (+2.9 points, p = 0.0002), and RoBERTa-base beats BERT and ELECTRA on top-5 (p <= 0.0001). Together these results indicate that, within the accuracy range studied, **detection accuracy is not the factor limiting reply quality**.

[Table: error_propagation.csv] [Table: d1_backbone_significance.csv]

### 5.3.4 The Stage-1 Emotion Pick

The generator's stage-1 choice matches the gold label in 42.3% of conversations, versus 50.3% for the detector's top-1 on the same conversations. The difference of -8.0 points (95% CI [-13.3, -2.7]; McNemar p = 0.005; 46 conversations where only the top-1 was right, 22 where only the pick was right) is significant. The two-stage design's improvements in 5.3.2 therefore cannot be explained by more accurate emotion selection (Section 5.6).

[Table: two_stage_selection_accuracy.csv]

### 5.3.5 Confounding by Reply Length

WASSA scores correlate positively with reply length within every system (Spearman rho from 0.14 to 0.54, all p < 0.02; Table 5.x). A system that writes longer replies can therefore gain WASSA points without being more empathetic. This affects two comparisons. It explains Gemma-3-4B's higher WASSA score (Section 5.4). And it strengthens, rather than weakens, the final system's WASSA advantage over top-1 Phi prompting, because the final system achieves the higher score with *shorter* replies (57.1 vs 80.4 words). The larger single model's advantage cannot be explained by length either: its replies average 52.9 words.

[Table: length_vs_wassa_correlation.csv]

### 5.3.6 AI-Judge Reliability and Agreement

For the systems judged before the final design, the initial pipeline's AI-judge overall score (8.23) was significantly higher than zero-shot (7.73; Wilcoxon p = 0.0004) and the fine-tuned model (4.44; p < 0.001), not significantly different from the same-size single model (8.38; p = 0.21), and significantly lower than the larger single model (9.02; p < 0.001), on 64 conversations (Table 5.x). [[PENDING: (1) the same tests for the final system once its scores return; (2) per-example Spearman correlation between each pair of judges; (3) judge averages after z-score normalisation per judge, to rule out a lenient or harsh judge driving the ranking.]]

### 5.3.7 Human Evaluation Statistics

[[PENDING: (1) exclusions from attention checks; (2) selection rate per system with a two-sided binomial test against 1/3; (3) per-rater paired comparison of how often the final system was chosen over each baseline, using the Wilcoxon signed-rank test; (4) Fleiss' kappa for agreement between raters; (5) share of items where the human majority choice equals the AI judges' top-scored system.]]

### 5.3.8 Limitations of the Analysis

The generation sample has 300 conversations, which detects differences of a few percentage points on binary metrics but may miss smaller effects. The AI-judge comparisons use 64 conversations and are correspondingly less powerful. Emotion validation relies on our own detector reading replies it was not trained on. Holm correction was applied within each metric family, not across all metrics, and the choice of seven baselines as one family is a design decision that we report explicitly.

---

## Traps

- **Always give the Holm-adjusted p-value** when citing the final-system comparisons. The raw p for rules vs same-size (0.032) looks significant, but it is not after correction.
- The WASSA vs same-size p-value is **0.012 adjusted** (0.006 raw). Older notes quote 0.006; use the adjusted one.
- Error propagation's `validation_match` rows differ by construction (a wrong label steers the reply's emotion). Don't present them as "damage".
- Spearman rho 0.54 is zero-shot, 0.14 is top-1 Phi. Quote the range, not one value as if it applied to all systems.
