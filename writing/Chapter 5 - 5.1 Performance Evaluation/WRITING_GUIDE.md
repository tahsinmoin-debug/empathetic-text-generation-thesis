# Chapter 5 - 5.1 Performance Evaluation

> **Update (14 Sep 2026): AI-judge scores for all 8 systems are in.** The AI-judge panel is two judges (Nemotron 3 Ultra + gpt-oss-120b), and every judge number in the draft below is superseded - use `FINAL_TEXT.docx` and the regenerated `tables/` for all AI-judge and WASSA-validity numbers.

**Rubric:** CO6, "Analyze and assess alternative design solutions ... considering factors such as cost, efficiency, usability ... and performance in relation to provided specifications, prerequisites, and benchmarks." Criterion: perform analysis of alternative designs to obtain the best suitable solution. Marks: 5 from the defence panel + 5 from supervisor evaluation of the final report.

## What this section must do

Report how the **final system** performs on every evaluation signal, with every alternative alongside for reference. This is the results section. Interpretation of *why* one design beats another belongs in 5.2, and significance in 5.3. Follow the COLING results pattern: open each subsection by pointing at its table or figure, state the key number, then give a one-sentence reason.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/d1_final_overall.csv` | Detector headline metrics |
| `tables/d1_final_per_class.csv` | Per-emotion F1 (appendix if long) |
| `tables/d1_most_confused_pairs.csv` | Error analysis |
| `tables/all_systems_master_comparison.csv` | **Main results table**, all 11 systems |
| `tables/final_system_scorecard.csv` | Final system at a glance |
| `figures/fig_rule_adherence_all_systems.png` | Rule adherence |
| `tables/wassa_empathy_all_systems.csv` + `figures/fig_wassa_empathy_ci.png` | Empathy classifier |
| `tables/two_stage_selection_accuracy.csv` | Behaviour of the emotion-pick stage |
| `tables/diversity_and_repetition.csv` | Diversity and boilerplate |
| `tables/ai_judge_scores.csv`, `ai_judge_per_dimension.csv` | AI-judge panel, all 7 systems (complete) |
| `tables/human_eval_item_design.csv`, `human_eval_results_PENDING.csv` | Human study (**pending**) |

## Checklist

- [ ] 5.1.1 Experimental setup (short; refer back to 4.1.5)
- [ ] 5.1.2 Emotion detection
- [ ] 5.1.3 Rule adherence
- [ ] 5.1.4 Empathy (WASSA)
- [ ] 5.1.5 Emotion fidelity (validation, context awareness), **with the circularity caveat**
- [ ] 5.1.6 Two-stage selection behaviour
- [ ] 5.1.7 Diversity and repetition
- [x] 5.1.8 AI-judge evaluation (complete - see FINAL_TEXT)
- [ ] 5.1.9 Human evaluation `[[PENDING]]`
- [ ] 5.1.10 Case study (qualitative examples)

---

## Draft

### 5.1.1 Experimental Setup

All generation systems answer the same 300 conversations, sampled with seed 42 from the EmpatheticDialogues test split, using identical decoding settings (Section 4.1.4). The detector is evaluated on all 2,542 test conversations. We refer to the systems by the short names in Table 5.1. The **final system** is the two-stage pipeline with a RoBERTa-base detector and Phi-4-mini generator.

### 5.1.2 Emotion Detection

Table 5.2 reports the final detector's performance. RoBERTa-base reaches **50.3% top-1**, **76.5% top-3** and **84.3% top-5** accuracy, with a macro-F1 of 49.4%, against a chance level of 3.1% for 32 classes. Performance varies widely by emotion (Table 5.3): *caring* (F1 70.1%), *lonely* (69.5%) and *jealous* (64.7%) are recognised most reliably, while *angry* (18.2%), *ashamed* (22.2%) and *joyful* (30.9%) are the hardest. Table 5.4 shows that the most frequent errors are between closely related labels: *sentimental* read as *nostalgic* (31 cases), *afraid* as *terrified* (29), *angry* as *furious* (24) and *sad* as *devastated* (23). Many errors are therefore differences of intensity or nuance within one emotion family, not confusions between unrelated feelings. This is why the final design passes the top-3 candidates rather than a single label.

[Table: d1_final_overall.csv] [Table: d1_final_per_class.csv] [Table: d1_most_confused_pairs.csv]

### 5.1.3 Rule Adherence

Table 5.1 and Figure 5.1 report how often each system's reply acknowledges the user's feeling, asks exactly one question and stays within 120 words. The final system follows all three rules in **60.0%** of replies: 66.3% acknowledge, 92.7% ask exactly one question, and 100% stay within length, with an average of 57.1 words. Among systems of comparable size it is exceeded by the top-1 Phi-4-mini configuration (65.3%) and the initial pipeline (63.7%), and it is well ahead of the same-size single model (51.3%), zero-shot prompting (19.0%) and the monolithic fine-tuned model (0.0%). The larger single model (Qwen2.5-7B) has the highest rule adherence at 82.3%. The monolithic fine-tuned model shows the failure the rules were designed to prevent: it averages 7.07 questions and 200.7 words per reply, and 98.0% of its replies ask more than one question.

[Figure: fig_rule_adherence_all_systems.png] [Table: all_systems_master_comparison.csv]

### 5.1.4 Empathy Classifier Score

Figure 5.2 shows WASSA empathy scores with 95% confidence intervals. The final system scores **5.05**, the highest of all systems at or below Phi-4-mini's size and above the same-size single model (4.84), the initial pipeline (4.80), zero-shot (4.25) and the fine-tuned model (4.19). Gemma-3-4B (5.30) and the larger single model (5.76) score higher. Section 5.4 shows that Gemma's advantage is explained by reply length. The classifier outputs coarse ordinal classes (our replies fall in classes 1-7) and was trained on empathy ratings of turns in peer discussions of news articles, so we interpret differences between system averages rather than absolute values. Section 5.3.6 shows it ranks systems exactly as the AI judges do.

[Figure: fig_wassa_empathy_ci.png] [Table: wassa_empathy_all_systems.csv]

### 5.1.5 Emotion Fidelity

To check whether a reply expresses the *right* emotion, and not merely a well-formed one, we pass each reply back through the detector and compare its reading with the true label (*emotion validation*) and with the emotion the system conditioned on (*context awareness*). The final system has the highest emotion validation of all systems, **36.7%**, against 32.3% for the larger single model, 28.3% for the top-1 Phi configuration, 26.0% for the initial pipeline and 25.7% for the same-size single model. Its context awareness is 54.3%.

Two caveats limit how far these numbers can be pushed. The detector was trained on user messages, not replies, so its reading of a reply is noisy. And context awareness uses the same detector that produced the signal for our pipelines, which can favour them. Emotion validation compares against the dataset's independent gold label and is the more trustworthy of the two.

### 5.1.6 Behaviour of the Two-Stage Emotion Pick

Table 5.5 examines the stage-1 decision. The true emotion is among d1's top-3 in 78.0% of the 300 evaluation conversations, and every stage-1 output parses to a valid candidate. The generator overrides d1's top-1 choice in 40.7% of cases. Its pick matches the gold label in **42.3%** of conversations, which is *lower* than d1's own top-1 accuracy on the same sample (50.3%). The final system's gains in empathy and emotion validation therefore do not come from choosing the gold label more often. Section 5.6 discusses this finding.

[Table: two_stage_selection_accuracy.csv]

### 5.1.7 Diversity and Repetition

Table 5.6 reports distinct-n diversity and a boilerplate rate: the share of replies containing one of that system's ten most frequent five-word phrases. The final system's Dist-2 of 47.8% is close to the initial pipeline's (49.5%), but its boilerplate rate is higher (58.7% vs 23.7%). Its most repeated phrase, "how are you feeling right", appears in 49 replies. Phi-4-mini with a single generic prompt is the most repetitive system (77.0%), so the per-emotion prompts reduce repetition relative to the same generator, but do not eliminate it. These values use 300 replies and are not comparable to published distinct-n scores computed on the full test set.

[Table: diversity_and_repetition.csv]

### 5.1.8 AI-Judge Evaluation

Two AI judges, Nemotron 3 Ultra and gpt-oss-120b (selection rationale in Section 4.1.5.1), score each reply blind from 1 to 10 for empathy, relevance and safety, on a stratified sample of 64 conversations (two per emotion, the shortest and longest message). Table 5.7 reports the results. Among the systems already judged, the larger single model has the highest average (9.02), followed by the same-size single model (8.38), the initial pipeline (8.23), zero-shot (7.73) and the fine-tuned model (4.44). [[PENDING: final system and top-1 Phi configuration averages; one sentence on where the final system ranks and whether its empathy dimension differs from relevance and safety.]]

[Table: ai_judge_scores.csv] [Table: ai_judge_per_dimension.csv]

### 5.1.9 Human Evaluation

To validate the automatic and AI-judge results, we conducted a human forced-choice study (design in Table 5.8). Participants read 13 situations and, for each, chose which of three replies "best understands and responds to how this person is feeling": the final system, the same-size single model or the larger single model, in counterbalanced order. Situations cover 13 different emotions, including four central to support conversations (sad, lonely, angry, anxious) and three student-life scenarios. Two items include an obviously poor reply as an attention check. [[PENDING: number of participants; number excluded by attention checks; selection rate per system (chance = 33.3%) with binomial tests; Fleiss' kappa; agreement between the human majority choice and the AI judges' top-scored system.]]

[Table: human_eval_item_design.csv] [Table: human_eval_results_PENDING.csv]

### 5.1.10 Case Study

[[TEAM TO FILL: show 2-3 conversations as a table in the COLING main.598 style (user message, then the reply from each system). To avoid cherry-picking, use the human-evaluation items, which were drawn at random, and say so. The replies are in `human_evaluation/Human_Evaluation_Study_Plan.docx` and the system for each reply is in `human_evaluation/ANSWER_KEY_do_not_share_with_raters.csv`.]]

---

## Traps

- **Top-3 76.5% and top-5 84.3% are the deployed model's numbers.** The comparison table in 4.2 shows 85.0% top-5 for RoBERTa-base because it comes from a re-run under identical comparison settings. Say which run each number comes from if both appear.
- On the 300-conversation sample, gold-in-top-3 is 78.0%. On the full test set it is 76.5%. Keep the two apart.
- **Do not describe the final system as best on rule adherence.** It is not. Its strengths are empathy score and emotion fidelity.
- Context awareness is partly circular (see 5.1.5). Lead with emotion validation.
