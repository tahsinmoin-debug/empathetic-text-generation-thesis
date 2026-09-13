# Chapter 5 - 5.2 Analysis of Design Solutions

**Rubric:** CO6. Assess the alternative designs "to determine the most appropriate one, considering factors such as cost, efficiency, usability, manufacturability, impact, sustainability, maintainability, and performance". Criterion: analyse the alternatives to obtain the best suitable solution.

## What this section must do

5.1 reported numbers. This section **compares design alternatives and explains the outcomes**, using the ablation logic of the COLING papers ("to substantiate the role of X, we remove it..."), and ends with a multi-criteria judgement covering more than raw performance. Each paragraph answers: which alternatives were compared, what won, by how much, and why.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/all_systems_master_comparison.csv` | Source for every comparison below |
| `figures/fig_rule_adherence_all_systems.png` | May be cross-referenced from 5.1 instead of repeated |
| `tables/d1_backbone_comparison.csv`, `figures/fig_d1_backbone_comparison.png` | Detector analysis |
| `tables/gemma_length_band_analysis.csv`, `figures/fig_wassa_by_length_band.png` | Generator analysis |
| `tables/rag_negative_result.csv` | RAG analysis |

## Checklist

- [ ] 5.2.1 Decoupled vs monolithic designs
- [ ] 5.2.2 Detector alternatives
- [ ] 5.2.3 Generator alternatives
- [ ] 5.2.4 Integration alternatives (top-1 vs two-stage)
- [ ] 5.2.5 Retrieval augmentation
- [ ] 5.2.6 Multi-criteria assessment and selection

---

## Draft

### 5.2.1 Decoupled versus Monolithic Designs

The central design question is whether emotion detection should be a separate model. We compare the decoupled designs with four monolithic alternatives answering the same conversations.

**Fine-tuning a single model end to end is the worst option.** The LoRA fine-tuned generator follows all rules in 0% of replies, averaging 7.07 questions and 200.7 words. Trained to imitate human listener replies without any explicit structure, it reproduces the dataset's conversational habit of asking follow-up questions and exaggerates it. **Zero-shot prompting without emotion or rules** keeps replies moderate in length but asks the required single question in only 22.7% of replies (19.0% all rules) and scores lowest but one on empathy (WASSA 4.25). **A same-size single model instructed to detect and respond in one pass** closes part of the gap (51.3% all rules, WASSA 4.84), but its replies are barely more faithful to the true emotion than zero-shot (25.7% vs 24.7% emotion validation). The final decoupled system, with a generator 2.5 times the size of this baseline, reaches 60.0% all rules, WASSA 5.05 and 36.7% emotion validation. The empathy and emotion-validation gains are statistically significant after correction. The rule-adherence gain is not (Section 5.3).

**A larger single model remains stronger.** Qwen2.5-7B, prompted to detect and respond in one pass, achieves the best rule adherence (82.3%) and WASSA score (5.76) of all systems, and the highest AI-judge average among the systems judged so far (9.02). The same-size and larger single models share one prompt, so model capacity, not the decoupled structure, explains most of this difference. The decoupled design's advantage is therefore not raw quality. It is that its emotional reasoning is explicit and inspectable at a smaller scale, and that the generator can be given emotion-specific guidance, which Section 5.2.4 shows to matter.

### 5.2.2 Detector Alternatives

Two findings shape the detector choice. First, **what the detector is trained on matters more than which network is used.** Training GPT-2 Medium on situation descriptions but deploying it on user messages cost about 9 points of accuracy (56.9% vs 48.2%). Retraining on the deployment input was the single largest improvement to the detector. Second, among encoders trained on the right input, **backbones of the same size are statistically indistinguishable at top-1** (48.8-50.2%), and RoBERTa-base differs only in ranking quality (top-5 85.0% vs 81.7-82.7%). Doubling capacity with RoBERTa-large adds 2.9 points top-1 at 7.4 times the training cost (Figure 5.3). When RoBERTa-large was swapped into the final pipeline, rule adherence, WASSA empathy and emotion validation were all unchanged (Section 5.3.3), so RoBERTa-base offers the best balance.

### 5.2.3 Generator Alternatives

With the detector and prompt fixed, the generator determines rule adherence: Phi-4-mini 65.3%, Qwen2.5-1.5B 57.7%, Gemma-3-4B 22.0%. Gemma is the instructive case. It acknowledges feelings in 96.7% of replies and has a higher WASSA score than Phi (5.30 vs 4.89), which could suggest it is the more empathetic model. But it ignores the length rule, averaging 133 words with 72% of replies over the limit, and WASSA scores rise with reply length in every system (Section 5.4). Comparing replies of similar length (Figure 5.4), Phi matches or exceeds Gemma in every length band where both have replies, and Gemma's advantage appears only above 120 words. Gemma's apparent empathy advantage is therefore a length effect, and its inability to respect the length constraint disqualifies it for a support setting where brevity is a requirement.

[Figure: fig_wassa_by_length_band.png] [Table: gemma_length_band_analysis.csv]

### 5.2.4 Integration Alternatives

To substantiate the role of the two-stage design, we compare it with the same detector and generator using only the top-1 emotion and a generic prompt (w/o two-stage). With Phi-4-mini, the two-stage design raises emotion validation from 28.3% to 36.7% and WASSA from 4.89 to 5.05, both significant after correction, and it shortens replies from 80.4 to 57.1 words. All-rules adherence is 5.3 points lower (60.0% vs 65.3%), a difference that is not significant. The design depends on generator capacity: with Qwen2.5-1.5B, two-stage prompting lowers all-rules adherence to 36.7% (vs 57.7% for top-1) and acknowledgement to 47.0%, consistent with the earlier finding that a small model loses rule-following when given more to reason about (the candidate-pick variant, 54.0%). The larger instruction-tuned generator can use the richer, emotion-specific prompt without losing control of the output format. Note that this comparison removes three things together: the top-3 pick, the per-emotion guidance and the attention cues. Their individual contributions were not ablated separately, so the gain should be attributed to the combined design.

### 5.2.5 Retrieval Augmentation

Showing the generator one or two similar training conversations reduced all-rules adherence in both versions tried (54.3% and 43.7% vs 63.7% without retrieval). The retrieved human replies often contained several questions, and the model imitated them despite the rule (one-question compliance fell to 76.0% in the first version). Filtering the examples and adding an extra reminder about questions reduced acknowledgement from 70.3% to 57.0% instead. RAG also produced the least consistent empathy scores of any system (WASSA standard deviation 1.46). For this task, examples of real human replies conflict with explicit behavioural rules rather than supporting them.

### 5.2.6 Multi-Criteria Assessment

Table 5.x summarises the main alternatives against the rubric's criteria.

| Design | Performance (all rules / WASSA / emotion validation) | Cost to build | Efficiency (model size, GPU memory) | Usability and inspectability | Maintainability |
|---|---|---|---|---|---|
| Monolithic fine-tuned 1.5B | 0.0% / 4.19 / 20.0% | GPU training | 1.5B | No emotion exposed; uncontrolled length | Retrain to change behaviour |
| Zero-shot 1.5B | 19.0% / 4.25 / 24.7% | None | 1.5B | No emotion exposed | Prompt edit |
| Single model 1.5B (detect + respond) | 51.3% / 4.84 / 25.7% | None | 1.5B | Emotion hidden inside the model | Prompt edit |
| Single model 7B (detect + respond) | **82.3% / 5.76** / 32.3% | None | 7.6B, about 5.6 GB (4-bit) | Emotion hidden inside the model | Prompt edit |
| Initial pipeline (GPT-2 + Qwen 1.5B) | 63.7% / 4.80 / 26.0% | Detector training | 0.36B + 1.5B | Top-1 emotion visible | Components independent |
| + RAG | 43.7% / 4.85 / 27.7% | Retrieval index | + embedding model | Retrieved examples visible | Extra component |
| RoBERTa + Gemma-4B | 22.0% / 5.30 / 35.0% | Detector training | 0.13B + 4B (4-bit) | Top-1 emotion visible | Components independent |
| **Final: two-stage RoBERTa + Phi-4-mini** | 60.0% / 5.05 / **36.7%** | Detector training (8.8 min) | 0.13B + 3.8B (4-bit) | Top-3, chosen emotion and cue words all visible | Detector, prompts and generator independently replaceable |

We select the two-stage RoBERTa-base + Phi-4-mini design. Among designs of comparable size it provides the most emotionally faithful and highest-scoring empathetic replies, it keeps rule adherence within statistical noise of the best small configuration, and it is the only design that exposes its complete emotional reasoning. The larger single model outperforms it on raw quality. That trade-off is acknowledged rather than hidden, and Chapter 5.6 discusses when the larger model would be the better choice.

---

## Traps

- The **"2.5 times the size"** comparison (3.8B vs 1.5B) is exactly why the same-size comparison is not perfectly same-size for the final system. Say it openly. The initial pipeline (Qwen 1.5B generator) is the truly size-matched decoupled design.
- GPU memory for the final pipeline was **not measured**. Only the 7B model's 5.6 GB was recorded. Don't fill that cell with a guess.
- 5.2.4 compares two-stage against top-1 **with the same generator**. That is the clean ablation; keep it that way.
- 5.2.4 is a **bundled** ablation (pick + per-emotion prompt + cues removed together). If asked which part helps most, the honest answer is that it was not separated. The stage-1 accuracy result (5.1.6) suggests the pick itself is not the source.
