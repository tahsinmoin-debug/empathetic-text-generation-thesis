# 5.2 Analysis of Design Solutions

> Note for the team (delete before submission): the rubric lists both this section and "Final Design Adjustments" as 5.2. Confirm the numbering with the supervisor and adjust headings and cross-references together.

Section 5.1 reported how each design performs. This section compares the design alternatives, explains the differences between them, and closes with an assessment against criteria beyond raw performance.

## 5.2.1 Decoupled versus Monolithic Designs

The central design question is whether emotion detection should be performed by a separate model. To answer it, we compare the decoupled designs with four monolithic alternatives responding to the same conversations (Table 5.1).

**Fine-tuning a single model end to end teaches the dataset's style, not the desired behaviour.** The generator fine-tuned with LoRA writes short, casual replies (10.0 words on average) that usually ask a question (74.0% ask exactly one) but seldom acknowledge the user's feeling (30.0%), so it follows all rules in only 22.3% of replies and has the lowest WASSA score (3.36), emotion validation (18.3%) and AI-judge scores (empathy 3.88, overall 5.96) of all systems. It faithfully learns its training data: real listener replies in EmpatheticDialogues acknowledge the feeling in only 10.6% of cases and satisfy all three rules in 4.3%. **Zero-shot prompting without an emotion or rules** keeps replies to a moderate length but asks the required single question in only 22.7% of replies (19.0% satisfy all rules), and has the second-lowest empathy score (WASSA 4.25). **A same-size single model instructed to detect the emotion and respond in one step** closes part of the gap (51.3% of replies satisfy all rules, WASSA 4.84), but its replies are barely more faithful to the true emotion than those of zero-shot prompting (25.7% against 24.7% emotion validation). The final decoupled system, whose generator is about 2.5 times the size of this baseline, reaches 60.0% rule adherence, WASSA 5.05 and 36.7% emotion validation. The improvements in empathy and emotion validation are statistically significant after correction for multiple comparisons, whereas the improvement in rule adherence is not (Section 5.3).

**The decoupled design, not the generator, produces the empathy gain.** The strongest test of the design is the single model built on the final system's own generator, Phi-4-mini, prompted to detect the emotion and respond in one pass. It reaches high rule adherence (73.7%), but its empathy and emotional fidelity stay at the level of the smaller single model (WASSA 4.75 against 4.84; emotion validation 27.0% against 25.7%). Adding the detector and emotion-specific guidance to that same generator raises WASSA to 5.05 and emotion validation to 36.7%, both significant after correction (Section 5.4.9). The decoupled system does, however, follow all three rules less often than this baseline (60.0% against 73.7%), because its prompt must carry emotion-specific guidance as well as the rules. At an equal generator the design therefore buys empathy and emotional fidelity at some cost in rule adherence.

**A larger single model remains stronger.** Qwen2.5-7B [8], prompted to detect the emotion and respond in one step, achieves the best rule adherence (82.3%) and WASSA score (5.76) of all systems, and the highest AI-judge score of all judged systems (8.90, against 8.68 for the final system). The same-size and larger single models use the same prompt, so model capacity, rather than the decoupled structure, explains most of this difference. The advantage of the decoupled design is therefore not raw output quality. It lies in making the emotional reasoning explicit and inspectable at a smaller scale, and in allowing the generator to receive emotion-specific guidance, which Section 5.2.4 shows to be important.

## 5.2.2 Detector Alternatives

Two findings shaped the choice of detector. First, **the text a detector is trained on matters more than the network used**. Training GPT-2 Medium on situation descriptions but deploying it on user messages cost about nine percentage points of accuracy (56.9% against 48.2%), and retraining on the input received in use was the largest single improvement to the detector (Table 4.4). Second, among encoders trained on the correct input, **models of the same size are statistically indistinguishable on top-1 accuracy** (48.8-50.2%), and RoBERTa-base differs only in the quality of its ranking, with top-5 accuracy of 85.0% against 81.7-82.7% (Table 4.5). RoBERTa-large adds 2.9 points of top-1 accuracy at 7.4 times the training cost. When RoBERTa-large was substituted into the final pipeline, rule adherence, WASSA empathy and emotion validation were all unchanged (Section 5.3.3), so RoBERTa-base offers the best balance.

## 5.2.3 Generator Alternatives

With the detector and prompt held fixed, the choice of generator determines rule adherence: 65.3% for Phi-4-mini, 57.7% for Qwen2.5-1.5B and 22.0% for Gemma-3-4B (Table 4.6). Gemma is the instructive case. It acknowledges feelings in 96.7% of replies and has a higher WASSA score than Phi-4-mini (5.30 against 4.89), which might suggest that it is the more empathetic model. However, it disregards the length rule, averaging 133 words with 72% of replies over the limit, and WASSA scores rise with reply length in every system (Section 5.4). When replies of similar length are compared (Figure 5.3, Table 5.10), Phi-4-mini matches or exceeds Gemma in every length band in which both have replies, and Gemma's advantage appears only above 120 words. Gemma's apparent advantage in empathy is therefore an effect of reply length, echoing the observation that longer LLM responses may be mistaken for greater empathy [29]. Its failure to respect the length constraint rules it out for a support setting in which brevity is a requirement.


![Figure 5.3](figures/fig_wassa_by_length_band.png)

*Figure 5.3: WASSA empathy score by reply length for Gemma-3-4B, Phi-4-mini (top-1) and the final system*



| Length band (words) | Gemma WASSA | Gemma n | Phi (top-1) WASSA | Phi (top-1) n | Final two-stage Phi WASSA | Final two-stage Phi n |
|---|---|---|---|---|---|---|
| <60 |  | 0 | 4.73 | 70 | 4.84 | 197 |
| 60-80 |  | 0 | 4.86 | 94 | 5.47 | 93 |
| 80-100 | 4.71 | 7 | 4.9 | 87 | 5.43 | 7 |
| 100-120 | 4.97 | 76 | 5.24 | 37 | 5 | 3 |
| 120-150 | 5.37 | 163 | 5 | 7 |  | 0 |
| 150+ | 5.63 | 54 | 5 | 5 |  | 0 |

*Table 5.10: WASSA empathy score within each reply-length band*


## 5.2.4 Integration Alternatives

To assess the contribution of the two-stage design, we compare it with a system that uses the same detector and generator but passes only the top-1 emotion with a generic prompt. With Phi-4-mini, the two-stage design raises emotion validation from 28.3% to 36.7% and WASSA empathy from 4.89 to 5.05, both significant after correction, and shortens replies from 80.4 to 57.1 words. Its rule adherence is 5.3 points lower (60.0% against 65.3%), a difference that is not statistically significant. The benefit depends on the capacity of the generator: with Qwen2.5-1.5B, two-stage prompting lowers rule adherence to 36.7% (against 57.7% with top-1) and acknowledgement to 47.0%, consistent with the earlier finding that a small model loses rule-following when given more to reason about, as with the candidate-pick variant (54.0%, Table 4.7). The larger instruction-tuned generator can use the richer, emotion-specific prompt without losing control of the output format. This comparison removes three elements together: the top-3 selection, the emotion-specific guidance and the attention cues. Their individual contributions were not separated, so the improvement is attributed to the combined design.

## 5.2.5 Retrieval Augmentation

Providing the generator with one or two similar training conversations [11] reduced rule adherence in both versions tested, to 54.3% and 43.7% against 63.7% without retrieval (Table 4.8). The retrieved listener replies model a different style from the one the rules require, since only 4.3% of EmpatheticDialogues listener replies satisfy all three rules, and compliance with the one-question rule fell to 76.0% in the first version. Filtering the examples and adding a further reminder about questions instead reduced acknowledgement from 70.3% to 57.0%. Retrieval also produced the least consistent empathy scores of any system, with a WASSA standard deviation of 1.46. For this task, examples of real human replies conflict with explicit behavioural rules rather than reinforcing them.

## 5.2.6 Multi-Criteria Assessment

Table 5.11 assesses the main alternatives against criteria beyond performance: cost, efficiency, usability and maintainability.

| Design | Performance (all rules / WASSA / emotion validation) | Cost to build | Efficiency (model size, GPU memory) | Usability and inspectability | Maintainability |
|---|---|---|---|---|---|
| Monolithic fine-tuned 1.5B | 22.3% / 3.36 / 18.3% | GPU training | 1.5B | No emotion exposed; casual replies that rarely acknowledge feelings | Retraining needed to change behaviour |
| Zero-shot 1.5B | 19.0% / 4.25 / 24.7% | None | 1.5B | No emotion exposed | Prompt edit |
| Single model 1.5B (detect and respond) | 51.3% / 4.84 / 25.7% | None | 1.5B | Emotion hidden inside the model | Prompt edit |
| Single model 7.6B (detect and respond) | 82.3% / 5.76 / 32.3% | None | 7.6B, about 5.6 GB (4-bit) | Emotion hidden inside the model | Prompt edit |
| Initial pipeline (GPT-2 + Qwen 1.5B) | 63.7% / 4.80 / 26.0% | Detector training | 0.36B + 1.5B | Top-1 emotion visible | Components independent |
| Initial pipeline with retrieval | 43.7% / 4.85 / 27.7% | Retrieval index | Adds an embedding model | Retrieved examples visible | Additional component |
| RoBERTa + Gemma-3-4B | 22.0% / 5.30 / 35.0% | Detector training | 0.13B + 4B (4-bit) | Top-1 emotion visible | Components independent |
| **Final: two-stage RoBERTa + Phi-4-mini** | 60.0% / 5.05 / 36.7% | Detector training (8.8 min) | 0.13B + 3.8B (4-bit) | Top-3 candidates, chosen emotion and cue words visible | Detector, prompts and generator independently replaceable |

*Table 5.11: Assessment of the main design alternatives against multiple criteria*

We select the two-stage design with a RoBERTa-base detector and a Phi-4-mini generator. Among designs of comparable size, it produces the most emotionally faithful replies and the highest empathy scores, its rule adherence is statistically indistinguishable from that of the best small configuration, and it is the only design that exposes its complete emotional reasoning. The larger single model outperforms it in raw quality. This trade-off is reported openly, and Section 5.6 discusses when the larger model would be the better choice.

## References

[8] A. Yang, B. Yang, B. Zhang, B. Hui et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.  
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 9459-9474.  
[29] V. Sorin, D. Brin, Y. Barash, E. Konen, A. Charney, G. Nadkarni, and E. Klang, "Large language models and empathy: Systematic review," J. Med. Internet Res., vol. 26, p. e52597, 2024, doi: 10.2196/52597.  
