# 4.2 Preliminary Design or Design (Model) Specification

This section presents the alternative designs considered for each part of the system, the simulation evidence that decided between them, and the full specification of the selected design.

## 4.2.1 Overview

Figure 4.2 shows the selected design. The user's latest message, together with their preceding turns, is read by the emotion detector d1, a fine-tuned RoBERTa-base classifier [6], which outputs the three most probable of the 32 emotions. In a first, deterministic stage, the generator Phi-4-mini-instruct [7] chooses which of these three emotions best fits the message. The system prompt is then assembled from guidance specific to the chosen emotion, three behavioural rules, and the words that received the most attention within d1. In the second stage, Phi-4-mini generates the reply. Every intermediate decision is recorded. The following subsections describe the alternatives simulated for each component and the evidence behind each choice.


![Figure 4.2](figures/fig_final_architecture.png)

*Figure 4.2: Architecture of the selected system*


## 4.2.2 Emotion Detector Alternatives (d1)

**GPT-2 Medium (initial design).** The initial detector was GPT-2 Medium, a 355-million-parameter decoder-only model, fine-tuned on the dataset's situation descriptions. On those descriptions it reached 56.9% top-1 accuracy, but when evaluated on the input it actually receives, the user's first message, its accuracy fell to 48.2% (Table 4.4). This design was rejected for two reasons: it was trained on the wrong kind of text, and a left-to-right decoder is a weaker architecture for classification than a bidirectional encoder [5].


| d1 version | Trained on | Evaluated on | Top-1 (%) | Top-5 (%) | Note |
|---|---|---|---|---|---|
| GPT-2 Medium (initial) | situation `prompt` field | situation text | 56.9 | 90.5 | Matches training text, but NOT what the system receives at inference |
| GPT-2 Medium (initial) | situation `prompt` field | first user message (real input) | 48.2 | 82.7 | Honest deployment number - train/inference mismatch costs ~9 points |
| RoBERTa-base (final) | first user message | first user message (real input) | 50.3 | 84.3 | Deployed checkpoint; top-3 = 76.5% |

*Table 4.4: Effect of training the detector on the text it receives in use*


**Encoder candidates.** We then trained four encoder models under identical conditions: the same first-utterance data, four epochs, a maximum length of 128 tokens, a warmup ratio of 0.06, and the same test partition (Table 4.5, Figure 4.3). BERT-base [5] reaches 48.8% top-1 and 81.7% top-5 accuracy, ELECTRA-base 50.2% and 82.7%, and RoBERTa-base [6] 50.1% and 85.0%. The three base-sized models are statistically tied on top-1 accuracy, but RoBERTa-base is significantly better than both of the others on top-5 accuracy (Table 5.15). Because the selected design passes the detector's three best candidates to the generator, the quality of the ranking matters more than the single top label, and **RoBERTa-base was therefore selected**. RoBERTa-large, with 355 million parameters, is significantly more accurate still (53.0% top-1 and 87.2% top-5), but it requires 7.4 times the training time. When it was substituted into the final pipeline, no measure of reply quality improved (Section 5.3), so RoBERTa-base was retained.


| Backbone | Parameters (M) | Learning rate | Batch size | Top-1 acc (%) | Top-5 acc (%) | Macro-F1 (%) | Train time (min) |
|---|---|---|---|---|---|---|---|
| bert-base-uncased | 110 | 0 | 8 | 48.8 | 81.7 | 47.9 | 8.6 |
| google/electra-base-discriminator | 110 | 0 | 8 | 50.2 | 82.7 | 48.9 | 8.7 |
| roberta-base | 125 | 0 | 8 | 50.1 | 85 | 49.1 | 8.8 |
| roberta-large | 355 | 0 | 4 | 53 | 87.2 | 52.4 | 64.7 |

*Table 4.5: Comparison of detector models trained under identical settings*



![Figure 4.3](figures/fig_d1_backbone_comparison.png)

*Figure 4.3: Top-1 and top-5 accuracy of the candidate detectors*


## 4.2.3 Response Generator Alternatives (d2)

To isolate the effect of the generator, we ran a controlled comparison in which the detector (RoBERTa-base), the prompt and rules, the 300 conversations and the decoding settings were all held fixed and only d2 was changed (Table 4.6). **Qwen2.5-1.5B-Instruct** [8], the initial generator, followed all rules in 57.7% of replies. **Gemma-3-4B-it** acknowledged feelings most often (96.7%) but kept to the length limit in only 27.7% of replies, averaging 133 words, so only 22.0% of its replies satisfied all rules. **Phi-4-mini-instruct**, with 3.8 billion parameters and 4-bit quantisation, achieved the highest rule adherence at 65.3% while keeping 96.0% of replies within the length limit, and **was therefore selected**. An earlier candidate, Phi-2, was excluded because it lacks a chat template and could not reliably follow system-prompt instructions.


| Setting ID | System | Ack (%) | One question (%) | Length OK (%) | All rules (%) | Mean words | Questions / reply | >1 question (%) | WASSA empathy (0-10) | WASSA SD | Emotion validation (%) | Context awareness (%) | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bakeoff_qwen | RoBERTa d1 + Qwen2.5-1.5B d2 (top-1 emotion) | 67.67 | 84.33 | 100 | 57.67 | 34.72 | 1.07 | 10.33 | 4.78 | 1.31 | 25.33 | 31 | 300 |
| bakeoff_gemma | RoBERTa d1 + Gemma-3-4B d2 (top-1 emotion) | 96.67 | 92 | 27.67 | 22 | 133.25 | 1.08 | 8 | 5.3 | 0.84 | 35 | 48.67 | 300 |
| bakeoff_phi | RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 72 | 94.67 | 96 | 65.33 | 80.45 | 1.05 | 5 | 4.89 | 1.04 | 28.33 | 36.67 | 300 |

*Table 4.6: Generator candidates with the detector, prompt and conversations held fixed*


## 4.2.4 Integration Alternatives: How d1's Output Reaches d2

Because d1 produces a full probability distribution, its output can be passed to the generator in several ways. Using the initial detector and generator, we simulated four variants on the same 300 conversations (Table 4.7). *Argmax* passes only the most probable emotion and achieved 63.7% rule adherence. *Confidence-gated* switches to tentative wording when d1 is uncertain and achieved 63.3%, but only 28.7% of the low-confidence replies actually used tentative wording. *Distribution* lists every emotion with probability above 0.05 and achieved 62.0%. *Candidate-pick* asks the generator to choose among the candidates and write the reply in a single call, and achieved 54.0%. None of the variants outperformed argmax, and candidate-pick reduced rule adherence, suggesting that a single generation call cannot both resolve the emotion and follow the rules reliably.

This analysis motivated the **two-stage** design adopted in the final system. The correct emotion lies among d1's three best candidates in 76.5% of test cases, compared with 50.3% for its top candidate alone, so the wider candidate set is valuable. However, combining several emotions in one prompt produces contradictory guidance: responding well to *disappointed*, for instance, is very different from responding well to *impressed*. The two-stage design therefore separates the choice of emotion into its own short, deterministic call, and gives the generation call a single, coherent prompt for that emotion.


| Variant | Ack% | OneQ% | LenOK% | AllRules% | LowConfHedge% |
|---|---|---|---|---|---|
| argmax - top-1 emotion only (adopted) | 70.3 | 87.3 | 100 | 63.7 | - |
| confidence-gated - hedge when d1 is unsure | 71 | 88.3 | 100 | 63.3 | 28.7 |
| distribution - list all emotions above p=0.05 | 72.7 | 83.7 | 100 | 62 | - |
| candidate-pick - d2 chooses AND generates in one pass | 65.3 | 82.7 | 100 | 54 | - |

*Table 4.7: Alternative ways of passing the detector's output to the generator*


## 4.2.5 Monolithic and Retrieval Alternatives

To test whether a separate detector is needed at all, four monolithic alternatives were simulated using the same rules and conversations. In the *zero-shot* alternative, Qwen2.5-1.5B receives a generic caring prompt and no detected emotion. In the *monolithic fine-tuned* alternative, Qwen2.5-1.5B is fine-tuned with LoRA [10] directly on pairs of user messages and human replies. In the *single model of the same size*, Qwen2.5-1.5B is instructed to identify the emotion silently and then respond under the three rules. In the *larger single model*, the same prompt is given to Qwen2.5-7B in 4-bit form. Their results are analysed in Section 5.2. In brief, the fine-tuned model failed every rule, the zero-shot model rarely asked the single required question, the same-size single model trailed the decoupled designs, and the larger single model performed best overall.

Following the Pre-Thesis 2 panel's suggestion, we also simulated a **retrieval-augmented** variant [11] that retrieves one or two similar training conversations with the same detected emotion, using all-MiniLM-L6-v2 sentence embeddings [12], and shows them to the generator as examples. Across two versions, retrieval reduced rule adherence from 63.7% to 54.3% and then to 43.7%, so it was not adopted (Table 4.8).


| Variant | Ack (%) | One question (%) | All rules (%) | WASSA SD | Source |
|---|---|---|---|---|---|
| Initial pipeline (no retrieval) | 70.33 | 87.33 | 63.67 | 1.33 | result files |
| RAG attempt 1 (1-2 retrieved ED examples) |  | 76 | 54.3 |  | recorded in Full_Project_Documentation (outputs overwritten by attempt 2) |
| RAG attempt 2 (+ single-question filter + reminder) | 57 | 78 | 43.67 | 1.46 | result files |

*Table 4.8: Effect of retrieval augmentation on rule adherence*


## 4.2.6 Specification of the Selected Design

**Detector.** The detector is RoBERTa-base with a 32-way classification head, fine-tuned on first utterances with a learning rate of 2e-5, an effective batch size of 16, up to four epochs with early stopping, a warmup ratio of 0.06, weight decay of 0.01, gradient clipping at 1.0 and 16-bit mixed precision. In conversation, d1 reads the user's last four turns joined together. Conversational context is known to be important for recognising emotions in dialogue [44], and among the windows we tested on long conversational scenarios, four turns was best or joint best in three of four scenarios, whereas six turns degraded two scenarios because emotions from earlier turns competed with the current one.

**Attention cues.** d1 is loaded with an attention implementation that returns attention weights, because the default implementation in the Transformers library [40] returned none in our verification. We take the attention from the first token to every other token in the final layer [4], averaged across attention heads, keep the highest-scoring 40% of word spans (between 3 and 14), and expand sub-word pieces to whole words.

**Stage 1: emotion selection.** Phi-4-mini receives the user's message and the three candidate emotions with the instruction: *"An automatic detector narrowed the person's likely feeling down to these options: {a}, {b}, {c}. Read their message and decide which ONE of those options best fits what they are actually expressing. Reply with ONLY that single word."* Decoding is greedy with at most 10 new tokens. If the output does not name one of the candidates, the system falls back to d1's most probable emotion; in simulation, every output named a valid candidate.

**Stage 2: generation.** The system prompt combines, in order, the chosen emotion, the cue words, emotion-specific guidance on what to focus on and what to avoid, and the three universal rules. For example, for *lonely* the prompt instructs the model to "acknowledge the isolation itself, gently - the weight of it, not just 'you seem sad'" and not to "offer a surface-level fix like 'go meet people'", followed by the rules to acknowledge and validate the feeling in two to three sentences, share a reassuring thought specific to that feeling, and ask exactly one short, gentle question at the end, all within 100 words. Guidance of this kind was written for all 32 emotions [[TEAM TO FILL: name the source document from which the per-emotion guidance was adapted]]. Phi-4-mini is loaded with 4-bit NF4 quantisation, double quantisation and 16-bit computation [9], and samples with temperature 0.7 and top-p 0.9, up to 250 new tokens.

**Conversation memory.** The generator receives the last ten exchanges verbatim. From the second turn onward, the question becomes optional and the model is asked not to reuse earlier phrasing, so that a multi-turn conversation does not become an interrogation. The first turn keeps the original rules, so single-turn results remain comparable.

Table 4.9 lists every setting of the selected design together with its justification.


| Component | Setting | Value | Justification |
|---|---|---|---|
| d1 | Backbone | RoBERTa-base (125M) | Top-5 significantly above BERT/ELECTRA at equal top-1; pipeline uses top-3; RoBERTa-large tested in-pipeline with no reply gain |
| d1 | Input text | First user utterance, max 128 tokens | Matches inference-time input (fixes mismatch) |
| d1 | Learning rate / epochs | 2e-5 / 4 (early stopping, patience 1) | Standard BERT-family fine-tuning range |
| d1 | Batch | 8 x 2 accumulation = 16 | Fits 8 GB VRAM |
| d1 | Warmup / weight decay / grad clip | 0.06 / 0.01 / 1.0 | Warmup 0.06 from RoBERTa paper; prevents large-model divergence |
| d1 (large, comparison only) | Learning rate / batch | 1e-5 / 4 | Base settings diverged |
| Two-stage | Candidates passed to d2 | Top-3 | Gold in top-3 76.5% vs top-1 50.3% |
| Two-stage | Stage-1 decoding | Greedy, max 10 tokens | Classification decision, deterministic |
| d2 | Model | Phi-4-mini-instruct, 4-bit NF4, double quant, fp16 compute | Best rule adherence in bake-off; fits 8 GB |
| d2 | Sampling | temperature 0.7, top-p 0.9, max 250 new tokens, seed 42 | Same across every generation experiment for fairness |
| d2 | Prompt | Per-emotion focus/avoid guidance (32) + 3 rules + attention cues, <= 100 words | Rules scored at 120 words (scorer tolerance) |
| Attention cues | Source / amount | d1 last-layer [CLS] attention, top 40% tokens, word-boundary expanded | Eager attention required |
| Conversational agent | d1 history window | Last 4 user turns | Best or tied-best in 3 of 4 long scenarios; 6 turns degraded |
| Conversational agent | d2 history window | Last 10 exchanges, verbatim | Avoids repetition; no lossy summarisation |
| AI judges | Decoding | temperature 0, JSON output | Deterministic scoring |
| Evaluation | Sample | 300 test conversations (seed 42); judges: 64 stratified (2 per emotion) | Same sample for every system |
| Statistics | Tests | McNemar exact, Wilcoxon, 10k paired bootstrap, Holm correction | Paired data; binary vs continuous metrics |

*Table 4.9: Configuration of the selected design and the justification for each setting*


## 4.2.7 Simulation and Verification

Each alternative was functionally verified before full evaluation. Small trial runs confirmed that outputs were well formed and that model-specific details were handled correctly: Phi-4-mini uses an end-of-turn token different from its end-of-text token, Gemma's chat template has no system role, and Phi-4-mini combines its attention projections into a single layer. Every generation design was then simulated on the same 300 held-out conversations with identical decoding settings, and every detector candidate on the full test partition of 2,542 conversations. Two automatic safeguards protect the selected design: execution stops if the detector returns no attention weights, and the outcome of every stage-1 decision is recorded, all of which parsed successfully. The results of these simulations determined the selections described above and are analysed in full in Chapter 5.

## References

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 5998-6008.  
[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. 2019 Conf. North American Chapter Assoc. Comput. Linguistics (NAACL-HLT), 2019, pp. 4171-4186, doi: 10.18653/v1/N19-1423.  
[6] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.  
[7] A. Abouelenin, A. Ashfaq, A. Atkinson et al. (Microsoft), "Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs," arXiv:2503.01743, 2025.  
[8] A. Yang, B. Yang, B. Zhang, B. Hui et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.  
[9] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.  
[10] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. Learning Representations (ICLR), 2022.  
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 9459-9474.  
[12] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. 2019 Conf. Empirical Methods Natural Language Processing and 9th Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982-3992, doi: 10.18653/v1/D19-1410.  
[40] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., "Transformers: State-of-the-art natural language processing," in Proc. 2020 Conf. Empirical Methods Natural Language Processing: System Demonstrations, 2020, pp. 38-45, doi: 10.18653/v1/2020.emnlp-demos.6.  
[44] S. Poria, N. Majumder, R. Mihalcea, and E. Hovy, "Emotion recognition in conversation: Research challenges, datasets, and recent advances," IEEE Access, vol. 7, pp. 100943-100953, 2019, doi: 10.1109/ACCESS.2019.2929050.  
