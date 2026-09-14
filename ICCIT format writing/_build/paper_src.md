TITLE: Does a Separate Emotion Detector Help Small Language Models Respond Empathetically? A Controlled Study on a Consumer GPU

## Abstract

Small language models that run on local hardware are attractive for emotionally supportive conversation, because a user's disclosure never has to leave the device. It is unclear, however, whether such models benefit from a separate emotion detector or should simply be prompted to recognise the emotion and respond in one step. We present a controlled study on the EmpatheticDialogues benchmark in which every system uses the same 300 test conversations, the same decoding settings and a single 8 GB GPU. Our decoupled system passes the three most probable of 32 emotions from a RoBERTa-base detector (50.3% top-1 accuracy), together with emotion-specific guidance and attention-derived cue words, to a 4-bit Phi-4-mini generator. We evaluate replies with rule adherence, the WASSA 2024 empathy classifier, emotion validation, a blind two-judge LLM panel and a human forced-choice study, using paired tests with Holm correction. The decoupled system (4.0B parameters) is significantly more empathetic and emotionally faithful than a 1.5B single model, but a 7.6B single model remains stronger on rule adherence and empathy; with 52% of its parameters, our system reaches 97.5% of its overall AI-judge score. Three analyses show that detection accuracy is not the bottleneck: incorrect labels do not degrade replies, a more accurate detector brings no gain, and the generator's own emotion choice is less accurate than the detector's yet yields better replies. The benefit comes from how the detected emotion is used rather than how accurately it is detected.

## Index Terms

empathetic response generation, emotion recognition, small language models, EmpatheticDialogues, LLM-as-a-judge

## I. Introduction

People increasingly turn to conversational agents when they want to talk about how they feel. A useful reply to such a message should recognise the feeling, validate it, and invite the person to continue without overwhelming them with questions {cite:rashkin2019}. Large language models (LLMs) produce fluent and often empathetic replies {cite:qian2023llm, sorin2024jmir}, but the strongest models run on remote servers, which means that sensitive disclosures are sent to a third party. Small open models that fit on a consumer GPU avoid this problem, yet they are weaker at recognising fine-grained emotions {cite:belay2025}.

A natural response is to *decouple* the task: a small encoder detects the emotion and a generator writes the reply conditioned on it. Earlier empathetic dialogue models predicted the emotion as an auxiliary objective of a single generation model {cite:lin2019moel, sabour2022cem, zhou2023case}, and Yang et al. recently proposed using the emotion predictions of such models to guide an LLM {cite:yang2024hef}. What remains unclear is whether a separate detector is worth its cost for *small, locally run* generators, and, if it helps, whether the benefit comes from detecting the emotion more accurately or from the way the detected emotion is used.

We answer these questions with a controlled comparison in which data, decoding and hardware are held fixed. Our contributions are:

- A decoupled pipeline for a single 8 GB GPU, in which a RoBERTa-base detector passes its top-3 emotions, emotion-specific guidance and attention-derived cue words to a 4-bit Phi-4-mini generator, compared against monolithic baselines at 1.5B and 7.6B parameters.
- Controlled ablations separating the effect of *decoupling* (same generator with and without a detector) from the effect of *emotion-specific guidance* (same detector and generator, with and without it).
- Three independent analyses showing that detector accuracy is not what limits reply quality.
- A multi-signal evaluation (rule adherence, WASSA empathy, emotion validation, a blind LLM-judge panel and human preference) with paired significance tests and correction for multiple comparisons, including negative results for retrieval augmentation and fine-tuning.

## II. Related Work

**Empathetic response generation.** EmpatheticDialogues (ED) {cite:rashkin2019} contains about 25k conversations grounded in 32 emotions. Models trained on it, such as MoEL {cite:lin2019moel}, MIME {cite:majumder2020mime}, EmpDG {cite:li2020empdg}, KEMP {cite:li2022kemp}, CEM {cite:sabour2022cem}, CASE {cite:zhou2023case} and ESCM {cite:yang2023escm}, predict the emotion as an auxiliary task and report emotion accuracies between 30.6% and 41.2% {cite:qian2023llm, yang2023escm}.

**LLMs for empathy.** Qian et al. found that LLMs outperform these specialised models in human evaluation {cite:qian2023llm}, and TOOL-ED improves LLM empathy by letting the model call an emotion knowledge tool {cite:cao2025tooled}. A systematic review reports that LLM replies are often rated as empathetic but tend to be repetitive and lengthy {cite:sorin2024jmir}.

**Combining small models with LLMs.** The closest work to ours is HEF {cite:yang2024hef}, which gives an LLM the top-*k* emotions and the attention-weighted cause words of a small empathetic model and reports higher emotion accuracy for ChatGPT and 6-13B open models. Our study differs in four respects: we use a dedicated classifier rather than a joint generation model; every component runs locally with at most 3.8B generator parameters; we compare against monolithic single models of smaller and larger size under identical conditions; and we test directly whether detection accuracy drives the improvement.

**Evaluation.** Strong LLM judges agree with human preferences in over 80% of cases but favour longer answers and their own outputs {cite:zheng2023judge, panickssery2024}; panels of judges from different model families reduce this bias {cite:verga2024}. The WASSA 2024 shared task provides empathy classifiers trained on human ratings of conversation turns {cite:giorgi2024wassa}.

## III. Method

### A. Task and Data

Given a user's message, the system must produce a reply that satisfies three rules derived from supportive-listening practice: it (R1) acknowledges the feeling, (R2) asks exactly one question and (R3) stays within 120 words. The rules make over-questioning measurable: a model fine-tuned end to end on ED listener replies asks 7.07 questions per reply (Section V). We use the official ED train, validation and test partitions. Each conversation contributes its *first user utterance* labelled with the conversation's emotion, which gives 2,542 test examples. We deliberately do not use the situation descriptions, because they are written after the fact and state the feeling explicitly: a GPT-2 detector trained on them scored 56.9% on situation text but only 48.2% on the user messages that a deployed system actually receives.

### B. Emotion Detector (d1)

We fine-tune RoBERTa-base {cite:liu2019roberta} (125M parameters) as a 32-way classifier on first utterances (learning rate 2e-5, effective batch size 16, linear warm-up over 6% of steps, early stopping on validation accuracy, at most 128 tokens). Among equally tuned base-size encoders, RoBERTa-base was comparable to ELECTRA in top-1 accuracy (50.1% against 50.2%), above BERT (48.8%), and gave the highest top-5 accuracy (85.0%); RoBERTa-large reached 53.0% but required 7x the training time. The deployed checkpoint reaches **50.3% top-1, 76.5% top-3 and 84.3% top-5** accuracy (macro-F1 49.4%) on the test partition, against a chance level of 3.1%. Its most frequent errors are between neighbouring emotions such as *sentimental*/*nostalgic* and *afraid*/*terrified*, which motivates passing several candidates rather than one label.

### C. Two-Stage Generator (d2)

[[FIGURE fig:arch | fig_architecture.png | Decoupled two-stage pipeline. The detector's top-3 emotions and attention cue words condition a 4-bit Phi-4-mini generator.]]

Fig. {ref:fig:arch} shows the pipeline. The generator is Phi-4-mini-instruct {cite:abouelenin2025phi4mini} (3.8B) loaded with 4-bit NF4 quantisation {cite:dettmers2023qlora}. It was selected in a bake-off against Qwen2.5-1.5B {cite:qwen2024} and Gemma-3-4B under the same detector and prompt: Phi-4-mini gave the highest rule adherence (65.3% against 57.7% and 22.0%), whereas Gemma-3-4B exceeded the length limit in 72.3% of replies. In **stage 1**, the generator is shown the user message and the detector's three most probable emotions and chooses one by greedy decoding. In **stage 2**, it writes the reply under a system prompt containing (i) guidance specific to the chosen emotion, stating what to focus on and what to avoid for each of the 32 emotions, (ii) the three rules, and (iii) *cue words*, the 40% of tokens that received the most attention from the detector's final-layer [CLS] representation. Stage-2 decoding uses temperature 0.7, top-p 0.9 and at most 250 new tokens. The full system has 4.0B parameters.

### D. Baselines

All baselines respond to the same conversations with the same decoding settings. *Zero-shot*: Qwen2.5-1.5B with a generic empathetic prompt and no detector. *Fine-tuned*: Qwen2.5-1.5B fine-tuned end to end on ED replies with LoRA {cite:hu2022lora}. *Single model (1.5B)* and *single model (7.6B)*: Qwen2.5-1.5B and Qwen2.5-7B (4-bit), each prompted with the same rules to first name the emotion and then respond in one pass. *Initial pipeline*: a GPT-2 Medium detector with a Qwen2.5-1.5B generator and a generic emotion-conditioned prompt. *Top-1 Phi*: our detector and generator, passing only the top-1 emotion with a generic prompt. We also tested retrieval augmentation {cite:lewis2020rag} of the initial pipeline with similar ED conversations.

## IV. Experimental Setup

**Evaluation sample.** Generation experiments use 300 test conversations sampled with seed 42. The detector is evaluated on all 2,542 test examples.

**Measures.** (1) *Rule adherence*: automatic checks for an acknowledgement phrase, exactly one question mark and at most 120 words; we report the share of replies satisfying all three. (2) *WASSA empathy*: the hyy33 system from the WASSA 2024 shared task {cite:yang2024hyy33}, trained on human empathy ratings of conversation turns and correlating with them at r = 0.544 {cite:giorgi2024wassa}, mapped to a 0-10 scale. (3) *Emotion validation*: whether the detector, applied to the reply, recovers the gold emotion of the conversation. (4) *AI judges*: Nemotron 3 Ultra and gpt-oss-120b, from developers unrelated to our models {cite:panickssery2024}, score each reply blind from 1 to 10 for empathy, relevance and safety at temperature 0; overall is the mean of the three. Judges are applied to a stratified subset of 64 conversations (the shortest and longest for each emotion). A third planned judge, Mistral Large, became unavailable before the final systems were judged; on the 320 replies it did score, the two- and three-judge averages correlate at Spearman rho = 0.98 with identical system rankings. (5) *Human evaluation*: a three-way forced choice between our system and the two single models on 13 situations, including two attention checks.

**Statistics.** All comparisons are paired. We use McNemar's exact test for binary measures {cite:mcnemar1947}, the Wilcoxon signed-rank test for scores {cite:wilcoxon1945} and 95% bootstrap confidence intervals from 10,000 resamples {cite:efron1993}. P-values are Holm-corrected {cite:holm1979} across the baselines compared on each measure. We call a difference significant only if the corrected p < 0.05 and the interval excludes zero.

**Implementation.** All experiments use Python with PyTorch {cite:paszke2019pytorch} and Hugging Face Transformers {cite:wolf2020transformers} on one NVIDIA RTX 4060 with 8 GB of memory.

## V. Results

### A. Overall Comparison

[[TABLE tab:main | Results on 300 ED test conversations (judge scores on the 64-conversation subset). Params: total parameters in billions. Rules: replies satisfying all three rules. Val.: emotion validation. Emp./All: AI-judge empathy and overall (1-10). | wide]]
| System | Params | Rules (%) | 1 Q (%) | Words | WASSA | Val. (%) | Emp. | All |
|---|---|---|---|---|---|---|---|---|
| Zero-shot Qwen2.5-1.5B | 1.5 | 19.0 | 22.7 | 75.9 | 4.25 | 24.7 | 5.98 | 7.51 |
| Fine-tuned Qwen2.5-1.5B | 1.5 | 0.0 | 1.7 | 200.7 | 4.19 | 20.0 | 3.44 | 4.44 |
| Single model Qwen2.5-1.5B | 1.5 | 51.3 | 73.7 | 42.0 | 4.84 | 25.7 | 7.15 | 8.24 |
| Single model Qwen2.5-7B | 7.6 | **82.3** | **99.0** | 53.0 | **5.76** | 32.3 | **8.15** | **8.90** |
| Initial pipeline (GPT-2 + Qwen-1.5B) | 1.9 | 63.7 | 87.3 | 35.2 | 4.80 | 26.0 | 6.98 | 8.04 |
| RoBERTa + Qwen-1.5B, top-1 | 1.7 | 57.7 | 84.3 | 34.7 | 4.78 | 25.3 | - | - |
| RoBERTa + Phi-4-mini, top-1 | 4.0 | 65.3 | 94.7 | 80.5 | 4.89 | 28.3 | 7.84 | 8.65 |
| **Ours: two-stage RoBERTa + Phi-4-mini** | 4.0 | 60.0 | 92.7 | 57.1 | 5.05 | **36.7** | 7.84 | 8.68 |
[[/TABLE]]

Table {ref:tab:main} summarises all systems. Our system achieves the highest emotion validation (36.7%) and the second-highest WASSA and AI-judge scores, behind only the 7.6B single model. The fine-tuned model reproduces the questioning habit of ED listeners, averaging 7.07 questions and 200.7 words per reply, and fails every rule. Retrieval augmentation lowered the initial pipeline's rule adherence from 63.7% to 43.7%, likely because the retrieved human replies model the multi-question behaviour that the rules prohibit.

### B. Significance Against Key Baselines

[[TABLE tab:sig | Paired differences (ours minus baseline) with 95% bootstrap CIs. Rules and validation in percentage points. * significant after Holm correction. | narrow]]
| Measure | vs 1.5B single | vs 7.6B single | vs top-1 Phi |
|---|---|---|---|
| Rules | +8.7 [+1.0, +16.3] | -22.3 [-29.3, -15.3]* | -5.3 [-13.0, +2.0] |
| Validation | +11.0 [+5.0, +17.0]* | +4.3 [-1.7, +10.3] | +8.3 [+3.0, +13.7]* |
| WASSA | +0.21 [+0.08, +0.35]* | -0.71 [-0.84, -0.58]* | +0.16 [+0.05, +0.26]* |
| Judge emp. | +0.70 [+0.39, +1.01]* | -0.30 [-0.55, -0.07]* | +0.01 [-0.23, +0.25] |
| Judge all | +0.44 [+0.17, +0.72]* | -0.21 [-0.40, -0.03] | +0.04 [-0.16, +0.24] |
[[/TABLE]]

Table {ref:tab:sig} reports the key paired comparisons. **Against the 1.5B single model**, our system is significantly more empathetic by both WASSA (+0.21) and the AI judges (+0.70 for empathy, +0.44 overall) and more emotionally faithful (+11.0 points). Its higher rule adherence (+8.7 points) is not significant after correction (corrected p = 0.095). **Against the 7.6B single model**, it is significantly worse on rule adherence, WASSA and judge-rated empathy; the gap in the judges' overall score (-0.21) does not survive correction (corrected p = 0.073). **Against top-1 Phi**, which uses the same detector and generator, emotion-specific guidance significantly raises WASSA (+0.16) and emotion validation (+8.3 points) while writing 29% fewer words, but the AI judges rate the two configurations alike.

### C. Decoupling Versus Guidance

Because our generator is larger than the 1.5B single model, Table {ref:tab:main} alone cannot attribute the gains to decoupling. Two controlled comparisons separate the effects. *Decoupling with the generator fixed*: adding a detector to Qwen2.5-1.5B (initial pipeline against the 1.5B single model) raises rule adherence from 51.3% to 63.7% (McNemar p = 0.002), driven by the one-question rule (87.3% against 73.7%), but changes neither WASSA (4.80 against 4.84, p = 0.69), emotion validation (26.0% against 25.7%, p = 1.0) nor the judges' overall score (8.04 against 8.24, p = 0.14). *Guidance with detector and generator fixed*: the two-stage prompt improves empathy and emotion fidelity over top-1 prompting (Table {ref:tab:sig}). Decoupling therefore mainly improves *structural control* of the reply, whereas emotion-specific guidance improves its *empathic content*. The two-stage design also depends on generator capacity: with Qwen2.5-1.5B it reduced rule adherence to 36.7%.

### D. Is the Detector the Bottleneck?

[[TABLE tab:bottleneck | Three tests of whether detector accuracy limits reply quality. | narrow]]
| Test | Result | p |
|---|---|---|
| Rules when d1 label is wrong vs correct (ours) | 64.2% vs 54.3% | 0.096 |
| WASSA when label is wrong vs correct (initial) | 4.78 vs 4.83 | 0.72 |
| GPT-2 to RoBERTa d1, Qwen-1.5B fixed: validation | 26.0% to 25.3% | 0.91 |
| RoBERTa-base to -large in ours: accuracy | 50.3% to 53.3% | 0.21 |
| ... all rules | 60.0% to 54.7% | 0.60 |
| ... WASSA | 5.05 to 5.01 | 0.85 |
| ... emotion validation | 36.7% to 38.0% | 0.85 |
| Stage-1 pick vs d1 top-1 accuracy | 42.3% vs 50.3% | 0.005 |
[[/TABLE]]

Table {ref:tab:bottleneck} presents three analyses. First, replies generated from an incorrect detector label are no worse in rule adherence or empathy (Fisher and Mann-Whitney tests). Second, replacing the detector with a more accurate one leaves reply quality unchanged, both when GPT-2 is replaced by RoBERTa and when RoBERTa-large is substituted into our system (Holm-corrected p shown). Third, the generator's stage-1 choice agrees with the gold label *less* often than the detector's top-1 prediction (42.3% against 50.3%), yet the two-stage system produces the most emotionally faithful replies. Many disagreements are plausible neighbouring emotions, and ED assigns a single label per conversation {cite:rashkin2019}. Together these results indicate that a reply needs an emotion close enough to respond to, and that the gain comes from the emotion-specific guidance rather than from selecting the gold label.

For reference, our detector reads only the first user message yet exceeds the emotion accuracy reported by joint generation models that use the full dialogue context (30.6-48.3%) {cite:qian2023llm, yang2023escm, wu2025traits}. It thus achieves higher accuracy with less input. Because those models predict emotion as an auxiliary task and were not re-trained under our setting, this comparison supports the use of a dedicated detector rather than a claim of the best detector on ED.

### E. Judge Agreement and Human Evaluation

The two judges agree with each other at Spearman rho = 0.73 for empathy and 0.74 overall across 448 replies, and rank the seven systems identically except for exchanging our system and top-1 Phi, whose scores differ by 0.03. At the system level, WASSA ranks the seven systems in nearly the same order as the judges (rho = 0.96). [[PENDING: human evaluation results: number of valid participants after attention checks; selection rate of each system against chance (33.3%) with binomial tests; Fleiss' kappa; agreement between the human majority choice and the judges' top system.]]

## VI. Discussion and Limitations

Our results give a qualified answer. At comparable or smaller scale, a separate detector combined with emotion-specific guidance yields replies that are more empathetic and more faithful to the user's emotion, and the detector's decision remains visible and auditable. It does not, however, outperform a single model with nearly twice as many parameters, which remains preferable when only output quality matters and such hardware is available.

**Parameter efficiency.** With 52% of the parameters of the 7.6B single model, our system reaches 97.5% of its overall AI-judge score (difference not significant) and 96.2% of its judge-rated empathy (difference significant). At the detector level, RoBERTa-large has 2.8x the parameters of RoBERTa-base and 2.9 points higher test accuracy (Section III-B) but brings no gain in reply quality (Table {ref:tab:bottleneck}), so the smaller detector is the efficient choice.

**Application orientation.** The system runs entirely on one 8 GB consumer GPU, so disclosures need not leave the device. We extended it into a multi-turn agent in which the detector reads the user's last four turns, the generator keeps the last ten exchanges, and every turn displays the candidate emotions, the chosen emotion and the cue words. This audit trail lets developers or clinicians see why a reply took a given tone and insert safeguards before generation, which a single end-to-end model cannot offer.

This study has several limitations. We did not evaluate a single-model baseline built on Phi-4-mini, so the comparison with the 1.5B single model combines decoupling, guidance and generator capacity; Section V-C separates these only with the Qwen2.5-1.5B generator. Automatic measures are proxies: the acknowledgement check is keyword-based, WASSA rewards length (Spearman rho = 0.14-0.54 with word count within systems) and was trained on discussions of news articles {cite:giorgi2024wassa}, emotion validation uses our own detector, and LLM judges prefer longer answers {cite:zheng2023judge}. Judge results rest on 64 conversations. Evaluation covers single turns in English. Our system reuses stock phrases in 58.7% of replies.

**Ethical considerations.** ED is used under its CC BY-NC 4.0 licence, and no data from real users was collected; only ED text was sent to the judge APIs. The label set has no category for suicidal ideation and the system has no crisis safeguard, so it must not be deployed as a mental-health service without a reviewed crisis filter and referral mechanism {cite:who2023suicide}. [[PENDING: statement on consent and anonymity of human-evaluation participants.]]

## VII. Conclusion

We compared decoupled and monolithic small language models for empathetic response generation under identical data, decoding and hardware. A RoBERTa-base detector feeding its top-3 emotions and emotion-specific guidance to a 4-bit Phi-4-mini produces replies that are significantly more empathetic and emotionally faithful than a 1.5B single model, although a 7.6B single model remains stronger. Detection accuracy is not the limiting factor; how the detected emotion is used matters more. Future work should add a Phi-4-mini single-model baseline, combine emotion-specific guidance with larger generators, and evaluate multi-turn conversations with mental-health professionals.

## Acknowledgment

[[PENDING: funding or supervision acknowledgment, added after review.]] The authors used Claude (Anthropic) to assist with analysis scripts and with drafting and language editing of this manuscript. All experiments, results and text were reviewed, verified and revised by the authors, who take full responsibility for the content.
