# Chapter 2: Literature Review

## 2.1 Preliminaries

This section introduces the concepts on which the rest of the thesis builds: the task of empathetic response generation, the dataset used, the model families involved, and the evaluation and statistical methods applied in later chapters.

**Empathetic Response Generation** is the task of producing a conversational reply that recognises and responds appropriately to the emotional state of the speaker. Rashkin et al. [2] established it as a benchmark task and showed that models trained on empathetic conversations are perceived as more empathetic by human evaluators. Later work distinguishes *affective* empathy, which reflects and validates what the speaker feels, from *cognitive* empathy, which understands why the speaker feels it [11]. In a mental-health support setting the aim is supportive listening rather than diagnosis or therapy.

**EmpatheticDialogues (ED)** is a crowd-sourced corpus of 24,850 two-person English conversations. For each conversation, a speaker chooses one of 32 fine-grained emotion labels, such as *afraid*, *terrified*, *lonely* or *proud*, writes a short description of a situation in which they felt that emotion, and then discusses the situation with a listener. The corpus is divided into 19,533 training, 2,770 validation and 2,547 test conversations, and all conversations started by the same speaker are kept within one partition so that similar situations do not appear in more than one split. The dataset is released under the CC BY-NC 4.0 licence [20].

**Transformer** models [4] process text with self-attention: each token computes a set of weights over all tokens in the input, and these weights determine how much each token contributes to its updated representation. Because the weights are explicit, the attention of a trained model can be inspected to see which words most influenced its output. This property is used in our system to extract the words that drove an emotion prediction.

**Encoder-only and Decoder-only Models** are the two main transformer families. Encoder-only models such as BERT [5] read the entire input in both directions at once and are well suited to classification. Decoder-only models such as GPT-2, Qwen and Phi process text from left to right and generate one token at a time, which suits response generation. The system in this thesis uses an encoder for emotion detection and a decoder for response generation.

**RoBERTa** [6] is an encoder with the BERT architecture that is pre-trained for longer, on more data, with dynamic masking and without the next-sentence prediction objective. RoBERTa-base has 125 million parameters and RoBERTa-large 355 million. For classification, a linear layer is added on top of the representation of the first token, and the whole network is fine-tuned with a cross-entropy loss.

**Instruction-tuned Language Models** are decoder-only models that have been further trained to follow natural-language instructions within a chat format. Phi-4-mini-instruct, with 3.8 billion parameters [21], and the Qwen2.5-Instruct models [22] are examples. Because they follow a *system prompt*, their behaviour can be shaped at inference time without changing their weights.

**Prompt Engineering** is the design of the instructions given to a language model. A system prompt can include the detected emotion, explicit behavioural rules such as "ask exactly one question", and guidance specific to the situation. Prompting is far cheaper than fine-tuning and can change from one request to the next, but models do not always comply, so compliance must be measured rather than assumed.

**Quantisation and Low-Rank Adaptation.** Four-bit NormalFloat (NF4) quantisation stores model weights in four bits, reducing memory use roughly fourfold, so that a model with several billion parameters fits in 8 GB of GPU memory [23]. Low-Rank Adaptation (LoRA) [24] fine-tunes a model by training small low-rank matrices added to its frozen weights. In this thesis LoRA is used only for a monolithic baseline.

**Top-k Prediction** returns a classifier's *k* most probable labels rather than only the single most likely one. When the label set contains many closely related emotions, such as *angry* and *furious*, the correct label is frequently ranked second or third, so top-3 and top-5 accuracy indicate whether the correct emotion lies within a small set of candidates.

**Retrieval-Augmented Generation (RAG)** [25] supplements a model's input with retrieved documents, in our case similar past conversations, so that the model can ground its output in them. Retrieval typically relies on sentence embeddings compared by cosine similarity [26].

**LLM-as-a-Judge** evaluation uses a strong language model to score generated text against a rubric. Strong LLM judges agree with human preferences in over 80% of cases, but they exhibit position bias, a preference for longer answers and a tendency to favour their own outputs [18]. Judges should therefore be blind to the identity of the system being evaluated and should not share a model family with it.

**WASSA Empathy Classifier.** WASSA is a long-running workshop on computational approaches to subjectivity, sentiment and social media analysis, held with the Association for Computational Linguistics. Its 2024 shared task included a track on predicting third-party human ratings of empathy for individual turns of a conversation [16]. We use a publicly released DeBERTa-based system from that track [17] as an automatic empathy signal that is independent of the AI judges.

**Paired Significance Testing.** When two systems respond to the same conversations, their results form matched pairs. **McNemar's test** compares paired binary outcomes, such as whether each reply followed all rules, using only the pairs on which the systems disagree. The **Wilcoxon signed-rank test** compares paired continuous scores without assuming normality. A **bootstrap** confidence interval is obtained by repeatedly resampling the conversations and recomputing the difference. Because running many tests increases the chance of a false positive, the **Holm-Bonferroni procedure** adjusts p-values for multiple comparisons.

## 2.2 Review of Existing Research

This section reviews prior work on empathetic dialogue in four groups: the benchmark and early models, knowledge-enhanced models, systems based on large language models (LLMs), and work on emotion recognition and evaluation.

### 2.2.1 Empathetic Dialogue Benchmarks and Early Models

Rashkin et al. [2] introduce EmpatheticDialogues, a benchmark of 24,850 crowd-sourced conversations grounded in 32 emotion labels, and show that retrieval and generation models trained on it are perceived as more empathetic by human evaluators. The authors also train emotion classifiers on the situation descriptions written by speakers, reporting 43% accuracy for fastText and 48% for a fine-tuned DeepMoji model. Although the dataset established the task, these classifiers read an explicit, after-the-fact description of the situation rather than what the speaker actually says in conversation, which is the harder input a deployed system receives.

Lin et al. [7] propose a mixture of empathetic listeners (**MoEL**), in which the model first detects the user's emotion and then combines the outputs of listener components specialised for different emotions. Majumder et al. [8] observe that empathetic responses often mimic the user's emotion to a degree that depends on its polarity, and model this through polarity-based emotion groups and stochastic emotion mixing (**MIME**). Li et al. [9] model emotion at both the dialogue and token levels and use adversarial learning with user feedback (**EmpDG**). When these models are re-evaluated under a common setting, they recognise the speaker's emotion with only 30.6-31.2% accuracy and produce replies of low lexical diversity [13]. Their main drawback is that emotion recognition is learned as an auxiliary task of small generators trained from scratch, which limits both recognition accuracy and fluency.

### 2.2.2 Knowledge-Enhanced Empathetic Generation

Later work adds external knowledge to capture why the user feels as they do. **KEMP** [10] enriches the dialogue history with commonsense and emotional lexical knowledge organised into an emotional context graph. **CEM** [11] uses commonsense inferences about the user's situation to model both the affective and cognitive aspects of empathy. **CASE** [12] aligns cognition and affection at both coarse and fine levels of granularity, and **ESCM** [27] models dynamic correlations between emotion and semantics. These methods raise emotion recognition accuracy to 36.6-41.2%. However, in human evaluation CEM and CASE receive empathy ratings of only 3.08 and 3.04 out of 5, showing that knowledge integration adds complexity to small generators without closing the gap to modern language models.

Cao et al. [28] carry this idea into the LLM era with the **EKTC** framework, in which a fine-tuned LLM decides when to call a commonsense knowledge base as a tool, and construct the **TOOL-ED** dataset by inserting tool-use trajectories into EmpatheticDialogues. Fine-tuning Qwen1.5-14B and Vicuna-7B on this data yields a BERTScore of 0.8765, and in A/B human evaluation by three researchers the tool-calling models win 61.3% (Qwen) and 72.3% (Vicuna) of empathy comparisons against the same models fine-tuned on plain EmpatheticDialogues. The approach requires fine-tuning models of 7-14 billion parameters, well beyond a consumer GPU, and does not measure whether replies obey explicit interaction constraints.

Wu et al. [29] argue that empathetic expression should adapt to the speaker's personality and incorporate visual and MBTI personality information into a GPT-2-based generator. On text-only EmpatheticDialogues their model reaches 45.4% emotion recognition accuracy, and it is evaluated by seven graduate annotators using both five-point ratings and pairwise comparisons. For general text-based support, however, the approach depends on multimodal input and personality labels that a user's message does not provide.

### 2.2.3 Large Language Models for Empathetic and Supportive Dialogue

Qian et al. [13] compare GPT-3.5 and ChatGPT with the trained state of the art on EmpatheticDialogues. Human evaluators rate ChatGPT at 4.64 out of 5 for empathy, against 3.08 for CEM and 3.04 for CASE, and its replies are several times more lexically diverse. A systematic review of twelve studies on LLM empathy [15] confirms that LLMs display elements of cognitive empathy, but also reports recurring weaknesses: repetitive use of empathic phrases, a tendency to forget initial instructions, and lengthy responses that "could potentially be erroneously associated with greater empathy". These behaviours motivate the explicit rule-adherence, repetition and length analyses used in this thesis.

Wang et al. [14] combine prompt engineering with LoRA and full-parameter fine-tuning of Qwen2.5 models for the NLPCC 2025 emotional-support task. Every fine-tuned configuration outperforms the prompted base model on validation, with the best configuration reaching a total score of 41.17 against 30.87 without fine-tuning, and their submission ranked second. These gains, however, required fine-tuning a 72-billion-parameter model. Rasool et al. [30] fuse emotion-lexicon and multi-level embeddings into Flan-T5, Llama 2, DeepSeek-R1 and GPT-4 for psychotherapy transcripts, using a BERT-based model for emotion detection and a generative LLM for the response. Their paired t-tests show that the improvements in empathy (p = 0.53) and coherence (p = 0.12) are not statistically significant, and they call for human evaluation in future work. Both studies support separating emotion understanding from generation, but neither enforces or measures controllable response behaviour, and the second illustrates how reported gains can fail a significance test.

### 2.2.4 Emotion Recognition and Evaluation Methodology

Paranjape et al. [31] describe Chirpy Cardinal, a socialbot developed for the Alexa Prize, which includes an emotion classifier obtained by fine-tuning RoBERTa on the first utterance of each EmpatheticDialogues conversation. The classifier reaches 61.5% top-1 and 92% top-5 accuracy, with most errors occurring between similar labels such as *angry* and *furious*. This is the published result most directly comparable to the emotion detector developed in this thesis. Belay et al. [32] evaluate encoder-only, encoder-decoder and decoder-only language models on multi-label emotion classification in four Ethiopian languages and English, and conclude that fine-tuned encoder models still outperform few-shot LLMs at emotion classification. This supports the use of a dedicated encoder, rather than a generative model, for emotion detection.

For evaluation, Zheng et al. [18] establish the use of strong LLMs as judges and document their position, verbosity and self-enhancement biases. Panickssery et al. [33] show that LLM evaluators can recognise their own generations and score them more favourably, and Verga et al. [19] find that a panel of judges drawn from different model families is less biased than a single large judge. For automatic empathy scoring, the WASSA 2024 shared task provides systems trained on human empathy ratings of conversation turns [16]. Across the EmpatheticDialogues literature, human evaluation uses either five-point ratings or A/B preference judgements, and Wu et al. note that pairwise comparison accounts for individual variation among annotators. Table 2.1 summarises the reviewed work.


| Work | Data | Approach and result | Limitation relevant to this thesis |
|---|---|---|---|
| Rashkin et al., 2019 (ACL) | EmpatheticDialogues | Introduces 25k-conversation, 32-emotion benchmark; emotion classifiers on situation text | Benchmark only; situation-text classifiers overstate real-input accuracy |
| Lin et al., 2019 - MoEL (EMNLP) | ED | Mixture of emotion-specific listeners; emotion acc 30.75% | Low recognition accuracy; small model generation |
| Majumder et al., 2020 - MIME (EMNLP) | ED | Emotion mimicry grouped by polarity | Coarse polarity groups; generic replies |
| Sabour et al., 2022 - CEM (AAAI) | ED | Adds COMET commonsense (cognitive + affective) | Human empathy 3.08/5, far below LLMs |
| Zhou et al., 2023 - CASE (ACL) | ED | Aligns cognition and affection at coarse and fine grain | Human empathy 3.04/5 |
| Qian et al., 2023 (arXiv 2310.05140) | ED | LLMs (ChatGPT) beat trained SOTA on automatic and human metrics | Uncontrolled, verbose; no rule/format control measured |
| Paranjape et al., 2020 - Chirpy Cardinal | ED + live Alexa users | RoBERTa-large emotion classifier on first utterance, 61.5% top-1 | Technical report; validation-set score |
| Cao et al., 2025 - TOOL-ED (COLING) | ED | LLM tool-calling for commonsense knowledge; A/B human eval | Fine-tuning 7-14B models; heavy resources |
| Wu et al., 2025 - Personality-aware ERG (COLING) | MELD-based multimodal + ED | Personality + visual cues; emotion acc 45.4% text-only | Needs multimodal input and personality labels |
| Belay et al., 2025 (COLING) | EthioEmo (4 Ethiopian languages) + SemEval-2018 English | Encoder, encoder-decoder, decoder LLMs for multi-label emotion | Fine-tuned encoders still beat few-shot LLMs at emotion classification |
| Rasool et al., 2025 (AI, MDPI) | Psychotherapy transcripts | Emotion-aware embedding fusion into Flan-T5/Llama 2/DeepSeek-R1/GPT-4; BERT for emotion | Paired t-tests non-significant; no human evaluation |
| Wang et al., 2025 (arXiv 2507.12820) | NLPCC 2025 ESC task | Prompting + LoRA / full fine-tuning of Qwen2.5; total 41.17 vs 30.87 base, ranked 2nd | Needs fine-tuning up to 72B parameters |

*Table 2.1: Summary of the existing research reviewed in this chapter*


## 2.3 Summary of Key Findings

Research on empathetic response generation has moved from small models trained from scratch to large pre-trained language models. Early joint models such as MoEL, MIME and EmpDG treat emotion recognition as an auxiliary task and identify the speaker's emotion in only 30.6-31.2% of cases [7], [8], [9], [13]. Knowledge-enhanced successors, including KEMP, CEM, CASE and ESCM, raise this to 36.6-41.2% by adding commonsense and emotional knowledge [10], [11], [12], [27], and personality-aware multimodal generation reaches 45.4% on text-only EmpatheticDialogues [29]. Dedicated classifiers do better still: a fine-tuned RoBERTa model reaches 61.5% on the first utterance of each conversation [31], and fine-tuned encoders continue to outperform prompted LLMs at emotion classification [32]. On the generation side, instruction-following LLMs have changed the landscape. Human evaluators rate ChatGPT at 4.64 out of 5 for empathy against about 3.0 for the best trained models, and recent systems fine-tune LLMs of 7 to 72 billion parameters with knowledge tools or emotional-support data to improve further [28], [14].

These advances leave four problems open. First, **LLM empathy is fluent but difficult to control**. LLMs repeat empathic phrases, forget initial instructions and produce lengthy responses that may be mistaken for greater empathy [15], yet none of the EmpatheticDialogues studies reviewed measures whether replies follow explicit interaction rules. Second, **the strongest results depend on resources beyond a consumer GPU**, such as fine-tuning models of tens of billions of parameters, multimodal input, or commercial APIs to which a user's disclosure must be sent. Third, **emotion recognition is often evaluated on an easier input than a deployed system receives**. Classifiers trained on situation descriptions read an explicit summary rather than the user's message [2], and joint models do not expose the emotion they recognise for inspection. Fourth, **the statistical evidence is often thin**. Most studies report differences in automatic metrics or small human studies without paired significance testing, and at least one reports empathy gains that are not statistically significant [30]. Future work should therefore pursue controllable and inspectable behaviour under realistic resource limits, emotion recognition evaluated on real conversational input, and conclusions supported by appropriate statistical tests.

This thesis addresses these gaps as summarised in Table 2.2.

| Gap | What the literature lacks | How this thesis addresses it | Section |
|---|---|---|---|
| G1 | Measured control over interaction behaviour | Explicit response rules scored for every reply and compared against monolithic alternatives | 4.2, 5.1 |
| G2 | Inspectable emotional reasoning | A separate detector whose top-3 candidates, chosen emotion and influential words are exposed | 4.1, 4.2 |
| G3 | Small, self-hostable systems | A 125M-parameter detector and a 3.8B-parameter generator on a single 8 GB GPU, without fine-tuning the generator | 4.2, 3.8 |
| G4 | Statistical rigour | Paired tests with confidence intervals and multiple-comparison correction, error-propagation analysis, and several independent evaluation signals | 5.3 |

*Table 2.2: Research gaps and how this thesis addresses them*
