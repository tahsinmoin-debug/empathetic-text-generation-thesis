# Chapter 2 - 2.1 Preliminaries

**Rubric:** CO9, "Conduct independent research, literature survey and learning of new technologies and concepts..." Assessed in the Pre-Thesis 1 and Final Thesis reports.

## What this section must do

Define every concept a reader needs before the literature review and the method make sense, one short paragraph each. The BRAC template's 2.1 uses exactly this format: a **bold term** followed by a 3-5 sentence definition, grouped from general to specific.

**Only define things the thesis actually uses.** The June draft's framing no longer applies (GoEmotions, LoRA as the main d2 method, brain analogies).

## Checklist, in this order

1. Empathetic response generation, and cognitive vs affective empathy
2. The EmpatheticDialogues dataset
3. Transformers and self-attention
4. Encoder-only vs decoder-only models (why d1 is an encoder and d2 a decoder)
5. RoBERTa
6. Instruction-tuned small language models (Phi-4-mini, Qwen2.5)
7. Prompt engineering and system prompts
8. 4-bit quantisation (NF4) and LoRA / QLoRA (LoRA only as the rejected monolithic baseline)
9. Top-k prediction
10. Retrieval-augmented generation (RAG), needed because the panel asked about it
11. LLM-as-a-judge evaluation
12. WASSA empathy classifier
13. Paired significance testing (McNemar, Wilcoxon, bootstrap confidence intervals, Holm correction)

## Style notes

- Template pattern: "**Convolutional Neural Network (CNN)** is a deep learning model built to...". Copy this exactly, including spelling out the acronym at first use.
- One figure is enough, e.g. a small encoder vs decoder diagram. Don't pad the section.
- Cite the original paper for each concept (`[[CITE]]` markers below).

---

## Draft

**Empathetic Response Generation** is the task of producing a conversational reply that recognises and responds appropriately to the emotional state of the speaker. Rashkin et al. [[CITE: Rashkin et al., 2019, ACL]] introduced the task as a benchmark. Later work distinguishes *affective* empathy, which reflects and validates what the speaker feels, from *cognitive* empathy, which understands why they feel it [[CITE: Sabour et al., 2022, CEM, AAAI]]. In a mental-health support setting the goal is supportive listening rather than diagnosis or therapy.

**EmpatheticDialogues (ED)** is a crowd-sourced corpus of 24,850 two-person English conversations collected through Amazon Mechanical Turk [[CITE: Rashkin et al., 2019]]. For each conversation, a speaker selects one of 32 fine-grained emotion labels (for example *afraid*, *terrified*, *lonely*, *proud*), writes a short situation description, and then talks about that situation with a listener. The authors split the data into 19,533 / 2,770 / 2,547 conversations for training, validation and test, keeping all conversations started by the same speaker in one partition to prevent situation overlap. ED is released under the CC BY-NC 4.0 licence.

**Transformer** models [[CITE: Vaswani et al., 2017]] process text with *self-attention*: every token computes weights over all other tokens, and those weights decide how much each token contributes to its new representation. The attention weights of a trained classifier can be inspected to see which words most influenced a prediction. Our system uses this to extract context cues (Chapter 4).

**Encoder-only and decoder-only models** are the two main transformer families. Encoder-only models such as BERT [[CITE: Devlin et al., 2019]] read the whole input in both directions at once, which suits classification. Decoder-only models such as GPT-2, Qwen and Phi read left-to-right and generate text one token at a time, which suits response generation. Our design uses an encoder for emotion detection (d1) and a decoder for response generation (d2).

**RoBERTa** [[CITE: Liu et al., 2019, RoBERTa]] is a BERT-style encoder pre-trained longer, on more data, with dynamic masking and without the next-sentence objective. RoBERTa-base has 125 million parameters. For classification, a linear layer is placed on the representation of the first token (`[CLS]`, written `<s>` in RoBERTa) and the whole network is fine-tuned with cross-entropy loss.

**Instruction-tuned small language models** are decoder-only models further trained to follow natural-language instructions and chat templates. Phi-4-mini-instruct (3.8 billion parameters) [[CITE: Microsoft, 2025, Phi-4-Mini Technical Report]] and Qwen2.5-Instruct (1.5 and 7 billion parameters) [[CITE: Qwen Team, 2024, Qwen2.5 Technical Report]] are examples. Because they follow a *system prompt*, their behaviour can be shaped at inference time without retraining.

**Prompt Engineering** is the design of the instructions given to a language model. A system prompt can include the detected emotion, explicit behavioural rules ("ask exactly one question") and task-specific guidance. It is cheaper than fine-tuning and can be changed per request, but models do not always comply, so compliance has to be measured.

**Quantisation and Low-Rank Adaptation.** 4-bit NormalFloat (NF4) quantisation stores model weights in 4 bits, cutting memory roughly fourfold, so a 3.8B-parameter model fits in 8 GB of GPU memory [[CITE: Dettmers et al., 2023, QLoRA]]. Low-Rank Adaptation (LoRA) [[CITE: Hu et al., 2022]] fine-tunes a model by training small low-rank matrices added to frozen weights. We use LoRA only for a monolithic baseline, which is later rejected.

**Top-k Prediction** returns a classifier's *k* most probable labels instead of only the single most likely one. With 32 closely related labels the correct emotion is often ranked second or third (for example *angry* vs *furious*), so top-3 or top-5 accuracy measures whether the right answer is among a small candidate set.

**Retrieval-Augmented Generation (RAG)** [[CITE: Lewis et al., 2020]] adds retrieved documents, here similar past conversations, to a model's prompt so it can imitate or ground itself in them. Retrieval uses sentence embeddings such as all-MiniLM-L6-v2 [[CITE: Reimers and Gurevych, 2019]] and cosine similarity.

**LLM-as-a-Judge** evaluation uses a strong language model to score generated text against a rubric [[CITE: Zheng et al., 2023, Judging LLM-as-a-judge]]. It scales beyond human annotation but can be biased toward longer answers or toward models from its own family, so judges should be blind to system identity and drawn from different providers.

**WASSA Empathy Classifier.** The WASSA 2024 shared task on empathy detection [[CITE: Giorgi et al., 2024, WASSA 2024 shared task]] produced models that predict an empathy score for a piece of text. We use a publicly released DeBERTa-based model from that task (`hyy-33/hyy33-WASSA-2024-Track-2`) as an automatic empathy signal independent of the AI judges.

**Paired Significance Testing.** When two systems answer the *same* conversations, their results are paired. **McNemar's exact test** [[CITE: McNemar, 1947]] compares paired binary outcomes, such as whether a reply followed all rules. The **Wilcoxon signed-rank test** [[CITE: Wilcoxon, 1945]] compares paired continuous scores. A **paired bootstrap** resamples conversations to build a 95% confidence interval for the difference. Testing many comparisons at once raises the chance of a false positive, so the **Holm-Bonferroni correction** [[CITE: Holm, 1979]] adjusts the p-values.

---

## Traps

- Don't describe ED as "33,090 conversations" (the June draft said this). The paper reports 24,850.
- Don't call Phi-4-mini "fine-tuned". The final d2 is zero-shot.
- Keep this section to definitions. Results and comparisons belong in 2.2 and Chapter 5.
