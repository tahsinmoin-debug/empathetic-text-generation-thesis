# Chapter 4 - 4.2 Preliminary Design or Design (Model) Specification

**Rubric:** CO5. Criteria: *preliminary design of multiple alternative solutions of the system*; *perform simulation of the alternative design solutions for functional verification*. Marks: 5 from the report + 10 from the defence panel.

## What this section must do

Present the **alternative designs** considered for every part of the system, show the simulation evidence that decided between them, and then specify the selected design precisely. The BRAC template gives one sub-subsection per alternative and ends with "Simulation and Verification" plus a comparison table. Keep the *detailed* statistical analysis for Chapter 5. Here, show enough evidence to justify each selection.

## Tables and figures in this folder

| File | Use |
|---|---|
| `figures/fig_final_architecture.png` | Opens 4.2.1 |
| `tables/d1_evolution_mismatch_fix.csv` | Why the initial GPT-2 detector was replaced |
| `tables/d1_backbone_comparison.csv` + `figures/fig_d1_backbone_comparison.png` | Detector alternatives |
| `tables/d1_backbone_significance.csv` | Short support for the detector choice |
| `tables/d2_bakeoff_candidates.csv` | Generator alternatives |
| `tables/integration_variant_ablation.csv` | Ways of passing d1's output to d2 |
| `tables/rag_negative_result.csv` | RAG alternative |
| `tables/configuration_and_hyperparameters.csv` | Full specification of the selected design with a justification for each setting |

## Checklist

- [ ] 4.2.1 Overview of the selected architecture (figure)
- [ ] 4.2.2 Emotion detector alternatives (d1)
- [ ] 4.2.3 Response generator alternatives (d2)
- [ ] 4.2.4 Integration alternatives (how d1 reaches d2)
- [ ] 4.2.5 Monolithic and retrieval alternatives
- [ ] 4.2.6 Specification of the selected design (prompts, cues, memory, configuration table)
- [ ] 4.2.7 Simulation and verification

---

## Draft

### 4.2.1 Overview

Figure 4.2 shows the selected design. A user's latest message, together with their preceding turns, is read by the emotion detector d1, a fine-tuned RoBERTa-base classifier that outputs the three most probable of 32 emotions. In a first, deterministic stage, the generator Phi-4-mini-instruct chooses the emotion that best fits the message from these three candidates. The system prompt is then assembled from emotion-specific guidance for that emotion, three behavioural rules, and the words that received the most attention inside d1. In the second stage Phi-4-mini generates the reply. Every intermediate decision is logged. The following subsections present the alternatives simulated for each component and the evidence behind each selection.

[Figure: fig_final_architecture.png] *Figure 4.2: Selected system architecture.*

### 4.2.2 Emotion Detector Alternatives (d1)

**GPT-2 Medium (initial design).** The initial detector was GPT-2 Medium (355M parameters), a decoder-only model, fine-tuned on the dataset's situation descriptions. On those descriptions it reached 56.9% top-1 accuracy. When evaluated on the input it actually receives, the user's first message, accuracy fell to 48.2% (Table 4.x). The design was rejected for two reasons: it was trained on the wrong text, and a left-to-right decoder is a weaker architecture for classification than a bidirectional encoder.

**Encoder candidates.** We then trained four encoder backbones under identical conditions: the same first-utterance data, 4 epochs, maximum length 128, warmup ratio 0.06 and the same test split (Table 4.x, Figure 4.3). BERT-base reaches 48.8% top-1 and 81.7% top-5, ELECTRA-base 50.2% and 82.7%, and RoBERTa-base 50.1% and 85.0%. The three base models are statistically tied on top-1 accuracy (RoBERTa vs ELECTRA, p = 0.96; RoBERTa vs BERT, p = 0.11), but RoBERTa-base is significantly better on top-5 than both (p <= 0.0001). Because the selected design passes the detector's top-3 candidates to the generator, ranking quality matters more than the single top label, so **RoBERTa-base was selected**. RoBERTa-large (355M) is significantly better still (53.0% top-1, 87.2% top-5), but it costs 7.4 times the training time, and Chapter 5 shows that detector accuracy gains do not reach response quality.

[Table: d1_evolution_mismatch_fix.csv] [Table: d1_backbone_comparison.csv] [Figure: fig_d1_backbone_comparison.png]

### 4.2.3 Response Generator Alternatives (d2)

To isolate the effect of the generator, we ran a controlled comparison: the same detector (RoBERTa-base), the same prompt and rules, the same 300 conversations and the same decoding settings, changing only d2 (Table 4.x). **Qwen2.5-1.5B-Instruct**, the initial generator, follows all rules in 57.7% of replies. **Gemma-3-4B-it** acknowledges feelings most often (96.7%) but obeys the length limit in only 27.7% of replies, averaging 133 words, so only 22.0% satisfy all rules. **Phi-4-mini-instruct** (3.8B, 4-bit) achieves the highest rule adherence, 65.3%, while keeping 96.0% of replies within length. **Phi-4-mini was selected.** An earlier base (non-instruction-tuned) candidate, Phi-2, was excluded because it has no chat template and could not follow system-prompt constraints.

[Table: d2_bakeoff_candidates.csv]

### 4.2.4 Integration Alternatives: How d1's Output Reaches d2

d1 produces a full probability distribution, so there are several ways to use it. With the initial detector and generator, we simulated four variants on the same 300 conversations (Table 4.x). *Argmax* passes only the top emotion (63.7% all rules). *Confidence-gated* switches to hedged wording when d1 is unsure (63.3%), but only 28.7% of low-confidence replies actually hedged. *Distribution* lists every emotion above probability 0.05 (62.0%). *Candidate-pick* asks the generator to choose among the candidates and write the reply in the same call (54.0%). None beat argmax, and candidate-pick hurt rule adherence, which suggests that one generation call cannot both disambiguate the emotion and follow the rules.

This analysis motivated the **two-stage** design adopted in the final system. The true emotion is among d1's top-3 in 76.5% of test cases against 50.3% for top-1, so the wider candidate set is valuable, but merging several emotions into one prompt produces contradictory guidance. For example, *disappointed* and *impressed* call for opposite responses. The two-stage design therefore separates the choice into its own short, deterministic call and gives the generation call a single, coherent emotion-specific prompt.

[Table: integration_variant_ablation.csv]

### 4.2.5 Monolithic and Retrieval Alternatives

To test whether a separate detector is necessary at all, four monolithic alternatives were simulated under the same rules and sample. *Zero-shot*: Qwen2.5-1.5B with a generic caring prompt and no detector. *Monolithic fine-tuned*: Qwen2.5-1.5B LoRA-tuned end to end on (user message, human reply) pairs. *Single model, same size*: Qwen2.5-1.5B instructed to silently identify the emotion and then respond under the three rules. *Single model, bigger*: the same prompt on Qwen2.5-7B in 4-bit. Their results are analysed in Section 5.2. In brief, the fine-tuned model failed every rule, zero-shot rarely asked the single required question, the same-size single model trailed the decoupled designs, and the larger single model performed best overall. Following the Pre-Thesis 2 panel's suggestion, we also simulated a **retrieval-augmented** variant that retrieves one or two similar training conversations with the same detected emotion (all-MiniLM-L6-v2 embeddings) and shows them to the generator as examples. Across two versions it reduced rule adherence from 63.7% to 54.3% and then to 43.7%, so it was not adopted (Table 4.x).

[Table: rag_negative_result.csv]

### 4.2.6 Specification of the Selected Design

**Detector.** RoBERTa-base with a 32-way classification head, fine-tuned on first utterances (learning rate 2e-5, effective batch 16, 4 epochs with early stopping, warmup 0.06, weight decay 0.01, gradient clipping 1.0, fp16). In conversation, d1 reads the concatenation of the user's last four turns. This window was best or tied-best in three of four long test scenarios. Six turns degraded two scenarios, because emotions from earlier turns competed with the current one.

**Attention cues.** d1 is loaded with the `eager` attention implementation, because the default `sdpa` backend returns no attention weights. We take the last layer's attention from the `[CLS]` position, average over heads, keep the top 40% of word spans (between 3 and 14), and expand sub-word pieces to whole words.

**Stage 1 (emotion pick).** Phi-4-mini receives the message and the three candidates with the instruction: *"An automatic detector narrowed the person's likely feeling down to these options: {a}, {b}, {c}. Read their message and decide which ONE of those options best fits what they are actually expressing. Reply with ONLY that single word..."* Decoding is greedy with at most 10 tokens. If the output does not name a candidate, the system falls back to d1's top-1. In simulation, every output parsed to a valid candidate.

**Stage 2 (generation).** The system prompt combines, in order, the chosen emotion, the cue words, the emotion-specific *focus* and *avoid* guidance, and the universal rules. Example for *lonely*:

> You are a highly empathetic counselor. The user is feeling lonely.
> The parts of their message that most signalled this feeling: {cues}.
> Ground your reply in those specifics rather than responding generically.
> Acknowledge the isolation itself, gently - the weight of it, not just 'you seem sad.'
> Don't offer a surface-level fix like 'go meet people.' Don't act overly cheerful.
> Follow these three rules strictly: 1. Acknowledge and validate their feelings deeply using 2 to 3 comforting sentences... 2. Share a reassuring thought that fits this specific feeling... 3. Ask EXACTLY ONE short, gentle follow-up question at the very end... Keep your entire reply under 100 words.

Guidance exists for all 32 emotions [[TEAM TO FILL: name the source document the per-emotion guidance was adapted from]]. Phi-4-mini runs in 4-bit NF4 with double quantisation and fp16 compute, sampling at temperature 0.7 and top-p 0.9 with at most 250 new tokens.

**Conversation memory.** d2 receives the last ten exchanges verbatim. From the second turn onward the question rule becomes optional and repeated phrasing is discouraged, so that a multi-turn conversation does not turn into an interrogation. The first turn keeps the original rules, so single-turn results remain comparable.

Table 4.x lists every setting of the selected design with its justification.

[Table: configuration_and_hyperparameters.csv]

### 4.2.7 Simulation and Verification

Each alternative was verified functionally before full evaluation. Small smoke-test runs confirmed that outputs were well formed and that model-specific details were handled: Phi-4-mini's end-of-turn token differs from its end-of-text token, Gemma's chat template has no system role, and Phi-4-mini uses fused attention projections. Every design was then simulated on the same 300 held-out conversations under identical decoding settings, and detector candidates on the full 2,542-example test split. Two automatic safeguards protect the selected design: a hard failure if the detector returns no attention weights, and logging of whether each stage-1 decision parsed successfully (100% did). The outcome of these simulations determined the selections above and is analysed in full in Chapter 5.

---

## Traps

- **Never reuse the June Table 3.1** (GPT-2 56.64% vs Qwen2.5-0.5B 35.93% vs TinyLlama 16.91%). The numbers come from different datasets.
- The integration ablation (4.2.4) used the *initial* GPT-2 + Qwen pipeline. Say so, as the draft does.
- Keep numbers here to what justifies each choice. Significance tests on the final system belong in 5.3.
- The history-window evidence (4 turns) comes from qualitative scenario tests recorded in the code, not a large-sample experiment. Describe it as scenario testing.
