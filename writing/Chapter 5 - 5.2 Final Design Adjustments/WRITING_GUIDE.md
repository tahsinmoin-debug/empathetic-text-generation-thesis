# Chapter 5 - 5.2 Final Design Adjustments

**Rubric:** CO7, "Complete the final design and development of the solution with necessary adjustment based on performance evaluation." Criteria: obtain the final design by making necessary adjustments to the selected design as per the performance evaluation result; develop the solution as per the final design. Marks: 5.

## What this section must do

Show a clear chain for each change: **evaluation result → diagnosed cause → adjustment → re-measured effect**. The rubric rewards adjustments *driven by evaluation*, so every change here must point to a number. Then describe the developed solution: the multi-turn conversational agent and the prepared demo.

**Numbering:** the rubric lists this and "Analysis of Design Solutions" both as 5.2. Confirm with the supervisor; the simplest fix is to make this 5.3 and renumber the rest.

## Tables in this folder

| File | Use |
|---|---|
| `tables/design_adjustment_log.csv` | One row per adjustment: problem, evidence, change, effect (the core table) |
| `tables/d1_evolution_mismatch_fix.csv` | Evidence for adjustment A1 |
| `tables/configuration_and_hyperparameters.csv` | Final configuration after all adjustments |

## Checklist

- [ ] 5.2.1 Adjustments to the detector (A1)
- [ ] 5.2.2 Adjustments to the generator and prompting (A2, A3, A4)
- [ ] 5.2.3 Adjustments implementing interpretability (A5)
- [ ] 5.2.4 Development of the final solution: the multi-turn agent (A6, A7)
- [ ] 5.2.5 Adjustments evaluated and rejected (A8), and deferred adjustments (A9, crisis filter)

---

## Draft

The performance evaluation of the initial pipeline (GPT-2 Medium detector with Qwen2.5-1.5B generator) identified several weaknesses. Each adjustment below was made in response to a measured result and was re-evaluated on the same conversations before being kept (Table 5.x).

[Table: design_adjustment_log.csv]

### 5.2.1 Detector: Correcting the Training Input

Evaluating the initial detector on the input it actually receives exposed a train/inference mismatch. The model had been trained on the dataset's situation descriptions, which state the feeling explicitly, but the pipeline feeds it the user's conversational message. The two texts are identical in only 11.7% of conversations, and accuracy dropped from 56.9% on situation text to 48.2% on user messages (Table 5.x). We retrained the detector on first utterances and, following the backbone comparison in Section 4.2.2, replaced GPT-2 Medium with RoBERTa-base. The adjusted detector reaches 50.3% top-1 and 84.3% top-5 on the real input. The deployment accuracy is now honest, and the ranking quality needed for top-3 selection is improved.

### 5.2.2 Generator and Prompting

**Generator replacement (A2).** With the adjusted detector fixed, the generator bake-off showed Phi-4-mini achieving higher rule adherence (65.3%) than the original Qwen2.5-1.5B (57.7%), so it replaced Qwen as d2, loaded in 4-bit to fit in memory.

**Two-stage emotion selection (A3).** The adjusted detector's top-1 label is wrong in about half of conversations, but the correct label is among its top-3 in 76.5%. The earlier candidate-pick experiment showed that asking the generator to choose *and* respond in one call degrades rule-following. We therefore split this into a deterministic selection call followed by a separate generation call.

**Emotion-specific prompts (A4).** A generic instruction such as "the user is feeling X" gives the generator no guidance on how responses to *lonely* and to *proud* should differ. We wrote focus and avoid guidance for each of the 32 emotions. With the detector and Phi-4-mini held fixed, adjustments A3 and A4, together with the attention cues (A5), raised emotion validation from 28.3% to 36.7% and WASSA empathy from 4.89 to 5.05 relative to top-1 prompting without them (both significant after correction), while all-rules adherence changed by a non-significant -5.3 points. Replies also became shorter (57.1 vs 80.4 words) and every reply stayed within the length limit.

### 5.2.3 Implementing the Attention Mechanism

The original design specified an attention mechanism, but it had not been implemented. When we added extraction of the detector's attention weights, verification revealed that the library's default attention backend silently returns no weights, which would have left the feature inactive while appearing to work. The adjustment forces the eager attention implementation, fails loudly if weights are missing, and passes the highest-attention words to the generator as context cues. It also expands sub-word fragments to whole words and limits the cues to the top 40% of word spans, so the cues do not simply repeat the entire message.

### 5.2.4 Developing the Final Solution: A Multi-Turn Conversational Agent

The single-turn evaluation assumes one message and one reply. Developing the solution into a conversational agent required three further adjustments.

**Detector context window (A6).** Classifying only the latest message loses context. For example, "I am okay. Well, not properly." read alone is classified as *content* with 72% confidence, although in context the user was angry. Guided by emotion-recognition-in-conversation work that finds windows of two to five utterances effective [[CITE: an ERC context-window study - team to verify the exact source]], we tested windows on long conversational scenarios. A four-turn window was best or tied-best in three of four scenarios, while six turns degraded two scenarios, in one case flipping a breakup scenario to the wrong emotion at 64% confidence, as stale emotions from earlier turns competed with the current one. The final agent classifies the last four user turns.

**Generator memory (A7).** The generator receives the last ten exchanges verbatim rather than a summary, so it can avoid repeating itself without losing emotional detail to summarisation.

**Turn-aware rules.** The mandatory single question suits a first reply but turns a long conversation into an interrogation. From the second turn onward, the question is optional and repeated phrasing is discouraged. The first turn keeps the original rules so that single-turn results remain valid. The agent also detects when the user's emotion changes between turns and tells the generator to respond to the current emotion.

A web demonstration of the agent was prepared for Hugging Face Spaces, with disclaimers and crisis resources on the page. [[TEAM TO FILL: deployed URL and date, or state that it was prepared but not deployed.]]

### 5.2.5 Rejected and Deferred Adjustments

Not every candidate adjustment was adopted. **Retrieval augmentation (A8)**, suggested by the Pre-Thesis 2 panel, reduced all-rules adherence from 63.7% to 43.7% and was rejected (Section 5.2). **A larger detector (A10)** was also tested: RoBERTa-large is 2.9 points more accurate on the test split, but swapped into the final pipeline it improved no reply-quality measure (all rules 60.0% to 54.7%, p = 0.16; WASSA 5.05 to 5.01, p = 0.43; emotion validation 36.7% to 38.0%, p = 0.68), so RoBERTa-base was kept and the conversational agent was not changed. Two further adjustments were **deferred to future work**. A repetition penalty (A9) could reduce the final system's boilerplate rate of 58.7%, but regenerating the final outputs would have invalidated AI-judge scores already being collected. A pre-generation crisis filter is needed before any public deployment (Section 3.4).

---

## Traps

- The **11.7%** identical-text figure and the **56.9% / 48.2%** accuracies come from measurements recorded in `train_roberta_ed_d1/train_roberta_ed_d1.py`. Cite them as our measurements.
- The history-window results are **scenario tests**, not a statistical experiment. The draft words them accordingly; keep it that way.
- A3, A4 and A5 were evaluated **together** against top-1 Phi (A2, the generator switch, is on both sides of that comparison). Don't assign the gain to any one of them.
- `[[CITE]]` for the context-window literature: find and read the actual paper before citing it. The "2-5 turns" claim came from a search during development and has not been checked against a specific paper.
