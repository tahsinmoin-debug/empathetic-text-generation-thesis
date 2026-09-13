# Chapter 5 - 5.6 Discussion

**Rubric:** CO6 (Chapter 5: 5.6 Discussion). There is no 5.5 in the rubric; confirm numbering with the supervisor.

## What this section must do

Interpret the results as a whole: what they mean for the research question, why the surprising results happened, how the work answers the Pre-Thesis 2 panel's concerns, what the limitations are, and what should come next. *AI* (MDPI) 2025 structures its discussion as **summary of contributions → challenges → practical implications → future work → closing statement**. Follow that order. The discussion introduces no new numbers, only ones already reported in 5.1-5.4.

## Tables in this folder

| File | Use |
|---|---|
| `tables/two_stage_selection_accuracy.csv` | Reference for 5.6.2 |
| `tables/d1_most_confused_pairs.csv` | Reference for 5.6.2 |

## Checklist

- [ ] 5.6.1 Answer to the research question
- [ ] 5.6.2 Why the two-stage design works even though its pick is less accurate
- [ ] 5.6.3 Response to the panel: why multiple models, and why not RAG
- [ ] 5.6.4 When a larger single model is the better choice
- [ ] 5.6.5 Limitations
- [ ] 5.6.6 Future work
- [ ] Closing paragraph `[[PENDING: final sentence depends on human evaluation and AI-judge results]]`

---

## Draft

### 5.6.1 Answer to the Research Question

This thesis asked whether separating emotion detection from response generation produces better empathetic replies than asking one model to do both, under the constraint of a single consumer GPU. The answer from our experiments is qualified. **At comparable scale, the decoupled design produces replies that are significantly more empathetic by the WASSA classifier and significantly more faithful to the user's actual emotion**, while its rule adherence is statistically indistinguishable from a same-size single model after correcting for multiple comparisons. **A larger single model, however, outperforms it** on rule adherence and empathy. The advantage of decoupling is therefore not that separation alone beats scale. It lies in what separation makes possible: an explicit and inspectable emotional decision that can be conditioned on, audited and intercepted.

### 5.6.2 Why the Two-Stage Design Works

The most surprising result is that the final system's emotion pick agrees with the dataset's gold label *less* often than the detector's own top choice (42.3% vs 50.3%), yet the final system has the highest emotion validation and the strongest empathy scores among comparable designs. Three observations reconcile this. First, EmpatheticDialogues assigns each conversation a single fine-grained label chosen by the speaker before the conversation, while many messages plausibly express a neighbouring feeling. Most detector errors are exactly such neighbours (afraid/terrified, sentimental/nostalgic), so a "wrong" pick is often a reasonable reading of the text. Second, the error-propagation analysis shows that replies built on a wrong label are no worse on rules or empathy. What the reply needs is an emotion close enough to respond to, not the exact gold label. Third, the gain appears when emotion-specific guidance is added: telling the generator *how* to respond to loneliness, not just *that* the user is lonely, changes the substance of the reply. Our ablation removed the pick, the guidance and the attention cues together, so we cannot separate their contributions. The accuracy result suggests the guidance, not the pick, does most of the work.

### 5.6.3 Response to the Pre-Thesis 2 Panel

The panel questioned whether multiple models were necessary and suggested that a simpler agentic, RAG-based design might work better. We tested both suggestions directly. A single model of the same size, prompted to identify the emotion and respond in one pass, produced less empathetic and less emotionally faithful replies than the decoupled design. A single larger model performed better on raw quality, which we report as a genuine trade-off (5.6.4). Retrieval of similar human conversations reduced rule adherence in both versions we built (from 63.7% to 54.3% and 43.7%), because real listener replies model the multi-question behaviour the rules are meant to prevent. We did not build the full multi-turn agentic version the panel described, so our evidence covers retrieval-augmented prompting, not agentic systems in general. The second model earns its place as a transparency and control component: it exposes the emotional decision and costs 125M parameters and under ten minutes of training. A larger detector would not change this conclusion. Swapping RoBERTa-large into the final system left reply quality unchanged.

### 5.6.4 When a Larger Single Model Is the Better Choice

If the only goal is the highest-quality reply on hardware that can run a 7B model, our results favour the larger single model. It led on rule adherence (82.3%), WASSA (5.76) and AI-judge scores. The decoupled design is preferable when the system's emotional reasoning must be visible, for example to audit why a reply took a certain tone, to show users or clinicians which emotion was detected, or to insert safeguards such as a crisis filter before any text is generated. It is also preferable when each component must be replaceable independently. The two approaches are not exclusive: the detector and emotion-specific prompting could be combined with a larger generator, which we leave to future work.

### 5.6.5 Limitations

(1) **Data.** EmpatheticDialogues is English, crowd-sourced and single-label, so neither detection accuracy nor reply appropriateness can be assumed to transfer to Bangla speakers or to people in genuine distress. (2) **Safety.** The label set has no category for suicidal ideation and the system has no crisis safeguard (Section 3.4). (3) **Evaluation.** Automatic metrics are proxies: the acknowledgement check is keyword-based and biased toward negative emotions, WASSA was trained on peer discussions of news articles, outputs coarse classes and rewards length, emotion validation uses our own detector, and AI judges may share length biases. (4) **Scale of evidence.** Generation results use 300 conversations and AI-judge results 64, and the human study uses a small item set with non-expert raters [[PENDING: final participant count]]. (5) **Single-turn evaluation.** Quantitative results are single-turn. The multi-turn agent was validated only on scenario tests. (6) **Repetition.** The final system reuses stock phrases in 58.7% of replies. (7) **Efficiency** (latency and memory at inference) was not measured.

### 5.6.6 Future Work

In order of priority: (i) a pre-generation crisis filter with reviewed safe responses, required before any public deployment; (ii) a component-wise ablation separating the stage-1 pick, emotion-specific guidance and attention cues; (iii) the detector and emotion-specific prompting combined with a larger generator; (iv) decoding-level repetition control, such as a repetition penalty of 1.1; (v) evaluation on real multi-turn conversations and with mental-health professionals; (vi) a Bangla or code-mixed dataset for the detector; and (vii) measurement of inference latency and memory for deployment decisions.

### Closing

[[PENDING: one paragraph restating the contribution, updated once the human evaluation and the final system's AI-judge scores are in. If human raters prefer the final system over the same-size single model, say so with the selection rate. If they prefer the larger model, keep the inspectability framing and report it plainly.]]

---

## Traps

- Don't turn 5.6.2 into a claim that the gold labels are "wrong". They are a single fine-grained choice; the draft frames it that way.
- Don't write the closing before the pending results are in. The framing depends on them.
- "Transparency and control" is the defensible headline. "Beats single models" is not.
