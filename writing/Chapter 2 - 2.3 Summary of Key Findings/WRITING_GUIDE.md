# Chapter 2 - 2.3 Summary of Key Findings

**Rubric:** CO9. Pre-Thesis 1 and Final Thesis reports.

## What this section must do

Condense 2.2 into what the field knows, what it has not solved, and where this thesis fits. In the BRAC template this is 1-2 dense paragraphs of findings followed by the drawbacks that motivate the work ("However, these models also have several drawbacks including... Future improvements should focus on..."). End with the gaps this thesis addresses, so that Chapter 4 follows naturally.

## Tables in this folder

| File | Use |
|---|---|
| `tables/literature_review_matrix.csv` | Use it here if 2.2 does not already end with it (don't print it twice) |

## Checklist

- [ ] Trends with numbers: emotion accuracy of joint models (30.6-48.3%), LLMs rated far above trained models (4.64 vs about 3.0 out of 5).
- [ ] Unsolved problems: controllability, verbosity and repetition, heavy fine-tuning, weak statistical evidence, emotion recognition on real input.
- [ ] Gaps G1-G4, each tied to a chapter where the thesis addresses it.
- [ ] Nothing here that is not already in 2.2. The summary introduces no new citations.

---

## Draft

Research on empathetic response generation has moved from small models trained from scratch on EmpatheticDialogues to large pre-trained language models. Early joint models such as MoEL, MIME and EmpDG treat emotion recognition as an auxiliary task and identify the speaker's emotion in only 30.6-31.2% of cases. Knowledge-enhanced successors (KEMP, CEM, CASE, ESCM) raise this to 36.6-41.2% by adding commonsense and emotion concepts, and multimodal, personality-aware generators reach 45.4-48.3% on text-only ED. Dedicated classifiers do better: a fine-tuned RoBERTa-large reaches 61.5% on the first utterance of each conversation (validation split), and fine-tuned encoders continue to outperform prompted LLMs at emotion classification. On the generation side, instruction-following LLMs changed the picture. Human evaluators rate ChatGPT at 4.64 out of 5 for empathy against about 3.0 for the best trained models, and recent systems fine-tune 7-72B-parameter LLMs, with knowledge tools or emotional-support data, to gain further.

However, these advances leave several problems open. First, **LLM empathy is fluent but hard to control**: reviews report repetitive empathic phrases, overly long answers and poor adherence to initial instructions, yet no ED study we reviewed measures whether a reply follows explicit interaction rules, such as validating before advising or asking one question rather than many. Second, **the strongest results rely on resources beyond a consumer GPU**: fine-tuning 14B-72B models, multimodal inputs, or commercial APIs that require sending sensitive disclosures to a third party. Third, **emotion recognition is often measured on the wrong input**. Classifiers trained on ED's situation descriptions read an explicit, after-the-fact summary rather than the message a user actually types, and joint models do not expose their recognised emotion for inspection. Fourth, **evaluation evidence is thin**: most studies report differences in automatic metrics or small human studies without paired significance tests, and at least one reports empathy gains that are not statistically significant (p = 0.53). Future work should therefore focus on controllable and inspectable behaviour under realistic resource limits, emotion recognition evaluated on real conversational input, and evidence that survives statistical testing.

### Gaps addressed by this thesis

| Gap | What the literature lacks | How this thesis addresses it | Where |
|---|---|---|---|
| G1 | Measured control over interaction behaviour | Explicit rules, scored per reply, compared against monolithic alternatives | 4.2, 5.1 |
| G2 | Inspectable emotion reasoning | Decoupled detector with visible top-3 emotions, attention cues and the emotion actually used | 4.1, 4.2 |
| G3 | Small-scale, self-hostable systems | d1 (125M) + d2 (3.8B, 4-bit) on one 8 GB GPU, with no fine-tuning of the generator | 4.2, 3.8 |
| G4 | Statistical rigour | Paired tests with confidence intervals and multiple-comparison correction, error-propagation analysis, and several independent evaluation signals | 5.3 |

---

## Traps

- **Don't claim a gap the literature has filled.** "Nobody separates emotion detection from generation" is false: Rasool et al. use BERT for detection and an LLM for generation. The gap is *measured control and inspectability under small-scale constraints*.
- Keep "consumer GPU" and "privacy" framed as design constraints, not as the headline novelty. Earlier panel feedback found the bare "8 GB constraint" framing weak.
