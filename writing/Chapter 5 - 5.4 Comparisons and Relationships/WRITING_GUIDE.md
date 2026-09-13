# Chapter 5 - 5.4 Comparisons and Relationships

**Rubric:** CO6 (Chapter 5: 5.4 Comparisons and Relationships).

## What this section must do

Two jobs. **(a) Compare against published work** on EmpatheticDialogues, honestly, only where the comparison is valid, and say where it is not. **(b) Report relationships between variables** found in our experiments: what drives what. Examiners test the first for overclaiming and value the second as insight.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/emotion_accuracy_vs_literature.csv` | Detector vs published ED emotion accuracy |
| `tables/human_empathy_vs_literature.csv` | Human-evaluation protocols and results in the literature (our row pending) |
| `tables/controlled_d1_swap_same_d2.csv` | Relationship: detector accuracy → reply quality (initial pipeline) |
| `tables/detector_sensitivity_robertalarge.csv` | Same relationship tested in the final system |
| `tables/error_propagation.csv` | Relationship: label correctness → reply quality |
| `tables/length_vs_wassa_correlation.csv` | Relationship: reply length → WASSA |
| `tables/gemma_length_band_analysis.csv`, `figures/fig_wassa_by_length_band.png` | Length-matched comparison |

## Checklist

- [ ] 5.4.1 Emotion recognition vs the literature (with the Chirpy Cardinal comparison)
- [ ] 5.4.2 Response generation vs the literature (what is and isn't comparable)
- [ ] 5.4.3 Detection accuracy vs reply quality
- [ ] 5.4.4 Reply length vs measured empathy
- [ ] 5.4.5 Generator capacity vs prompt complexity
- [ ] 5.4.6 Emotion similarity vs detection errors

---

## Draft

### 5.4.1 Emotion Recognition Compared with Published Work

Table 5.x places our detector among published emotion-recognition results on EmpatheticDialogues's 32 classes. Comparisons must account for two differences: **what text the system reads** (a full conversation, the explicit situation description, or only the speaker's first message) and **which split is reported**.

Our detector (50.3% top-1) exceeds all joint recognition-and-generation models listed: MoEL, MIME, EmpDG, KEMP, CEM, CASE and ESCM at 30.6-41.2%, and multimodal and personality-aware models at 45.4-48.3%. It is on par with RoBERTa-GPT2 (51.5%). These models read the whole conversation but learn emotion as a side task of generation, so the gap reflects the benefit of a dedicated classifier. Classifiers on situation descriptions (fastText 43%, DeepMoji 48%, RoBERTa 65.9%) read an easier, explicitly emotional text and are not directly comparable. Our own GPT-2 detector scored 56.9% on situation text but 48.2% on user messages.

The most directly comparable result is the RoBERTa classifier of Chirpy Cardinal [[CITE: Paranjape et al., 2020]], trained on the same input, the first utterance, which reports **61.5% top-1 and 92% top-5**. Inspecting its released checkpoint shows that it is **RoBERTa-large** and that its saved evaluation, whose per-class counts match the official validation partition, is a **validation-set** score. The fair counterpart in our work is our RoBERTa-large detector at **53.0% top-1 and 87.2% top-5 on the held-out test split**, which remains about 8 points lower. Differences in training settings and in validation vs test evaluation may explain part of this gap. We did not pursue it further because our analysis shows that detector accuracy does not propagate to response quality (5.4.3).

[Table: emotion_accuracy_vs_literature.csv]

### 5.4.2 Response Generation Compared with Published Work

Most automatic generation metrics reported on ED are **not comparable** with ours. Distinct-n is computed over the full test set in published work but over 300 replies here, and distinct-n falls as the sample grows (on our own output, from 32.3% at 50 replies to 15.6% at 300). BERTScore in published work compares a reply to the gold response after multi-turn context, while our systems respond to the first message only. We therefore do not rank our system against published automatic scores. Our evaluation instead measures properties published work does not report: explicit rule adherence, an independent empathy classifier, emotion validation and paired significance tests.

Human evaluation offers a qualitative comparison of direction. In the literature, human raters strongly prefer LLM replies to trained empathetic models (ChatGPT 4.64 vs CEM 3.08 out of 5), and fine-tuned tool-augmented LLMs win 61-72% of A/B empathy comparisons against LoRA baselines (Table 5.x). [[PENDING: our human-evaluation selection rate for the final system against the same-size and larger single models, and how its direction compares with these findings.]] Because protocols differ (Likert, A/B, and our three-way choice), only the direction of results is comparable, not the magnitude.

[Table: human_empathy_vs_literature.csv]

### 5.4.3 Relationship between Detection Accuracy and Reply Quality

Detection accuracy and reply quality are only weakly related in this system. In the controlled comparison, where the generator (Qwen2.5-1.5B), prompt and conversations are held fixed and only the detector changes from GPT-2 (48.2% on real input) to RoBERTa (50.3%), neither emotion validation (26.0% vs 25.3%, p = 0.91) nor rule adherence (63.7% vs 57.7%, p = 0.09) improves. Within a single pipeline, replies built on a wrong emotion label follow the rules as often as those built on a correct label, and are rated as empathetic (Section 5.3.3). The replies are robust to label errors partly because most errors are between neighbouring emotions (5.4.6), whose appropriate responses are similar. By contrast, **how the emotion is used** matters. With the same detector and generator, emotion-specific two-stage prompting raises emotion validation by 8.3 points. Conditioning a small detector's output well matters more than squeezing out more detection accuracy.

The same relationship holds in the final system. Replacing RoBERTa-base with RoBERTa-large raised detector accuracy on the evaluation conversations from 50.3% to 53.3%, but the stage-1 pick changed in only 31% of conversations, and rule adherence, WASSA empathy and emotion validation did not change significantly (Table 5.x). Across three independent tests (label errors, a GPT-2 to RoBERTa swap and a RoBERTa-base to RoBERTa-large swap), gains in detection accuracy of 2-3 points never translated into better replies.

[Table: controlled_d1_swap_same_d2.csv] [Table: error_propagation.csv] [Table: detector_sensitivity_robertalarge.csv]

### 5.4.4 Relationship between Reply Length and Measured Empathy

WASSA empathy increases with reply length within every system, with Spearman correlations from 0.14 (top-1 Phi) to 0.54 (zero-shot). Measured empathy is therefore partly a function of verbosity. Gemma-3-4B shows why this matters: it has the second-highest WASSA score overall (5.30), but at matched lengths it does not beat Phi-4-mini. In every length band where both have replies, Phi matches or exceeds it, and Gemma's advantage appears only above 120 words (Figure 5.x). The final system reverses the usual direction: it achieves higher WASSA than top-1 Phi prompting while writing about 30% fewer words. This relationship means empathy classifiers and LLM judges, which are also known to favour longer answers, should be interpreted alongside reply length.

[Figure: fig_wassa_by_length_band.png] [Table: length_vs_wassa_correlation.csv] [Table: gemma_length_band_analysis.csv]

### 5.4.5 Relationship between Generator Capacity and Prompt Complexity

Richer prompts help only generators with enough capacity to follow them. With Qwen2.5-1.5B, adding the stage-1 pick, emotion-specific guidance and cue words *lowers* all-rules adherence from 57.7% to 36.7% and acknowledgement from 67.7% to 47.0%. The same additions with Phi-4-mini (3.8B) keep rule adherence statistically unchanged and improve empathy and emotion validation. The earlier candidate-pick experiment showed the same pattern at 1.5B: asking one call to both disambiguate and respond reduced adherence from 63.7% to 54.0%. Across model sizes, rule adherence under the single-model prompt rises from 51.3% (1.5B) to 82.3% (7.6B). Our results therefore suggest a practical threshold: instruction complexity should grow with generator capacity, and below a few billion parameters the simpler prompt is safer.

### 5.4.6 Relationship between Emotion Similarity and Detection Errors

Detection errors concentrate between emotions that differ in intensity or nuance rather than in kind: *sentimental* → *nostalgic*, *afraid* → *terrified*, *angry* → *furious*, *sad* → *devastated*, *apprehensive* → *anxious* (Table 5.4). The weakest class, *angry* (F1 18.2%), is largely absorbed by its stronger neighbour *furious*. Because the right label is usually a near neighbour, the correct emotion falls within the top-3 in 76.5% of cases against 50.3% at top-1. This is the relationship the two-stage design exploits, and it helps explain why label errors cause little damage downstream.

---

## Traps

- **Never write "our detector outperforms all published results."** The draft's wording (exceeds joint models, on par with RoBERTa-GPT2, below Chirpy Cardinal's RoBERTa-large) is the defensible version.
- Don't attribute the whole 8-point Chirpy gap to "validation vs test". That is a plausible factor, not a measured one. The draft says "may explain part of this gap".
- "Practical threshold" in 5.4.5 is an interpretation from two generator sizes plus one 7B single model. Keep the hedge "suggest".
- The claim that LLM judges favour longer answers comes from the LLM-as-a-judge literature (Zheng et al., 2023). Cite it; we did not measure it for our judges.
