# Thesis Writing Kit

Everything needed to write the final report sections listed in the course rubric (CO5-CO14), with one folder per rubric section. Each folder contains:

| Item | What it is |
|---|---|
| `WRITING_GUIDE.md` | What the rubric asks, what to cover, a **draft in the style of our reference papers**, and the traps to avoid |
| `tables/` | CSV tables plus `TABLES.md` (the same tables formatted, with their source and caveats) |
| `figures/` | 300-dpi PNG figures ready to insert into Word (only in folders that need them) |

Folder names follow the rubric exactly, except that `:` became ` -` because Windows does not allow colons in folder names.

**Nothing in `tables/` or `figures/` was typed by hand.** Every number is regenerated from the raw experiment outputs by two scripts:

```
python writing/_scripts/build_writing_tables.py     # tables (CPU only, a few seconds)
python writing/_scripts/build_writing_figures.py    # figures (run after the tables)
python writing/_scripts/build_writing_docx.py       # Word versions with tables and figures embedded
```

When the AI-judge scores for the final system arrive, merge them into `llm_judge_evaluation/` as usual, re-run both scripts, and the judge tables fill themselves.

---

## 1. Folder index and rubric mapping

| Rubric | Folder | Status |
|---|---|---|
| CO9 | `Chapter 2 - 2.1 Preliminaries` | Ready to write |
| CO9 | `Chapter 2 - 2.2 Review of Existing Research` | Ready to write |
| CO9 | `Chapter 2 - 2.3 Summary of Key Findings` | Ready to write |
| CO12 | `Chapter 3 - 3.4 Ethical Issues` | Ready to write |
| CO10 | `Chapter 3 - 3.6 Project Management Plan` | Needs future dates from the team |
| CO10 | `Chapter 3 - 3.7 Risk Management` | Ready to write |
| CO11 | `Chapter 3 - 3.8 Economic Analysis` | Needs prices (BDT) from the team |
| CO5 | `Chapter 4 - 4.1 Design Process or Methodology Overview` | Ready to write |
| CO5 | `Chapter 4 - 4.2 Preliminary Design or Design (Model) Specification` | Ready to write |
| CO6 | `Chapter 5 - 5.1 Performance Evaluation` | AI-judge and human-evaluation rows **pending** |
| CO6 | `Chapter 5 - 5.2 Analysis of Design Solutions` | Ready to write |
| CO7 | `Chapter 5 - 5.2 Final Design Adjustments` | Ready to write |
| CO6 | `Chapter 5 - 5.3 Statistical Analysis` | Ready; final-system judge rows **pending** |
| CO6 | `Chapter 5 - 5.4 Comparisons and Relationships` | Human-evaluation row **pending** |
| CO6 | `Chapter 5 - 5.6 Discussion` | Ready to write; final paragraph waits for pending results |
| CO8 | `Presentation - Tools Used` | Ready |

**Numbering conflict in the rubric:** it lists both "5.2 Analysis of Design Solutions" and "5.2 Final Design Adjustments", and there is no 5.5. Two folders are provided. Ask the supervisor which numbering to use. The simplest fix is 5.2 Analysis, 5.3 Final Design Adjustments, then renumber the rest.

---

## 2. Placeholders used in the drafts

| Marker | Meaning |
|---|---|
| `[[PENDING: ...]]` | Result not collected yet (AI-judge scores for the final system, human evaluation). Leave the sentence and fill the value later. |
| `[[TEAM TO FILL: ...]]` | A fact only the team knows (dates, prices, names, roles). Do not guess. |
| `[[CITE: ...]]` | Insert the citation in the thesis's reference style. |
| `[Table: file.csv]` / `[Figure: file.png]` | Insert the table or figure from this folder and give it a chapter number. |

---

## 3. Writing style (taken from our reference papers)

The drafts copy the patterns used in COLING 2025 main.598 (Wu et al.), COLING 2025 main.355 (TOOL-ED, Cao et al.), *AI* 2025 6(56) (Rasool et al.), arXiv 2507.12820 (Wang et al.), and the BRAC CSE400 thesis template. Follow these rules when editing:

1. **First-person plural, active voice.** "We fine-tune RoBERTa-base on..." Not "RoBERTa-base was fine-tuned by the authors...". Both COLING papers write this way.
2. **Every results paragraph opens by pointing at its table or figure.** "Table 5.3 shows the automatic evaluation results." "As illustrated in Figure 5.2, ...". This is the most consistent pattern across all four papers.
3. **Claim, then number, then reason.** Pattern from main.598: "Compared with X, our method obtains ... Concretely, we achieve A in absolute ... The improvements show that ...". State the direction, give the exact number, then explain with "This indicates that / This suggests that / This could be attributed to".
4. **Name baselines as `Author (year) did X (ACRONYM)`.** From main.598's related work: "Lin et al. (2019) employed ... (MoEL)."
5. **Ablation language:** "To substantiate the role of component X, we remove it (w/o X)..." Use it for the design-alternative comparisons.
6. **Report negative results plainly.** main.355 writes "the performances ... are slightly lower than some of the LLMs. This could be attributed to...". Do the same for RAG, Gemma, and the same-size comparison.
7. **Implementation details form one dense factual paragraph:** framework, hardware, model sizes, decoding settings, optimiser, learning rate, seed.
8. **Statistical claims name the test, the p-value and the direction.** From *AI* 2025: "Coherence exhibits a marginal decline (T-statistic: -2.10, p-value: 0.12)". We add the confidence interval and the correction method.
9. **Contributions as a numbered (i), (ii), (iii) list** at the end of the introduction (main.355).
10. **Thesis chapters (BRAC template) use short sub-subsections** (3.1.1, 3.1.2, ...), each one or two paragraphs, and end a section with an "In summary," paragraph.

**Words to avoid** (the June Chapter 3 draft used them, and the panel will push back): "mathematically overcome", "mimics professional human counseling", "vastly more efficient", "massive", "perfectly", "completely resolved", brain-analogy framing (amygdala / prefrontal cortex). Replace them with the measured number.

**Do not mention BLEU or ROUGE anywhere.** They were dropped project-wide because reference-overlap metrics cannot rank empathetic replies.

---

## 4. Corrections: old material that must NOT be reused

| Old claim (June Chapter 3 / earlier docs) | Why it is wrong | Use instead |
|---|---|---|
| d1 = GPT-2 Medium, 56.64% accuracy | Measured on the situation `prompt` text, not the user message the system actually receives | Final d1 = RoBERTa-base, **50.3% top-1 / 76.5% top-3 / 84.3% top-5** on real input (`Chapter 4 - 4.2/tables/d1_evolution_mismatch_fix.csv`) |
| Table 3.1: GPT-2 56.64% vs Qwen2.5-0.5B 35.93% vs TinyLlama 16.91% | GPT-2's number is from EmpatheticDialogues, while Qwen's and TinyLlama's come from a **GoEmotions** test split. It compares different datasets. | The d1 backbone comparison (same data, same settings) |
| d2 = Qwen2.5-1.5B with LoRA | Final d2 is Phi-4-mini-instruct, zero-shot, 4-bit; the LoRA fine-tune collapsed (0% all-rules) | `Chapter 5 - 5.2 Final Design Adjustments` |
| "d1 beats all published ED results" | Only true against joint generation models; Chirpy Cardinal's roberta-large reports 61.5% on validation | `Chapter 5 - 5.4/tables/emotion_accuracy_vs_literature.csv` |
| "Decoupled design significantly beats a same-size single model on rule adherence" | True for the *initial* pipeline (p = 0.002). For the **final** system: 60.0% vs 51.3%, raw p = 0.032, **Holm-adjusted p = 0.095, not significant** | Claim significance on WASSA empathy and emotion validation instead, both significant after correction |
| "Temperature 0.9" | The code uses **temperature 0.7, top-p 0.9** | `configuration_and_hyperparameters.csv` |
| "WASSA was trained on reaction essays, not chat replies" | The model used is from **Track 2**, trained on human empathy ratings of **conversation turns** | `Chapter 5 - 5.3/tables/wassa_validity.csv` |
| "hyy33 ranked 4th" | 4th is the authors' own claim; the findings table lists them **5th of 9** by average | same table |
| 28-class GoEmotions results | Side research, not the thesis pipeline | Mention only as "initial dataset exploration" in 3.6 |

---

## 5. The five findings the whole thesis rests on

Keep these consistent in every chapter:

1. **Detection accuracy is not the bottleneck.** A wrong d1 label does not reduce rule adherence (p = 0.40-0.55) or WASSA empathy (p = 0.72). Swapping GPT-2 for the more accurate RoBERTa while keeping d2 fixed changes neither emotion validation (p = 0.91) nor rule adherence (p = 0.09). **Tested directly in the final system:** replacing RoBERTa-base with RoBERTa-large (3.0 points more accurate on the same conversations) changed no reply-quality measure (all rules p = 0.16, WASSA p = 0.43, emotion validation p = 0.68), so RoBERTa-base is kept.
2. **How the emotion is used matters more than how accurately it is detected.** The two-stage design (d2 picks from d1's top-3, then an emotion-specific prompt) raises emotion validation from 28.3% to 36.7% and WASSA from 4.89 to 5.05 over the same models with top-1 only. Both remain significant after Holm correction.
3. **Honest caveat on (2):** d2's stage-1 pick matches the gold label *less* often than d1's top-1 (42.3% vs 50.3%, p = 0.005). The gain comes from emotion-specific guidance, not from better label selection.
4. **Against baselines of the same size** the final system is significantly more empathetic (WASSA +0.21) and more emotionally faithful (+11.0 pp). It is not significantly better on rule adherence after correction.
5. **A 4.7x larger single model (Qwen2.5-7B) is still better** on rules, WASSA and AI-judge scores. The thesis argues for inspectability and controllability at small scale, not superiority over bigger models.

---

## 6. Still pending (leave space)

- AI-judge scores for `two_stage_phi` and `bakeoff_phi` (teammates running). Affects 5.1, 5.3 and 5.6.
- Human evaluation (13-item Google Form, `human_evaluation/`). Affects 5.1, 5.3, 5.4 and 5.6.
- Inter-judge agreement statistics. Compute after the scores arrive.
- Team facts: member roles, future dates, hardware price, electricity tariff.
