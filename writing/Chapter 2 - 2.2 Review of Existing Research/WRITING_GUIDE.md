# Chapter 2 - 2.2 Review of Existing Research

**Rubric:** CO9 (literature survey). Pre-Thesis 1 and Final Thesis reports.

## What this section must do

Review prior work closely enough that Chapter 5.4 can compare against it and 2.3 can name the gap. The BRAC template gives each study one paragraph: **what problem → data → method → result (with a number) → drawback**. The COLING papers group works into themed sub-headings. The draft below does both.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/literature_review_matrix.csv` | A summary table at the end of 2.2 (template sample reports end 2.2 or 2.3 with one) |
| `tables/emotion_accuracy_vs_literature.csv` | Can also be used here; required in 5.4 |

## Checklist

- [ ] Four themes: (A) benchmark and early models, (B) knowledge-enhanced models, (C) LLM-based empathetic generation, (D) emotion recognition and evaluation methods.
- [ ] One number per study wherever the paper gives one.
- [ ] Every paragraph ends with the limitation that matters for *our* design. Those limitations feed 2.3.
- [ ] Verify every bibliographic entry (authors, venue, pages) in Google Scholar before submission. Entries marked "verified" below were checked against the PDF or arXiv page.

## Style notes

- Template: "The study [1] mainly focuses on... This study used... The model had shown... Despite showing remarkable results the main drawback of this model is...".
- COLING main.598 related work: "Lin et al. (2019) employed ... (**MoEL**). Ghosal et al. (2020) divided emotions into two groups according to their polarity (**MIME**)."
- Use present tense for what a paper proposes and past tense for what it found.

---

## Draft

### 2.2.1 Empathetic Dialogue Benchmarks and Early Models

Rashkin et al. [[CITE: Rashkin et al., 2019, ACL; verified]] introduce EmpatheticDialogues, a benchmark of 24,850 crowd-sourced conversations grounded in 32 emotion labels, and show that retrieval and generation models trained on it are rated as more empathetic by humans. The authors also train emotion classifiers on the situation descriptions: fastText reaches 43% accuracy and a fine-tuned DeepMoji model 48%. The dataset established the task, but these classifiers read the explicit, after-the-fact situation description rather than what the speaker actually says in the conversation, which is the harder input a deployed system receives.

Lin et al. [[CITE: Lin et al., 2019, EMNLP]] propose a mixture of empathetic listeners (**MoEL**), in which a separate decoder specialises in responding to each emotion and a meta-listener combines them according to the predicted emotion distribution. Majumder et al. [[CITE: Majumder et al., 2020, EMNLP]] group emotions by polarity and mimic the user's emotion with stochastic variation (**MIME**). Li et al. [[CITE: Li et al., 2020, COLING]] add an adversarial framework with coarse and fine-grained emotion signals (**EmpDG**). As re-evaluated by Qian et al. [[CITE: Qian et al., 2023, arXiv 2310.05140; verified]], these models recognise the speaker's emotion with only 30.6-31.2% accuracy and produce low-diversity replies (Dist-2 of 1.83-2.16). Their main drawback is that emotion recognition is learned as a side task of generation in small models trained from scratch, which limits both recognition accuracy and fluency.

### 2.2.2 Knowledge-Enhanced Empathetic Generation

Later work adds external knowledge to capture *why* the user feels something. **KEMP** [[CITE: Li et al., 2022, AAAI]] injects emotion-related concepts from knowledge graphs. **CEM** [[CITE: Sabour et al., 2022, AAAI]] uses COMET commonsense inferences to model both affective and cognitive empathy. **CASE** [[CITE: Zhou et al., 2023, ACL]] aligns cognition and affection at coarse and fine granularity. **ESCM** [[CITE: Yang et al., 2023, Findings of EMNLP - numbers taken from search summary, confirm in paper]] models dynamic correlations between emotion and semantics. These methods raise emotion accuracy to 36.6-41.2% and Dist-2 to 2.86-4.11. Human evaluation by Qian et al. still rates CEM and CASE at only 3.08 and 3.04 out of 5 for empathy. The limitation is that knowledge integration adds complexity to small generators without closing the quality gap to modern language models.

Cao et al. [[CITE: Cao et al., 2025, TOOL-ED, COLING, pp. 5305-5320; verified]] take this idea into the LLM era. Their **EKTC** framework lets a fine-tuned LLM decide when to call a commonsense knowledge base as a tool, and they build the **TOOL-ED** dataset by inserting tool-use trajectories into ED. Fine-tuning Qwen1.5-14B and Vicuna-7B with this data gives a BERTScore of 0.8765. In A/B human evaluation by three researchers, the tool-calling model wins 61.3% of empathy comparisons for Qwen and 72.3% for Vicuna against the same model LoRA-tuned on plain ED (a GPT-4 judge gives a similar 71% for Qwen). The approach requires fine-tuning 7-14B-parameter models, well beyond an 8 GB consumer GPU, and it does not measure whether replies obey explicit interaction constraints.

Wu et al. [[CITE: Wu et al., 2025, From Traits to Empathy, COLING, pp. 8925-8938; verified]] argue that empathy should adapt to the listener's personality and add visual and MBTI personality information to a GPT-2-based generator. On text-only ED their model reaches 45.4% emotion accuracy and Dist-2 of 6.22, improving on earlier non-LLM models, and it is evaluated by seven graduate annotators with both 1-5 ratings and pairwise comparisons. The drawback for general deployment is its dependence on multimodal input and personality labels, which a text support conversation does not provide.

### 2.2.3 Large Language Models for Empathetic and Supportive Dialogue

Qian et al. [[CITE: Qian et al., 2023; verified]] compare ChatGPT and GPT-3.5 with the trained state of the art on ED. ChatGPT is rated 4.64 out of 5 for empathy by human evaluators against 3.08 for CEM and 3.04 for CASE, and its diversity is several times higher (Dist-2 of 18.29). They conclude that LLMs "significantly outperform state-of-the-art models". However, a 2024 systematic review of LLM empathy [[CITE: "Large Language Models and Empathy: Systematic Review", J Med Internet Res 2024;26:e52597 - confirm authors]] notes recurring problems with LLM empathy: repetitive empathic phrases, overly long responses, difficulty following initial instructions, and sensitivity to prompt wording. These are exactly the behaviours our rule-adherence and repetition measurements target.

Wang et al. [[CITE: Wang et al., 2025, arXiv 2507.12820; verified]] combine prompt engineering with LoRA and full-parameter fine-tuning of Qwen2.5 models for the NLPCC 2025 emotional-support task. Every fine-tuned configuration beats the prompted base model on validation (best total score 41.17 for LoRA rank 32 on Qwen2.5-72B, against 30.87 without fine-tuning), and their submission ranked second, but the gains require fine-tuning a 72B-parameter model. Rasool et al. [[CITE: Rasool et al., 2025, AI 6(3):56; verified]] fuse emotion-lexicon and multi-level embeddings into Flan-T5, Llama 2, DeepSeek-R1 and GPT-4 for psychotherapy transcripts, using a BERT-based model for emotion detection and a generative LLM for the response. Their paired t-tests show that the improvements in empathy (p = 0.53) and coherence (p = 0.12) are not statistically significant, and they explicitly call for human evaluation. Both works support separating emotion understanding from generation, but neither enforces or measures controllable response behaviour, and the second shows how easily reported gains fail a significance test.

### 2.2.4 Emotion Recognition and Evaluation Methodology

Paranjape et al. [[CITE: Paranjape et al., 2020, Neural Generation Meets Real People, arXiv 2008.12348; verified]] build Stanford's Chirpy Cardinal socialbot and, as a component, fine-tune RoBERTa on the first utterance of each ED conversation, reporting 61.5% top-1 and 92% top-5 accuracy. Most errors are confusions between similar labels such as *angry* and *furious*. Inspecting the released checkpoint shows it is RoBERTa-large and that the reported figure is on the official validation split. This is the result most directly comparable to our detector. Belay et al. [[CITE: Belay et al., 2025, COLING, pp. 3523-3540; verified]] evaluate encoder-only, encoder-decoder and decoder-only models on multi-label emotion classification in four Ethiopian languages and English (SemEval 2018), concluding that "fine-tuning encoder-only language models can still outperform the few-shot approaches of LLMs". This supports using a dedicated encoder rather than a generative model for detection.

For evaluation, Zheng et al. [[CITE: Zheng et al., 2023]] establish LLM-as-a-judge methodology and document its biases, including a preference for verbose answers and for the judge's own model family. The WASSA 2024 shared task [[CITE: Giorgi et al., 2024]] provides trained empathy predictors. Across the ED literature, human evaluation uses either 1-5 Likert ratings (Qian et al., Wu et al.) or A/B preference with Win/Lose/Tie (Cao et al.; Wu et al.). Wu et al. note that pairwise comparison accounts for individual variation among annotators.

[Table: literature_review_matrix.csv] *Summary of existing research reviewed in this chapter.*

---

## Traps

- **Chirpy Cardinal is the one comparison a knowledgeable examiner may raise.** Present it here honestly (see 5.4). Never write that our detector "outperforms all published work".
- Qian et al.'s emotion accuracies for MoEL, MIME and the rest are **their re-evaluation**. Other papers report slightly different numbers for the same models (COLING 598 lists MoEL at 32.2%). Cite the table you take a number from.
- The JMIR systematic review was found through search, not read in full. Confirm the authors and the quoted limitations in the paper itself before relying on it, or drop the sentence.
