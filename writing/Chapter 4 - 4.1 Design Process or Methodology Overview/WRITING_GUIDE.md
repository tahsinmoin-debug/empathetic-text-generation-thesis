# Chapter 4 - 4.1 Design Process or Methodology Overview

**Rubric:** CO5, "Design multiple engineering/theoretical solutions ... to meet the desired objectives, need and requirements within the given constraints." Criterion: prepare the design process as per objectives, specifications, requirements and constraints. Marks: 5 from the Chapter 4 report + 10 from the defence panel.

## What this section must do

Explain *how* the solution was designed, step by step, before 4.2 describes *what* was designed. The BRAC template splits this into short sub-subsections (Problem Definition → Data → Architecture → Training Strategy → Evaluation) and ends with "In summary...". Follow that structure, and make the evidence-gated iteration visible, because it is what the panel will probe.

## Tables and figures in this folder

| File | Use |
|---|---|
| `figures/fig_design_process.png` | Design-process flow with the iteration loop (Figure 4.1) |
| `figures/fig_final_architecture.png` | Final system overview (Figure 4.2; may instead open 4.2) |
| `tables/tools_and_versions.csv` | Implementation environment (one short table or one paragraph) |
| `tables/ai_judge_panel.csv` | The three AI judges (4.1.5.1) |
| `tables/ai_judge_selection_criteria.csv` | Why these judges, with evidence (4.1.5.1) |
| `tables/wassa_validity.csv` | Why the WASSA classifier is used and how far it can be trusted (4.1.5.2) |

## Checklist

- [ ] 4.1.1 Problem definition and objectives, tied to 3.6.1 requirements
- [ ] 4.1.2 Data acquisition and preprocessing (ED, first utterance, `_comma_` cleaning, the official split, the 300-conversation evaluation sample)
- [ ] 4.1.3 Architecture design strategy (why decoupled; which alternatives were planned)
- [ ] 4.1.4 Training and generation strategy (fine-tune the detector only; prompt the generator)
- [ ] 4.1.5 Evaluation and interpretation strategy (five independent signals + statistics)
- [ ] "In summary" paragraph

---

## Draft

The design process follows an iterative, evidence-gated methodology (Figure 4.1). Starting from the problem definition and constraints, we prepare the data, design several alternative solutions for each component, simulate every alternative on the same held-out conversations, and evaluate them with multiple independent signals and paired statistical tests. A design adjustment is adopted only when this evidence supports it, and each adjustment re-enters the loop at the design stage.

[Figure: fig_design_process.png] *Figure 4.1: Design process.*

### 4.1.1 Problem Definition and Objective Formulation

The objective is a text-based conversational system that responds to a person sharing an emotional experience with a short, validating and non-intrusive reply, and that makes its emotional reasoning inspectable. This translates into three measurable design targets: (i) identify the expressed emotion among 32 fine-grained categories and expose the candidates and the final choice; (ii) generate replies that acknowledge the feeling, ask at most one question and stay short; and (iii) run both components on a single 8 GB consumer GPU without sending user messages to an external service. The central design question, raised directly by the Pre-Thesis 2 panel, is whether emotion detection and response generation should be separate models or whether a single model is sufficient. The methodology is therefore built to test that question rather than assume the answer.

### 4.1.2 Data Acquisition and Preprocessing

We use EmpatheticDialogues with its official speaker-disjoint train, validation and test partitions. Each conversation contributes one example: the **first utterance** written by the speaker, labelled with the conversation's emotion. We deliberately do not use the dataset's situation description field. That text is written after the fact and names the feeling far more explicitly, so a model trained on it learns an easier task than the one it faces in deployment (Section 5.2 quantifies this). The preprocessing restores commas that the dataset encodes as the string `_comma_`, drops rows that fail to parse, and keeps the first utterance of each conversation, giving 2,542 test examples for detector evaluation. For response generation, each test conversation is paired as (first user utterance, human listener reply), and a fixed random sample of 300 conversations (seed 42) is used for every generation experiment, so that all systems answer exactly the same inputs.

### 4.1.3 Model Architecture Design

We adopt a decoupled architecture with two components. **d1**, an encoder classifier, reads the user's text and outputs a probability distribution over the 32 emotions. **d2**, an instruction-tuned decoder, writes the reply conditioned on d1's output through its system prompt. The motivation is threefold. Encoders fine-tuned for classification outperform prompted generative models at emotion recognition. A separate detection step exposes the system's emotional reasoning for inspection. And each component can be chosen, replaced and evaluated independently within the memory budget. To test the design question rather than assume it, the architecture design explicitly includes **monolithic alternatives**: a zero-shot generator with no detector, a generator fine-tuned end to end on ED replies, and single models of the same and larger size prompted to detect the emotion and respond in one pass. It also includes alternative ways of connecting d1 to d2 (Section 4.2).

### 4.1.4 Model Training and Generation Strategy

Only the detector is trained. d1 candidates are fully fine-tuned with cross-entropy loss, early stopping on validation accuracy and a linear warmup of 6% of training steps. The generator is used **zero-shot**, controlled entirely by its system prompt, for two reasons established in the experiments: a LoRA fine-tuned generator imitated the dataset's habit of asking many questions and failed every rule, and prompting can be changed per emotion at no training cost. Generators larger than 1.5B parameters are loaded with 4-bit NF4 quantisation to fit in memory. All generation uses identical sampling settings (temperature 0.7, top-p 0.9, at most 250 new tokens, seed 42), so differences between systems come from the design, not from decoding.

### 4.1.5 Model Evaluation and Interpretation

No single automatic metric captures empathy, so each design is evaluated with **five independent signals**: (1) rule adherence, measuring whether a reply acknowledges the feeling, asks exactly one question and stays within 120 words; (2) the WASSA 2024 empathy classifier; (3) emotion validation, measuring whether the emotion expressed in the reply matches the true label as read by the final detector; (4) a blind panel of three AI judges from different providers scoring empathy, relevance and safety; and (5) a human forced-choice study. Detector candidates are evaluated with top-1, top-5 and macro-F1 on the full test split. Because all systems answer the same conversations, differences are tested with paired tests (McNemar's exact test for binary outcomes, the Wilcoxon signed-rank test for scores), 10,000-sample bootstrap confidence intervals and Holm correction for multiple comparisons. A difference is called significant only when the corrected p-value is below 0.05 and the confidence interval excludes zero. We additionally analyse error propagation, meaning whether a wrong detector label damages the final reply, to establish which component limits overall quality.

#### 4.1.5.1 Selection of the AI-Judge Panel

Using language models as evaluators is established practice: strong LLM judges agree with human preferences more than 80% of the time [[CITE: Zheng et al., 2023, Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS]]. They also carry known biases, and the panel was designed around them. First, **LLM evaluators recognise and favour their own generations** [[CITE: Panickssery, Bowman and Feng, 2024, LLM Evaluators Recognize and Favor Their Own Generations, NeurIPS]]. We therefore chose judges from three developers with no relation to the models being evaluated: **Nemotron 3 Ultra** (NVIDIA, 550B-parameter mixture-of-experts), **Mistral Large** (Mistral AI) and **gpt-oss-120b** (OpenAI, open weights). None shares a model family with our generator (Phi-4-mini, Microsoft) or our baselines (Qwen, Alibaba). Second, **a panel of judges from disjoint model families shows less intra-model bias than a single large judge** [[CITE: Verga et al., 2024, Replacing Judges with Juries, arXiv 2404.18796]], so we average three judges rather than relying on one. Third, following the co-supervisor's requirement, all judges are flagship-tier; smaller local judges considered earlier were dropped. Each judge sees only the user's message and one reply, never the system's name or the gold reply, and scores empathy, relevance and safety from 1 to 10 against explicit anchors (1-3, 4-6, 7-10) at temperature 0. All three were accessible through free API tiers, so non-GPU team members could run the evaluation. Earlier candidates were replaced when they became unavailable (Gemini quota failures, the retirement of DeepSeek on GitHub Models and of Llama 3.3 on Groq); valid scores collected before a retirement were kept. Table 4.x lists the panel and Table 4.x the selection criteria.

[Table: ai_judge_panel.csv] [Table: ai_judge_selection_criteria.csv]

#### 4.1.5.2 Use of the WASSA Empathy Classifier

As a signal independent of both our rules and the AI judges, we use the hyy33 system from the WASSA 2024 shared task, held at ACL 2024 [[CITE: hyy33 at WASSA 2024, ACL Anthology 2024.wassa-1.39]]. It suits this project for three reasons. It was trained on **third-party human ratings of empathy for individual conversation turns**, the same unit as our replies. It is a peer-reviewed, publicly released system whose predictions correlate with human empathy ratings at r = .544 on the task's test set. And it comes from a different model family, dataset and organisation than any component of our system, so it cannot favour our pipeline. Its limits are stated alongside: it was not the most accurate empathy predictor in its own task (the organisers' RoBERTa baseline reached r = .694), its training dialogues are peer discussions of news articles rather than supportive replies, and it outputs coarse ordinal classes. Section 5.3 shows that on our data it reproduces the AI judges' ranking of systems exactly but is too noisy to rank individual replies, so it is used **only to compare system averages** over 300 conversations.

[Table: wassa_validity.csv]

### 4.1.6 Implementation Environment

All components are implemented in Python 3.10 with PyTorch 2.5.1 and Hugging Face Transformers 5.16.1, using bitsandbytes for 4-bit quantisation and SciPy for statistical testing, on a single NVIDIA RTX 4060 (8 GB) with 16 GB of system memory (Table 4.x).

In summary, the design process treats the choice between a decoupled and a monolithic system as an empirical question. It holds data, sampling and decoding constant across alternatives and adopts only changes supported by several independent measurements that survive paired statistical testing.

---

## Traps

- Say **"first utterance"**, never "prompt", when describing d1's input. The distinction is the whole point of the mismatch fix.
- Don't describe the attention mechanism as "cross-attention between d1 and d2". It is the detector's own self-attention, read out to extract cue words.
- Human evaluation and AI-judge results are pending. This section only describes the *method*, so it can be written in full now.
