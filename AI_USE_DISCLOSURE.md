# Disclosure of AI assistance

This file records where AI assistance was used in this project, so that the disclosure in the
conference paper and in the thesis can be checked against the repository. It is written to be
accurate rather than flattering: where an AI assistant did most of the work, this file says so.

**Assistant used:** Claude (Anthropic), through Claude Code, during 2026.
**Not used:** no AI system produced any experimental result. Every model was trained and every
system evaluated by the authors on their own hardware, and every number in the thesis and the paper
comes from those runs and from result files kept in this repository.

## 1. Writing

| Item | AI role | Author role |
|---|---|---|
| Conference paper (`ICCIT format writing/`), Abstract and Sections I-VII | Wrote the complete first draft from the authors' results and notes | Chose the framing and claims, revised the text, verified every number and reference |
| Thesis writing kit (`writing/*/FINAL_TEXT`), Chapters 2-5 sections | Wrote the draft text, including the tables, captions and reference list | Same as above |
| Reference list (`writing/_scripts/references.py`) | Assembled and formatted entries in IEEE style | Checked entries against publisher pages |

Claimed contributions, claim wording and the decision about what may and may not be claimed were
discussed with the authors, and several claims were removed or weakened after those discussions.

## 2. Code

| Area | AI role |
|---|---|
| Analysis and reporting: `writing/_scripts/` (tables, figures, final text), `resource_benchmark/` | Written by the assistant |
| Evaluation orchestration: additions to `llm_judge_evaluation/run_llm_judge.py` and `run_empathy_classifier.py` (new settings, `--settings`, guards), `teammate_nemotron_package/` | Written by the assistant |
| Corrected LoRA baseline `train_qwen2_5_monolithic_d2_fixed/`, single-model Phi baseline `single_model_baseline/generate_single_model_phi.py` | Written by the assistant |
| Original pipeline, detector training, generation and scoring scripts (`train_*`, `emotion_prompts/`, `d2_settings_comparison/`, `conversational_agent/`) | Written by the authors; the assistant later made specific additions, such as the `--d1-dir` and overwrite guards |
| Running every script, on the authors' GPU | Authors only |

## 3. Analysis and error-finding

The assistant computed the reported statistics, and also found and reported errors in the authors'
earlier work, including a training-configuration error in the first fine-tuned baseline (documented
as adjustment A11), a mismatch between the detector's training text and its inference input, and
several overstated claims in earlier drafts. Decisions on how to respond to each finding were taken
by the authors.

## 4. Statement used in the paper

> In accordance with IEEE policy on AI-generated text, the authors disclose the following. Claude
> (Anthropic) was used to produce the initial draft of the Abstract and of Sections I to VII of this
> manuscript from the authors' experimental results and notes, and the authors then revised that
> draft. Claude was also used to write parts of the analysis code, namely the scripts that compute
> the reported statistics and generate the tables and figures, and the scripts that run and merge the
> LLM-judge evaluation. It was not used to generate any experimental result: all models were trained
> and all systems evaluated by the authors on their own hardware, and every number reported here comes
> from those runs. The authors verified all results, references and claims, and take full
> responsibility for the content of this paper.

## 5. To complete before submission

- If any other AI writing tool was used, for example a built-in assistant in a reference manager or
  editor such as Prism, add one sentence naming that tool and the sections it touched.
- The university may require a separate declaration for the thesis; check the current BRAC University
  policy and add it to the thesis front matter.
