# 3.6 Project Management Plan

This section sets out the project's requirements, the development method followed, the schedule and resources, and the revisions made to the plan as the work progressed. Risks are analysed in Section 3.7 and costs in Section 3.8.

## 3.6.1 Requirements and Specifications

The requirements were refined from the project proposal, supervisor feedback and the remarks of the Pre-Thesis 2 defence panel into the following specification.

**Functional requirements.** (F1) Given a user's message, the system identifies the most likely emotion among the 32 EmpatheticDialogues categories [1] and exposes this decision for inspection. (F2) It generates a reply that acknowledges the feeling, offers a reassuring thought and asks at most one follow-up question. (F3) Replies are short: at most 100 words as instructed, and scored against a limit of 120 words. (F4) The system supports multi-turn conversation and tracks changes in the user's emotion.

**Non-functional requirements.** (N1) Both models run on a single consumer GPU with 8 GB of memory. (N2) No user message needs to be sent to a third-party service at inference time. (N3) Every reported result can be reproduced from saved outputs with fixed random seeds.

**Constraints.** The project was bound by a fixed academic calendar, a single GPU-equipped machine, a three-member team in which one member ran the GPU experiments, and the rate limits of free evaluation APIs.

## 3.6.2 Development Methodology

The project followed an iterative, evidence-gated methodology, described in Section 4.1. Each design alternative was implemented, simulated on the same 300 held-out conversations, scored with the same measures, and adopted only if the evidence supported it. Inexpensive local measures, such as rule adherence and emotion validation, served as a first gate before costly or rate-limited evaluation with AI judges and human participants, so that limited evaluation capacity was spent only on candidates that had already passed.

## 3.6.3 Work Breakdown and Schedule

Table 3.2 divides the work into ten phases and Figure 3.1 presents them as a Gantt chart. Because the dates are taken from when each set of project artefacts was produced, the schedule reflects the work as it actually happened. Phases P0 to P2 cover background study, candidate models and the move to EmpatheticDialogues that produced the Pre-Thesis 2 report in June 2026. Phases P3 to P5 cover evaluation of the initial pipeline, external evaluation, and the design adjustments made in response, from July to September 2026. Phases P6 to P9 cover the conversational agent, robustness analyses, human evaluation and thesis writing.


| Phase | Work package | Start | End | Main outputs | Evidence in repository |
|---|---|---|---|---|---|
| P0 | Background study, dataset exploration (GoEmotions, ED) | Jan 2026 | Mar 2026 | GoEmotions EDA; first GPT-2 Medium training | `GoEmotions/`, `GPT 2 medium training/` |
| P1 | Detector and generator candidates | Mar 2026 | May 2026 | Qwen2 / Qwen2.5 / TinyLlama detectors; Qwen2.5, Phi-2, Gemma responders | `train_qwen2/`, `train_response_*` |
| P2 | Switch to EmpatheticDialogues; P2 report and presentation | Jun 2026 | Jun 2026 | GPT-2 d1 on ED; Chapter 3 draft v1-v6; poster; speech | `Chapter_3_Final_Submission_v6.docx`, `Posters/` |
| P3 | Initial pipeline evaluation | Jul 2026 | Aug 2026 | Confidence-gating ablation; zero-shot / fine-tuned baselines; interrogative bias | `evaluation_results_ablation_confidence_gating/` |
| P4 | External evaluation | Aug 2026 | Sep 2026 | AI-judge panel (3 judges); WASSA; RAG test; single-model baselines; significance tests | `llm_judge_evaluation/`, `single_model_baseline/` |
| P5 | Design adjustment | Aug 2026 | Sep 2026 | d1 mismatch fix (RoBERTa); d2 bake-off; two-stage + per-emotion prompts; attention cues | `train_roberta_ed_d1/`, `emotion_prompts/` |
| P6 | Conversational agent and deployment prep | Sep 2026 | Sep 2026 | Context memory (4 / 10 turns); Hugging Face Space (not deployed) | `conversational_agent/`, `huggingface_space/` |
| P7 | Validation and robustness | Sep 2026 | Sep 2026 | Backbone comparison; error propagation; diversity; literature comparison | `d1_model_comparison/`, `evaluation_results_diversity/` |
| P8 | Human evaluation and final judging | Sep 2026 | [[TEAM TO FILL]] | Google Form (13 items); AI-judge scores for final system (completed, two-judge panel) | `human_evaluation/`, `teammate_judge_package/` |
| P9 | Thesis writing, defence preparation | Sep 2026 | [[TEAM TO FILL - submission / defence date]] | Chapters 2-5, final report, slides | `writing/` |

*Table 3.2: Project phases, outputs and supporting evidence*



![Figure 3.1](figures/fig_project_gantt.png)

*Figure 3.1: Project timeline*


## 3.6.4 Team Roles and Responsibilities

[[TEAM TO FILL: one line per member giving name, student ID, main responsibilities and supporting responsibilities, reflecting the real division of work, for example model development and experiments; AI-judge evaluation runs and literature review; human-evaluation design, data collection and writing.]]

Because only one member's computer had a suitable GPU, tasks were divided so that work not requiring a GPU, namely AI-judge evaluation through APIs, the human study, the literature review and writing, could proceed in parallel with the GPU experiments. Step-by-step command guides were prepared so that members without a programming background could run evaluations independently.

## 3.6.5 Resource Plan

**Compute:** a single NVIDIA GeForce RTX 4060 with 8 GB of memory [38] and 16 GB of system memory, used for all training and generation. Heavy jobs were scheduled when memory was free, following two system freezes (risk R1, Section 3.7). **Evaluation services:** three free API tiers from different providers, with rate-limited work divided between two team members' accounts. **Data:** the public EmpatheticDialogues corpus [3]. **People:** as described in Section 3.6.4. Costs are analysed in Section 3.8.

## 3.6.6 Revisions to the Plan

The plan was revised whenever evidence overturned an assumption. The five main revisions were as follows.

1. **Change of dataset (June 2026).** Early work used GoEmotions [39], a dataset of Reddit comments labelled with 28 emotions. Because these are isolated comments rather than conversations, the project moved to EmpatheticDialogues, and the detector and generator work was restarted on the new data.
2. **Expanded evaluation (July-August 2026).** After the Pre-Thesis 2 panel questioned whether multiple models were needed, single-model baselines of equal and larger size, a retrieval-augmented variant and an AI-judge panel were added to the schedule before any further development.
3. **Replacement of the judge panel (August 2026).** Several planned judges became unavailable because of quota limits and model retirements. The panel was rebuilt from other providers and fixed at two judges that remained accessible for the whole evaluation, and judging was divided between two accounts to work within a daily limit of 50 requests, so that every system was scored by the same two judges.
4. **Redesign of the detector (late August 2026).** A mismatch was discovered between the text the detector was trained on and the text it received in use. The detector was retrained, and a comparison of generator models was added to the schedule, leading to the final two-stage design.
5. **Scope control (September 2026).** A repetition-penalty improvement was deferred to future work and the final system's outputs were frozen, because regenerating them would have invalidated AI-judge scores that were already being collected.

## 3.6.7 Remaining Schedule

Table 3.3 lists the milestones remaining at the time of writing.

| Milestone | Target date | Status |
|---|---|---|
| AI-judge scores for the final system returned | September 2026 | Completed |
| Human evaluation questionnaire closed | [[TEAM TO FILL]] | In progress |
| Complete draft submitted to supervisor | [[TEAM TO FILL]] | In progress |
| Final report submitted | [[TEAM TO FILL]] | Planned |
| Final defence | [[TEAM TO FILL]] | Planned |

*Table 3.3: Remaining milestones*

## References

[1] H. Rashkin, E. M. Smith, M. Li, and Y.-L. Boureau, "Towards empathetic open-domain conversation models: A new benchmark and dataset," in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), Florence, Italy, 2019, pp. 5370-5381, doi: 10.18653/v1/P19-1534.  
[3] Facebook Research, "EmpatheticDialogues," GitHub repository (CC BY-NC 4.0 licence). [Online]. Available: https://github.com/facebookresearch/EmpatheticDialogues  
[38] NVIDIA, "GeForce RTX 4060 Ti & 4060 graphics cards: Specifications." [Online]. Available: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/  
[39] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, "GoEmotions: A dataset of fine-grained emotions," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4040-4054, doi: 10.18653/v1/2020.acl-main.372.  
