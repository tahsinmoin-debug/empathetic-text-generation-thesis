# Chapter 3 - 3.6 Project Management Plan

**Rubric:** CO10, "Demonstrate resource management skill in various stages of developing the solution." Final Thesis Report. Criteria: obtain specifications and requirements; choose the analysis and design method; identify resources; identify risks (covered in 3.7); prepare a **complete project management plan including schedule, budget, resource and risk management**; **prepare a revised plan and reschedule as required**.

## What this section must do

Present the plan the project *actually followed*, not a tidy plan invented afterwards, and show where it was revised and why. The rubric explicitly rewards rescheduling, and this project has several well-documented revisions with evidence.

## Tables and figures in this folder

| File | Use |
|---|---|
| `tables/project_timeline_from_repository.csv` | Phase table: work package, dates, outputs, and evidence folder |
| `figures/fig_project_gantt.png` | Gantt chart of the same phases |

The dates come from **file modification dates in the project folder**, meaning when each artefact was actually produced. Future end dates are marked `[[TEAM TO FILL]]`.

## Checklist

- [ ] 3.6.1 Requirements and specifications (functional, non-functional, constraints)
- [ ] 3.6.2 Development methodology (iterative, evidence-gated)
- [ ] 3.6.3 Work breakdown and schedule (table + Gantt)
- [ ] 3.6.4 Team roles and responsibilities `[[TEAM TO FILL]]`
- [ ] 3.6.5 Resource plan (hardware, compute, APIs, people). Cross-reference 3.8 for cost.
- [ ] 3.6.6 Revisions to the plan (at least four, each with trigger, change and impact)
- [ ] 3.6.7 Remaining schedule to submission and defence `[[TEAM TO FILL]]`

---

## Draft

### 3.6.1 Requirements and Specifications

The project requirements were refined from the proposal, supervisor feedback and the Pre-Thesis 2 panel remarks into the following specification.

**Functional requirements.** (F1) Given a user's message, identify the most likely emotion among the 32 EmpatheticDialogues categories and expose that decision for inspection. (F2) Generate a reply that acknowledges the feeling, offers a reassuring thought and asks at most one follow-up question. (F3) Keep replies short (at most about 100 words in the prompt, scored at 120). (F4) Support a multi-turn conversation that tracks emotional change.

**Non-functional requirements.** (N1) Both models run on a single consumer GPU with 8 GB of memory. (N2) No user message needs to be sent to a third-party service at inference time. (N3) Every reported result is reproducible from saved outputs with fixed seeds.

**Constraints.** A fixed academic calendar, one GPU-capable machine, a three-member team of whom one member runs GPU experiments, and free-tier API limits for evaluation.

### 3.6.2 Development Methodology

We followed an iterative, evidence-gated process (Figure 4.1). Each design alternative was implemented, simulated on the same 300 held-out conversations, scored with the same metrics, and adopted only if the evidence supported it. Cheap, local checks (rule adherence, emotion validation) served as a gate before expensive or rate-limited evaluation (AI judges, human study), so rate-limited resources were spent only on candidates that had already passed.

### 3.6.3 Work Breakdown and Schedule

Table 3.x breaks the work into ten phases, and Figure 3.x shows them as a Gantt chart. Phases P0-P2 cover background study, candidate models and the switch to EmpatheticDialogues that produced the Pre-Thesis 2 report in June 2026. P3-P5 cover evaluation of the initial pipeline, external evaluation and the design adjustments made in response (July-September 2026). P6-P9 cover the conversational agent, robustness checks, human evaluation and writing.

[Table: project_timeline_from_repository.csv]
[Figure: fig_project_gantt.png]

### 3.6.4 Team Roles and Responsibilities

[[TEAM TO FILL - one row per member: name, ID, primary responsibilities, secondary responsibilities. Reflect the real division of work, for example: model development and experiments; AI-judge evaluation runs (Nemotron short/long split, gpt-oss) and literature review; human evaluation form, data collection and writing.]]

Because only one member's machine has a suitable GPU, tasks were divided so that no-GPU work (AI-judge evaluation through APIs, the human study, literature review and writing) ran in parallel with GPU experiments. Command-by-command guides were prepared so that non-coding members could run evaluations independently.

### 3.6.5 Resource Plan

**Compute:** one NVIDIA RTX 4060 (8 GB) with 16 GB system RAM, used for all training and generation. Heavy jobs were scheduled when memory was free, after two system freezes (Risk R1, Section 3.7). **Evaluation services:** three free-tier APIs from different providers, with rate-limited work split across two team members' accounts. **Data:** the public EmpatheticDialogues corpus. **People:** see 3.6.4. Costs are analysed in Section 3.8.

### 3.6.6 Revisions to the Plan

The plan was revised whenever evidence invalidated an assumption. The main revisions were:

1. **Dataset change (June 2026).** Early work used GoEmotions (Reddit comments, 28 labels). Because these are isolated comments rather than conversations, the project moved to EmpatheticDialogues, and the detector and generator schedules were restarted on the new data.
2. **Evaluation expansion (July-August 2026).** After the Pre-Thesis 2 panel questioned whether multiple models were needed, single-model baselines of the same size and larger (Qwen2.5-1.5B and Qwen2.5-7B), a RAG variant and an AI-judge panel were added to the schedule before any further development.
3. **Judge panel replacement (August 2026).** Planned judges became unavailable (Gemini quota limits, retirement of DeepSeek on GitHub Models and of Llama 3.3 on Groq). The panel was rebuilt as Nemotron 3 Ultra and gpt-oss-120b, and judging was split across two accounts to work around a 50-request daily limit.
4. **Detector redesign (late August 2026).** A train/inference mismatch was found in the detector: it was trained on situation descriptions but fed user messages. The detector was retrained and a d2 bake-off inserted into the schedule, which led to the two-stage final design.
5. **Scope control (September 2026).** A repetition-penalty improvement was deferred to future work, and the final system's outputs were frozen, because regenerating them would have invalidated AI-judge scores already being collected.

### 3.6.7 Remaining Schedule

| Milestone | Target date | Status |
|---|---|---|
| AI-judge scores for final system returned | [[TEAM TO FILL]] | In progress |
| Human evaluation form closed | [[TEAM TO FILL]] | In progress |
| Complete draft to supervisor | [[TEAM TO FILL]] | In progress |
| Final report submission | [[TEAM TO FILL]] | Planned |
| Final defence | [[TEAM TO FILL]] | Planned |

---

## Traps

- Don't present GoEmotions results as part of the final system. They belong only in revision 1.
- Revisions are a strength under this rubric. Don't hide them, and don't dramatise them either.
- Budget belongs in 3.8 and risks in 3.7. Here, cross-reference them in one line each.
