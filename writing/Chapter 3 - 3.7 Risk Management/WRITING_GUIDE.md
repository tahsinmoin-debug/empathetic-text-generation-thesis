# Chapter 3 - 3.7 Risk Management

**Rubric:** CO10, "Identify project risk issues and how to address that." Final Thesis Report.

## What this section must do

Identify the risks, rate them, and show how each was handled. Most risk sections are hypothetical. Ours can be better because **most of these risks actually happened**, and each has a traceable mitigation. Present the process (identify → assess → mitigate → monitor) and then the register.

## Tables in this folder

| File | Use |
|---|---|
| `tables/risk_register.csv` | The risk register: 12 risks with category, likelihood, impact, status and mitigation |

## Checklist

- [ ] 3.7.1 Risk management process, including the likelihood × impact scale definitions
- [ ] 3.7.2 Risk register (table)
- [ ] 3.7.3 Discussion of the three most consequential risks, with evidence
- [ ] 3.7.4 Residual risks that remain open

---

## Draft

### 3.7.1 Risk Management Process

Risks were identified at each phase review and whenever an experiment behaved unexpectedly. Each risk was rated for **likelihood** (Low, Medium, High: how probable within the project period) and **impact** (Low, Medium, High, Very High: effect on the validity of results, schedule, or user safety). Risks with High impact were mitigated immediately. The others were monitored and acted on if they occurred. Mitigations were preferably *structural*, such as a code assertion, a fixed seed or a pinned version, rather than relying on care, so that the same failure could not silently recur.

### 3.7.2 Risk Register

Table 3.x summarises the twelve risks identified. They fall into seven categories: technical, methodological, external, schedule, evaluation, ethical and team. Eight of them occurred during the project, two are currently active, and two were identified in advance and mitigated or partly mitigated.

[Table: risk_register.csv]

### 3.7.3 Key Risks and Their Handling

**Silent technical failure (R3).** The design uses the detector's attention weights to extract the words that most influenced its prediction. With the default attention implementation in the Transformers library (`sdpa`), requesting attention weights returns an empty result without raising an error. Cue extraction therefore fell back to plain text splitting while appearing to work. Had this gone unnoticed, the thesis would have claimed an attention mechanism that did not operate. We switched to the `eager` implementation and added a hard-fail assertion that stops the run if attention weights are missing. We treat this as the model case for structural mitigation.

**Methodological validity (R4, R5).** Two risks threatened the correctness of reported numbers rather than the schedule. First, the original detector was trained on situation descriptions but evaluated and deployed on user messages. Evaluated on the real input, its accuracy fell from 56.9% to 48.2%. It was retrained on the correct input. Second, roberta-large diverged to 5.0% accuracy without learning-rate warmup. Instead of reporting that run as a result, we identified the cause, applied the warmup ratio of 0.06 used in the RoBERTa paper with a lower learning rate, and obtained 53.0%.

**External service dependency (R7, R8).** AI-judge evaluation depended on free APIs outside the team's control. During the project, three candidate judge models became unavailable (Gemini quota limits, DeepSeek retired on GitHub Models, Llama 3.3 decommissioned on Groq), and OpenRouter limited the Nemotron judge to 50 requests per day. We mitigated this by drawing judges from three independent providers, keeping valid scores from retired judges, and splitting rate-limited judging between two team members using a stratified short/long partition that keeps every emotion covered in each half.

### 3.7.4 Residual Risks

Two risks remain open at the time of writing. **R9 (schedule):** a late design change would invalidate AI-judge scores already being collected, so the final system's outputs are frozen and the repetition-penalty improvement is deferred. **R11 (ethical):** the system has no crisis-disclosure safeguard beyond interface disclaimers. This risk is low in likelihood for the evaluation setting but very high in impact for any public deployment, and it must be resolved before such deployment (Section 3.4).

---

## Traps

- "Occurred" rows are evidence of good management, not an admission of failure. Present them that way.
- Don't invent probabilities or percentages. Qualitative levels are enough, and they are what we actually know.
