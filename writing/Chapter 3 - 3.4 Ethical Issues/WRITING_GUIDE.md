# Chapter 3 - 3.4 Ethical Issues

**Rubric:** CO12, "Incorporate ethical considerations and uphold academic responsibilities during the solution design process and throughout all phases of the work." Assessed through instructor observation and phase reports.

## What this section must do

Show that ethics shaped **design decisions**, not just a closing disclaimer. For a mental-health-adjacent system, examiners look for four things: user safety (especially crisis disclosure), privacy, honest representation of capability, and responsible data and research practice. The strongest material here is concrete: an unsafe case we actually observed, licences we checked, and results we reported even when they were negative.

## Tables in this folder

| File | Use |
|---|---|
| `tables/licences.csv` | Table of data and model licences (3.4.4) |

## Checklist

- [ ] Safety: scope (support, not therapy), the observed crisis failure, and the mitigation that exists versus the one that doesn't.
- [ ] Privacy: what data left the machine during evaluation (only public ED text), and what the demo stores (nothing).
- [ ] Human evaluation ethics: consent, anonymity, content warning, voluntary participation.
- [ ] Data and licences: ED is CC BY-NC 4.0, meaning non-commercial only.
- [ ] Bias: English, US-crowdsourced data, and metric bias (the acknowledgement keywords favour negative emotions).
- [ ] Research integrity: failed runs not reported as results, negative results reported, AI-assistance disclosure per university policy.

---

## Draft

### 3.4.1 Scope and User Safety

The system is designed for supportive listening in low-risk conversations. It is not a therapeutic, diagnostic or crisis service and must not be presented as one. This boundary shapes the design: the response rules ask the model to acknowledge and validate feelings, offer a reassuring thought and invite the person to continue, rather than give clinical advice.

The most serious ethical risk is a user disclosing suicidal thoughts. The 32 emotion categories of EmpatheticDialogues contain no label for suicidal ideation or self-harm, so the detector cannot represent such a message. In interactive testing, the input "I want to die" was classified as *afraid* (45%), *terrified* (28%) and *sad* (4%), and the generated reply was warm and invited the user to keep talking, but it offered no crisis resource. For comparison, general-purpose commercial assistants tested with the same message responded mainly with helpline information. We regard neither response alone as adequate: guidance on safe messaging favours acknowledging the person *and* connecting them to support [[CITE: a safe-messaging guideline, e.g. WHO / national suicide-prevention guidance - team to select and verify]].

Mitigation in the current prototype is limited to the interface: the prepared demo page states that it is a student research prototype, is not monitored, is not a substitute for professional help, and lists crisis services ([[TEAM TO FILL: verify the current Kaan Pete Roi number on its official website - two different numbers have appeared during the project]]; findahelpline.com). A pre-generation crisis filter that routes self-harm language to a fixed, reviewed response was identified but **not implemented**, and is listed as essential future work before any public deployment. The decoupled architecture makes such a filter straightforward, because detection happens in a separate, inspectable step before any text is generated.

### 3.4.2 Privacy and Data Protection

All experiments use the public EmpatheticDialogues corpus; no personal data was collected to build or tune the models. During evaluation, only ED conversation text and the corresponding generated replies were sent to third-party AI-judge APIs (OpenRouter, Mistral, Groq). No user data was involved. Both models run on local hardware, so a deployed instance does not need to send a user's disclosure to an external provider. This is one practical motivation for a small, self-hostable pipeline. The prepared demonstration interface (not yet publicly deployed) holds conversation history only in per-session memory on the server and writes nothing to disk or logs; closing the page discards it.

### 3.4.3 Human Evaluation

The human evaluation (Chapter 5) asks volunteers, mostly students and some faculty members, to compare replies to 13 short situations. Participation is voluntary and anonymous. The form collects no names or email addresses, states its purpose and expected duration (about 12-17 minutes), warns that some situations describe sad or stressful experiences, and lets participants stop at any time. System identities are hidden and reply order is counterbalanced, so participants cannot be steered toward the authors' system. [[TEAM TO FILL: state the department's position on ethics review for anonymous, minimal-risk surveys - the team confirmed formal approval is not required; cite the policy or supervisor confirmation.]]

### 3.4.4 Data and Model Licensing

Table 3.x lists the licences of every dataset and model used. EmpatheticDialogues is released under **CC BY-NC 4.0**, which requires attribution and forbids commercial use. The trained detector is a derivative of this data, so it may be shared for research but not built into a commercial product without a separately licensed dataset. The final generator, Phi-4-mini-instruct, and the RoBERTa detector backbone are MIT-licensed. The Qwen2.5 baselines are Apache-2.0, and Gemma-3 is covered by Google's Gemma Terms of Use. The WASSA empathy model does not state a licence on its model card and is used for evaluation only, with credit to the shared task.

[Table: licences.csv]

### 3.4.5 Bias and Fairness

EmpatheticDialogues is English-only and was written by crowdworkers on Amazon Mechanical Turk. Its situations and ways of expressing emotion may not reflect users in Bangladesh or speakers of other languages, so detection accuracy and reply appropriateness cannot be assumed to transfer. The evaluation itself carries a known bias: the rule-based acknowledgement check looks for phrases such as "I'm sorry" and "that sounds", which suit negative emotions and under-credit valid responses to positive ones. We therefore never rely on this metric alone and use three blind AI judges from different companies, an independent empathy classifier and a human study. The judges come from providers different from the generator's (Phi-4-mini, Microsoft) and the baselines' (Qwen, Alibaba), which avoids a judge scoring its own model family.

### 3.4.6 Research Integrity

Failed experiments were treated as failures, not results. For example, a roberta-large run that diverged to 5.0% accuracy because of missing learning-rate warmup was re-run correctly and the broken run was not reported. Negative findings are reported in full: retrieval-augmented generation, the monolithic fine-tuned generator, the Gemma-3 candidate, and the result that the final system's rule-adherence advantage over a same-size single model is not significant after multiple-comparison correction. All reported numbers are regenerated from saved experiment outputs by scripts kept in the project repository. [[TEAM TO FILL: disclose any use of AI writing or coding assistants as required by BRAC University policy.]]

---

## Traps

- **Do not say the system is "safe for mental-health use".** It has a known, documented crisis gap.
- **Do not print a helpline number without checking the official source on the day you finalise the report.**
- The screenshot comparison with commercial assistants is a single anecdote. Use it to illustrate the gap, never as evidence that our system is "more empathetic than ChatGPT/Gemini".
