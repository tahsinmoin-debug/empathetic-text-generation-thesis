# 3.7 Risk Management

This section describes how project risks were identified, assessed and mitigated. Most of the risks listed were encountered during the project, and each has a documented mitigation.

## 3.7.1 Risk Management Process

Risks were identified at each phase review and whenever an experiment behaved unexpectedly. Each risk was rated for **likelihood** (Low, Medium or High), meaning how probable it was within the project period, and for **impact** (Low, Medium, High or Very High), meaning its effect on the validity of results, the schedule or user safety. Risks with high impact were mitigated immediately, while the others were monitored and addressed if they occurred. Wherever possible, mitigations were made *structural*, such as an automatic check in the code, a fixed random seed or a pinned library version, so that the same failure could not recur unnoticed.

## 3.7.2 Risk Register

Table 3.4 summarises the twelve risks identified, grouped into seven categories: technical, methodological, external, schedule, evaluation, ethical and team. Eight of these risks occurred during the project, two remain active, and two were identified in advance and mitigated or partly mitigated.


| ID | Category | Risk | Likelihood | Impact | Status | Mitigation |
|---|---|---|---|---|---|---|
| R1 | Technical | GPU out-of-memory / system freeze on 8 GB VRAM, 16 GB RAM | High | High | Occurred twice | 4-bit NF4 quantisation; pre-flight VRAM/RAM checks; heavy jobs run only when RAM is free |
| R2 | Technical | Library upgrade breaks code (transformers 5.x removed StableDropout, warmup_ratio) | Medium | High | Occurred | Compatibility shim; pinned versions; WASSA re-validated at 100% match on 1,800 pre-upgrade scores |
| R3 | Technical | Silent failure - default 'sdpa' attention returned no attention weights | Low | High | Occurred | Forced 'eager' attention + hard-fail assertion so the feature cannot silently switch off |
| R4 | Methodological | Train/inference mismatch in d1 (trained on situation text, fed user messages) | Medium | High | Occurred | Retrained on first user message; honest accuracy reported |
| R5 | Methodological | Unstable training of larger model (roberta-large diverged to 5.0%) | Medium | Medium | Occurred | Warmup 0.06 + lr 1e-5; failed run not reported as a result |
| R6 | Methodological | Non-reproducible generation (no fixed seed) | Medium | Medium | Occurred | Fixed seed 42 for sampling and data; earlier noisy numbers discarded |
| R7 | External | Free API models retired or withdrawn mid-project (Llama 3.3 on Groq, DeepSeek on GitHub Models, Gemini quota, Mistral Large 403 on both team accounts) | High | Medium | Occurred | Judge panel from several providers; final comparison on the two judges that scored every system; two- vs three-judge panel checked (rho 0.98, same ranking) |
| R8 | External | API rate limits (OpenRouter 50 requests/day) | High | Medium | Occurred | Stratified short/long split across two teammates' accounts with dedup merge |
| R9 | Schedule | Late design changes invalidate in-progress judging | Medium | High | Active | Freeze the final system outputs while teammates judge; repetition penalty moved to future work |
| R10 | Evaluation | Automatic metrics reward length or keywords, not empathy | High | Medium | Identified | Multiple independent signals: rules, WASSA, AI-judge panel, human study; length-matched analysis |
| R11 | Ethical | Unsafe reply to crisis disclosure (no self-harm category in ED labels) | Low | Very high | Identified | Disclaimer and helplines in demo UI; stated as limitation; crisis gate listed as future work |
| R12 | Team | Only one member can run GPU work | High | Medium | Active | No-GPU tasks (judging, writing, human eval) assigned to teammates with written command guides |

*Table 3.4: Risk register*


## 3.7.3 Key Risks and Their Handling

**Silent technical failure (R3).** The design uses the detector's attention weights to identify the words that most influenced its prediction. With the default attention implementation of the Transformers library [40], requesting attention weights returns an empty result without raising an error, so cue extraction silently fell back to simple text splitting while appearing to work. Had this gone unnoticed, the thesis would have described an attention mechanism that was not operating. The implementation was switched to one that returns attention weights, and an automatic check was added that stops execution if the weights are missing. This is the clearest example in the project of a structural mitigation.

**Validity of results (R4, R5).** Two risks threatened the correctness of reported figures rather than the schedule. First, the original detector was trained on situation descriptions but was evaluated and deployed on user messages. When evaluated on the real input, its accuracy fell from 56.9% to 48.2%, and it was retrained on the correct input. Second, a RoBERTa-large training run diverged to 5.0% accuracy because learning-rate warmup was missing. Rather than report this as a result, the cause was identified and the run repeated with warmup over 6% of training steps and a lower learning rate, reaching 53.0% accuracy.

**Dependence on external services (R7, R8).** Evaluation with AI judges relied on free APIs outside the team's control. During the project, several candidate judge models became unavailable through quota limits, retirement or subscription restrictions, including Mistral Large, which could no longer be accessed from either team account before the final systems were judged, and one provider limited its judge to 50 requests per day. These risks were mitigated by drawing the judges from independent providers, comparing every system on the judges that scored all of them, verifying that the reduced two-judge panel ranks the systems exactly as the three-judge panel did, and dividing rate-limited judging between two team members using a split that keeps every emotion category represented in each half.

## 3.7.4 Residual Risks

Two risks remain open at the time of writing. **R9 (schedule):** a late change to the design would invalidate the collected AI-judge scores and the human-evaluation items built from the final system's replies, so these outputs are frozen and the repetition-penalty improvement has been deferred. **R11 (ethical):** apart from interface disclaimers, the system has no safeguard for disclosures of suicidal thoughts. This risk has low likelihood in the evaluation setting but very high impact in any public deployment, and it must be resolved before such deployment (Section 3.4).

## References

[40] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., "Transformers: State-of-the-art natural language processing," in Proc. 2020 Conf. Empirical Methods Natural Language Processing: System Demonstrations, 2020, pp. 38-45, doi: 10.18653/v1/2020.emnlp-demos.6.  
