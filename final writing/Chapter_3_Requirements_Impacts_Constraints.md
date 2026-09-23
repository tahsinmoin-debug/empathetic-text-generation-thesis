# Chapter 3: Requirements, Impacts and Constraints

A system that replies to people describing painful experiences is judged by more than its accuracy. It must work within the hardware a student team owns, respect the licence of the data it learns from, protect what a person types, and never present itself as help it is not qualified to give. This chapter sets out the requirements built to meet those demands, the effect the system could have on people and on the environment, the ethical limits placed on it, the standards it follows, and how the project was planned, its risks handled and its costs contained.

## 3.1 Final Specifications and Requirements

This work had to satisfy technical and methodological criteria at once. Technically, two models had to run together on a single consumer graphics card and answer within a few seconds: a RoBERTa-base classifier that reads the user's message and returns the three most likely emotions together with the words that drove the decision, and a 3.8-billion-parameter instruction model, quantised to 4 bits, that chooses one of those emotions and writes the reply. Methodologically, every design choice had to be settled by measurement on the same held-out conversations rather than by preference, and every number in this thesis had to be reproducible from saved outputs. The requirements below were fixed before the final round of development and did not change afterwards; each is stated so that it can be checked by measurement rather than opinion, and Table A.2, in the appendix, records how every one was tested and what the test produced.

### 3.1.1 Functional Requirements

- **F1 - Emotion detection from the user's own words:** the emotion is read from the message the user actually writes, among the 32 categories of EmpatheticDialogues [2].
- **F2 - Candidate emotions, not a single guess:** the detector returns several likely emotions with a confidence each.
- **F3 - A visible emotional decision:** every reply carries the candidates, the emotion used and the words behind it.
- **F4 - Guided generation:** the generator receives wording prepared for the selected emotion.
- **F5 - Response rules:** every reply acknowledges the feeling, asks exactly one question and stays within 120 words.
- **F6 - Multi-turn conversation:** the components hold a conversation over several turns and notice when the emotion changes.

### 3.1.2 Non-Functional Requirements

- **N1 - Hardware budget:** the whole system runs on one consumer GPU with 8 GB of memory [34].
- **N2 - Privacy by locality:** no user message has to leave the device.
- **N3 - Response time:** a reply arrives within a few seconds on that hardware.
- **N4 - Reproducibility:** every number can be regenerated from saved outputs, with fixed seeds.
- **N5 - Inspectability:** each component can be replaced and tested alone.
- **N6 - Cost:** only open models, datasets and free service tiers are used.

### 3.1.3 Hardware and Software Requirements

**Hardware specifications:** all training, generation and benchmarking ran on one machine with an NVIDIA GeForce RTX 4060 holding 8 GB of video memory, 16 GB of system memory and a solid-state disk for the model cache, under Windows 11. That card is a requirement rather than a limitation, because the design target was a system a counselling service could host on an ordinary computer. It sets the ceiling the design works under, since the detector, the quantised generator and their activations must fit in 8 GB together, and it is the hardware on which the 3.40 GB peak memory and 4.17 seconds per reply of Chapter 5 were measured.

**Software requirements:** the system was built in Python 3.10 with PyTorch 2.5.1 on CUDA 12.1 [35] and Hugging Face Transformers 5.16.1 [36], which supply the pre-trained encoder and the instruction-tuned generator. Fine-tuning used PEFT for low-rank adaptation and bitsandbytes for 4-bit NF4 quantisation, without which the generator would not fit in 8 GB. Evaluation used SciPy [37] and scikit-learn [38] for the paired tests and classification metrics, sentence-transformers for the retrieval baseline, the WASSA 2024 empathy classifier [17], and the OpenRouter and Groq APIs for the judges. The demonstration uses Gradio and the human study Google Forms, and exact library versions are listed in the appendix.

### 3.1.4 Data Requirements

The work needed conversational data labelled with fine-grained emotions, which EmpatheticDialogues supplies: 23,144 conversations and 88,687 utterances, each conversation carrying one of 32 emotion labels, split by its authors into 17,839 conversations for training, 2,763 for validation and 2,542 for testing [2]. The detector is trained on the first message of each conversation, the text it receives in use, giving 17,780 training pairs. Reply quality is measured on 300 held-out conversations answered by every system, of which the AI judges score a stratified sample of 64. The corpus is released under a non-commercial research licence [20].

### 3.1.5 Design Constraints

Several constraints shaped the design rather than merely limiting it. The 8 GB memory budget ruled out an unquantised generator and any model much above 8 billion parameters, which is why the generator is quantised and the detector deliberately small. The dataset gives one emotion label per conversation although a message often carries several feelings at once, so detection accuracy is measured against a reference that is itself imperfect. The free evaluation services impose daily request limits, which capped the AI-judge sample at 64 conversations per system, and access to volunteers limited the human study to a short questionnaire. The project also had one GPU shared between three members and a fixed academic deadline, so experiments were sequenced by cost: cheap local measures first, and rate-limited evaluation only for designs that had already earned it.


## 3.2 Societal Impact

### 3.2.1 Potential Benefits

The system is built for the moment when someone wants to be heard and no trained listener is free. Six benefits follow from the way it is designed.

- **Privacy for people who will not talk to a cloud service:** both models run on the user's own machine, so a disclosure about grief, loneliness or fear never travels to a company's server, which for many people is the difference between writing the message and not writing it at all.
- **A listener available at every hour:** distress keeps no office hours, and a locally hosted system answers at three in the morning, on a holiday, or after the counselling centre has closed.
- **A first step towards professional help:** it is not a therapist and must not replace one, but it can hold the first conversation and make contacting a counsellor feel like a smaller step.
- **Affordable for the institutions that need it most:** a counselling service, a small clinic or a helpline can host it on one ordinary computer with no per-message fee. In Bangladesh, where mental-health professionals are few and private counselling is beyond most students, the price of entry matters as much as the quality of the reply.
- **A way around stigma:** many people will type a worry they would never say aloud to a relative or a doctor.
- **Replies that can be checked before they are shown:** the emotion behind every turn is written down, so a supervisor can review it and a safeguard can stop a reply before the user sees it. A single end-to-end model offers nothing to inspect.

### 3.2.2 Risks and Limits of Use

A system that sounds understanding can be trusted further than it deserves, so three risks are stated plainly.

- **Substitution for real help:** a person in distress may keep talking to a machine instead of contacting a professional. The emotion categories contain nothing for suicidal thinking and there is no crisis filter, so the system must never be presented as a mental-health service; any deployment needs a reviewed filter, a referral path and a visible helpline number [39].
- **Wrong or shallow empathy:** the detector's first choice is correct 50.3% of the time and the reply names the user's true emotion in 36.7% of cases, and a reply built on the wrong emotion can feel dismissive. Passing three candidates instead of one reduces that cost, but for one person on one occasion the risk is real.
- **Cultural and linguistic fit:** EmpatheticDialogues was written by English-speaking crowdworkers [2], and how feelings are expressed differs across languages, so use in Bangladesh would require data and evaluation in Bangla first.

In short, the design gives an institution what a large hosted model cannot, namely privacy, affordability and an auditable emotional decision, and pays for it with replies the AI judges rate no higher than a single model's.

## 3.3 Environmental Impact

The environmental cost of this work is the electricity used to train and evaluate the models, and it is small because every model involved is small. All training and local inference ran on one RTX 4060 with a rated board power of 115 W [34]. Taking rated power times measured GPU hours, which overstates the true draw, training the final detector used about 0.017 kWh and the RoBERTa-large alternative about 0.124 kWh. The heaviest items were generation for fourteen configurations over 300 conversations each and the repeated fine-tuning runs, a few kilowatt-hours in total, comparable to running a desktop for a few days. No cloud GPU was rented and nothing was trained from scratch.

The design also has a direct environmental consequence in use: within statistical error the system matches the AI-judge scores of a single model with 1.9 times its parameters while using 38% less GPU memory, so a service that would otherwise need a data-centre GPU can run on a mid-range consumer card, and because everything runs locally no message travels to a data centre and back.

The project also bought no new hardware, and the 8 GB limit was set precisely so that none would be needed.

## 3.4 Ethical Issues

Responding to someone who is upset carries obligations that an accuracy score does not capture. Seven of them shaped this project.

### 3.4.1 Scope and User Safety

The system is meant for supportive listening in low-risk conversations, and is neither a therapeutic nor a diagnostic nor a crisis service. The gravest risk is a user disclosing suicidal thoughts, and the 32 emotion categories of EmpatheticDialogues [2] contain no label for suicidal ideation, so the detector cannot represent such a message at all. In testing, "I want to die" was read as *afraid* (45%) and *terrified* (28%): the reply was warm but named no crisis service, while commercial assistants answered mainly with helpline information. Neither is adequate alone, since World Health Organization guidance asks for both understanding and accurate information on where to find help [39]. Mitigation here is limited to the interface, which states that the system is an unmonitored student project and lists crisis services ([[TEAM TO FILL: current Kaan Pete Roi number, verified on its official website]]; findahelpline.com). A pre-generation crisis filter is essential before any deployment, and the decoupled design makes it easy to add, because the emotion is decided before any text is written.

### 3.4.2 Clinical Responsibility

The system produces text that reads like emotional support, yet nobody clinically qualified is answering. Three commitments follow: it is never described to a user as a counsellor, therapist or helpline; responsibility for what it says rests with the developers and any institution hosting it, which is why the emotional decision behind each reply is recorded and can be audited; and before any deployment it must be reviewed by a qualified professional, run with a crisis filter and a referral path, and operated under a named person answerable for it.

### 3.4.3 Privacy and Data Protection

All models were developed and evaluated on the public EmpatheticDialogues corpus, and no personal data was collected; only dataset text and the replies generated from it were sent to the judge services. Both models run locally, so a deployed instance need not send a disclosure to an external provider, and the demonstration keeps conversation history in memory only, writing nothing to disk.

### 3.4.4 Human Evaluation

The human evaluation in Chapter 5 asks volunteers, mostly students with some faculty members, to rate replies to 8 short situations. Participation is voluntary and anonymous: no names or email addresses are collected, the purpose and the expected 15 minutes are stated up front, participants are warned that some situations describe stressful experiences, and anyone may stop at any point. The system behind each reply is hidden and reply order counterbalanced, so nobody can be steered toward the authors' system. [[TEAM TO FILL: state the department's policy or the supervisor's confirmation that formal ethics review is not required for this anonymous, minimal-risk survey.]]

### 3.4.5 Data and Model Licensing

EmpatheticDialogues is released under CC BY-NC 4.0 [20], which requires attribution and forbids commercial use, so the detector trained on it may be shared for research but not built into a commercial product. Phi-4-mini-instruct and the RoBERTa backbone are MIT-licensed, the Qwen2.5 baselines Apache-2.0, and the Gemma-3 candidate carries Google's Gemma Terms of Use [40]. The WASSA empathy model states no licence and is used for evaluation only, with credit to its authors; Table A.3 lists every licence and what it permits.


### 3.4.6 Bias and Fairness

EmpatheticDialogues is an English-only corpus written by crowdworkers, so neither detection accuracy nor the suitability of replies can be assumed to carry over to users in Bangladesh. The evaluation carries its own bias: the acknowledgement check looks for phrases such as "I'm sorry", which fit negative emotions and under-credit good responses to positive ones, so it is never used alone but alongside the AI-judge panel, the empathy classifier and the human study.

### 3.4.7 Research Integrity

Failed experiments were treated as failures rather than results, and negative findings are reported in full: retrieval augmentation, the fine-tuned generator, the Gemma-3 candidate, the larger detector, and the finding that our advantage in rule adherence over a same-size single model does not survive correction. When a configuration error was found in the fine-tuned baseline, it was retrained, every affected result replaced, and the error documented rather than hidden. Every figure in this thesis is regenerated from saved outputs by scripts in the project repository. [[TEAM TO FILL: disclosure of any use of AI writing or coding assistants, as required by BRAC University policy.]]

## 3.5 Standards

No engineering standard governs an empathetic dialogue system, so the project followed the published practice that applies to its parts. The data and every model are used under their own licences (Table A.3), with EmpatheticDialogues restricted to non-commercial research. The response rules follow established guidance on supportive conversation: acknowledging the feeling implements the emotional-reaction mechanism of the EPITOME framework [1], and the single-question rule follows counselling guidance against the question-answer trap [41]. Empathy is measured with the classifier from the WASSA 2024 shared task [16], and the AI-judge evaluation follows accepted practice: blind scoring, an anchored rubric, temperature 0 and judges from unrelated developers [19]. Statistical claims use paired tests with correction for multiple comparisons (Section 5.4.1). World Health Organization guidance sets the one standard the system does not meet, since it has no crisis safeguard and must not be deployed as a mental-health service [39]. The work also follows BRAC University's academic conduct requirements: sources cited, results from the team's own runs, errors reported rather than removed.

## 3.6 Project Management Plan

The project was carried out by a three-member team over nine months, with one GPU-equipped machine shared between three people and a fixed academic deadline. Work was organised into overlapping phases rather than a strict sequence, so that evaluation, writing and experimentation could run at once and a result that overturned an assumption could be absorbed without stopping everything else.

### 3.6.1 Development Methodology

Development followed an iterative, evidence-gated method. Each design alternative was implemented, run on the same 300 held-out conversations, scored with the same measures, and kept only if the numbers supported it. Cheap local measures acted as a first gate, so the rate-limited AI judges and the human study were spent only on candidates that had already passed.

### 3.6.2 Work Breakdown and Schedule

The work divided into five phases, shown as a Gantt chart in Figure 3.1. The dates come from when each set of artefacts was produced, so the chart reflects what happened.

- **Phase 1 - Background study and candidate selection (January to May 2026):** the problem was defined, the literature reviewed, and first detectors and generators trained on GoEmotions.
- **Phase 2 - Move to EmpatheticDialogues (June 2026):** isolated comments proved the wrong material for a conversational system, so the work restarted on a conversational corpus, producing the initial detector on the new data.
- **Phase 3 - Evaluation of the initial pipeline (July to August 2026):** the first complete pipeline was measured against zero-shot, fine-tuned and single-model baselines with an AI-judge panel, an empathy classifier and paired tests.
- **Phase 4 - Design adjustment (August to September 2026):** that evidence drove the corrected detector, the generator comparison, the two-stage emotion selection, per-emotion guidance and attention cues, giving the final system.
- **Phase 5 - Validation, agent and writing (September 2026):** robustness checks, the multi-turn agent, the human evaluation and this report.


![Figure 3.1](figures/fig_project_gantt.png)

*Figure 3.1: Project timeline*


### 3.6.3 Team Roles and Resources

[[TEAM TO FILL: one line per member giving name, student ID, main responsibilities and supporting responsibilities, reflecting the real division of work, for example model development and experiments; AI-judge evaluation runs and literature review; human-evaluation design, data collection and writing.]]

Because only one member's computer had the GPU described in Section 3.1.3, everything not needing one ran in parallel: AI-judge evaluation through free APIs, the human study, the literature review and the writing, with command guides so members without a programming background could run evaluations themselves. Heavy jobs were scheduled when memory was free, judging was split between two accounts to stay within a 50-request daily limit, and no funding was required.

### 3.6.4 Revisions to the Plan

The plan was revised whenever evidence overturned an assumption, four times in all.

1. **Change of dataset (June 2026):** GoEmotions [42] holds isolated Reddit comments rather than conversations, so the project moved to EmpatheticDialogues.
2. **Expanded evaluation (July to August 2026):** once the question arose of whether two models were needed at all, single-model baselines, a retrieval variant and an AI-judge panel were added.
3. **Redesign of the detector (late August 2026):** the detector had been trained on different text from the input it received in use, so it was retrained and a generator comparison was added, leading to the two-stage design.
4. **Scope control (September 2026):** a repetition-penalty improvement was deferred and the final outputs frozen, because regenerating them would have invalidated judge scores already being collected.

## 3.7 Risk Management

Risks were recorded whenever an experiment behaved unexpectedly or a phase was reviewed, and each was rated for likelihood and for impact on the results, the schedule or user safety. Wherever possible the mitigation was structural rather than procedural, meaning an automatic check, a fixed seed or a pinned version, so the same failure could not recur unnoticed. Of the twelve risks identified, eight occurred, two were prevented and two remain open. The full register is Table A.2; three entries changed how the project works.

- **Silent technical failure (R3):** the design reads the detector's attention weights to find the words that drove its prediction, but with the library's default attention implementation asking for those weights returns nothing without raising an error, so cue extraction quietly fell back to plain text splitting while appearing to work. Left unnoticed, the thesis would have described a mechanism that was not running. The implementation was changed and an automatic check now halts execution if the weights are missing.
- **Validity of results (R4, R5):** the original detector was trained on situation descriptions but used on the user's own messages, and its accuracy fell from 56.9% to 48.2% on the real input, so it was retrained. Separately, a RoBERTa-large run diverged to 5.0% through a missing warmup; rather than report that number, the cause was found and the run repeated, reaching 53.0%.
- **Dependence on external services (R7, R8):** the AI judges were free APIs outside the team's control, and several candidates were retired or restricted mid-project. The panel was fixed on the two that stayed available throughout, so every system is scored by identical judges, and rate-limited work was split between two accounts with every emotion represented in each half.

Two risks remain open. A late design change would invalidate the collected judge scores and the human-evaluation items, so those outputs are frozen and the repetition-penalty improvement is deferred. And apart from interface disclaimers the system still has no safeguard for disclosures of suicidal thinking, which must be resolved before any deployment (Section 3.4).


## 3.8 Economic Analysis

The project was designed to cost as close to nothing as possible, because the constraint a student team faces is the one a small clinic also faces. Every model is openly released, every library open source, the dataset free for research, the judges free-tier, and the human study runs on a free questionnaire service, so direct spending is limited to electricity, internet access and a GPU workstation the team already owned, [[TEAM TO FILL: purchase price in BDT, or a statement that it is an existing personal asset with no additional cost]]; Table A.5 gives the breakdown. Energy is equally small: at the card's rated 115 W, fine-tuning the final detector used about 0.017 kWh in 8.8 minutes and RoBERTa-large about 0.124 kWh in 64.7 minutes, so at [[TEAM TO FILL: BDT per kWh]] the cost of training is negligible.


### 3.8.1 Cost-Benefit of the Main Design Choices

- **Where the cheaper option also won:** RoBERTa-large is 2.9 points more accurate than RoBERTa-base but carries 2.8 times the parameters and 7.4 times the training time, and changed no measure of reply quality when substituted into the final system. Fine-tuning the generator cost 43 minutes of GPU time and satisfied all three rules in only 22.3% of replies, while prompting an instruction-tuned model needs no training and reaches 60.0%.
- **Where it did not:** a single Qwen2.5-7B model beats our pipeline on rule adherence, empathy-classifier score and judge-rated empathy, so on quality alone it would be the better choice; but on the same GPU our system needs 38% less peak memory and reaches 97.5% of that model's overall judge score.
- **Evaluation method:** human annotation in comparable work is paid, roughly USD 6 per hour in one published study, whereas the free-tier AI judges produced several hundred ratings each at no cost, and the human study checks those judges rather than replacing them.

### 3.8.2 Operating and Commercial Constraints

A self-hosted deployment has no per-message fee, whereas a commercial LLM API pays for every message, [[TEAM TO FILL: published price per million input and output tokens for a named commercial model, with the date checked]].
