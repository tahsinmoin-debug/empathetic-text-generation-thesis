# 1.1 Background

People increasingly bring their worries to a screen. Text-based support, whether in a counselling service, a student helpline or an everyday chat application, depends on one thing before anything else: the person on the other side must feel understood. In human support work this is called empathy, and studies of online support describe it as a set of concrete behaviours rather than a single quality. The EPITOME framework, built from thousands of real support conversations, identifies three mechanisms: reacting emotionally to what the person said, exploring their situation, and interpreting or naming the feeling behind it {cite:sharma2020epitome}.

Teaching a machine to do this has been studied for about as long as neural dialogue systems have existed. The field's main resource is EmpatheticDialogues, a corpus of 24,850 crowd-sourced conversations in which one participant describes a situation tied to one of 32 emotions and the other replies as a listener {cite:rashkin2019}. Alongside it, research on emotion recognition in conversation has produced models that label the feeling expressed in a message {cite:poria2019erc}, and the transformer architecture {cite:vaswani2017} with large-scale pre-training {cite:devlin2019,liu2019roberta} has made such classifiers both accurate and cheap to fine-tune.

Two lines of work have grown from this. The first builds a single neural model that reads the conversation and writes the reply, learning the emotion implicitly inside its own layers; MoEL, MIME, EmpDG, KEMP, CEM and CASE all follow this pattern {cite:lin2019moel,majumder2020mime,li2020empdg,li2022kemp,sabour2022cem,zhou2023case}. The second, more recent, prompts a large language model to respond supportively {cite:qian2023llm,wang2025esc}. Large language models write fluent and pleasant replies, but reviews of their use in health settings report that they repeat the same supportive formulas and offer little evidence of understanding the individual {cite:sorin2024jmir}.

# 1.2 Rationale of the Study or Motivation

Three practical problems motivated this thesis.

**The emotional decision is invisible.** When one model reads the message and writes the reply, nothing in the system states which emotion it believed the person was feeling. If the reply strikes the wrong tone, there is no intermediate decision to inspect, log or correct. For any application that a supervisor, a clinician or a university would have to stand behind, this is a serious limitation, and it cannot be solved by making the model larger.

**Good models are expensive.** The strongest conversational models are served from data centres, which puts them beyond a student project, a small clinic or a campus counselling service, and means that a person's disclosure must leave the device. Our own measurements show the size of the gap: the 7.6-billion-parameter model used as a reference in this thesis needs 5.47 GB of GPU memory, whereas the proposed system needs 3.40 GB and runs on a single 8 GB consumer graphics card.

**Fluency is not the same as understanding.** In our evaluation, the same generator prompted as a single model reused stock supportive phrases in 72.3% of its replies and named the user's actual emotion in only 27.0% of them. The replies read well; they simply were not specific to the person.

These three problems share a cause: the emotion is never treated as a separate decision. This thesis asks whether making it separate, and small enough to run locally, helps.

# 1.3 Problem Statement

A supportive reply has to do two different jobs: work out what the person is feeling, and write something that responds to that feeling. Current systems merge both jobs into one model. Merging them makes the emotional decision unobservable and ties the quality of the emotional reading to the size of the generator, which is exactly the resource that a low-cost, private, on-device deployment cannot afford.

This thesis therefore addresses the following question:

> Can a small emotion detector, combined with a small instruction-following generator that is told which emotion to respond to, produce replies that are more empathetic and more faithful to the user's emotion than a single model of comparable size, while running on one 8 GB consumer GPU and keeping the emotional decision visible?

The question is answered with measurements rather than impressions, and the trade-offs of the design are reported alongside its advantages.

# 1.4 Objectives

**General objective.** To design, build and evaluate a decoupled empathetic response system in which emotion detection and response generation are performed by separate, individually inspectable components that together fit within a consumer hardware budget.

**Specific objectives.**

1. To train an emotion detector on the 32 emotion categories of EmpatheticDialogues that reads the user's own message, and to select its backbone by controlled comparison rather than by assumption.
2. To design a way of passing the detector's decision to the generator that improves the reply, and to compare the alternatives (single emotion, top-3 candidates, emotion-specific guidance, attention cues) against each other.
3. To keep the complete system within 8 GB of GPU memory so that it runs on hardware the team already owns and no user message has to leave the device.
4. To evaluate the system against single-model baselines of both comparable and larger size, using several independent measures rather than one, and to test every reported difference with paired statistics.
5. To establish whether any measured advantage comes from the design or merely from the size of the generator, by running the same generator alone as a baseline.
6. To demonstrate the practical value of the decoupled design by extending it into a multi-turn conversational agent that displays the emotional decision behind every reply.

# 1.5 Methodology in Brief

The system has two components. The first, referred to as d1, is a RoBERTa-base classifier fine-tuned on the 32 emotions of EmpatheticDialogues; it reads the user's message and returns the three most probable emotions together with the words that most influenced its decision, taken from its own attention. The second, referred to as d2, is Phi-4-mini-instruct loaded in 4-bit form; it first chooses one of the three candidate emotions and then writes the reply, guided by wording written for that specific emotion and by three response rules: acknowledge the feeling, ask exactly one question, and stay within 120 words.

Evaluation uses four automatic signals and one human signal: automatic checks of the three rules; the WASSA 2024 shared-task empathy classifier {cite:giorgi2024wassa,yang2024hyy33}; emotion validation, which checks whether the emotion expressed in the reply matches the true label; a blind panel of two AI judges from unrelated developers {cite:zheng2023judge,verga2024}; and a forced-choice study with human participants. All systems answer the same 300 conversations, and differences are tested with paired methods, corrected for multiple comparisons.

# 1.6 Scopes and Challenges

**What this study covers.** Single-turn English replies to a person describing an emotional situation; 32 emotion categories; models small enough for one 8 GB consumer GPU; a comparison against single-model baselines of 1.5, 3.8 and 7.6 billion parameters; and a working multi-turn agent built from the same components.

**What this study does not cover.** It is not a mental-health service and has no crisis-detection safeguard; the emotion label set contains no category for suicidal ideation. It does not cover languages other than English, spoken interaction, long-term memory across sessions, or clinical outcomes of any kind.

**Challenges.** Four difficulties shaped the work. The dataset gives one emotion label per conversation, although a message can carry several feelings at once. The text the detector was originally trained on was not the text it receives in use, a mismatch that cost 8.7 points of accuracy before it was found and corrected. Automatic measures of empathy are proxies: the empathy classifier and the AI judges both give higher scores to longer replies, which has to be controlled for rather than ignored. Finally, the free AI-judge services used for evaluation were withdrawn or rate-limited during the project, which constrained the size of the judged sample.

# 1.7 Thesis Structure

Chapter 2 reviews existing work on empathetic dialogue, emotion recognition and the evaluation of empathy, and summarises what is still missing. Chapter 3 states the final specifications and requirements, the societal and environmental impact, the ethical issues, the standards followed, the project management plan, the risks and the economic analysis. Chapter 4 describes the design process and specifies the selected design, including every parameter and the reason for it. Chapter 5 reports the results: performance of the final system and all baselines, the analysis behind each design choice, the final adjustments, the statistical tests, the comparisons with published work, and the discussion. Chapter 6 concludes and states what remains to be done.
