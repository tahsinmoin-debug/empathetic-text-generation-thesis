# 1.1 Background

People increasingly bring their worries to a screen. A student who cannot sleep before a viva, someone who has just lost a pet, a person who has moved to a new city and has spoken to nobody outside work for a fortnight: each of them is more likely to type the sentence than to say it aloud to another person. The World Health Organization's most recent assessment of global mental health describes a large and widening gap between the number of people who need support and the services available to provide it {cite:who2022report}, and that gap is what has pushed text-based support, from counselling services and student helplines to everyday chat applications, into the space between having nobody to tell and seeing a professional.

Whatever such a system does next, it has to do one thing first: the person on the other side must feel understood. A reply that is factually sensible but emotionally wrong, offering a solution to someone who wanted acknowledgement, or cheerfulness to someone describing a loss, does not merely fail to help. Empathy in text-based support is not a vague quality; it has identifiable and measurable parts. The EPITOME framework, built on established empathy scales and applied to online mental-health conversations, separates a reply's *emotional reactions*, its *interpretations* of what the person means, and its *explorations* of what they have not yet said {cite:sharma2020epitome}, and counselling guidance is equally specific about what to avoid, warning that a sequence of questions turns support into an interrogation {cite:samhsa2019tip35}.

Teaching a machine to do this has been studied for about as long as neural dialogue systems have existed, and the field's main resource is EmpatheticDialogues, a corpus of 24,850 crowd-sourced conversations in which one person describes a situation tied to one of 32 emotions and a second replies as a listener {cite:rashkin2019, ed_github}. Two lines of work have grown from it. The first builds a **single neural model** that reads the conversation and writes the reply, learning the emotion implicitly inside its own layers: MoEL, MIME, EmpDG, KEMP, CEM, CASE and ESCM all follow this pattern, adding mixtures of listeners, emotion mimicry, external commonsense knowledge or graph reasoning {cite:lin2019moel, majumder2020mime, li2020empdg, li2022kemp, sabour2022cem, zhou2023case, yang2023escm}. The second, more recent line treats a **large language model** as the generator and asks how to steer it, whether by prompting {cite:qian2023llm, loh2023harnessing}, by giving it tools {cite:cao2025tooled}, by conditioning on personality {cite:wu2025traits}, or by reasoning about emotion step by step before answering {cite:li2024emotioncot}.

Recognising the emotion at all is a research problem in its own right. Emotion recognition in conversation has its own literature and its own difficulties, chiefly that a feeling is carried by context rather than by any single sentence {cite:poria2019erc}; emotions in a conversation may be implicit and have to be inferred rather than read off the words {cite:koga2023implicit}; a single message frequently carries more than one feeling at once {cite:belay2025}; and large language models, though capable, are not automatically better at this than a purpose-trained classifier {cite:feng2023affect}. Alongside this, a growing body of work applies the same machinery to clinical screening, from depression detection on social media {cite:lan2024depression} to shared tasks on mental-health timelines {cite:tseriotou2025clpsych}, while systematic review finds that large language models, for all their fluency, produce empathy that is noticeably repetitive {cite:sorin2024jmir} and that data scarcity and bias remain unsolved in this domain {cite:kumar2024unlocking}.

What almost all of this work shares is that the emotional decision is never written down. Whether the emotion is learned inside a joint model's hidden layers or inferred silently by a large language model, nobody, neither developer, supervisor, nor clinician, can point at the system and say which feeling it believed the person had, or intervene before it answers. This thesis takes the opposite position: that in a system whose purpose is emotional, the emotional decision should be a separate, visible step.

# 1.2 Rationale of the Study or Motivation

Our motivation is practical rather than architectural. If a support system of this kind is ever to be used by a university well-being service or a small organisation, it has to run on the hardware such places already own, it has to keep what people type on the machine they typed it on, and somebody has to be able to check why it said what it said. None of those requirements is met by an approach that needs a data-centre GPU and hides its reasoning.

By studying existing research and by measuring our own early attempts, we identified seven gaps that this work addresses:

1. **The emotional decision is invisible:** in both joint models and prompted large language models, the emotion is inferred internally and never exposed, so it cannot be logged, audited, corrected, or used to trigger a safeguard before any text is generated.
2. **Quality is tied to scale, and scale is tied to hardware:** the strongest reported results come from models too large for a consumer GPU, which puts local, private operation out of reach for exactly the settings that most need it.
3. **Evaluation does not measure what a support reply must do:** published work is compared mainly on overlap with a reference reply, which says nothing about whether a reply acknowledged the feeling, asked one question rather than five, or stayed short enough to read while distressed.
4. **Machine empathy repeats itself:** review of the field identifies repetitiveness as the characteristic weakness of large-model empathy {cite:sorin2024jmir}, yet it is rarely measured.
5. **One label per conversation does not match how people feel:** the benchmark assigns a single emotion to a whole conversation, while a real message often carries two or three at once {cite:belay2025}.
6. **Detectors are often measured on text they will never see:** a detector trained and reported on a dataset's situation description is solving an easier problem than the one it meets in use, a mismatch we found in our own first pipeline and measured at roughly nine points of accuracy.
7. **The design is rarely separated from the model:** when a system built on a larger generator performs better, published comparisons seldom establish whether the gain came from the design or simply from the size of the generator.

# 1.3 Problem Statement

Existing empathetic response systems either bury the emotional decision inside a single model or rely on a generator too large to run locally, and are evaluated by measures that do not correspond to what a supportive reply is supposed to do. The result is a system that cannot be inspected, cannot be deployed privately on ordinary hardware, and cannot be shown to be better in any way that matters to the person receiving the reply.

This thesis therefore asks a specific question: **does separating emotion detection from response generation produce better empathetic replies than asking a single model to do both, under the constraint of a single consumer GPU?** Answering it requires more than building the system. It requires measuring the replies against what a support reply must do rather than against a reference string; establishing whether any advantage comes from the design or from the size of the generator; and testing every reported difference statistically rather than reporting the better average and stopping.

# 1.4 Objective

The general objective of this study is to design, build and evaluate a decoupled empathetic response system in which emotion detection and response generation are performed by separate, individually inspectable components that together fit within a consumer hardware budget. Within that aim we set six specific objectives:

- **Build a detector that reads what users actually write:** train an emotion detector over the 32 categories of EmpatheticDialogues on the user's own message rather than on the dataset's situation description, and select its backbone by controlled comparison rather than by assumption.
- **Design the link between the two models:** devise a way of passing the detector's decision to the generator that measurably improves the reply, and compare the alternatives, namely a single emotion, three candidates, per-emotion guidance and attention-derived cue words, against one another.
- **Stay inside 8 GB of GPU memory:** keep the complete system within the memory of hardware the team already owns, so that no user message has to leave the device on which it was typed.
- **Evaluate against single models of comparable and larger size:** measure the system on several independent signals rather than one, including rule adherence, an independent empathy classifier, emotional fidelity, a panel of AI judges and human raters, and test every reported difference with paired statistics.
- **Separate the effect of the design from the effect of scale:** run the system's own generator alone as a baseline, so that any advantage can be attributed to the decoupled design rather than to the generator's size.
- **Demonstrate that the design is usable, not only measurable:** extend the system into a multi-turn conversational agent that displays the emotional decision behind every reply.

# 1.5 Methodology in Brief

The proposed system has two components, which is where its name comes from, and the work was carried out in four stages.

**The detector, d1:** A RoBERTa-base encoder is fine-tuned over the 32 emotions of EmpatheticDialogues on the first message of each conversation, which is the text the system receives in use. For any message it returns the three most probable emotions with their confidence, together with the words that most influenced its own decision, taken from the attention its classification token paid to each word. Five backbones were trained under identical settings and compared before this one was selected.

**The generator, d2:** Phi-4-mini-instruct, loaded in 4-bit form so that it fits beside the detector inside 8 GB, writes the reply in two calls. The first is a short, deterministic call that asks the model to choose one of the detector's three candidate emotions. The second builds a prompt from that single emotion, the cue words, guidance written specifically for that emotion, and three universal rules, and generates the reply. Three generators were compared with everything else held fixed before this one was selected.

**The alternatives:** Because a design is only as convincing as what it was compared against, six designs without a detector were also built and measured on the same conversations: zero-shot prompting, a model fine-tuned with LoRA on the dataset's own listener replies, single models of 1.5, 3.8 and 7.6 billion parameters prompted to detect and respond in one step, and a retrieval-augmented pipeline.

**The evaluation:** Every system answers the same 300 conversations with identical decoding settings, and is measured by five independent signals: automatic checks of the three response rules; the WASSA 2024 shared-task empathy classifier {cite:giorgi2024wassa, yang2024hyy33}; emotional fidelity, which reads the reply back through the detector and compares it with the conversation's true emotion; a blind panel of two large language models from unrelated developers {cite:zheng2023judge, verga2024}; and a human rating study. Differences are tested with paired statistics and corrected for multiple comparisons.

Figure 1.1 sets out the whole of this work on one page: the data it starts from, the two strands of building the system and testing it against the alternatives, the design that was measured and rejected, the two findings that settled the argument, and the system that was finally adopted. Chapter 4 specifies that system in detail, and Chapter 5 reports the measurement behind every box.

[[FIGURE 1.1 | fig_workflow.png | The workflow of this thesis, from the corpus to the adopted system]]

# 1.6 Scopes and Challenges

- **Scopes:**

1. **Emotionally aware support on ordinary hardware:** the complete system runs within 8 GB of GPU memory, so it can be deployed on a consumer machine and a user's disclosure need never be sent to an external service.
2. **An inspectable emotional decision:** every reply carries an audit trail, namely the three candidate emotions with their confidence, the emotion chosen, and the words that drove the detection, which a single end-to-end model cannot provide.
3. **Replaceable components:** the detector, the per-emotion guidance and the generator can each be changed without touching the others, which we demonstrate by substituting a larger detector into the finished system.
4. **A multi-turn agent, not only a benchmark entry:** the same components run as a conversational agent that reads the user's recent turns, keeps the conversation in memory, and adapts its rules after the first exchange.
5. **An evaluation that matches the use case:** replies are measured on rule adherence, an independent empathy classifier, emotional fidelity, a judge panel and human ratings, rather than on similarity to a reference reply.
6. **Reproducibility:** every parameter, seed and measurement is recorded, and every table and figure in this report is generated from saved experiment outputs by script.

- **Challenges:**

1. **A single label for a mixture of feelings:** the dataset assigns one emotion to an entire conversation, although a message can carry several at once, which places a ceiling on what any detector trained on it can be asked to do.
2. **A mismatch between training text and deployment text:** the detector's original training input was not the text it receives in use, a defect that cost roughly nine points of accuracy before it was found and corrected.
3. **A memory budget that constrains every choice:** 8 GB has to hold both models, which rules out the larger generators that published work relies on and forces 4-bit quantisation.
4. **Measures that can be gamed by length:** both the empathy classifier and the AI judges reward longer replies, so any comparison between systems of different verbosity has to control for length before it can be believed.
5. **Judges that may favour their own kind:** large language models used as evaluators tend to prefer text from their own family {cite:panickssery2024}, which constrains which models may judge which systems.
6. **No category for crisis:** the emotion set contains nothing corresponding to suicidal ideation, so the system must not be presented, or deployed, as a mental-health service.

# 1.7 Thesis Structure

**Chapter 1** provides the Introduction, giving the background to empathetic response generation and text-based support, the rationale and motivation for the research, the problem statement, the objectives, a brief overview of the methodology, the scopes and challenges of the work, and this outline of the thesis.

**Chapter 2** provides the Literature Review, covering the preliminaries of empathetic dialogue, emotion recognition and the model families used in this work, a review of existing research on empathetic response generation with both purpose-built models and large language models, and a summary of the key findings and the gaps that follow from them.

**Chapter 3** provides the Requirements, Impacts and Constraints, covering the final specifications and requirements, the societal and environmental impacts of the work, its ethical issues, the standards followed, the project management plan, the risk management plan and the economic analysis.

**Chapter 4** provides the Proposed Methodology, covering the design process, the specification of every design that was built, including the detector candidates, the generator candidates, the ways of connecting them and the designs built without a detector, the collection and preparation of the data, and the implementation of the selected design.

**Chapter 5** provides the Results and Analysis, covering the performance of every system on the same conversations, the analysis of the design alternatives, the adjustments made to the final design, the statistical analysis of every reported difference, the comparison of the results with published work, and a discussion of what the results mean and what they do not establish.

**Chapter 6** provides the Conclusion, summarising the findings, stating the contributions of the work, and giving recommendations for future research.
