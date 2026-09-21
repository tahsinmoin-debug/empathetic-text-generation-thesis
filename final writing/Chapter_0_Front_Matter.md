# Declaration

It is hereby declared that this thesis is our own original work while completing a degree at BRAC University. The thesis does not contain material previously published or written by a third party, except where this is appropriately cited through full and accurate referencing. The thesis does not contain material which has been accepted, or submitted, for any other degree or diploma at a university or other institution. We have acknowledged all main sources of help.

[[TEAM TO FILL: student name, ID and signature for each member, in the order used on the cover page.]]

# Approval

The thesis titled "[[TEAM TO FILL: final title as submitted]]" submitted by the students named above, of Summer 2026, has been accepted as satisfactory in partial fulfilment of the requirement for the degree of Bachelor of Science in Computer Science and Engineering.

[[TEAM TO FILL: supervisor name and designation, co-supervisor if applicable, programme coordinator and department head blocks as given in the department template.]]

# Abstract

Systems that reply supportively to a person in distress normally use one neural model to read the message and write the answer, which hides the emotional judgement inside the model and ties the quality of that judgement to the size of the generator. This thesis asks whether separating the two jobs produces better replies at a size that fits an ordinary computer. The proposed system uses a RoBERTa-base classifier, fine-tuned on the 32 emotions of EmpatheticDialogues, to return the three most likely emotions for the user's own message together with the words that drove the decision; a 3.8-billion-parameter instruction model, quantised to 4 bits, then chooses one of the three and writes the reply using wording prepared for that specific emotion and three response rules. Everything runs on a single 8 GB consumer GPU, and the emotion the system responded to is displayed with every reply.

The system is compared with eight alternatives, including single models of 1.5, 3.8 and 7.6 billion parameters, on 300 held-out conversations, using rule checks, the WASSA 2024 empathy classifier, emotion validation, a blind two-judge AI panel and a human forced-choice study. Against a single model of comparable size it is significantly more empathetic by the classifier (5.05 against 4.84) and by the AI judges (+0.70 empathy), and names the user's true emotion far more often (36.7% against 25.7%). Against the same generator prompted alone, the design raises empathy-classifier scores by 0.30 and emotion fidelity by 9.7 points and repeats itself less, at the cost of 13.7 points of rule adherence; the judges rate the two configurations alike. Against a single model with 1.9 times the parameters it uses 38% less GPU memory with no significant difference in response time or in judge-rated quality. The gains come from the design rather than from generator size: the same generator used alone scores no better on empathy than a model less than half its size.

**Keywords:** empathetic response generation; emotion detection; EmpatheticDialogues; two-stage pipeline; parameter efficiency; explainable dialogue systems.

# Acknowledgment

[[TEAM TO FILL: acknowledgements to the supervisor, co-supervisor, department, family and anyone who assisted, in the team's own words.]]

# Nomenclature

- **d1** - the emotion detector, the first component of the proposed system
- **d2** - the response generator, the second component
- **ED** - EmpatheticDialogues, the dataset used throughout
- **LoRA** - low-rank adaptation, a parameter-efficient fine-tuning method
- **QLoRA** - LoRA applied to a 4-bit quantised model
- **RAG** - retrieval-augmented generation
- **WASSA** - Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis, whose 2024 shared-task classifier is used to score empathy
- **EPITOME** - a published framework of empathy mechanisms in text-based support
- **Top-1 / top-3 / top-5 accuracy** - how often the correct emotion is the first, among the first three, or among the first five predictions
- **Emotion validation** - how often the emotion expressed in a reply matches the conversation's true emotion
- **Rule adherence** - the share of replies that acknowledge the feeling, ask exactly one question and stay within 120 words
