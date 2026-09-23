# 5.3 Final Design Adjustments

Evaluating the initial pipeline, a GPT-2 Medium detector with a Qwen2.5-1.5B generator, exposed several weaknesses. Each adjustment below was made in answer to a measured result, and each was re-evaluated on the same 300 conversations before it was kept. Table 5.19 is the full log, with the evidence behind every change and its measured effect.

[[TABLE 5.19 | design_adjustment_log.csv | Every design adjustment, the evidence behind it and its measured effect]]

## 5.3.1 Adjustments to the Detector, the Generator and the Prompt

Measuring the first detector on the input it actually receives revealed a mismatch between training and use: it had been trained on the dataset's situation descriptions, which name the feeling almost explicitly, while the pipeline hands it the user's conversational message. The detector was retrained on first messages and, following the encoder comparison in Section 5.2.2, GPT-2 Medium was replaced by RoBERTa-base. Its accuracy now describes the job it actually does, and its ranking is good enough for top-3 selection.

Three changes followed from the comparisons in Section 5.2, each kept only after it was re-measured on the same conversations. **The generator was replaced:** with the corrected detector held fixed, Phi-4-mini followed the rules more often than Qwen2.5-1.5B, so it became d2 in 4-bit form. **The emotion decision was given its own call:** the detector's top label is wrong in about half of conversations while the correct one is among its top three in 76.5%, and an earlier variant that chose and replied in a single call had damaged rule-following. **Guidance was written for each of the 32 emotions:** a generic "the user is feeling X" says nothing about how a reply to *lonely* should differ from a reply to *proud*. Together, and with detector and generator fixed, these raised emotion fidelity from 28.3% to 36.7% and empathy from 4.89 to 5.05, both reliable after correction, while the 5.3-point drop in rule adherence was not; replies also fell from 80.4 to 57.1 words, all within the limit.

The original design specified an attention mechanism but never implemented one. When extraction of the detector's attention weights was added, a verification check found that the library's default attention implementation returns no weights at all and raises no error, which would have left the feature silently inactive while appearing to work. The adjustment selects an implementation that returns weights, stops execution if they are missing, merges sub-word fragments back into whole words, and passes only the highest-scoring 40% of words to the generator so that the cues do not simply repeat the message.

## 5.3.2 From Single Replies to a Conversation

The evaluation assumes one message and one reply. Turning the design into the agent specified in Section 4.2.7 raised one question that had to be measured rather than assumed: **how much of the conversation should the detector read?**

**Why the dataset cannot answer it:** EmpatheticDialogues conversations have a median of 2 turns and a maximum of 4, and the gold emotion never changes inside one, in none of the 2,542 test conversations, so there is no multi-turn emotion shift in the data to tune a window against. We instead wrote conversations containing the shifts a real support conversation produces, above all the "I'm fine, actually I'm not" pattern in which the latest message alone says the opposite of what the person means. The column saying what a listener would hear is our own judgement, not an annotation, so this is **illustrative evidence for a design choice and not a statistical result**.

**Reading only the latest message loses the thread:** across ten short conversations with a deliberate shift (Table 5.36), moving from one turn to four moved the prediction to what a listener would hear in five of them, left it unchanged in three, and in two exchanged one neighbour of the intended emotion for another; in none did a correct single-message reading become wrong. The clearest case is the one that started this test: "I am okay. Well, not properly." alone is read as *content* at 72% confidence, and with the previous turn, "Someone yelled at me today very badly", as *angry*.

**But a window can also be too wide:** Table 5.37 widens the window on four long conversations of six to eight turns, and four turns is the widest setting that degrades none of them. At six turns two of the four move away from the listener's reading, one of them confidently: the break-up conversation, correct at four turns (*lonely*, 61%), becomes *guilty* at 64% once the anger of the earlier turns re-enters the window. The control stays *anxious* at every width with rising confidence, so the damage comes from stale emotions competing with the current one and not from the length of the input. Neither setting recovers the burnout conversation, which we report rather than leave out.

**Four turns, fixed:** widening the window only when the detector was unsure, using the median detector confidence of 0.632 as the threshold, produced the same answer as the fixed four-turn window in all ten scenarios, so the simpler of the two was kept. The generator's memory needed no such test, being an instruction-following model with a large context window rather than a classifier trained on short messages.

[[TABLE 5.36 | context_window_shift_scenarios.csv | What the detector predicts as the window widens, on ten short conversations containing an emotion shift]]

[[TABLE 5.37 | context_window_long_scenarios.csv | The same test on four long conversations, where a window can become too wide]]

A web demonstration of the agent was prepared for Hugging Face Spaces, with disclaimers and crisis resources shown on the page. [[TEAM TO FILL: the public URL and date of deployment, or a statement that the demonstration was prepared but not deployed.]]

## 5.3.3 Adjustments That Were Rejected or Deferred

Not every candidate change survived its evaluation. **Retrieval augmentation** cut rule adherence from 63.7% to 43.7% and was rejected (Section 5.2.5), and **a larger detector** improved no measure of reply quality inside the pipeline despite being 2.9 points more accurate on its own (Section 5.4.3).

Two changes were **deferred**. A repetition penalty would probably reduce the boilerplate rate of 58.7%, but regenerating the replies would have invalidated the judge scores and the human-evaluation items already built on them. A pre-generation crisis filter is required before any public deployment and is the first item of future work (Section 3.4).

**Correcting the fine-tuned baseline:** a check of the monolithic LoRA baseline found a configuration error. Its training script used the end-of-turn token as the padding token, and the collator excludes padding from the loss, so the model was never trained to stop: 99% of its replies ran to the 250-token limit, averaging 200.7 words and 7.07 questions, and none followed all three rules. We retrained it with the model's own padding token, added a start-up check that aborts training if the end-of-turn token would be masked, and selected the best epoch by validation loss. The corrected model ends its replies normally (10.0 words, none reaching the limit) and follows all rules in 22.3% of cases. Every result in this thesis uses the corrected baseline; the original run is recorded here only as a documented error.

The complete configuration after all adjustments, with the justification of every value, is listed in Table 4.9 in the appendix.
