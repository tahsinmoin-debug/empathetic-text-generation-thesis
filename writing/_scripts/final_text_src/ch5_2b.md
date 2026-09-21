# 5.2 Final Design Adjustments

> Note for the team (delete before submission): the rubric lists both this section and "Analysis of Design Solutions" as 5.2. Confirm the numbering with the supervisor; if this becomes 5.3, renumber its subsections and all later sections and cross-references.

The performance evaluation of the initial pipeline, which combined a GPT-2 Medium detector with a Qwen2.5-1.5B generator, identified several weaknesses. Each adjustment described in this section was made in response to a measured result and was re-evaluated on the same conversations before being kept. Table 5.12 summarises all adjustments.

[[TABLE 5.12 | design_adjustment_log.csv | Summary of design adjustments, the evidence behind each and its measured effect]]

## 5.2.1 Detector: Correcting the Training Input

Evaluating the initial detector on the input it actually receives revealed a mismatch between training and use. The model had been trained on the dataset's situation descriptions, which state the feeling explicitly, but the pipeline supplied the user's conversational message instead. The two texts are identical in only 11.7% of conversations, and accuracy fell from 56.9% on situation descriptions to 48.2% on user messages (Table 4.4). We retrained the detector on first utterances and, following the comparison of encoder models in Section 4.2.2, replaced GPT-2 Medium with RoBERTa-base {cite:liu2019roberta}. The adjusted detector reaches 50.3% top-1 and 84.3% top-5 accuracy on the real input, so its reported accuracy now reflects its use, and the quality of ranking needed for top-3 selection is improved.

## 5.2.2 Generator and Prompting

**Generator replacement (A2).** With the adjusted detector held fixed, the comparison of generators showed Phi-4-mini {cite:abouelenin2025phi4mini} achieving higher rule adherence (65.3%) than the original Qwen2.5-1.5B (57.7%), so Phi-4-mini replaced Qwen as d2, loaded with 4-bit quantisation to fit in memory.

**Two-stage emotion selection (A3).** The adjusted detector's top-1 label is wrong in about half of conversations, but the correct label is among its three best candidates in 76.5% of cases. The earlier candidate-pick experiment showed that asking the generator to choose the emotion and respond in a single call degrades rule-following, so the choice was separated into a deterministic selection call followed by a separate generation call.

**Emotion-specific prompts (A4).** A generic instruction such as "the user is feeling X" gives the generator no guidance on how a response to *lonely* should differ from a response to *proud*. We therefore wrote guidance on what to focus on and what to avoid for each of the 32 emotions. With the detector and Phi-4-mini held fixed, adjustments A3 and A4, together with the attention cues (A5), raised emotion validation from 28.3% to 36.7% and WASSA empathy from 4.89 to 5.05 relative to top-1 prompting without them. Both improvements are significant after correction, while the change in rule adherence (-5.3 points) is not. Replies also became shorter (57.1 against 80.4 words), and every reply stayed within the length limit.

## 5.2.3 Implementing the Attention Mechanism

The original design specified an attention mechanism, but it had not been implemented. When extraction of the detector's attention weights {cite:vaswani2017} was added, verification revealed that the default attention implementation of the Transformers library {cite:wolf2020transformers} returned no weights without raising an error, which would have left the feature inactive while appearing to work. The adjustment selects an implementation that returns attention weights, stops execution if the weights are missing, and passes the words receiving the most attention to the generator as context cues. It also expands sub-word fragments to whole words and limits the cues to the top 40% of word spans, so that the cues do not simply repeat the entire message.

## 5.2.4 Developing the Final Solution: A Multi-Turn Conversational Agent

The single-turn evaluation assumes one message and one reply. Extending the solution into a conversational agent required three further adjustments.

**Detector context window (A6).** Classifying only the latest message discards context. For example, "I am okay. Well, not properly." read on its own is classified as *content* with 72% confidence, although in context the user was expressing anger. Conversational context is known to be important for recognising emotions in dialogue {cite:poria2019erc}, so we tested several context windows on long conversational scenarios. A window of four user turns was best or joint best in three of four scenarios, whereas six turns degraded two scenarios, in one case changing a breakup scenario to the wrong emotion with 64% confidence as emotions from earlier turns competed with the current one. The final agent therefore classifies the user's last four turns.

**Generator memory (A7).** The generator receives the last ten exchanges verbatim rather than a summary, so that it can avoid repeating itself without losing emotional detail through summarisation.

**Turn-aware rules.** A mandatory single question suits a first reply, but over a long conversation it turns the exchange into an interrogation. From the second turn onward, the question is therefore optional and repetition of earlier phrasing is discouraged, while the first turn keeps the original rules so that single-turn results remain valid. The agent also detects when the user's emotion changes between turns and instructs the generator to respond to the current emotion.

A web demonstration of the agent was prepared for Hugging Face Spaces, with disclaimers and crisis resources displayed on the page. [[TEAM TO FILL: the public URL and date of deployment, or a statement that the demonstration was prepared but not deployed.]]

## 5.2.5 Rejected and Deferred Adjustments

Not every candidate adjustment was adopted. **Retrieval augmentation (A8)** {cite:lewis2020rag} reduced rule adherence from 63.7% to 43.7% and was rejected (Section 5.2). **A larger detector (A10)** was also tested. RoBERTa-large is 2.9 points more accurate on the test partition, but when substituted into the final pipeline it improved no measure of reply quality: rule adherence changed from 60.0% to 54.7% (p = 0.16), WASSA empathy from 5.05 to 5.01 (p = 0.43), and emotion validation from 36.7% to 38.0% (p = 0.68). RoBERTa-base was therefore kept, and the conversational agent was not changed. Two further adjustments were **deferred to future work**. A repetition penalty (A9) could reduce the final system's boilerplate rate of 58.7%, but regenerating the final outputs would have invalidated the AI-judge scores and the human-evaluation items that were based on them. A pre-generation crisis filter is required before any public deployment (Section 3.4).

**Correcting the fine-tuned baseline (A11).** Checking the monolithic LoRA baseline revealed a configuration error. Its training script used the end-of-turn token `<|im_end|>` as the padding token, and the data collator excludes padding tokens from the loss, so the model was never trained to end a reply. As a result, 99% of its replies ran to the 250-token limit, averaging 200.7 words and 7.07 questions, and none followed all three rules. We retrained the baseline with Qwen's own padding token, a start-up check that aborts training if the end-of-turn token would be masked, and the published LoRA settings described in Section 4.2.7; the best epoch was selected by validation loss. The corrected model ends its replies normally (10.0 words on average, none reaching the limit) and follows all rules in 22.3% of replies. All results in this thesis use the corrected baseline; the original run is reported here only as a documented error.

The final configuration after all adjustments is listed in Table 4.9.
