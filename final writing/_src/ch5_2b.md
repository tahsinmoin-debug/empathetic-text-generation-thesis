# 5.3 Final Design Adjustments

Evaluating the initial pipeline, a GPT-2 Medium detector with a Qwen2.5-1.5B generator, exposed several weaknesses. Each adjustment below was made in answer to a measured result, and each was re-evaluated on the same 300 conversations before it was kept. Table 5.19 is the full log, with the evidence behind every change and its measured effect.

[[TABLE 5.19 | design_adjustment_log.csv | Every design adjustment, the evidence behind it and its measured effect]]

## 5.3.1 Correcting What the Detector Is Trained On

Measuring the first detector on the input it actually receives revealed a mismatch between training and use: it had been trained on the dataset's situation descriptions, which name the feeling almost explicitly, while the pipeline hands it the user's conversational message. The two texts are identical in only 11.7% of conversations, and accuracy fell from 56.9% to 48.2% across that gap (Table 5.10). The detector was retrained on first messages and, following the encoder comparison in Section 5.2.2, GPT-2 Medium was replaced by RoBERTa-base. The corrected detector reaches 50.3% top-1 and 84.3% top-5 on the real input, so its reported accuracy now describes its actual job and its ranking is good enough for top-3 selection.

## 5.3.2 Generator and Prompting

**Replacing the generator:** with the corrected detector held fixed, Phi-4-mini followed the rules more often than Qwen2.5-1.5B (65.3% against 57.7%), so it became d2, loaded in 4-bit form to fit beside the detector.

**Splitting the emotion decision into its own call:** the detector's top label is wrong in about half of conversations, but the correct one is among its top three in 76.5%. An earlier variant that asked the generator to choose the emotion and write the reply in a single call had damaged rule-following, so the choice was given a short, greedy call of its own, leaving the generation prompt to describe exactly one feeling.

**Writing guidance for each emotion:** a generic instruction such as "the user is feeling X" tells the generator nothing about how a reply to *lonely* should differ from a reply to *proud*, so we wrote guidance on what to focus on and what to avoid for all 32 emotions. Together with the two-stage selection and the attention cues, and with the detector and generator held fixed, these changes raised emotion fidelity from 28.3% to 36.7% and empathy from 4.89 to 5.05, both reliable after correction, while the drop in rule adherence (-5.3 points) was not. Replies also became shorter, 57.1 against 80.4 words, and every one stayed within the length limit.

## 5.3.3 Making the Attention Mechanism Real

The original design specified an attention mechanism but never implemented one. When extraction of the detector's attention weights was added, a verification check found that the library's default attention implementation returns no weights at all and raises no error, which would have left the feature silently inactive while appearing to work. The adjustment selects an implementation that returns weights, stops execution if they are missing, merges sub-word fragments back into whole words, and passes only the highest-scoring 40% of words to the generator so that the cues do not simply repeat the message.

## 5.3.4 From Single Replies to a Conversation

The evaluation assumes one message and one reply. Turning the design into an agent people can talk to needed three further adjustments.

**The detector reads four turns, not one:** classifying only the latest message throws away context. "I am okay. Well, not properly." on its own is classified as *content* with 72% confidence, although in context the person was angry. We tested several context windows on long scenarios; four user turns was best or joint best in three of four, while six turns degraded two of them, in one case flipping a break-up scenario to the wrong emotion with 64% confidence as older turns competed with the current one.

**The generator remembers ten exchanges verbatim:** the last ten exchanges are passed as they were written rather than summarised, so that emotional detail is not lost in compression.

**The rules become turn-aware:** one mandatory question suits a first reply but turns a long conversation into an interrogation, so from the second turn the question is optional and reuse of earlier phrasing is discouraged. The first turn keeps the original rules, which is what keeps the single-turn results in this chapter comparable. The agent also notices when the user's emotion changes between turns and instructs the generator to answer the current one.

A web demonstration of the agent was prepared for Hugging Face Spaces, with disclaimers and crisis resources shown on the page. [[TEAM TO FILL: the public URL and date of deployment, or a statement that the demonstration was prepared but not deployed.]]

## 5.3.5 Adjustments That Were Rejected or Deferred

Not every candidate change survived its evaluation. **Retrieval augmentation** cut rule adherence from 63.7% to 43.7% and was rejected (Section 5.2.5). **A larger detector** was tested and rejected as well: RoBERTa-large is 2.9 points more accurate on the test partition, but inside the pipeline it improved nothing, moving rule adherence from 60.0% to 54.7% (p = 0.16), empathy from 5.05 to 5.01 (p = 0.43) and emotion fidelity from 36.7% to 38.0% (p = 0.68).

Two changes were **deferred**. A repetition penalty would probably reduce the boilerplate rate of 58.7%, but regenerating the replies would have invalidated the judge scores and the human-evaluation items already built on them. A pre-generation crisis filter is required before any public deployment and is the first item of future work (Section 3.4).

**Correcting the fine-tuned baseline:** a check of the monolithic LoRA baseline found a configuration error. Its training script used the end-of-turn token as the padding token, and the collator excludes padding from the loss, so the model was never trained to stop: 99% of its replies ran to the 250-token limit, averaging 200.7 words and 7.07 questions, and none followed all three rules. We retrained it with the model's own padding token, added a start-up check that aborts training if the end-of-turn token would be masked, and selected the best epoch by validation loss. The corrected model ends its replies normally (10.0 words, none reaching the limit) and follows all rules in 22.3% of cases. Every result in this thesis uses the corrected baseline; the original run is recorded here only as a documented error.

The complete configuration after all adjustments, with the justification of every value, is listed in Table 4.9.
