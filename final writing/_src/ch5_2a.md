# 5.2 Analysis of Design Solutions

Section 5.1 reported what each design achieved. This section explains why they differ, taking one design question at a time, and ends by weighing the alternatives against criteria that performance figures do not capture.

## 5.2.1 Decoupled against Monolithic Designs

The central question of the thesis is whether emotion detection deserves a model of its own. Four designs without a detector answer the same conversations, and they fail in three different ways.

**Fine-tuning a single model teaches the dataset's style, not the wanted behaviour:** the LoRA-tuned generator writes short, casual replies that seldom acknowledge the feeling, and comes last on empathy, emotion fidelity and the judges; it learned its training data correctly, and the training data does not contain the behaviour we want. **Prompting without an emotion leaves the model nothing to work from:** zero-shot prompting asks the required single question in 22.7% of replies. **Asking one model to do both jobs closes part of the gap, but not the part that matters here:** the same-size single model satisfies all rules in 51.3% of replies, yet its replies are barely more faithful to the true emotion than zero-shot prompting (25.7% against 24.7%).

Because that comparison changes the generator as well as the design, we built the test that changes only the design: **the final system's own generator, given the single-model prompt and no detector at all.** It follows the rules more often than the decoupled system (73.7% against 60.0%), but its empathy and emotional fidelity stay at the level of the smaller single model (4.75 and 27.0%); adding the detector and the per-emotion guidance to that same generator raises them to 5.05 and 36.7%. Section 5.5.5 reports it in full, and it is the strongest evidence in this thesis that the second model earns its place.

**A larger single model is still stronger overall:** Qwen2.5-7B leads on rule adherence, empathy and the judges, and since it shares a prompt with the same-size single model, capacity rather than structure explains that gap. Section 5.2.6 weighs what decoupling costs against what it makes possible.

## 5.2.2 Detector Alternatives

Two findings settled the detector. First, **what the detector is trained on matters more than which network is used.** Training GPT-2 Medium on the dataset's situation descriptions but deploying it on the user's own message cost about nine points of accuracy (56.9% against 48.2%); retraining on the text the model actually receives was the single largest improvement made to the detector.

Second, **among encoders trained on the right input, models of the same size are indistinguishable on top-1 accuracy** (48.8-50.2%); RoBERTa-base separates itself only on the quality of its ranking, with 85.0% top-5 against 81.7-82.7% for BERT and ELECTRA (Figure 5.10, Tables 5.11 and 5.12). Ranking is exactly what the two-stage design consumes. RoBERTa-large adds 2.9 points of top-1 accuracy for 7.4 times the training cost and changed no measure of reply quality inside the pipeline (Section 5.4.3), so RoBERTa-base was kept.

[[TABLE 5.10 | d1_evolution_mismatch_fix.csv | Effect of training the detector on the text it actually receives]]

[[FIGURE 5.10 | fig_d1_backbone_comparison.png | Top-1 and top-5 accuracy of the detector candidates under identical training]]

[[TABLE 5.11 | d1_backbone_comparison.csv | Detector candidates trained and tested under identical settings]]

[[TABLE 5.12 | d1_backbone_significance.csv | Paired significance tests between the detector candidates]]

## 5.2.3 Generator Alternatives

With the detector, the prompt and the conversations held fixed, the generator decides how well the rules are followed: 65.3% for Phi-4-mini, 57.7% for Qwen2.5-1.5B and 22.0% for Gemma-3-4B.

Gemma is the instructive case, because on the surface it looks like the most empathetic model: it acknowledges the feeling in 96.7% of replies and scores higher than Phi-4-mini on the empathy classifier (5.30 against 4.89). But it averages 133 words, with 72% of replies over the limit, and the classifier rewards length in every system we measured (Section 5.4.4). Comparing replies of similar length removes that advantage: in every band where both have replies, Phi-4-mini matches or beats Gemma, and Gemma leads only above 120 words, where Phi-4-mini has no replies at all. Its apparent empathy is verbosity, and a system meant to answer someone in distress briefly cannot ignore the length rule.

[[TABLE 5.13 | d2_bakeoff_candidates.csv | Generator candidates with the detector, prompt and conversations held fixed]]

[[TABLE 5.14 | gemma_length_band_analysis.csv | Empathy score within each reply-length band, with the number of replies in each]]

## 5.2.4 Integration Alternatives

The link between the two models is itself a design choice, and the four ways of making it differ mainly in how much reasoning they ask of the generator. Against the same detector and generator with only the top-1 emotion and a generic prompt, the two-stage design raises emotion fidelity from 28.3% to 36.7% and empathy from 4.89 to 5.05, both reliable after correction, and shortens replies from 80.4 to 57.1 words. It costs 5.3 points of rule adherence, which is not a reliable difference.

The benefit depends on the generator's capacity. The same change applied to Qwen2.5-1.5B *lowers* rule adherence to 36.7% and acknowledgement to 47.0%, which repeats what the earlier candidate-pick variant showed for that model (54.0%): a small model given more to reason about loses control of the output format. Only the larger instruction-tuned generator can absorb a richer prompt without dropping the rules.

One limitation is worth stating plainly: this comparison removes three things at once, the top-3 selection, the emotion-specific guidance and the attention cues. Their individual contributions were not separated, so the improvement is credited to the combined design, and separating them is listed as future work (Section 5.6.6).

[[TABLE 5.15 | integration_variant_ablation.csv | The four ways of passing the detector's output to the generator]]

## 5.2.5 Retrieval Augmentation

**How retrieval was used:** every training conversation was embedded as a message-and-reply pair with all-MiniLM-L6-v2 {cite:reimers2019sbert}. At generation time the detector's emotion narrowed the pool to conversations carrying that emotion, the two most similar messages were retrieved by cosine similarity, and their human replies were pasted into the system prompt as worked examples above the same three rules {cite:lewis2020rag}. The second attempt narrowed the pool further to replies asking at most one question, and added an explicit reminder that the reply must contain exactly one question mark.

It lowered rule adherence in both versions, to 54.3% and 43.7% against 63.7% without retrieval (Table 5.16). The reason is visible in the data: real listener replies satisfy all three rules in only 4.3% of cases, so the retrieved examples teach a style that contradicts the instructions given in the same prompt. Compliance with the one-question rule fell to 76.0% in the first version; filtering the examples and repeating the rule recovered that but cut acknowledgement from 70.3% to 57.0%.

**What the cost bought was nothing:** Table 5.16 reports retrieval on the same measures as every other system, because a design that loses rule adherence could still be worth keeping if it gained elsewhere. It did not: empathy is 4.85 against 4.80 without retrieval and emotion fidelity 27.7% against 26.0%, both well inside the confidence intervals of Figure 5.2, while the spread of its empathy scores is the widest of any system. Retrieval therefore cost 20 points of rule adherence and returned no measurable gain, which is why it was rejected before the judge panel was run; it is the one design in Table 5.1 the judges never scored, and its absence from the judge table and the composite follows from that rather than from selection. Three cells of the first attempt cannot be recovered, because its outputs were overwritten before those measures were computed, and the table says so instead of leaving them blank. For this task, examples of how people actually reply work against explicit behavioural rules rather than reinforcing them.

[[TABLE 5.16 | rag_negative_result.csv | Retrieval augmentation on every measure, against the same pipeline without it]]

## 5.2.6 Assessment against Criteria beyond Performance

Performance is not the only criterion, and the alternatives separate cleanly on the others. The two designs that need training, the fine-tuned baseline and the retrieval pipeline, are also the two that cost most to change. Every single-model design, of any size, keeps its emotional decision inside itself, so there is nothing to inspect and nowhere to place a safeguard. The decoupled designs expose the emotion, and the selected one exposes the most: three candidates, the emotion chosen and the words that drove it, with the detector, the guidance and the generator each replaceable on their own.

Because the performance measures disagree with one another, Table 5.17 combines five of them into one ranking: rule adherence, empathy, emotion fidelity, judge-rated empathy and freedom from boilerplate, each standardised across the systems and averaged with equal weight. Nothing is tuned: the weights are equal, every judged system is included, and the measures are those already in Table 5.1.

On that ranking the final system is **second of eight**, behind only the 7.6-billion-parameter single model, and **first among the systems that fit the 8 GB budget set as requirement N1 in Section 3.1.2**. The result does not rest on one favourable measure: dropping any single measure leaves it second or third, always behind the larger single model. A composite depends on what goes into it, so we report it as a summary and not as evidence in its own right; the per-measure comparisons in Sections 5.4.2 and 5.5.7, with their significance tests, remain the basis for every claim in this thesis.

[[TABLE 5.17 | composite_ranking.csv | Equal-weight composite of five independent measures across the judged systems]]
