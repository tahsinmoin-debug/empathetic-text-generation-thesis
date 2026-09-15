# 3.4 Ethical Issues

A system that responds to people describing their feelings raises ethical questions that shaped its design from the outset. This section addresses user safety, privacy, the ethics of the human evaluation, data and model licensing, bias, and research integrity.

## 3.4.1 Scope and User Safety

The system is designed for supportive listening in low-risk conversations. It is not a therapeutic, diagnostic or crisis service and must not be presented as one. This boundary is reflected in the response rules, which instruct the generator to acknowledge and validate the user's feelings, offer a reassuring thought and invite the user to continue, rather than to give clinical advice.

The most serious ethical risk is a user disclosing suicidal thoughts. The 32 emotion categories of EmpatheticDialogues [1] include no label for suicidal ideation or self-harm, so the detector cannot represent such a message. During interactive testing, the input "I want to die" was classified as *afraid* (45%), *terrified* (28%) and *sad* (4%). The generated reply was warm and invited the user to keep talking, but it did not point the user to any crisis resource. General-purpose commercial assistants given the same message responded mainly with helpline information. We consider neither response adequate on its own. World Health Organization guidance on communicating about suicide stresses both fostering understanding and empathy and providing accurate information on where and how to seek help [36]; although written for media professionals, the same principle applies to a system that responds to such disclosures.

In the current prototype, mitigation is limited to the interface. The prepared demonstration page states that the system is a student research prototype, is not monitored, and is not a substitute for professional help, and it lists crisis services ([[TEAM TO FILL: current Kaan Pete Roi number, verified on its official website]]; findahelpline.com). A pre-generation crisis filter, which would route messages containing self-harm language to a fixed, reviewed response, was identified but not implemented, and is listed as essential future work before any public deployment. The decoupled architecture makes such a filter straightforward to add, because emotion detection takes place in a separate, inspectable step before any text is generated.

## 3.4.2 Privacy and Data Protection

All models were developed and evaluated using the public EmpatheticDialogues corpus, and no personal data was collected to train or tune them. During evaluation, only EmpatheticDialogues text and the corresponding generated replies were sent to the third-party AI-judge services, and no data from real users was involved. Both models run on local hardware, so a deployed instance would not need to send a user's disclosure to an external provider, which is one practical motivation for a small, self-hostable system. The prepared demonstration interface, which has not yet been publicly deployed, holds conversation history only in per-session memory on the server, writes nothing to disk or logs, and discards the conversation when the page is closed.

## 3.4.3 Human Evaluation

The human evaluation described in Chapter 5 asks volunteers, mostly students together with some faculty members, to compare replies to 13 short situations. Participation is voluntary and anonymous. The questionnaire collects no names or email addresses, explains its purpose and expected duration of about 12 to 17 minutes, warns that some situations describe sad or stressful experiences, and allows participants to stop at any time. The identity of the system behind each reply is hidden and reply order is counterbalanced, so participants cannot be steered toward the authors' system. [[TEAM TO FILL: state the department's policy or the supervisor's confirmation that formal ethics review is not required for this anonymous, minimal-risk survey.]]

## 3.4.4 Data and Model Licensing

Table 3.1 lists the licences of the dataset and models used. EmpatheticDialogues is released under the Creative Commons Attribution-NonCommercial 4.0 licence [3], which requires attribution and forbids commercial use. Because the emotion detector is trained on this data, it may be shared for research but may not be incorporated into a commercial product without separately licensed training data. The final generator, Phi-4-mini-instruct [7], and the RoBERTa detector backbone [6] are released under the MIT licence, the Qwen2.5 baselines [8] under Apache-2.0, and the Gemma-3 candidate under Google's Gemma Terms of Use [37]. The WASSA empathy model [15] does not state a licence on its model card and is used for evaluation only, with credit to its authors.


| Asset | Role | Licence | Implication |
|---|---|---|---|
| EmpatheticDialogues | Dataset | CC BY-NC 4.0 | Attribution required; no commercial use of the data or derived commercial product |
| FacebookAI/roberta-base, roberta-large | d1 backbone | MIT | Permissive |
| microsoft/Phi-4-mini-instruct | Final d2 | MIT | Permissive |
| Qwen/Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct | Baselines / initial d2 | Apache-2.0 | Permissive, keep notice |
| google/gemma-3-4b-it | Bake-off candidate | Gemma Terms of Use | Use-restriction policy applies |
| google-bert/bert-base-uncased, google/electra-base-discriminator | d1 comparison | Apache-2.0 | Permissive |
| openai-community/gpt2-medium | Initial d1 | MIT | Permissive |
| sentence-transformers/all-MiniLM-L6-v2 | RAG retriever | Apache-2.0 | Permissive |
| hyy-33/hyy33-WASSA-2024-Track-2 | Empathy scorer | Not stated on model card | Evaluation use only; cite the shared task |
| openai/gpt-oss-120b | AI judge | Apache-2.0 | Accessed via Groq API terms |
| NVIDIA Nemotron 3 Ultra, Mistral Large | AI judges | Provider API terms | Only public ED text was sent |

*Table 3.1: Licences of the dataset and models used in this work*


## 3.4.5 Bias and Fairness

EmpatheticDialogues is an English-only corpus written by crowdworkers [1]. Its situations and ways of expressing emotion may not reflect users in Bangladesh or speakers of other languages, so neither detection accuracy nor the appropriateness of replies can be assumed to transfer to those users. The evaluation itself carries a known bias. The rule-based acknowledgement check searches for phrases such as "I'm sorry" and "that sounds", which suit negative emotions and under-credit valid responses to positive ones. We therefore never rely on this measure alone, and complement it with a blind panel of AI judges from different developers, an independent empathy classifier and a human study. The AI judges were chosen from developers unrelated to both the generator (Microsoft) and the baselines (Alibaba), because LLM evaluators tend to favour outputs from their own model family [34].

## 3.4.6 Research Integrity

Failed experiments were treated as failures, not as results. For example, a RoBERTa-large training run that diverged to 5.0% accuracy because learning-rate warmup was missing was repeated with corrected settings, and the failed run was not reported. Negative findings are reported in full, including those for retrieval-augmented generation, the monolithic fine-tuned generator, the Gemma-3 candidate, the larger RoBERTa-large detector, and the result that the final system's advantage in rule adherence over a same-size single model is not significant after correction for multiple comparisons. When a configuration error was found in the fine-tuned baseline, whose training script prevented the model from learning to end its replies, the baseline was retrained and every affected result was replaced, and the error is documented rather than hidden (Section 5.2.5). All reported figures are regenerated from saved experiment outputs by scripts kept in the project repository. [[TEAM TO FILL: disclosure of any use of AI writing or coding assistants, as required by BRAC University policy.]]

## References

[1] H. Rashkin, E. M. Smith, M. Li, and Y.-L. Boureau, "Towards empathetic open-domain conversation models: A new benchmark and dataset," in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), Florence, Italy, 2019, pp. 5370-5381, doi: 10.18653/v1/P19-1534.  
[3] Facebook Research, "EmpatheticDialogues," GitHub repository (CC BY-NC 4.0 licence). [Online]. Available: https://github.com/facebookresearch/EmpatheticDialogues  
[6] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.  
[7] A. Abouelenin, A. Ashfaq, A. Atkinson et al. (Microsoft), "Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs," arXiv:2503.01743, 2025.  
[8] A. Yang, B. Yang, B. Zhang, B. Hui et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.  
[15] H. Yang, L. Huang, T. Li, N. Rusnachenko, and H. Liang, "hyy33 at WASSA 2024 empathy and personality shared task: Using the CombinedLoss and FGM for enhancing BERT-based models in emotion and empathy prediction from conversation turns," in Proc. 14th Workshop Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA), 2024, pp. 430-434, doi: 10.18653/v1/2024.wassa-1.39.  
[34] A. Panickssery, S. R. Bowman, and S. Feng, "LLM evaluators recognize and favor their own generations," in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024; arXiv:2404.13076.  
[36] World Health Organization and International Association for Suicide Prevention, Preventing Suicide: A Resource for Media Professionals, Update 2023. Geneva, Switzerland: WHO, 2023. [Online]. Available: https://www.who.int/publications/i/item/9789240076846  
[37] Google, "Gemma terms of use." [Online]. Available: https://ai.google.dev/gemma/terms  
