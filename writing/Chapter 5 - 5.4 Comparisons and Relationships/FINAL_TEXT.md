# 5.4 Comparisons and Relationships

This section compares the results with published work on EmpatheticDialogues, where such comparisons are valid, and reports the relationships between variables that emerged from the experiments.

## 5.4.1 Emotion Recognition Compared with Published Work

Table 5.21 places our detector among published emotion-recognition results for the 32 classes of EmpatheticDialogues. Two differences must be taken into account when comparing them: **the text the system reads**, which may be a whole conversation, the explicit situation description, or only the speaker's first message; and **the data partition on which the result is reported**.

Our detector (50.3% top-1) exceeds all of the joint recognition-and-generation models listed, including MoEL [20], MIME [21], EmpDG [22], KEMP [24], CEM [2], CASE [25] and ESCM [26], at 30.6-41.2%, as well as personality-aware and related models at 45.4-48.3% [28]. It is on par with RoBERTa-GPT2 (51.5%) [45]. These models read the whole conversation but learn emotion recognition as an auxiliary task of generation, so the gap reflects the benefit of a dedicated classifier. Classifiers trained on situation descriptions, such as fastText (43%) and DeepMoji (48%) [1] and RoBERTa (65.9%) [46], read an easier and more explicitly emotional text and are not directly comparable; our own GPT-2 detector scored 56.9% on situation descriptions but only 48.2% on user messages.

The most directly comparable result is the RoBERTa classifier of Chirpy Cardinal [32], which was trained on the same input, the first utterance, and reports **61.5% top-1 and 92% top-5** accuracy. Inspection of its released checkpoint shows that it is **RoBERTa-large**, and that its saved evaluation, whose per-class counts match the official validation partition, is a **validation-set** score. The fair counterpart in our work is our RoBERTa-large detector, which reaches **53.0% top-1 and 87.2% top-5 on the held-out test partition**, about 8 points lower. Differences in training settings and between validation and test evaluation may explain part of this gap. We did not pursue it further because, as shown in Section 5.4.3, detection accuracy does not carry through to reply quality.


| System | Input and split | Top-1 (%) | Top-5 (%) | Source |
|---|---|---|---|---|
| EmpDG (Li et al., 2020) | Joint model, full conversation | 30.64 |  | Qian et al. 2023, Table 1 |
| MoEL (Lin et al., 2019) | Joint model, full conversation | 30.75 |  | Qian et al. 2023, Table 1 |
| MIME (Majumder et al., 2020) | Joint model, full conversation | 31.21 |  | Qian et al. 2023, Table 1 |
| KEMP (Li et al., 2022) | Joint model, full conversation | 36.57 |  | Qian et al. 2023, Table 1 |
| CEM (Sabour et al., 2022) | Joint model, full conversation | 37.81 |  | Qian et al. 2023, Table 1 |
| CASE (Zhou et al., 2023) | Joint model, full conversation | 38.92 |  | Qian et al. 2023, Table 1 |
| ESCM (Yang et al., 2023) | Joint model, full conversation | 41.19 |  | Findings of EMNLP 2023 |
| Personality-aware model (Wu et al., 2025) | Joint model, text-only ED | 45.4 |  | COLING 2025 main.598, Table 4 |
| EmpSOA | Joint model, text-only ED | 48.3 |  | COLING 2025 main.598, Table 4 |
| fastText (Rashkin et al., 2019) | Classifier, situation description | 43 |  | ED paper |
| DeepMoji (Rashkin et al., 2019) | Classifier, situation description | 48 |  | ED paper |
| RoBERTa-GPT2 (2021) | Joint model, full conversation | 51.51 |  | arXiv 2109.03004, Table 3 |
| Chirpy Cardinal RoBERTa-large (Paranjape et al., 2020) | Classifier, first utterance, VALIDATION split | 61.5 | 92 | arXiv 2008.12348, App. D; checkpoint verified as roberta-large, official split |
| RoBERTa (Welivita and Pu, 2020) | Classifier, situation description | 65.88 |  | arXiv 2012.04080 |
| OURS - RoBERTa-base d1 | Classifier, first utterance, TEST split | 50.3 | 84.3 | this work |
| OURS - RoBERTa-large (not deployed) | Classifier, first utterance, TEST split | 53 | 87.2 | this work |

*Table 5.21: Emotion recognition accuracy on EmpatheticDialogues reported in published work and in this thesis*


## 5.4.2 Response Generation Compared with Published Work

Most automatic generation measures reported for EmpatheticDialogues are **not comparable** with ours. Published distinct-n scores are computed over the full test set, whereas ours are computed over 300 replies, and distinct-n falls as the sample grows; on our own output it drops from 32.3% at 50 replies to 15.6% at 300. Published BERTScore values compare a reply with the reference response after a multi-turn context, whereas our systems respond to the first message only. We therefore do not rank our system against published automatic scores. Instead, our evaluation measures properties that published work does not report: explicit rule adherence, an independent empathy classifier, emotion validation and paired significance tests.

Human evaluation allows a comparison of direction rather than magnitude. In the literature, human raters strongly prefer LLM replies to those of trained empathetic models, rating ChatGPT at 4.64 out of 5 against 3.08 for CEM [23], and fine-tuned tool-augmented LLMs win 61-72% of A/B empathy comparisons against models fine-tuned without tools [27] (Table 5.22). [[PENDING: the human-evaluation selection rate of the final system against the same-size and larger single models, and how its direction compares with these findings.]] Because the protocols differ, using five-point ratings, A/B preferences and, in our study, a three-way choice, only the direction of the results can be compared.


| System | Human evaluation protocol | Empathy result | Source |
|---|---|---|---|
| CEM (Sabour et al., 2022) | 1-5 Likert, crowd | 3.08 | Qian et al. 2023, Table 2 |
| CASE (Zhou et al., 2023) | 1-5 Likert, crowd | 3.04 | Qian et al. 2023, Table 2 |
| KEMP (Li et al., 2022) | 1-5 Likert, crowd | 2.83 | Qian et al. 2023, Table 2 |
| ChatGPT (Qian et al., 2023) | 1-5 Likert, crowd | 4.64 | Qian et al. 2023, Table 2 |
| TOOL-ED Qwen1.5-14B (Cao et al., 2025) | A/B vs same model LoRA-tuned on ED, 3 researchers | 61.3% win / 32.7% lose / 6.0% tie | COLING 2025 main.355, Table 2 (human) |
| OURS - final system | 3-way forced choice, students and faculty | [[PENDING - human evaluation]] | this work |

*Table 5.22: Human evaluation of empathy in published work and in this thesis*


## 5.4.3 Relationship between Detection Accuracy and Reply Quality

In this system, detection accuracy and reply quality are only weakly related. In a controlled comparison in which the generator (Qwen2.5-1.5B), the prompt and the conversations are held fixed and only the detector changes, from GPT-2 (48.2% on real input) to RoBERTa (50.3%), neither emotion validation (26.0% against 25.3%, p = 0.91) nor rule adherence (63.7% against 57.7%, p = 0.09) improves (Table 5.23). Within a single pipeline, replies built on a wrong emotion label follow the rules as often as those built on a correct label, and are scored as equally empathetic (Table 5.14). Replies are robust to label errors partly because most errors occur between neighbouring emotions (Section 5.4.6), for which appropriate responses are similar. By contrast, **how the emotion is used** does matter: with the same detector and generator, emotion-specific two-stage prompting raises emotion validation by 8.3 points. Conditioning well on a small detector's output matters more than obtaining additional detection accuracy.

The same relationship holds in the final system. Replacing RoBERTa-base with RoBERTa-large raised detector accuracy on the evaluation conversations from 50.3% to 53.3%, but the emotion chosen in stage 1 changed in only 31% of conversations, and rule adherence, WASSA empathy and emotion validation did not change significantly (Table 5.16). Across three independent tests, namely label errors, a GPT-2 to RoBERTa swap and a RoBERTa-base to RoBERTa-large swap, gains of two to three points in detection accuracy never translated into better replies.


| Metric | RoBERTa d1 + Qwen (%) | GPT-2 d1 + Qwen (%) | Difference (pp) | 95% CI (pp) | McNemar p | Significant? |
|---|---|---|---|---|---|---|
| Emotion validation | 25.3 | 26 | -0.7 | [-6.3, +5.0] | 0.91 | no |
| All rules | 57.7 | 63.7 | -6 | [-12.7, +0.7] | 0.09 | no |

*Table 5.23: Effect of a more accurate detector with the generator held fixed*


## 5.4.4 Relationship between Reply Length and Measured Empathy

WASSA empathy increases with reply length within every system, with Spearman correlations from 0.14 (top-1 Phi-4-mini) to 0.54 (zero-shot prompting) (Table 5.17). Measured empathy is therefore partly a function of verbosity, in line with the observation that longer LLM responses may be mistaken for greater empathy [29]. Gemma-3-4B illustrates why this matters. It has the second-highest WASSA score overall (5.30), but when replies of similar length are compared it does not outperform Phi-4-mini: in every length band in which both have replies, Phi-4-mini matches or exceeds it, and Gemma's advantage appears only above 120 words (Figure 5.3). The final system reverses the usual pattern, achieving a higher WASSA score than top-1 Phi-4-mini prompting while writing about 30% fewer words. Empathy classifiers and LLM judges, which are also known to favour longer answers [13], should therefore be interpreted together with reply length.

## 5.4.5 Relationship between Generator Capacity and Prompt Complexity

Richer prompts help only generators with sufficient capacity to follow them. With Qwen2.5-1.5B, adding the stage-1 selection, emotion-specific guidance and cue words *lowers* rule adherence from 57.7% to 36.7% and acknowledgement from 67.7% to 47.0%. The same additions with Phi-4-mini, which has 3.8 billion parameters, leave rule adherence statistically unchanged while improving empathy and emotion validation. The earlier candidate-pick experiment showed the same pattern for the 1.5-billion-parameter model: asking a single call both to choose the emotion and to respond reduced adherence from 63.7% to 54.0% (Table 4.7). Across model sizes, rule adherence under the single-model prompt rises from 51.3% at 1.5 billion parameters to 82.3% at 7.6 billion. These results suggest that the complexity of instructions should increase with the capacity of the generator, and that for models below a few billion parameters a simpler prompt is the safer choice.

## 5.4.6 Relationship between Emotion Similarity and Detection Errors

Detection errors are concentrated among emotions that differ in intensity or nuance rather than in kind: *sentimental* is read as *nostalgic*, *afraid* as *terrified*, *angry* as *furious*, *sad* as *devastated*, and *apprehensive* as *anxious* (Table 5.4), a pattern also reported for a comparable classifier [32]. The weakest class, *angry* (F1 18.2%), is largely absorbed by its more intense neighbour *furious*. Because the correct label is usually a near neighbour, it lies within the three most probable candidates in 76.5% of cases, against 50.3% for the single most probable. This is the relationship the two-stage design exploits, and it helps to explain why label errors cause little damage to the final replies.

## References

[1] H. Rashkin, E. M. Smith, M. Li, and Y.-L. Boureau, "Towards empathetic open-domain conversation models: A new benchmark and dataset," in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), Florence, Italy, 2019, pp. 5370-5381, doi: 10.18653/v1/P19-1534.  
[2] S. Sabour, C. Zheng, and M. Huang, "CEM: Commonsense-aware empathetic response generation," in Proc. AAAI Conf. Artificial Intelligence, vol. 36, no. 10, 2022, pp. 11229-11237.  
[13] L. Zheng, W.-L. Chiang, Y. Sheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, vol. 36, 2023; arXiv:2306.05685.  
[20] Z. Lin, A. Madotto, J. Shin, P. Xu, and P. Fung, "MoEL: Mixture of empathetic listeners," in Proc. 2019 Conf. Empirical Methods Natural Language Processing and 9th Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 121-132, doi: 10.18653/v1/D19-1012.  
[21] N. Majumder, P. Hong, S. Peng, J. Lu, D. Ghosal, A. Gelbukh, R. Mihalcea, and S. Poria, "MIME: MIMicking emotions for empathetic response generation," in Proc. 2020 Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 8968-8979, doi: 10.18653/v1/2020.emnlp-main.721.  
[22] Q. Li, H. Chen, Z. Ren, P. Ren, Z. Tu, and Z. Chen, "EmpDG: Multi-resolution interactive empathetic dialogue generation," in Proc. 28th Int. Conf. Computational Linguistics (COLING), 2020, pp. 4454-4466, doi: 10.18653/v1/2020.coling-main.394.  
[23] Y. Qian, W.-N. Zhang, and T. Liu, "Harnessing the power of large language models for empathetic response generation: Empirical investigations and improvements," in Findings Assoc. Comput. Linguistics: EMNLP 2023, 2023; arXiv:2310.05140.  
[24] Q. Li, P. Li, Z. Ren, P. Ren, and Z. Chen, "Knowledge bridging for empathetic dialogue generation," in Proc. AAAI Conf. Artificial Intelligence, vol. 36, no. 10, 2022, pp. 10993-11001.  
[25] J. Zhou, C. Zheng, B. Wang, Z. Zhang, and M. Huang, "CASE: Aligning coarse-to-fine cognition and affection for empathetic response generation," in Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (ACL), vol. 1, 2023, pp. 8223-8237, doi: 10.18653/v1/2023.acl-long.457.  
[26] Z. Yang, Z. Ren, Y. Wang, X. Zhu, Z. Chen, T. Cai, Y. Wu, Y. Su, S. Ju, and X. Liao, "Exploiting emotion-semantic correlations for empathetic response generation," in Findings Assoc. Comput. Linguistics: EMNLP 2023, 2023, pp. 4826-4837, doi: 10.18653/v1/2023.findings-emnlp.320.  
[27] H. Cao, Y. Zhang, S. Feng, X. Yang, D. Wang, and Y. Zhang, "TOOL-ED: Enhancing empathetic response generation with the tool calling capability of LLM," in Proc. 31st Int. Conf. Computational Linguistics (COLING), Abu Dhabi, UAE, 2025, pp. 5305-5320.  
[28] J. Wu, X. Huang, Z. Zhu, and S. Wang, "From traits to empathy: Personality-aware multimodal empathetic response generation," in Proc. 31st Int. Conf. Computational Linguistics (COLING), Abu Dhabi, UAE, 2025, pp. 8925-8938.  
[29] V. Sorin, D. Brin, Y. Barash, E. Konen, A. Charney, G. Nadkarni, and E. Klang, "Large language models and empathy: Systematic review," J. Med. Internet Res., vol. 26, p. e52597, 2024, doi: 10.2196/52597.  
[32] A. Paranjape, A. See, K. Kenealy, H. Li, A. Hardy, P. Qi, K. R. Sadagopan, N. M. Phu, D. Soylu, and C. D. Manning, "Neural generation meets real people: Towards emotionally engaging mixed-initiative conversations," in 3rd Proc. Alexa Prize (Alexa Prize 2019), 2020; arXiv:2008.12348.  
[45] Y. Liu, W. Maier, W. Minker, and S. Ultes, "Empathetic dialogue generation with pre-trained RoBERTa-GPT2 and external knowledge," in Proc. Int. Workshop Spoken Dialogue System Technology (IWSDS), 2021; arXiv:2109.03004.  
[46] A. Welivita and P. Pu, "A taxonomy of empathetic response intents in human social conversations," in Proc. 28th Int. Conf. Computational Linguistics (COLING), 2020, pp. 4886-4899, doi: 10.18653/v1/2020.coling-main.429.  
