# 5.4 Comparisons and Relationships

This section compares the results with published work on EmpatheticDialogues, where such comparisons are valid, and reports the relationships between variables that emerged from the experiments.

## 5.4.1 Emotion Recognition Compared with Published Work

Table 5.21 places our detector among published emotion-recognition results for the 32 classes of EmpatheticDialogues. Two differences must be taken into account when comparing them: **the text the system reads**, which may be a whole conversation, the explicit situation description, or only the speaker's first message; and **the data partition on which the result is reported**.

Our detector (50.3% top-1) exceeds all of the joint recognition-and-generation models listed, including MoEL [20], MIME [21], EmpDG [22], KEMP [24], CEM [2], CASE [25] and ESCM [26], at 30.6-41.2%, as well as personality-aware and related models at 45.4-48.3% [28]. It is on par with RoBERTa-GPT2 (51.5%) [48]. Two differences explain this gap. First, our detector is initialised from RoBERTa, which was pre-trained on about 160 GB of text [6], whereas joint models such as MoEL and CASE initialise their word representations from GloVe vectors and train their encoders on EmpatheticDialogues alone [20], [25]. RoBERTa-GPT2, which is built on pre-trained RoBERTa and GPT-2 models, is the only joint model in Table 5.21 that matches our accuracy, showing that pre-training, rather than access to the whole conversation, drives the higher accuracy. Second, our detector is trained and selected only for emotion classification, whereas the joint models learn emotion recognition as an auxiliary task and share their capacity with response generation. Our detector reads only the user's first message, whereas these models read the whole conversation, so **it achieves higher accuracy with less input**. Because these models were not re-trained under our setting, we describe this as a higher accuracy than that reported on the same dataset, not as a controlled comparison, and we do not claim that our detector is the most accurate on EmpatheticDialogues. Classifiers trained on situation descriptions, such as fastText (43%) and DeepMoji (48%) [1] and RoBERTa (65.9%) [49], read an easier and more explicitly emotional text and are not directly comparable; our own GPT-2 detector scored 56.9% on situation descriptions but only 48.2% on user messages.

The most directly comparable result is the RoBERTa classifier of Chirpy Cardinal [32], which is described as trained on the first utterance and reports **61.5% top-1 and 92% top-5** accuracy. Inspection of its released checkpoint shows that it is **RoBERTa-large**, and that its saved evaluation, whose per-class counts match the official validation partition, is a **validation-set** score. The fair counterpart in our work is our RoBERTa-large detector, which reaches **53.0% top-1 and 87.2% top-5 on the held-out test partition**, about 8 points lower. Differences in training settings and between validation and test evaluation may explain part of this gap, and its input could not be fully verified: the example message in its released code is the situation description of a test conversation rather than a first utterance. We did not pursue it further because, as shown in Section 5.4.3, detection accuracy does not carry through to reply quality.


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

## 5.4.7 Parameter and Resource Efficiency

Table 5.24 relates reply quality to the total number of parameters. With 3.96 billion parameters, 52.0% of those of the Qwen2.5-7B single model, **the final system reaches 97.5% of its overall AI-judge score** (8.68 against 8.90, a difference that is not significant after correction; Section 5.3.7) and 96.2% of its judge-rated empathy, although this empathy difference is significant. It retains 87.7% of the larger model's WASSA score and 72.9% of its rule adherence, and its emotion validation is 4.3 points higher, a difference that is not significant. Compared with the 1.54-billion-parameter Qwen2.5-1.5B single model, the additional 2.4 billion parameters bring significant gains in WASSA empathy, emotion validation and AI-judge empathy (Tables 5.13 and 5.19). The final system therefore occupies an efficient middle point: most of the larger model's rated quality at about half its size, with a detector that accounts for only about 3% of its parameters.

The detector shows the same pattern (Table 5.25). RoBERTa-large has 2.8 times as many parameters as RoBERTa-base, took 7.4 times as long to train and is 2.9 points more accurate on the test partition, yet substituting it into the final system improved neither rule adherence, WASSA empathy nor emotion validation (Section 5.3.3). **The smaller detector is therefore the parameter-efficient choice**: additional detector capacity buys accuracy that does not reach the user.



| System | Parameters (B) | All rules (%) | WASSA empathy | Emotion validation (%) | AI-judge empathy | AI-judge overall |
|---|---|---|---|---|---|---|
| Single model, same size: Qwen2.5-1.5B detects + responds | 1.54 | 51.3 | 4.84 | 25.7 | 7.15 | 8.24 |
| FINAL SYSTEM - Two-stage: RoBERTa top-3 -> Phi-4-mini picks + per-emotion prompt | 3.96 | 60.0 | 5.05 | 36.7 | 7.84 | 8.68 |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 7.62 | 82.3 | 5.76 | 32.3 | 8.15 | 8.9 |
| Final system as % of Qwen2.5-7B | 52.0% | 72.9% | 87.7% | 113.6% | 96.2% | 97.5% |

*Table 5.24: Reply quality against total parameters for the final system and the single models*



| Detector | Parameters (M) | Training time (min) | Top-1 on test (%) | Top-5 on test (%) | Final system: all rules (%) | Final system: WASSA | Final system: emotion validation (%) |
|---|---|---|---|---|---|---|---|
| roberta-base | 125 | 8.8 | 50.1 | 85 | 60 | 5.05 | 36.7 |
| roberta-large | 355 | 64.7 | 53 | 87.2 | 54.7 | 5.01 | 38 |

*Table 5.25: Detector size, detection accuracy and final-system reply quality for RoBERTa-base and RoBERTa-large*


**Resource use.** Parameter counts do not determine running cost on their own: the final system loads two models and calls its generator twice per reply. We therefore measured peak GPU memory and response time for the final system and both single models on the same 50 conversations, on the RTX 4060 used throughout, with the evaluation's decoding settings (Tables 5.26 and 5.27). **Compared with the Qwen2.5-7B single model, the final system needs 38% less peak GPU memory** (3.40 GB against 5.47 GB) **with no significant difference in response time** (4.17 s against 3.91 s per reply, 95% CI of the difference [-0.03, +0.56] s, Wilcoxon p = 0.10). Per generated token the two are practically identical (65.7 ms against 65.4 ms, p = 0.91); the small difference per reply reflects the final system's slightly longer replies. The additional steps of the decoupled design are inexpensive: the detector and cue extraction take 0.016 s and the stage-1 emotion selection 0.29 s, together about 7% of the time per reply. The Qwen2.5-1.5B single model, by contrast, is lighter (2.90 GB) and 2.6 times faster than the final system, so the final system is resource-efficient relative to the larger model it approaches in quality, not relative to the smallest baseline, whose replies are significantly less empathetic. These measurements come from a single GPU and 50 messages and should be read as indicative.


| System | Parameters (B) | Peak GPU memory (GB) | Mean s per reply | Median s per reply | ms per generated token | Mean reply tokens | Model load (s) | N messages |
|---|---|---|---|---|---|---|---|---|
| FINAL SYSTEM - Two-stage: RoBERTa top-3 -> Phi-4-mini picks + per-emotion prompt | 3.96 | 3.4 | 4.17 | 4.07 | 65.7 | 63.9 | 15.9 | 50 |
| Single model, same size: Qwen2.5-1.5B detects + responds | 1.54 | 2.9 | 1.58 | 1.49 | 32.7 | 48.8 | 5.4 | 50 |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 7.62 | 5.47 | 3.91 | 3.75 | 65.4 | 59.8 | 21.4 | 50 |

*Table 5.26: GPU memory and response time of the final system and the single models*



| Final system vs | Measure | N | Final | Baseline | Difference | 95% CI | Wilcoxon p | Significant? |
|---|---|---|---|---|---|---|---|---|
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | Seconds per reply | 50 | 4.17 | 3.91 | 0.27 | [-0.03, +0.56] | 0.1 | no |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | ms per generated token | 50 | 65.68 | 65.44 | 0.24 | [-1.88, +2.39] | 0.91 | no |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | Peak GPU memory (GB) | 1 | 3.4 | 5.47 | -2.07 | -37.9% relative |  | single measurement |
| Single model, same size: Qwen2.5-1.5B detects + responds | Seconds per reply | 50 | 4.17 | 1.58 | 2.6 | [+2.36, +2.83] | 0 | YES |
| Single model, same size: Qwen2.5-1.5B detects + responds | ms per generated token | 50 | 65.68 | 32.65 | 33.03 | [+31.04, +35.03] | 0 | YES |
| Single model, same size: Qwen2.5-1.5B detects + responds | Peak GPU memory (GB) | 1 | 3.4 | 2.9 | 0.5 | +17.1% relative |  | single measurement |

*Table 5.27: Paired comparison of response time and memory between the final system and each single model*


## 5.4.8 Validity of the Response Rules

Because rule adherence is one of our main measures, the rules themselves require justification: a system that follows our rules is only better if the rules describe something that matters. Table 5.28 gives the origin of each rule and its measured effect. The effects are computed **within each system**, by removing each system's own mean before comparing replies that follow the rule with replies that do not, so they cannot be explained by differences between systems.

**Acknowledging the feeling (R1)** corresponds to the *emotional reactions* mechanism of EPITOME, a framework of empathy in text-based mental-health support derived from established empathy scales [41]. In our data, replies containing an acknowledgement score 0.36 points higher on WASSA empathy within the same system (p < 0.001, 3,300 replies). The AI judges show the same direction but no significant effect (+0.11, p = 0.91).

**Asking exactly one question (R2)** balances two considerations. Questions are the *explorations* mechanism of EPITOME [41] and are the most frequent listener behaviour in EmpatheticDialogues [49], so a reply with no question closes the conversation; but asking several questions is the "question-answer trap" that counselling guidance warns reduces engagement [42], and only 4.7% of EmpatheticDialogues listener replies ask more than one. **Neither automatic measure rewards this rule**: WASSA is 0.20 points lower for replies with exactly one question and the judges show no effect. We therefore present R2 as a usability and safety requirement that prevents the interrogation failure mode, visible in the uncorrected fine-tuned baseline, rather than as a route to a higher empathy score.

**Staying within 120 words (R3)** keeps a reply readable for a distressed user. Here the two measures disagree. The AI judges rate replies within the limit 0.65 points higher on empathy within the same system (p = 0.006), whereas WASSA rates them 0.22 points lower, consistent with its preference for longer replies (Section 5.4.4). Because WASSA's length preference is a known weakness and the judges read the text itself, we treat the judges' result as the more informative of the two.

In summary, R1 is supported by both a published framework and our data, R3 is supported by the judges and by the needs of the use case, and R2 is supported by counselling practice and by how people actually reply in the dataset, but not by our automatic empathy measures. The human evaluation, which asks people which reply best understands the speaker, is the test that can settle R2, and it is still in progress.


| Rule | Where the rule comes from | Effect on WASSA empathy (within system) | Effect on AI-judge empathy (within system) |
|---|---|---|---|
| R1: acknowledge the feeling | Emotional reactions are one of the three empathy mechanisms of the EPITOME framework for text-based support (Sharma et al., 2020) | +0.37 (2459 vs 1141 replies) p = 1.65e-17 | +0.25 (366 vs 146 replies) p = 0.123 |
| R2: ask exactly one question | Explorations are an EPITOME mechanism, but counselling guidance warns against the question-answer trap of asking several questions (SAMHSA TIP 35, 2019); only 4.7% of EmpatheticDialogues listener replies ask more than one | -0.19 (2906 vs 694 replies) p = 0.00144 | -0.08 (409 vs 103 replies) p = 0.759 |
| R3: stay within 120 words | Keeps the reply readable for a distressed user and prevents the runaway replies seen in the fine-tuned baseline (design requirement) | -0.22 (3319 vs 281 replies) p = 0.00796 | +0.64 (492 vs 20 replies) p = 0.00861 |

*Table 5.28: Origin of each response rule and its measured effect within systems*


Table 5.29 sets the two systems side by side on every measure we collect, and adds a length control, because the single model writes 29% longer replies (73.5 against 57.1 words) and both the AI judges and the acknowledgement check reward length (Sections 5.4.4 and 5.4.8).

**What the decoupled design wins.** Its replies score higher on the WASSA empathy classifier (5.05 against 4.75) and recover the user's true emotion more often (36.7% against 27.0%). The WASSA advantage is the more robust of the two: it holds *despite* shorter replies, and it becomes stronger, not weaker, when length is held constant (p < 0.001), whereas the emotion-validation advantage weakens under that control (p = 0.13). The decoupled system also repeats itself less, with a boilerplate rate of 58.7% against 72.3% and higher lexical diversity (distinct-2 47.8% against 39.1%), which matters because repetitiveness is the most frequently reported weakness of LLM empathy [29]. It reaches these results while generating 22% fewer words per reply, and it exposes the emotion it used, which can be logged, audited or intercepted before generation.

**What the single model wins, and what remains of it.** The single model follows all three rules more often (73.7% against 60.0%) and acknowledges the feeling more often (77.3% against 66.3%). Both advantages are largely explained by reply length: with length held constant, the acknowledgement difference disappears entirely (p = 0.36) and the rule-adherence difference shrinks to the edge of significance (p = 0.04). The remaining part is a real cost of the design: in the decoupled system, the three rules share the prompt with emotion-specific guidance and cue words, so the generator follows them slightly less closely.

**Where the two are equal.** The AI judges do not separate the systems on empathy (7.84 against 8.04, adjusted p = 0.077) or overall quality (8.68 against 8.78, p = 0.16), and at matched reply length their scores coincide almost exactly (7.92 against 7.89 for replies of 40 to 60 words, 7.83 against 7.89 for 60 to 80 words).

**Conclusion.** Against its own generator used alone, the decoupled design produces replies that express the user's emotion more accurately and are rated as more empathetic by an independent classifier, that repeat themselves less and are shorter, and whose emotional reasoning is inspectable; it pays for this with slightly lower adherence to the response rules. For a system whose purpose is to make someone feel understood, and whose emotional decision must be auditable, that is a favourable exchange, but it is an exchange rather than a clean win, and we report it as such.


| Measure | Decoupled (ours) | Same generator alone | Difference | 95% CI | Test | p | p, length held constant | Verdict |
|---|---|---|---|---|---|---|---|---|
| WASSA empathy (0-10) | 5.05 | 4.75 | 0.3 | [+0.19, +0.41] | Wilcoxon signed-rank | 0.0 | 0.0 | decoupled better |
| Emotion validation (%) | 36.67 | 27.0 | 9.67 | [+3.67, +15.33] | McNemar exact | 0.00169 | 0.1335 | decoupled better |
| AI-judge empathy (1-10) | 7.84 | 8.04 | -0.2 | [-0.45, +0.09] | Wilcoxon signed-rank | 0.07675 | 0.5539 | no difference |
| AI-judge overall (1-10) | 8.68 | 8.78 | -0.1 | [-0.32, +0.17] | Wilcoxon signed-rank | 0.16262 | 0.4855 | no difference |
| All three rules (%) | 60.0 | 73.67 | -13.67 | [-21.00, -6.33] | McNemar exact | 0.00048 | 0.0405 | single model better |
| Acknowledgement (%) | 66.33 | 77.33 | -11.0 | [-18.00, -4.00] | McNemar exact | 0.00269 | 0.3584 | single model better |
| Exactly one question (%) | 92.67 | 96.0 | -3.33 | [-7.00, +0.33] | McNemar exact | 0.11018 | 0.6265 | no difference |
| Reply length (words) | 57.11 | 73.48 | -16.37 | [-18.77, -14.01] | Wilcoxon signed-rank | 0.0 | 0.1056 | shorter replies by design |
| Boilerplate rate (%) | 58.7 | 72.3 | -13.6 | corpus-level measure | - | - | - | decoupled better |
| Distinct-2 (%) | 47.84 | 39.12 | 8.72 | corpus-level measure | - | - | - | decoupled better |
| Emotion used is visible and can be intercepted | yes | no | - | - | - | - | - | decoupled better |

*Table 5.29: The decoupled system against the same generator used alone, measure by measure, with reply length held constant*



## 5.4.9 Separating the Effect of the Detector from the Effect of the Generator

The comparisons in Section 5.3.2 hold the conversations and decoding fixed but not the generator, because the final system's generator (Phi-4-mini) is larger than the 1.5-billion-parameter single model. To attribute the results correctly, we ran a further baseline in which **the final system's own generator is given the single-model prompt with no detector at all**, so that the only difference from the final system is the decoupled design.

The result separates the two effects cleanly. **Model capacity explains the structural gains, and the decoupled design explains the empathic gains.** Moving from Qwen2.5-1.5B to Phi-4-mini as a single model raises rule adherence from 51.3% to 73.7% (+22.3 points, p < 0.001) but leaves empathy and emotional fidelity essentially unchanged: WASSA moves from 4.84 to 4.75 (p = 0.18) and emotion validation from 25.7% to 27.0% (p = 0.70). In other words, **a larger generator alone buys none of the empathy we report**. Adding the detector, the top-3 candidates and the emotion-specific guidance to that same generator raises **WASSA from 4.75 to 5.05 (+0.30, adjusted p < 0.001) and emotion validation from 27.0% to 36.7% (+9.7 points, adjusted p = 0.007)**. Both gains are therefore caused by the decoupled design, not by the size of the generator, which answers the question the Pre-Thesis 2 panel raised about whether the second model earns its place.

The same comparison shows the cost of the design. With the generator held fixed, the decoupled system follows all three rules less often than the single model (60.0% against 73.7%, adjusted p = 0.002), mainly through the acknowledgement check (66.3% against 77.3%, p = 0.013); the one-question rule does not differ (92.7% against 96.0%, p = 0.22) and every reply stays within the length limit. The prompt is the likely cause: in the decoupled system, the three rules share space with emotion-specific guidance and cue words, whereas the single-model prompt contains the rules alone. Two qualifications should be kept in mind. The acknowledgement check is keyword-based (Section 5.4.8), so replies that validate a feeling in wording outside the keyword list are not counted, and rule adherence measures the *form* of a reply, whereas WASSA and emotion validation measure what the reply expresses.

The AI judges, which read the replies themselves, place the two systems level: the single model scores 8.04 for empathy and 8.78 overall against 7.84 and 8.68 for the decoupled system, and neither difference is significant after correction (p = 0.077 and p = 0.16). The evidence at an equal generator is therefore mixed in the way one would expect from measures that look at different things: the decoupled system wins on the emotion the reply expresses (emotion validation, +9.7 points) and on the empathy classifier (+0.30), the single model wins on rule adherence (+13.7 points), and the judges see no reliable difference.

**In summary, at an equal generator the decoupled design is the better configuration for the goal of this thesis**, producing measurably more emotionally faithful replies and higher classifier-rated empathy, with the emotional decision exposed for inspection, while the judges rate the two configurations equally. It trades part of its rule adherence for that gain, and we report the trade-off rather than claiming an advantage on every measure.



## References

[1] H. Rashkin, E. M. Smith, M. Li, and Y.-L. Boureau, "Towards empathetic open-domain conversation models: A new benchmark and dataset," in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL), Florence, Italy, 2019, pp. 5370-5381, doi: 10.18653/v1/P19-1534.  
[2] S. Sabour, C. Zheng, and M. Huang, "CEM: Commonsense-aware empathetic response generation," in Proc. AAAI Conf. Artificial Intelligence, vol. 36, no. 10, 2022, pp. 11229-11237.  
[6] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.  
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
[41] A. Sharma, A. S. Miner, D. C. Atkins, and T. Althoff, "A computational approach to understanding empathy expressed in text-based mental health support," in Proc. 2020 Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 5263-5276, doi: 10.18653/v1/2020.emnlp-main.425.  
[42] Substance Abuse and Mental Health Services Administration, Enhancing Motivation for Change in Substance Use Disorder Treatment, Treatment Improvement Protocol (TIP) Series No. 35, SAMHSA Publication No. PEP19-02-01-003. Rockville, MD, USA: SAMHSA, 2019, ch. 3.  
[48] Y. Liu, W. Maier, W. Minker, and S. Ultes, "Empathetic dialogue generation with pre-trained RoBERTa-GPT2 and external knowledge," in Proc. Int. Workshop Spoken Dialogue System Technology (IWSDS), 2021; arXiv:2109.03004.  
[49] A. Welivita and P. Pu, "A taxonomy of empathetic response intents in human social conversations," in Proc. 28th Int. Conf. Computational Linguistics (COLING), 2020, pp. 4886-4899, doi: 10.18653/v1/2020.coling-main.429.  
