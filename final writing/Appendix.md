# Appendix

## Tools and Technologies Used


The system was implemented in Python 3.10 using PyTorch 2.5.1 and Hugging Face Transformers 5.16.1. Transformers provides pre-trained encoders, namely RoBERTa, BERT and ELECTRA, and instruction-tuned decoders, namely Phi-4-mini, Qwen2.5 and Gemma-3, behind a common interface, which allowed design alternatives to be exchanged without rewriting code. The bitsandbytes library provided 4-bit NF4 quantisation, allowing generators of 3.8 to 7.6 billion parameters to run on a single NVIDIA GeForce RTX 4060 with 8 GB of memory, and the PEFT library provided LoRA for the monolithic fine-tuning baseline.

Evaluation combined SciPy and scikit-learn for paired statistical tests and classification metrics, sentence-transformers for the retrieval baseline, the WASSA 2024 empathy classifier, and two AI judges accessed through the OpenRouter and Groq APIs, selected from different developers to avoid self-preference bias. A Gradio interface was prepared for demonstration on Hugging Face Spaces, the human study was conducted with Google Forms, and all tables and figures in this report were generated from saved experiment outputs by scripts using pandas and matplotlib, so that every reported number can be reproduced.

## A. Supplementary Tables

These tables support statements in the main text and are placed here to keep the chapters readable. Each caption states where the result is used.


| Criterion | How it was applied | Evidence or reason |
|---|---|---|
| Flagship-tier capability | Required by the co-supervisor; small local judges (Gemma-2-2B, Phi-3-mini) were dropped from an earlier plan | Zheng et al. (2023): strong LLM judges agree with human preferences over 80% of the time |
| Different developers, none related to our models | No judge shares a family with the generator (Phi, Microsoft) or the baselines (Qwen, Alibaba) | Panickssery et al. (2024): LLM evaluators recognise and favour their own generations |
| A panel instead of one judge | Scores averaged over the two judges that scored every system | Verga et al. (2024): a panel from disjoint model families shows less intra-model bias than one large judge |
| Same judges for every system | Both judges scored all 512 replies of all judged systems | Mixing judge sets across systems would make the scores incomparable |
| Blind scoring | Judges see only the user message and one reply - never the system name or the gold reply | Removes identity bias and any reward for word overlap with a reference reply |
| Anchored rubric, deterministic output | 1-3 / 4-6 / 7-10 anchors for empathy, relevance and safety; temperature 0; JSON-only answer | Anchors keep judges on the same scale; temperature 0 removes sampling noise |
| Free API access | No GPU needed, so non-coding teammates could run the judging | Project resource constraint (Section 3.8) |
| Documented replacements | Gemini (repeated quota failures), DeepSeek (retired on GitHub Models, removed from OpenRouter free tier), Llama 3.3 70B (decommissioned on Groq), Mistral Large (subscription tier) | See risk R7 in Section 3.7 |

*Table A.1: Criteria for selecting the AI judges (selection criteria behind Section 4.1.5.1).*


| Emotion | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---|---|---|---|
| caring | 71.4 | 68.8 | 70.1 | 80 |
| lonely | 65.2 | 74.4 | 69.5 | 78 |
| jealous | 80 | 54.3 | 64.7 | 81 |
| embarrassed | 69 | 60.5 | 64.5 | 81 |
| disgusted | 66.7 | 58.1 | 62.1 | 86 |
| nostalgic | 55 | 71.4 | 62.1 | 77 |
| surprised | 60.9 | 61.4 | 61.2 | 127 |
| guilty | 54.4 | 65.2 | 59.3 | 66 |
| content | 60.6 | 56.6 | 58.5 | 76 |
| prepared | 62.7 | 54.5 | 58.3 | 77 |
| impressed | 54.9 | 61.7 | 58.1 | 81 |
| proud | 51.2 | 65.6 | 57.5 | 96 |
| faithful | 53.4 | 62 | 57.4 | 50 |
| trusting | 54.4 | 57.8 | 56.1 | 64 |
| grateful | 54.7 | 54.7 | 54.7 | 95 |
| annoyed | 52.7 | 53.8 | 53.3 | 91 |
| devastated | 40.4 | 68.7 | 50.8 | 67 |
| confident | 41.2 | 56 | 47.5 | 75 |
| hopeful | 46.8 | 45.6 | 46.2 | 79 |
| excited | 42.7 | 48.4 | 45.4 | 91 |
| furious | 44.3 | 46.3 | 45.3 | 67 |
| terrified | 37.6 | 53.5 | 44.2 | 71 |
| disappointed | 44.9 | 43.2 | 44 | 81 |
| sentimental | 60.8 | 34.1 | 43.7 | 91 |
| anxious | 39.1 | 43.6 | 41.2 | 78 |
| sad | 42.3 | 34.1 | 37.7 | 88 |
| anticipating | 35.5 | 30.1 | 32.6 | 73 |
| afraid | 33.8 | 30.1 | 31.9 | 73 |
| apprehensive | 38 | 26.8 | 31.4 | 71 |
| joyful | 31.6 | 30.1 | 30.9 | 83 |
| ashamed | 27.3 | 18.8 | 22.2 | 64 |
| angry | 25 | 14.3 | 18.2 | 84 |

*Table A.2: Precision, recall and F1 of the final detector for each emotion (32 rows of per-emotion detail behind Section 5.1.2).*


| Item | Emotion | Scored by the AI judges | Situation type |
|---|---|---|---|
| 1 | anxious | True | core supportive emotion |
| 2 | anticipating | False | student situation |
| 3 | content | True | positive emotion |
| 4 | prepared | False | student situation |
| 5 | lonely | True | core supportive emotion |
| 6 | angry | True | core supportive emotion |
| 7 | furious | False | student situation |
| 8 | sad | True | core supportive emotion |

*Table A.3: Design of the human-evaluation items (item-by-item design of the human study described in Section 5.1.9).*


| Length band (words) | Gemma WASSA | Gemma n | Phi (top-1) WASSA | Phi (top-1) n | Final two-stage Phi WASSA | Final two-stage Phi n |
|---|---|---|---|---|---|---|
| <60 |  | 0 | 4.73 | 70 | 4.84 | 197 |
| 60-80 |  | 0 | 4.86 | 94 | 5.47 | 93 |
| 80-100 | 4.71 | 7 | 4.9 | 87 | 5.43 | 7 |
| 100-120 | 4.97 | 76 | 5.24 | 37 | 5 | 3 |
| 120-150 | 5.37 | 163 | 5 | 7 |  | 0 |
| 150+ | 5.63 | 54 | 5 | 5 |  | 0 |

*Table A.4: WASSA empathy score within each reply-length band (length-band detail behind Section 5.2.3).*


| comparison | n | pipeline_mean | baseline_mean | diff | ci_95_low | ci_95_high | test | p_value | significant | metric | question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| roberta-base vs google__electra-base-discriminator | 2542 | 0.5 | 0.5 | -0 | -0.02 | 0.01 | McNemar exact | 0.96 | False | top1 | Why RoBERTa over ELECTRA? (same size class) |
| roberta-base vs google__electra-base-discriminator | 2542 | 0.85 | 0.83 | 0.02 | 0.01 | 0.04 | McNemar exact | 0 | True | top5 | Why RoBERTa over ELECTRA? (same size class) |
| roberta-base vs bert-base-uncased | 2542 | 0.5 | 0.49 | 0.01 | -0 | 0.03 | McNemar exact | 0.11 | False | top1 | Why RoBERTa over BERT? (same size class) |
| roberta-base vs bert-base-uncased | 2542 | 0.85 | 0.82 | 0.03 | 0.02 | 0.05 | McNemar exact | 0 | True | top5 | Why RoBERTa over BERT? (same size class) |
| roberta-large vs roberta-base | 2542 | 0.53 | 0.5 | 0.03 | 0.01 | 0.04 | McNemar exact | 0 | True | top1 | Is the larger model's advantage real? (why not bigger) |
| roberta-large vs roberta-base | 2542 | 0.87 | 0.85 | 0.02 | 0.01 | 0.03 | McNemar exact | 0 | True | top5 | Is the larger model's advantage real? (why not bigger) |

*Table A.5: Paired significance tests between detector models (significance tests behind the detector comparison in Section 4.2.2).*


| System | Mean words | Length OK (%) | WASSA mean | Spearman rho (length vs WASSA) | p |
|---|---|---|---|---|---|
| Zero-shot Qwen2.5-1.5B (no detector, generic prompt) | 75.9 | 83.7 | 4.25 | 0.54 | 0 |
| Monolithic LoRA fine-tuned Qwen2.5-1.5B (no detector) | 10 | 100 | 3.36 | 0.57 | 0 |
| Single model, same size: Qwen2.5-1.5B detects + responds | 42 | 100 | 4.84 | 0.2 | 0 |
| Single model, same generator: Phi-4-mini (4-bit) detects + responds | 73.5 | 99.3 | 4.75 | 0.1 | 0.09 |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | 52.9 | 100 | 5.76 | 0.19 | 0 |
| Initial pipeline: GPT-2 Medium d1 + Qwen2.5-1.5B d2 | 35.2 | 100 | 4.8 | 0.33 | 0 |
| Initial pipeline + retrieval (RAG) | 34 | 100 | 4.85 | 0.24 | 0 |
| RoBERTa d1 + Qwen2.5-1.5B d2 (top-1 emotion) | 34.7 | 100 | 4.78 | 0.29 | 0 |
| RoBERTa d1 + Gemma-3-4B d2 (top-1 emotion) | 133.3 | 27.7 | 5.3 | 0.31 | 0 |
| RoBERTa d1 + Phi-4-mini d2 (top-1 emotion) | 80.4 | 96 | 4.89 | 0.14 | 0.02 |
| Two-stage: RoBERTa top-3 -> Qwen2.5-1.5B picks + per-emotion prompt | 44.1 | 99.7 | 4.76 | 0.21 | 0 |
| FINAL SYSTEM - Two-stage: RoBERTa top-3 -> Phi-4-mini picks + per-emotion prompt | 57.1 | 100 | 5.05 | 0.26 | 0 |

*Table A.6: Correlation between reply length and WASSA empathy score within each system (per-system correlations behind Section 5.4.5).*


| Aspect | Check | Result | Source |
|---|---|---|---|
| Recognition | Where it comes from | WASSA 2024 shared task at ACL 2024 (14 teams); system paper hyy33, ACL Anthology 2024.wassa-1.39 | Findings paper 2024.wassa-1.30 |
| Recognition | What it was trained on | Third-party human empathy ratings (ordinal 1-5) of individual conversation turns (Track 2, CONV-Turn) | Findings paper |
| Recognition | Accuracy against humans on the task test set | Pearson r = .544 for empathy; 5th of 9 teams on the track average in the findings table (the authors report 4th) | Findings paper, Table 3 |
| Recognition | Caveat | The organisers' RoBERTa-base baseline reached r = .694 on empathy; the best team reached .582 | Findings paper, Table 3 |
| Recognition | Output format | 11 ordinal classes (argmax); our replies receive classes 1-7; the authors do not document how ratings map to classes | Checkpoint metadata; our scores |
| Our data | System-level ranking vs AI-judge panel (8 systems) | Spearman rho = 0.74 | computed |
| Our data | Per-reply agreement with the judges' mean empathy | rho = 0.30 (p = 2.0e-12, N = 512) | computed |
| Our data | Same, controlling for reply length | partial rho = 0.25 (p = 1.7e-08) | computed |
| Our data | Agreement within a single system | rho = 0.05 to 0.41; 4 of 8 significant | computed |
| Our data | For reference: judge vs judge on empathy | rho = 0.69 | computed |
| Reproducibility | Re-implementation check after a library upgrade | 1,800 of 1,800 pre-upgrade scores reproduced exactly | `llm_judge_evaluation/wassa_empathy_scores_BACKUP_pre_upgrade.csv` |

*Table A.7: Evidence on the validity of the WASSA empathy classifier for this project (validity checks on the empathy classifier, summarised in Section 5.4.6).*


| Metric | RoBERTa d1 + Qwen (%) | GPT-2 d1 + Qwen (%) | Difference (pp) | 95% CI (pp) | McNemar p | Significant? |
|---|---|---|---|---|---|---|
| Emotion validation | 25.3 | 26 | -0.7 | [-6.3, +5.0] | 0.91 | no |
| All rules | 57.7 | 63.7 | -6 | [-12.7, +0.7] | 0.09 | no |

*Table A.8: Effect of a more accurate detector with the generator held fixed (detector-swap detail behind Section 5.5.3).*


| Detector | Parameters (M) | Training time (min) | Top-1 on test (%) | Top-5 on test (%) | Final system: all rules (%) | Final system: WASSA | Final system: emotion validation (%) |
|---|---|---|---|---|---|---|---|
| roberta-base | 125 | 8.8 | 50.1 | 85 | 60 | 5.05 | 36.7 |
| roberta-large | 355 | 64.7 | 53 | 87.2 | 54.7 | 5.01 | 38 |

*Table A.9: Detector size, detection accuracy and final-system reply quality for RoBERTa-base and RoBERTa-large (detector-level figures behind Section 5.5.7).*


| Final system vs | Measure | N | Final | Baseline | Difference | 95% CI | Wilcoxon p | Significant? |
|---|---|---|---|---|---|---|---|---|
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | Seconds per reply | 50 | 4.17 | 3.91 | 0.27 | [-0.03, +0.56] | 0.1 | no |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | ms per generated token | 50 | 65.68 | 65.44 | 0.24 | [-1.88, +2.39] | 0.91 | no |
| Single model, bigger: Qwen2.5-7B (4-bit) detects + responds | Peak GPU memory (GB) | 1 | 3.4 | 5.47 | -2.07 | -37.9% relative |  | single measurement |
| Single model, same size: Qwen2.5-1.5B detects + responds | Seconds per reply | 50 | 4.17 | 1.58 | 2.6 | [+2.36, +2.83] | 0 | YES |
| Single model, same size: Qwen2.5-1.5B detects + responds | ms per generated token | 50 | 65.68 | 32.65 | 33.03 | [+31.04, +35.03] | 0 | YES |
| Single model, same size: Qwen2.5-1.5B detects + responds | Peak GPU memory (GB) | 1 | 3.4 | 2.9 | 0.5 | +17.1% relative |  | single measurement |

*Table A.10: Paired comparison of response time and memory between the final system and each single model (paired tests behind the resource figures in Section 5.5.7).*

