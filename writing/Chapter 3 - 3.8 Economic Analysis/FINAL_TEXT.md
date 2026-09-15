# 3.8 Economic Analysis

This section analyses the cost of developing and evaluating the system, the computing resources used, the cost-effectiveness of the main design decisions, and the constraints on commercial use.

## 3.8.1 Development Cost

The project was designed to use hardware the team already owned together with free software, models and services (Table 3.5). All models are openly released and all software libraries are open source. The dataset is freely available for non-commercial research [3]. The AI judges were accessed through free API tiers, the demonstration interface targets the free Hugging Face Spaces CPU tier, and the human study uses Google Forms with volunteer participants. Direct monetary cost is therefore limited to electricity, internet access and the GPU workstation, [[TEAM TO FILL: purchase price in BDT, or a statement that it is an existing personal asset with no additional cost]].


| Item | Details | Cost (BDT) | Type |
|---|---|---|---|
| Hardware | Personal PC with RTX 4060 8 GB, 16 GB RAM (already owned) | [[TEAM TO FILL - BDT purchase price, or 'existing asset']] | One-time |
| Electricity | GPU board power 115 W (NVIDIA spec) x GPU hours x tariff | [[TEAM TO FILL - BDT/kWh]] | Per run |
| Pretrained models | RoBERTa (MIT), Phi-4-mini (MIT), Qwen2.5 (Apache-2.0), Gemma-3 (Gemma terms) | 0 | Open weights |
| Dataset | EmpatheticDialogues (CC BY-NC 4.0) | 0 | Non-commercial use only |
| AI judges | OpenRouter free tier, Mistral free tier, Groq free tier | 0 | Rate-limited |
| Hosting (demo) | Hugging Face Spaces free CPU tier (2 vCPU, 16 GB) | 0 | Not yet deployed |
| Human evaluation | Google Forms, volunteer raters | 0 | No payment |
| Internet | Model downloads, API calls | [[TEAM TO FILL]] | Monthly |

*Table 3.5: Development cost breakdown*


## 3.8.2 Compute Usage and Energy

Training time was recorded for each detector experiment (Table 3.6). Fine-tuning RoBERTa-base took 8.8 minutes, the BERT-base and ELECTRA-base comparison models took 8.6 and 8.7 minutes, and RoBERTa-large took 64.7 minutes. An upper-bound estimate of the GPU energy consumed by a training run is

E (kWh) = P (kW) x t (h),

where P = 0.115 kW is the rated board power of the RTX 4060 [38]. This gives about 0.017 kWh for RoBERTa-base and 0.124 kWh for RoBERTa-large, excluding the rest of the computer. At an electricity tariff of [[TEAM TO FILL: BDT per kWh]], the cost of training the detector is negligible. Generation time for the response experiments was not recorded and is therefore not reported.


| Task | Hardware | Wall-clock minutes | Status |
|---|---|---|---|
| d1 RoBERTa-base training | RTX 4060 8 GB | 8.8 | recorded (d1 comparison run) |
| d1 BERT-base training | RTX 4060 8 GB | 8.6 | recorded |
| d1 ELECTRA-base training | RTX 4060 8 GB | 8.7 | recorded |
| d1 RoBERTa-large training | RTX 4060 8 GB | 64.7 | recorded |
| Response generation, 300 conversations per system (x11 systems) | RTX 4060 8 GB | [[TEAM TO FILL - not logged]] | not recorded |
| AI judging (Nemotron, gpt-oss; Mistral for 5 systems) | Cloud API, free tier | network time only | free tier |
| WASSA scoring, 3,300 replies | RTX 4060 / CPU | [[TEAM TO FILL]] | not recorded |

*Table 3.6: Recorded compute usage*


## 3.8.3 Cost-Benefit Analysis of Design Choices

**Detector size.** Compared with RoBERTa-base, RoBERTa-large improves top-1 accuracy by 2.9 percentage points (53.0% against 50.1%) and top-5 accuracy by 2.2 points, both statistically significant, but it has 2.8 times as many parameters and needs 7.4 times the training time (Table 3.7). We measured the benefit directly by running the final system with RoBERTa-large as its detector (Table 5.16). Rule adherence (60.0% against 54.7%), WASSA empathy (5.05 against 5.01) and emotion validation (36.7% against 38.0%) showed no significant change. Because the additional cost bought no measurable improvement in the replies, RoBERTa-base was retained.


| Backbone | Parameters (M) | Top-1 acc (%) | Top-5 acc (%) | Train time (min) |
|---|---|---|---|---|
| bert-base-uncased | 110 | 48.8 | 81.7 | 8.6 |
| google/electra-base-discriminator | 110 | 50.2 | 82.7 | 8.7 |
| roberta-base | 125 | 50.1 | 85 | 8.8 |
| roberta-large | 355 | 53 | 87.2 | 64.7 |

*Table 3.7: Size, accuracy and training cost of the candidate detectors*


**Fine-tuning compared with prompting the generator.** Fine-tuning the generator with LoRA [10] consumed GPU time and produced the weakest system tested: no reply followed all three rules and 98% of replies asked more than one question. Prompting an instruction-tuned model requires no training and achieves 60.0% for the final system. For this task, prompting is superior in both cost and quality.

**Generator size.** Here the trade-off runs the other way. A single Qwen2.5-7B model [8], with 7.6 billion parameters and using about 5.6 GB of GPU memory in 4-bit form, outperforms our pipeline, whose generator has 3.8 billion parameters, on rule adherence (82.3% against 60.0%), WASSA empathy (5.76 against 5.05) and AI-judge empathy (8.15 against 7.84). If output quality were the only objective, the larger single model would be preferred. Measured on the same GPU, however, our pipeline needs 38% less peak GPU memory than the larger model (3.40 GB against 5.47 GB) with no significant difference in response time (4.17 s against 3.91 s per reply; Section 5.4.7), so it can run on hardware with less memory while reaching 97.5% of the larger model's overall AI-judge score. It is not cheaper to run than the Qwen2.5-1.5B single model, which is lighter and 2.6 times faster but produces significantly less empathetic replies. The decoupled design is retained for this balance and for its inspectability, since the candidate emotions, the chosen emotion and the words that drove the decision are all visible and can be acted upon.

**Evaluation method.** Human annotation in comparable research is paid; for example, Belay et al. report paying annotators about USD 6 per hour [33]. The free-tier AI judges in this project produced several hundred judgements each at no monetary cost, in exchange for rate limits and the need to replace unavailable judges (Section 3.7). The human study is used to validate these judges rather than to replace them.

## 3.8.4 Operating Cost

A self-hosted deployment incurs the hardware and electricity costs described above but no per-message fees, and no user message leaves the host. A deployment based on a commercial LLM API would instead incur a fee for every message, [[TEAM TO FILL: current published price per million input and output tokens for a named commercial model, with the date checked]]. A free CPU host with 2 virtual CPUs and 16 GB of memory costs nothing but responds considerably more slowly, [[TEAM TO FILL: measured response time if the demonstration is deployed; otherwise state that latency was not measured]].

## 3.8.5 Commercial Constraints

EmpatheticDialogues is licensed under CC BY-NC 4.0, which forbids commercial use [3]. Because the detector is trained on this data, the system as built cannot be sold or used in a paid service. Commercial use would require retraining the detector on commercially licensed or newly collected data. The generator and the detector backbone are MIT-licensed [7], [6] and impose no such restriction (Table 3.1).

## References

[3] Facebook Research, "EmpatheticDialogues," GitHub repository (CC BY-NC 4.0 licence). [Online]. Available: https://github.com/facebookresearch/EmpatheticDialogues  
[6] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.  
[7] A. Abouelenin, A. Ashfaq, A. Atkinson et al. (Microsoft), "Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs," arXiv:2503.01743, 2025.  
[8] A. Yang, B. Yang, B. Zhang, B. Hui et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.  
[10] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. Learning Representations (ICLR), 2022.  
[33] T. D. Belay, I. A. Azime, A. A. Ayele, G. Sidorov, D. Klakow, P. Slusallek, O. Kolesnikova, and S. M. Yimam, "Evaluating the capabilities of large language models for multi-label emotion understanding," in Proc. 31st Int. Conf. Computational Linguistics (COLING), Abu Dhabi, UAE, 2025, pp. 3523-3540.  
[38] NVIDIA, "GeForce RTX 4060 Ti & 4060 graphics cards: Specifications." [Online]. Available: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/  
