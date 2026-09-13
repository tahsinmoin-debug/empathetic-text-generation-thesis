# Chapter 3 - 3.8 Economic Analysis

**Rubric:** CO11, "Perform economic analysis of the resource management." Final Thesis Report. Criteria: conduct economic analysis; conduct cost-benefit estimation.

## What this section must do

Account for what the project cost to build and evaluate, what it would cost to run, and justify the design choices on cost-benefit grounds. Only claim figures you can back up: measured numbers from our logs, licence terms we checked, and prices the team fills in from receipts or official tariff pages.

## Tables in this folder

| File | Use |
|---|---|
| `tables/cost_breakdown.csv` | Development cost items; BDT cells are `[[TEAM TO FILL]]` |
| `tables/compute_usage.csv` | Measured training times; other rows marked as not logged |
| `tables/d1_backbone_cost_benefit.csv` | Cost vs accuracy of detector sizes |
| `tables/detector_sensitivity_robertalarge.csv` | The benefit side: what the larger detector changed in the final system (nothing significant) |
| `tables/licences.csv` | Licence terms, including the commercial-use restriction |

## Checklist

- [ ] 3.8.1 Development cost (hardware, electricity, software, data, evaluation)
- [ ] 3.8.2 Compute usage and energy estimate (formula + measured times)
- [ ] 3.8.3 Cost-benefit of design choices (detector size, generator size, fine-tuning vs prompting, evaluation method)
- [ ] 3.8.4 Operating cost and deployment options
- [ ] 3.8.5 Commercial constraints (CC BY-NC data)

---

## Draft

### 3.8.1 Development Cost

The project was designed to run on hardware the team already owned and on free-of-charge software, models and services (Table 3.x). All models are open-weight and all libraries are open source. The dataset is freely available for non-commercial research. The three AI judges were accessed through free API tiers, the demonstration interface targets the free Hugging Face Spaces CPU tier, and the human study uses Google Forms with volunteer participants. Direct monetary cost is therefore limited to electricity, internet access and the (pre-existing) GPU workstation: [[TEAM TO FILL: purchase price in BDT, or state that it is an existing personal asset with zero marginal cost]].

[Table: cost_breakdown.csv]

### 3.8.2 Compute Usage and Energy

Training times were logged for the detector experiments (Table 3.x). Fine-tuning RoBERTa-base took 8.8 minutes, and the comparison backbones BERT-base and ELECTRA-base took 8.6 and 8.7 minutes. RoBERTa-large took 64.7 minutes. An upper-bound estimate of the GPU energy for a training run is

  E (kWh) = P_GPU (kW) × t (h),

where P_GPU = 0.115 kW is the RTX 4060's rated board power. This gives about 0.017 kWh for RoBERTa-base and 0.124 kWh for RoBERTa-large, excluding the rest of the system. At a tariff of [[TEAM TO FILL: BDT per kWh]], detector training costs a negligible amount. Generation time for the response experiments was not logged, so we do not report it. [[TEAM TO FILL, optional: time one 300-conversation generation run and add it.]]

[Table: compute_usage.csv]

### 3.8.3 Cost-Benefit of Design Choices

**Detector size.** RoBERTa-large improves top-1 accuracy by 2.9 percentage points (53.0% vs 50.1%) and top-5 by 2.2 points, both statistically significant, but it needs 2.8 times as many parameters and 7.4 times the training time (Table 3.x). We measured the benefit directly by running the final system with RoBERTa-large (Table 3.x). Rule adherence (60.0% vs 54.7%), WASSA empathy (5.05 vs 5.01) and emotion validation (36.7% vs 38.0%) showed no significant change. The extra cost buys no measurable improvement in replies, so RoBERTa-base was kept.

**Fine-tuning vs prompting the generator.** Fine-tuning the generator with LoRA consumed GPU time and produced the worst system tested: 0% of replies followed all rules and 98% asked more than one question. Prompting an off-the-shelf instruction-tuned model needs no training at all and reaches 60.0% for the final system. On this task, prompting dominates on both cost and quality.

**Generator size.** The honest trade-off runs the other way here. A single Qwen2.5-7B model (7.6B parameters, about 5.6 GB of GPU memory in 4-bit) outperforms our pipeline, whose generator is Phi-4-mini at 3.8B parameters, on rule adherence (82.3% vs 60.0%), WASSA empathy (5.76 vs 5.05) and AI-judge scores. If quality per unit of compute were the only objective, the larger single model would be preferred. We keep the decoupled design for inspectability and controllability: every decision, including the emotion candidates, the chosen emotion and the influencing words, is visible and can be intercepted. We do not claim a cost advantage over the larger model.

**Evaluation method.** Human annotation in comparable work is paid (for example, about USD 6 per hour for annotators in Belay et al., 2025). Free-tier AI judges produced several hundred judgements per judge at no monetary cost, at the price of rate limits and service churn (Section 3.7). The human study is used to validate those judges rather than replace them.

### 3.8.4 Operating Cost

A self-hosted deployment costs the hardware and electricity above plus no per-message fees, and no user message leaves the host. A deployment built on a commercial LLM API instead costs a fee per token for every message: [[TEAM TO FILL: current published price per million input/output tokens for a named commercial model, with the date checked]]. A free CPU host (2 vCPU, 16 GB RAM) costs nothing but responds much more slowly [[TEAM TO FILL: measured latency if the Space is deployed; otherwise say latency was not measured]].

### 3.8.5 Commercial Constraints

EmpatheticDialogues is licensed CC BY-NC 4.0, which forbids commercial use. The trained detector is derived from this data, so the system as built cannot be sold or used in a paid service. Commercialisation would require re-training the detector on commercially licensed or newly collected data. The generator (Phi-4-mini, MIT) and detector backbone (RoBERTa, MIT) do not impose this restriction.

---

## Traps

- **Don't invent prices, tariffs or latency.** Every BDT figure and any per-token API price must come from a receipt or an official page, with the date you checked.
- **Don't claim the pipeline is "cheaper than a bigger model" without measurement.** We did not measure inference cost, and the bigger single model is better on quality.
- The 115 W figure is the GPU's rated maximum. Label energy numbers as upper-bound GPU-only estimates.
