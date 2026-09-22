# 6.1 Conclusion

This thesis set out to test whether separating emotion detection from response generation is worth doing, at a size that fits an ordinary computer. The answer, from 4,200 scored replies across fourteen system configurations, is a qualified yes.

Three findings support the design. First, at a comparable scale the decoupled system is significantly more empathetic than a single model that does both jobs: against the 1.5-billion-parameter single model it scores higher on the empathy classifier, higher on both AI judges and 11.0 points higher on emotion fidelity. Second, the advantage comes from the design and not from the size of the generator; when the same 3.8-billion-parameter generator is prompted alone, its empathy-classifier score and emotion fidelity fall to the level of a model less than half its size, and adding the detector with emotion-specific guidance raises them again. Third, the system does this within the resource budget it was designed for: 3.40 GB of GPU memory, 38% less than a single model with 1.9 times its parameters, with no significant difference in response time or in judge-rated quality.

Two findings limit the claim, and are reported as plainly as the positive ones. The larger single model remains better on rule adherence and on the empathy classifier, so a deployment with the hardware to run it and no need for an auditable emotional decision should use it. And at an equal generator the AI judges do not separate the two configurations: the case for the decoupled design rests on emotion fidelity, the empathy classifier, less repetition and the visible emotional decision, not on the judges' preference.

The contribution of this work is therefore not a new record on a benchmark. It is a measured account of what a decoupled design buys and what it costs, at a scale where the choice actually matters, together with a working multi-turn agent that shows the emotional decision behind every reply.

# 6.2 Future Work

Four directions follow directly from the results.

**Close the rule-adherence gap:** the decoupled system follows its own response rules less often than a single model because the rules share the prompt with emotion-specific guidance. Separating rule enforcement from generation, by checking and repairing the reply after it is written, should recover the gap without giving up the emotional guidance.

**Reduce repetition:** the system reuses stock supportive phrases in 58.7% of replies. This is better than the single-model baselines but still high, and a repetition penalty or a decoding constraint is the obvious next step; it was deliberately deferred in this project so that the evaluated outputs would remain fixed.

**Evaluate multi-turn behaviour:** all reported measurements are single-turn. The multi-turn agent works, but nothing here measures whether the emotional tracking across turns helps a real conversation.

**Test the design in Bangla and with real users:** the dataset, the models and the evaluation are entirely in English, and the evaluation uses crowd-sourced situations rather than people seeking support. Both would have to change before any claim about usefulness in Bangladesh could be made, and a crisis safeguard would have to be added first.
