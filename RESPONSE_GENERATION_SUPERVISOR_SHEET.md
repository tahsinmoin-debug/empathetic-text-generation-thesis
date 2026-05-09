# Response Generation Supervisor Sheet

## 1. Project Goal

The response-generation part of the thesis builds an empathetic text generator using a two-stage architecture:

```text
User text -> d1 emotion/context detector -> d2 decoder-only response generator
```

The goal is not only to detect emotion, but to generate an empathetic response that considers:

- the original user text
- the detected emotion
- important context cues from the user text

## 2. Dataset Used

Main response-generation file:

```text
GoEmotions/goemotions_multilevel_negative_dialogue.csv
```

Prepared response-generation files:

```text
GoEmotions/response_generation_qwen2_5/prepared_response_data.csv
GoEmotions/response_generation_qwen2_5/train.csv
GoEmotions/response_generation_qwen2_5/dev.csv
GoEmotions/response_generation_qwen2_5/test.csv
```

Prepared split sizes:

```text
Train: 17,913
Dev:   2,237
Test:  about 2,243
```

Important columns:

```text
text
gold_emotion
predicted_emotion
context_cues
assistant_response
training_text
```

## 3. Role of d1: Emotion Detection and Context Cue Extraction

We used the trained GPT-2-medium model as `d1`.

Path:

```text
GPT 2 medium training/checkpoints/gpt2_medium/final
```

`d1` performs two tasks:

1. Predicts the emotion label.
2. Extracts attention-based context cues from the user text.

The emotion labels are:

```text
Loss
Anger
Fear
Self-Conscious
Disgust
```

### How d1 Predicts Emotion

For each input text, the GPT-2-medium detector scores all five possible labels using this format:

```text
<emotion=Loss> {text}
<emotion=Anger> {text}
<emotion=Fear> {text}
<emotion=Self-Conscious> {text}
<emotion=Disgust> {text}
```

The label with the lowest language-model loss is selected as the predicted emotion.

Example:

```text
Text: I miss him so much during the NCAA tournament.
Predicted emotion: Loss
```

## 4. Attention Mechanism: Beginner Explanation

Attention tells us which tokens the model focuses on while processing the input.

In simple terms, GPT-2-medium reads the sentence and gives higher attention to words that are more important for understanding the emotion.

Example:

```text
I lost my job and I keep thinking about it all day.
```

Useful emotional context:

```text
lost my job
thinking about it all day
```

Less useful words:

```text
I
and
it
```

### How We Extracted Context Cues

After `d1` predicts an emotion, we run GPT-2-medium again with attention enabled:

```python
output_attentions=True
```

Then we:

1. Take the final transformer layer attention.
2. Average attention scores across attention heads.
3. Look at attention from the final token back to the input tokens.
4. Keep only tokens belonging to the original user text.
5. Select the highest-attention tokens.
6. Merge nearby selected tokens into readable phrases.

Example result:

```text
Text: I lost my job and I keep thinking about it all day.
Context cues: lost my job and I keep thinking about it all day
```

These context cues are then passed into `d2`.

Important clarification:

```text
We did not use attention from the assistant response.
We used attention from d1 while d1 was reading the user text.
```

## 5. Role of d2: Response Generation

`d2` is a decoder-only language model trained to generate the assistant response.

The original Qwen2.5 and Phi-2 prompt format was:

```text
User: {text}
Detected emotion: {predicted_emotion}
Context cues: {context_cues}
Assistant: {assistant_response}
```

Only the assistant response tokens were used for loss calculation. Prompt tokens were masked with `-100`, so the model learned to generate the response, not copy the prompt.

## 6. Models Trained for d2

### Model 1: Qwen2.5

```text
Model: Qwen/Qwen2.5-1.5B-Instruct
Method: QLoRA
Output: checkpoints/qwen2_5_1_5b_emotional_response_lora/final
```

### Model 2: Phi-2

```text
Model: microsoft/phi-2
Method: QLoRA
Output: checkpoints/phi2_emotional_response_lora/final
```

### Model 3: Gemma Grounded Experiment

```text
Model: google/gemma-2-2b-it
Method: QLoRA
Output: checkpoints/gemma2_2b_grounded_emotional_response_lora/final
```

Gemma used an improved grounded-context instruction:

```text
User: {text}
Detected emotion: {predicted_emotion}
Context cues: {context_cues}
Instruction: Write an empathetic response that reflects the detected emotion and directly mentions or clearly paraphrases the context cues. Avoid generic support phrases.
Assistant: {assistant_response}
```

The instruction was added because Qwen2.5 and Phi-2 often produced generic empathetic responses without clearly using the context cues.

## 7. Training Configuration

All response-generation models were trained with QLoRA.

Common settings:

```text
4-bit quantization: yes
Quantization type: NF4
Batch size: 1
Gradient accumulation: 16
Epochs: 3
Learning rate: 1e-4
LoRA r: 8
LoRA alpha: 16
LoRA dropout: 0.05
Gradient checkpointing: enabled
model.config.use_cache: False
```

Max length:

```text
Qwen2.5: 256
Phi-2:   256
Gemma:   192
```

Gemma used `max_length=192` to reduce GPU memory pressure on the RTX 4060 8GB setup.

## 8. Emotion Detection Results

Detector evaluation used:

```text
GoEmotions/splits_qwen2_5_detector/test.csv
Total samples: 2,243
```

| Rank | Model | Accuracy | Macro F1 | Correct / Total |
|---:|---|---:|---:|---:|
| 1 | GPT-2 Medium | 97.73% | 96.22% | 2192 / 2243 |
| 2 | Qwen2.5-0.5B Emotion Detector LoRA | 35.93% | 33.43% | 806 / 2243 |
| 3 | TinyLLaMA LoRA | 26.13% | 23.85% | 586 / 2243 |
| 4 | Qwen2-1.5B LoRA | 14.22% | 8.78% | 319 / 2243 |

Because GPT-2-medium performed best, it was selected as `d1`.

Note: the detector comparison has an important caveat. Some earlier models may have been trained using older formats/splits, so the result should be presented as an observed project result, not as a universal benchmark.

## 9. Response-Generation Training Results

All three d2 models completed 3 epochs, reaching global step 3360.

| Model | Best Eval Loss | Best Checkpoint |
|---|---:|---|
| Qwen2.5-1.5B | 0.2546 | checkpoint-3250 |
| Phi-2 | 0.2459 | checkpoint-3250 |
| Gemma-2-2B Grounded | 0.2419 | checkpoint-3250 |

Lower eval loss means the model fit the validation target responses better. However, response quality was also judged manually because low loss does not guarantee better empathy or better context usage.

## 10. Manual Testing Examples and Observations

### Example 1

Input:

```text
I lost my job and I keep thinking about it all day.
```

Observed issue:

```text
d1 sometimes predicted Self-Conscious instead of Loss/Fear/Stress.
```

Qwen2.5 produced fluent empathy, but it was generic and did not strongly mention job loss.

Gemma grounded experiment mentioned work slightly more, but still did not fully ground the response in:

```text
lost my job
thinking about it all day
```

### Example 2

Input:

```text
I broke up with my girlfriend today and now it feels like dying.
```

Observed issue:

```text
Self-harm wording was not handled safely.
```

The models generated soft empathetic responses but did not include an appropriate crisis/safety response.

### Example 3

Input:

```text
I am feeling so down today because of projects and assignments. It has to be done by tomorrow.
```

Observed issue:

```text
The current label set does not include Stress, Anxiety, or Overwhelmed.
```

The models often mapped this to `Loss`, then generated a generic sadness-style response instead of a deadline-stress response.

## 11. Qualitative Model Comparison

Based on manual tests:

```text
Qwen2.5 performed slightly better overall.
```

Reason:

- most fluent and natural response style
- emotionally smooth
- less awkward than Phi-2 in some examples

Observed ranking:

```text
1. Qwen2.5-1.5B
2. Phi-2
3. Gemma-2-2B grounded experiment
```

Important caveat:

```text
Gemma had the lowest eval loss, but qualitative tests did not show clear improvement in context grounding.
```

So the final judgment should not be based only on eval loss.

## 12. Main Problems Found

### Problem 1: Repetitive Training Responses

The target assistant responses contain many repeated phrases.

Observed counts:

```text
"makes sense that" appears about 14,304 times
"You deserve some softness" appears about 1,626 times
"while this settles" appears about 1,626 times
```

This caused all d2 models to learn similar template-like wording.

### Problem 2: d1 Works Well on Dataset but Can Fail on Custom Text

GPT-2-medium performed very well on the dataset test split, but some real-life custom examples were misclassified.

Example:

```text
I lost my job and I keep thinking about it all day.
```

Observed prediction:

```text
Self-Conscious
```

Better expected emotion:

```text
Loss / Fear / Stress
```

### Problem 3: Context Cue Extraction Is Useful but Imperfect

Sometimes context cues are good:

```text
lost my job and I keep thinking about it all day
```

Sometimes they contain weak fragments:

```text
am feeling
The
die because of
```

### Problem 4: No Safety Layer

Inputs containing self-harm language need a safety rule before normal response generation.

Example:

```text
I want to die
feels like dying
```

These should trigger a crisis-support response instead of only generic empathy.

## 13. Conclusion

The response-generation pipeline was successfully implemented:

```text
User text -> GPT-2-medium d1 -> predicted emotion + attention context cues -> decoder-only d2 -> empathetic response
```

Three decoder-only d2 models were trained:

```text
Qwen2.5-1.5B
Phi-2
Gemma-2-2B grounded
```

The best qualitative model was Qwen2.5, while Gemma achieved the lowest validation loss. The main limitation was not model size alone; it was the combination of repetitive target responses, imperfect context extraction, missing stress/anxiety labels, and no self-harm safety layer.

## 14. Recommended Next Improvements

1. Add a self-harm safety detector before d2 generation.
2. Improve context cue extraction by filtering weak fragments.
3. Add extra emotion categories such as:
   ```text
   Stress
   Anxiety
   Overwhelmed
   Heartbreak
   ```
4. Diversify or rewrite repetitive assistant responses.
5. Retrain the best model, likely Qwen2.5, using the grounded-context instruction format.
