# Empathetic Text Generation Thesis

This repository contains the thesis code and prepared data pipeline for context-aware empathetic text generation.

## Main Pipeline

The project uses a two-stage setup:

1. `d1`: GPT-2-medium emotion detector
   - Predicts one of: `Loss`, `Anger`, `Fear`, `Self-Conscious`, `Disgust`
   - Extracts attention-based context cues from the user text

2. `d2`: Decoder-only response generator
   - Trained variants include Qwen2.5, Phi-2, and Gemma
   - Input includes user text, detected emotion, and context cues
   - Output is an empathetic assistant response

## Important Folders

- `GoEmotions/`: dataset preparation and processed CSV files
- `GPT 2 medium training/`: GPT-2-medium emotion detector scripts
- `train_response_qwen2_5/`: Qwen2.5 response-generation training/testing
- `train_response_phi2/`: Phi-2 response-generation training/testing
- `train_response_gemma/`: Gemma grounded-context response-generation training/testing
- `train_qwen2/`, `train_tinyllama/`: earlier detector experiments
- `evaluation_results*/`: detector evaluation outputs

## Not Included In Git

The following are intentionally ignored because they are large/generated artifacts:

- virtual environments
- Hugging Face cache
- trained checkpoints
- model weights
- optimizer states

Recreate model artifacts by running the training scripts locally.
