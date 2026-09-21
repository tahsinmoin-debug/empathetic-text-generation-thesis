# Tools and Technologies Used

> Note for the team (delete before submission): the rubric assesses this item (CO8) through the presentation, demonstration and final report. Use the paragraph below in the report, for example within Section 4.1.6, and Table 4.3 on a presentation slide.

The system was implemented in Python 3.10 using PyTorch 2.5.1 and Hugging Face Transformers 5.16.1. Transformers provides pre-trained encoders, namely RoBERTa, BERT and ELECTRA, and instruction-tuned decoders, namely Phi-4-mini, Qwen2.5 and Gemma-3, behind a common interface, which allowed design alternatives to be exchanged without rewriting code. The bitsandbytes library provided 4-bit NF4 quantisation, allowing generators of 3.8 to 7.6 billion parameters to run on a single NVIDIA GeForce RTX 4060 with 8 GB of memory, and the PEFT library provided LoRA for the monolithic fine-tuning baseline.

Evaluation combined SciPy and scikit-learn for paired statistical tests and classification metrics, sentence-transformers for the retrieval baseline, the WASSA 2024 empathy classifier, and two AI judges accessed through the OpenRouter and Groq APIs, selected from different developers to avoid self-preference bias. A Gradio interface was prepared for demonstration on Hugging Face Spaces, the human study was conducted with Google Forms, and all tables and figures in this report were generated from saved experiment outputs by scripts using pandas and matplotlib, so that every reported number can be reproduced.

[[TABLE 4.3 | tools_and_versions.csv | Software and hardware used in this work]]
