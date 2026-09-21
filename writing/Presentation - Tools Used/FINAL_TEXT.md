# Tools and Technologies Used

> Note for the team (delete before submission): the rubric assesses this item (CO8) through the presentation, demonstration and final report. Use the paragraph below in the report, for example within Section 4.1.6, and Table 4.3 on a presentation slide.

The system was implemented in Python 3.10 using PyTorch 2.5.1 [43] and Hugging Face Transformers 5.16.1 [40]. Transformers provides pre-trained encoders, namely RoBERTa [6], BERT [5] and ELECTRA, and instruction-tuned decoders, namely Phi-4-mini [7], Qwen2.5 [8] and Gemma-3, behind a common interface, which allowed design alternatives to be exchanged without rewriting code. The bitsandbytes library provided 4-bit NF4 quantisation [9], allowing generators of 3.8 to 7.6 billion parameters to run on a single NVIDIA GeForce RTX 4060 with 8 GB of memory [38], and the PEFT library provided LoRA [10] for the monolithic fine-tuning baseline.

Evaluation combined SciPy [44] and scikit-learn [45] for paired statistical tests and classification metrics, sentence-transformers [12] for the retrieval baseline, the WASSA 2024 empathy classifier [15], and two AI judges accessed through the OpenRouter and Groq APIs, selected from different developers to avoid self-preference bias [34]. A Gradio interface was prepared for demonstration on Hugging Face Spaces, the human study was conducted with Google Forms, and all tables and figures in this report were generated from saved experiment outputs by scripts using pandas and matplotlib, so that every reported number can be reproduced.


| Purpose | Tool | Version |
|---|---|---|
| Language | Python | 3.10.9 |
| Deep learning | PyTorch (CUDA 12.1) | 2.5.1 |
| Models and training | Hugging Face Transformers | 5.16.1 |
| Parameter-efficient tuning | PEFT | 0.18.1 |
| Quantisation | bitsandbytes (4-bit NF4) | 0.49.2 |
| Training utilities | Accelerate / Datasets | 1.13.0 / 4.8.5 |
| Statistics | SciPy / scikit-learn | 1.15.3 / 1.7.2 |
| Data handling | pandas / NumPy | 2.3.3 / 2.2.6 |
| Retrieval | sentence-transformers | 5.7.0 |
| Evaluation | bert-score | 0.3.13 |
| AI-judge APIs | openai SDK (OpenRouter), groq SDK | 2.53.0 / 1.6.0 |
| Demo interface | Gradio on Hugging Face Spaces | 5.9.1 (planned) |
| Human evaluation | Google Forms | - |
| Documents | python-docx, Microsoft Word | 1.2.0 |
| Version control | Git | - |
| Hardware | NVIDIA RTX 4060 8 GB, 16 GB RAM, Windows 11 Pro | driver 616.56 |

*Table 4.3: Software and hardware used in this work*


## References

[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. 2019 Conf. North American Chapter Assoc. Comput. Linguistics (NAACL-HLT), 2019, pp. 4171-4186, doi: 10.18653/v1/N19-1423.  
[6] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.  
[7] A. Abouelenin, A. Ashfaq, A. Atkinson et al. (Microsoft), "Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs," arXiv:2503.01743, 2025.  
[8] A. Yang, B. Yang, B. Zhang, B. Hui et al. (Qwen Team), "Qwen2.5 technical report," arXiv:2412.15115, 2024.  
[9] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.  
[10] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. Learning Representations (ICLR), 2022.  
[12] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. 2019 Conf. Empirical Methods Natural Language Processing and 9th Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982-3992, doi: 10.18653/v1/D19-1410.  
[15] H. Yang, L. Huang, T. Li, N. Rusnachenko, and H. Liang, "hyy33 at WASSA 2024 empathy and personality shared task: Using the CombinedLoss and FGM for enhancing BERT-based models in emotion and empathy prediction from conversation turns," in Proc. 14th Workshop Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA), 2024, pp. 430-434, doi: 10.18653/v1/2024.wassa-1.39.  
[34] A. Panickssery, S. R. Bowman, and S. Feng, "LLM evaluators recognize and favor their own generations," in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024; arXiv:2404.13076.  
[38] NVIDIA, "GeForce RTX 4060 Ti & 4060 graphics cards: Specifications." [Online]. Available: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/  
[40] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., "Transformers: State-of-the-art natural language processing," in Proc. 2020 Conf. Empirical Methods Natural Language Processing: System Demonstrations, 2020, pp. 38-45, doi: 10.18653/v1/2020.emnlp-demos.6.  
[43] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 8024-8035.  
[44] P. Virtanen, R. Gommers, T. E. Oliphant et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261-272, 2020, doi: 10.1038/s41592-019-0686-2.  
[45] F. Pedregosa, G. Varoquaux, A. Gramfort et al., "Scikit-learn: Machine learning in Python," J. Machine Learning Research, vol. 12, pp. 2825-2830, 2011.  
