"""The two citation tables, written out rather than guessed.

File 1 (chapters 1-3) numbers its references exactly as our reference list does;
every one of its 46 numbers was checked against the list one by one, so that file
uses the builder's own numbering.

File 2 (chapters 4-6) numbers them its own way. Each of its 35 numbers was read in
the sentence that uses it and identified from what that sentence says, which is the
mapping below. Two of its numbers, 39 and 52, are the same WHO reference.
"""

# file 2: its number -> our bibtex key, with the sentence that fixes it
FILE2 = {
    "1":  ("sharma2020epitome",     "EPITOME, a framework based on existing empathy scales"),
    "2":  ("rashkin2019",           "the 32 emotions in EmpatheticDialogues"),
    "5":  ("devlin2019",            "BERT-base, masked-language modelling"),
    "6":  ("liu2019roberta",        "160 GB of text, dynamic masking, no next-sentence objective"),
    "7":  ("lin2019moel",           "joint models at 30.6-41.2%, first of four"),
    "8":  ("majumder2020mime",      "joint models at 30.6-41.2%, second of four"),
    "11": ("sabour2022cem",         "joint models at 30.6-41.2%, third of four"),
    "12": ("zhou2023case",          "joint models at 30.6-41.2%, fourth of four"),
    "13": ("qian2023llm",           "ChatGPT 4.64 against 3.08 for CEM"),
    "15": ("sorin2024jmir",         "machine-written empathy is repetitive"),
    "16": ("giorgi2024wassa",       "trained on human empathy ratings of conversation turns"),
    "17": ("yang2024hyy33",         "the hyy33 system from the WASSA 2024 shared task"),
    "18": ("zheng2023judge",        "more than 80% agreement with human evaluators"),
    "19": ("verga2024",             "a panel of different families has less bias"),
    "20": ("ed_github",             "the corpus's non-commercial research licence"),
    "21": ("abouelenin2025phi4mini", "Phi-4-mini-instruct, emphasis on instruction following"),
    "22": ("qwen2024",              "Qwen2.5-1.5B-Instruct"),
    "23": ("dettmers2023qlora",     "4-bit NF4 quantisation"),
    "24": ("hu2022lora",            "fine-tunes Qwen2.5-1.5B with LoRA"),
    "25": ("lewis2020rag",          "retrieval-augmented"),
    "26": ("reimers2019sbert",      "all-MiniLM-L6-v2 sentence embeddings"),
    "28": ("cao2025tooled",         "preference alongside ratings, as in similar studies"),
    "29": ("wu2025traits",          "the personality-aware models at 45.4-48.3%"),
    "31": ("paranjape2020chirpy",   "the same pattern found for a similar classifier"),
    "32": ("belay2025",             "encoders fine-tuned for the task beat the alternatives"),
    "33": ("panickssery2024",       "evaluators recognise and favour their own outputs"),
    "39": ("who2023suicide",        "crisis filter with reviewed safe responses"),
    "41": ("samhsa2019tip35",       "counselling guidance warns against several questions"),
    "42": ("demszky2020",           "the project started on GoEmotions"),
    "43": ("mcnemar1947",           "paired McNemar's exact test"),
    "44": ("wilcoxon1945",          "paired Wilcoxon signed-rank test"),
    "45": ("efron1993",             "bootstrap confidence intervals, 10,000 resamples"),
    "46": ("holm1979",              "Holm correction across families of comparisons"),
    "47": ("holtzman2020",          "greedy and beam search generate repetitive text"),
    "52": ("who2023suicide",        "the crisis filter is the first work we recommend"),
}
