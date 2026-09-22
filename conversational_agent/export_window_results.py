"""Write the two context-window experiments out as tables for the thesis.

Re-runs the scenarios of compare_history_windows.py (short emotion-shift conversations) and
test_long_windows.py (long arcs, 6 to 8 turns) and writes what d1 predicts at each window size.
Only the detector is loaded, so this is a light job: RoBERTa-base at 125M parameters, well
under a gigabyte, about a minute in total.

The scenarios are hand-written, because EmpatheticDialogues cannot answer this question: its
conversations have a median of 2 turns, a maximum of 4, and the gold emotion never changes
within a conversation (0 of 2,542 test conversations). The "what a listener would hear" column
is the authors' judgement, not a gold annotation, so these tables are illustrative evidence for
a design choice and are reported as such.

Usage:
    python conversational_agent/export_window_results.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "emotion_prompts"))

from attention_cues import assert_attention_available, load_d1_with_attention  # noqa: E402
from compare_history_windows import SCENARIOS as SHIFT_SCENARIOS  # noqa: E402
from test_long_windows import SCENARIOS as LONG_SCENARIOS, WINDOWS as LONG_WINDOWS  # noqa: E402

D1_DIR = ROOT / "train_roberta_ed_d1" / "checkpoints" / "final"
OUT = ROOT / "final writing" / "tables"
SHORT_WINDOWS = [1, 2, 4]


def predict(text, tokenizer, model, id2label, device):
    encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits[0], dim=-1).cpu()
    idx = int(torch.argmax(probs))
    return id2label[idx], float(probs[idx])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(D1_DIR)
    model = load_d1_with_attention(D1_DIR)
    model.to(device)
    assert_attention_available(model, tokenizer, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # --- short emotion-shift conversations -------------------------------------------------
    rows = []
    for sc in SHIFT_SCENARIOS:
        turns = sc["turns"]
        row = {"Conversation": sc["name"], "A listener would hear": sc["intended"]}
        for w in SHORT_WINDOWS:
            label, conf = predict(" ".join(turns[-w:]), tokenizer, model, id2label, device)
            row[f"Last {w} turn{'s' if w > 1 else ''}"] = f"{label} ({conf:.0%})"
        rows.append(row)
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / "context_window_shift_scenarios.csv", index=False)
    print("wrote context_window_shift_scenarios.csv", len(shift), "scenarios")

    # --- long arcs, where a window can be too wide -----------------------------------------
    rows = []
    for sc in LONG_SCENARIOS:
        turns = sc["turns"]
        row = {"Conversation": sc["name"].replace("Long arc: ", ""),
               "Turns": len(turns), "A listener would hear": sc["intended"]}
        for w in LONG_WINDOWS:
            if w > len(turns):
                row[f"Last {w}"] = "-"
                continue
            label, conf = predict(" ".join(turns[-w:]), tokenizer, model, id2label, device)
            row[f"Last {w}"] = f"{label} ({conf:.0%})"
        rows.append(row)
    long = pd.DataFrame(rows)
    keep = [c for c in long.columns if not (c.startswith("Last ") and (long[c] == "-").all())]
    long = long[keep]
    long.to_csv(OUT / "context_window_long_scenarios.csv", index=False)
    print("wrote context_window_long_scenarios.csv", len(long), "scenarios")


if __name__ == "__main__":
    main()
