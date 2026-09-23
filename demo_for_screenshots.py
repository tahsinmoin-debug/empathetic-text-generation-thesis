"""Two demonstrations meant to be screenshotted for the report.

    python demo_for_screenshots.py d1        the detector on one message: top-3 emotions,
                                             confidence, and the words its attention selected
    python demo_for_screenshots.py agent     the scripted conversation to type into the agent,
                                             printed as a checklist with what to expect

The d1 demonstration loads the 125M detector only: about half a gigabyte, a few seconds. It
does not load a generator, so it is safe to run on any machine. The agent itself is started
separately with conversational_agent/chat.py, which loads Phi-4-mini in 4-bit and needs about
3.4 GB of GPU memory.

Both scripts print the same audit trail the system keeps internally, so a screenshot of either
is evidence of what the design exposes, not a mock-up.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "emotion_prompts"))

D1_DIR = ROOT / "train_roberta_ed_d1" / "checkpoints" / "final"

MESSAGES = [
    "I moved to a new city for work and I haven't really spoken to anyone outside the office in weeks.",
    "My dog disappeared while I was away at college and nobody told me until I got home.",
    "I finally finished my thesis draft after months of putting it off!",
]

BAR = "#"


def show_d1():
    from attention_cues import (assert_attention_available, extract_context_cues,
                                load_d1_with_attention)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(D1_DIR)
    model = load_d1_with_attention(D1_DIR)
    model.to(device)
    assert_attention_available(model, tokenizer, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    print("=" * 78)
    print("  d1: emotion detection with attention cues")
    print(f"  RoBERTa-base, 32 emotions, fine-tuned on EmpatheticDialogues  [{device}]")
    print("=" * 78)

    for message in MESSAGES:
        encoded = tokenizer(message, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits[0], dim=-1).cpu()
        values, indices = torch.topk(probs, k=3)
        cues = extract_context_cues(message, tokenizer, model, device)

        print(f'\nUser message:\n  "{message}"\n')
        print("  d1 top-3 candidates passed to the generator")
        for rank, (v, i) in enumerate(zip(values, indices), start=1):
            share = float(v)
            print(f"    {rank}. {id2label[int(i)]:<14} {share:6.1%}  {BAR * int(share * 40)}")
        print(f"\n  words the detector's attention selected:\n    {cues}")
        print("\n" + "-" * 78)

    print("\n  Next in the pipeline: the generator picks one of the three candidates (stage 1),")
    print("  then writes the reply from that emotion's guidance plus these cue words (stage 2).")


SCRIPTED = [
    ("Proving the model remembers the previous turn", "angry", [
        ("Someone yelled at me today very badly. I don't know how to deal with this.",
         "d1 reads this on its own: expect angry or furious"),
        ("I am okay. Well, not properly.",
         "this line ALONE is read as 'content' at 72% confidence; with the previous turn in the "
         "window the agent still reads anger. This is the screenshot worth taking."),
    ]),
    ("Best single emotion to demonstrate: lonely", "lonely", [
        ("My partner and I broke up three weeks ago.", "expect sad or devastated"),
        ("I was so angry at first, I deleted every photo.", "expect angry or furious - watch for 'emotion shift'"),
        ("Now the flat is just very quiet.", "the window starts to carry the whole arc"),
        ("I keep reaching for my phone to text them.",
         "measured at 61% confidence for 'lonely' with the four-turn window, against the wrong "
         "answer when only this line is read"),
    ]),
]


def show_agent_script():
    print("=" * 78)
    print("  Screenshot script for the conversational agent")
    print("=" * 78)
    print("\n  Start it with:\n    python conversational_agent/chat.py")
    print("  It loads RoBERTa-base and Phi-4-mini in 4-bit, about 3.4 GB of GPU memory.\n")
    print("  Which emotion demonstrates the system best: LONELY.")
    print("    - the detector's second-strongest class of the 32 (F1 69.5%, against 49.4% average)")
    print("    - its per-emotion guidance is one of the most specific of the 32")
    print("    - and the four-turn window was measured as recovering it from a conversation")
    print("      whose latest line alone does not say it")
    print("  Second choice: SAD, which the judges score highest among the common emotions (8.75).")

    for title, emotion, turns in SCRIPTED:
        print("\n" + "-" * 78)
        print(f"  {title}   (target emotion: {emotion})")
        print("-" * 78)
        for i, (line, expect) in enumerate(turns, start=1):
            print(f"\n  Turn {i}, type exactly:")
            print(f'    {line}')
            print(f"    -> {expect}")
    print("\n" + "=" * 78)
    print("  Screenshot the whole exchange, including the audit trail the agent prints above")
    print("  each reply: the three candidates with their confidence, the emotion chosen, the")
    print("  cue words, and any emotion shift. That trail is the point of the design.")
    print("=" * 78)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "d1"
    if what == "d1":
        show_d1()
    elif what == "agent":
        show_agent_script()
    else:
        raise SystemExit("usage: python demo_for_screenshots.py [d1|agent]")
