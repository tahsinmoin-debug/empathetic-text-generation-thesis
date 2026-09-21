"""Build the human-evaluation study material: a print-ready plan document plus a
private answer key.

Design (each choice traced to evidence, not preference):

  Forced-choice preference, not solo Likert rating
      Both human-evaluation papers in this project's reading list use A/B preference with
      Win/Lose/Tie (COLING 2025 main.355; main.598). main.598 states the reason explicitly -
      pairwise comparison "accounts for individual variations among annotators", i.e. two
      people who use a 1-5 scale differently still agree on which of two replies is better.

  Three replies per item instead of two
      A deviation from those papers, forced by our constraints. With ~12 items of rater
      budget, a pairwise design spends the whole budget on ONE comparison, or splits it into
      4 items per comparison across three. Showing all three systems at once tests every
      comparison on every item. Cost: slightly heavier reading per item, and position bias
      is stronger with three options - mitigated by per-item randomised ordering below.

  Empathy only, one question per item
      Rater-fatigue control. main.355 used 4 aspects and main.598 used 3, but both had 3-7
      expert annotators; we trade aspect breadth for rater count. Empathy is the dimension
      the thesis actually claims.

  Sampled from the 64 conversations scored by all three AI judges
      This is what makes the study's most valuable output possible: measuring whether human
      preference agrees with the AI-judge panel that the rest of the results chapter rests on.

  Random selection with a fixed seed, one conversation per emotion
      Removes any possibility of cherry-picking, and the pool happens to hold exactly two
      conversations for each of the 32 emotions, so an emotion-balanced draw is clean.

Usage:
    python build_human_eval_form.py
"""

import random
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "evaluation_results_d2_settings_comparison"
JUDGE_DIR = ROOT / "llm_judge_evaluation"
OUT_DIR = Path(__file__).resolve().parent

SEED = 42

# Sizing is fatigue-driven, not arbitrary. A rater reads ~170 words per normal item (the
# person's message plus three replies) and ~320 on an attention-check item, because the decoy
# system's replies run 150-235 words. At 9 + 2 the form is ~2,100 words, i.e. 11-15 minutes.
# Going to 12 items pushes past 16 minutes, which is where volunteer answer quality drops.
# Power is not the binding constraint: 9 scored items x 25 raters is 225 judgments, enough to
# separate a 45% selection rate from the 33.3% chance level at p < 0.001.
N_SCORED_ITEMS = 11
N_ATTENTION_CHECKS = 2

# Emotions guaranteed a place in the item set; the rest of the draw stays random.
#
# Drawing all 13 at random from 32 categories reliably produced odd-looking sets - one had
# "faithful" and "prepared" in it but no plain sadness, no loneliness and no anger at all.
# Forcing one emotion at a time just moved the hole around: adding anger evicted the anxiety
# family, which is the single most common presenting concern in mental-health support.
#
# So the four families most central to this application are fixed, and the remaining nine
# items are still drawn at random. This is stratified sampling, and it is more defensible
# than a pure random draw, not less - it guarantees the coverage a reader checks for while
# leaving no room to hand-pick favourable examples: WHICH conversation represents each of
# these emotions is still chosen at random, exactly as for the unforced ones.
REQUIRED_EMOTIONS = ["sad", "lonely", "angry", "anxious"]

# Conversations pinned to specific item positions because most raters are students, and a
# study whose situations are all about spouses, children and jobs asks them to judge empathy
# for lives they have not lived. These three are student situations told from the student's
# own point of view.
#
# The cost, which has to be stated rather than buried: only 4 of the 64 AI-judge-scored
# conversations are student-related, and none from a student's own perspective, so these come
# from the wider 300-conversation test set instead. They are NOT AI-judge-scored, so they
# cannot feed the human-vs-AI-judge agreement analysis. That analysis still runs on the other
# 9 scored items, which is enough. Attention checks are therefore restricted to judged items,
# where the decoy's poor quality is backed by recorded scores rather than our own reading.
PINNED_ITEMS = {
    5: "hit:6423_conv:12846",   # prepared     - "My Calculus final was no joke!..."
    7: "hit:4693_conv:9386",    # furious      - classmate stole their ideas for a paper
    8: "hit:8061_conv:16123",   # anticipating - "I took a test for college... our midterm."
}

OUR_SYSTEM = "two_stage_phi"
BASELINES = ["single_model_same_size", "single_model_bigger"]
DECOY = "fine_tuned"          # used only in attention-check items

DISPLAY_NAME = {
    "two_stage_phi": "Our system (two-stage, Phi-4-mini)",
    "single_model_same_size": "Single model, same size (no detector)",
    "single_model_bigger": "Single bigger model (no detector)",
    "fine_tuned": "Fine-tuned monolithic model (ATTENTION-CHECK DECOY)",
}


JUDGE_FILES = [
    "nemotron_openrouter_scores.csv",
    "mistral_large_scores.csv",
    "gptoss_groq_scores.csv",
]


def core_conversations():
    """conv_ids scored by all three AI judges - the overlap that makes the
    human-vs-AI-judge agreement analysis possible."""
    sets = [set(pd.read_csv(JUDGE_DIR / f)["conv_id"]) for f in JUDGE_FILES]
    return set.intersection(*sets)


def decoy_judge_scores():
    """Mean AI-judge empathy score for the decoy system, per conversation.

    Used to choose WHICH items become attention checks. A check only works if the decoy reply
    is unmistakably poor: if it is merely mediocre, a careful rater may legitimately prefer it
    and then be wrongly excluded for inattention - worse than having no check at all. Picking
    the conversations where all three judges scored the decoy lowest makes that choice on
    recorded evidence rather than on our own reading of the replies.
    """
    frames = []
    for name in JUDGE_FILES:
        df = pd.read_csv(JUDGE_DIR / name)
        frames.append(df[df["setting"] == DECOY][["conv_id", "empathy"]])
    return pd.concat(frames).groupby("conv_id")["empathy"].mean()


def load_systems():
    systems = {}
    for name in [OUR_SYSTEM, *BASELINES, DECOY]:
        df = pd.read_csv(PRED_DIR / f"{name}_scored_predictions.csv")
        systems[name] = df.set_index("conv_id")
    return systems


def clean(text):
    return " ".join(str(text).replace("_comma_", ",").split())


def select_items():
    systems = load_systems()
    pool = core_conversations()
    for df in systems.values():
        pool &= set(df.index)

    base = systems[OUR_SYSTEM]
    rng = random.Random(SEED)

    # One conversation per emotion, emotions drawn at random - prevents cherry-picking and
    # guarantees the item set spans distinct emotional situations. Pinned items keep their own
    # emotion, and it is dropped from the random draw so no emotion appears twice.
    by_emotion = {}
    for conv_id, row in base.loc[sorted(pool)].iterrows():
        by_emotion.setdefault(row["gold_emotion"], []).append(conv_id)

    pinned_emotions = {base.loc[c, "gold_emotion"] for c in PINNED_ITEMS.values()}

    total = N_SCORED_ITEMS + N_ATTENTION_CHECKS
    n_random = total - len(PINNED_ITEMS)
    required = [e for e in REQUIRED_EMOTIONS if e in by_emotion and e not in pinned_emotions]
    remaining = [e for e in sorted(by_emotion)
                 if e not in required and e not in pinned_emotions]
    rng.shuffle(remaining)
    picked = required + remaining[: n_random - len(required)]
    rng.shuffle(picked)                     # so the guaranteed emotions are not always first

    drawn = iter(picked)
    chosen = []
    for position in range(1, total + 1):
        if position in PINNED_ITEMS:
            conv_id = PINNED_ITEMS[position]
            chosen.append((base.loc[conv_id, "gold_emotion"], conv_id))
        else:
            emotion = next(drawn)
            chosen.append((emotion, rng.choice(sorted(by_emotion[emotion]))))

    # Attention checks go on the conversations where the AI judges scored the decoy worst, so
    # only judged items are eligible - a pinned item has no judge score to justify the choice.
    # They are also spread across the form - one in each half. Taking simply the two lowest
    # scores put both checks in the last two items, which fails twice over: careless answering
    # in the first half would go undetected, and a rater who merely tires near the end would
    # trip both checks at once and be thrown out for inattention they did not show earlier.
    decoy_scores = decoy_judge_scores()
    # Item 1 is excluded: a rater is still working out what the task is on the very first
    # screen, so a check there risks excluding someone for inexperience rather than inattention.
    eligible = [i for i in range(1, total) if chosen[i][1] in pool]
    worst_first = lambda i: decoy_scores.get(chosen[i][1], 99)
    midpoint = total // 2
    check_positions = set()
    for group in ([i for i in eligible if i < midpoint], [i for i in eligible if i >= midpoint]):
        if group and len(check_positions) < N_ATTENTION_CHECKS:
            check_positions.add(min(group, key=worst_first))
    for index in sorted(eligible, key=worst_first):     # top up if a half had none
        if len(check_positions) >= N_ATTENTION_CHECKS:
            break
        check_positions.add(index)

    # Counterbalanced ordering, not free random shuffling. Pure shuffling on a set this small
    # comes out lumpy - an early draft put our system in the middle slot 6 times out of 11 -
    # and position preference would then be indistinguishable from quality preference. The
    # three cyclic rotations place each system in each slot exactly once, so repeating them
    # across the scored items balances position exactly. Which rotation lands on which item is
    # still randomised, so the ordering carries no information about content.
    rotations = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    scored_orders = [rotations[i % 3] for i in range(N_SCORED_ITEMS)]
    check_orders = [rotations[i % 3] for i in range(N_ATTENTION_CHECKS)]
    rng.shuffle(scored_orders)
    rng.shuffle(check_orders)

    items = []
    for idx, (emotion, conv_id) in enumerate(chosen):
        is_check = idx in check_positions
        shown = [OUR_SYSTEM, BASELINES[0], DECOY if is_check else BASELINES[1]]
        rotation = check_orders.pop() if is_check else scored_orders.pop()
        order = [shown[i] for i in rotation]
        replies = [
            {
                "label": "ABC"[pos],
                "system": sys_name,
                "text": clean(systems[sys_name].loc[conv_id, "generated_response"]),
                "words": int(systems[sys_name].loc[conv_id, "word_count"]),
            }
            for pos, sys_name in enumerate(order)
        ]
        items.append(
            {
                "number": idx + 1,
                "conv_id": conv_id,
                "gold_emotion": emotion,
                "user_text": clean(systems[OUR_SYSTEM].loc[conv_id, "user_text"]),
                "is_attention_check": is_check,
                "is_judged": conv_id in pool,
                "is_pinned": (idx + 1) in PINNED_ITEMS,
                "replies": replies,
            }
        )
    return items


# ---------------------------------------------------------------- document helpers

def h(doc, text, level):
    doc.add_heading(text, level=level)


def p(doc, text, bold=False, italic=False, size=11, color=None, space_after=6):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold, run.italic = bold, italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    para.paragraph_format.space_after = Pt(space_after)
    return para


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for cell, head in zip(t.rows[0].cells, headers):
        cell.text = str(head)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()
    return t


# ---------------------------------------------------------------- the document

N_TOTAL = N_SCORED_ITEMS + N_ATTENTION_CHECKS

CONSENT_TEXT = (
    "Thank you for helping with this study.\n\n"
    f"You will read {N_TOTAL} short messages from people describing something they are going "
    "through. For each one, you will see three possible replies. Your task is simply to "
    "choose which reply you feel responds best to that person's feelings.\n\n"
    "There are no right or wrong answers - we want your honest impression.\n\n"
    "This takes about 12-15 minutes. Your responses are completely anonymous; we do not "
    "collect your name or email. Some messages describe sad or stressful situations. You "
    "may stop at any time.\n\n"
    "By continuing, you agree to take part."
)

ITEM_LAYOUT = (
    f"[Title]  Situation 1 of {N_TOTAL}\n"
    "[Description]  Someone wrote: \"...the person's message...\"\n\n"
    "Response A: ...\n"
    "Response B: ...\n"
    "Response C: ...\n\n"
    "[Question - multiple choice - required]\n"
    "Which reply best understands and responds to how this person is feeling?\n"
    "    Response A  /  Response B  /  Response C"
)

THE_QUESTION = "Which reply best understands and responds to how this person is feeling?"


def build_document(items):
    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    h(doc, "Human Evaluation Study", 0)
    p(doc, "Context-Aware Empathetic Text Generation for Mental Health Support",
      italic=True, size=12)
    p(doc, "Build instructions, complete item set, and analysis plan.", size=10)

    # ---- 1. purpose
    h(doc, "1. What this study is for", 1)
    p(doc, "The study has two goals, and the second matters more than the first.")
    bullet(doc, "Goal 1 - Independent evidence for the main claim: do real people prefer our "
                "system's replies over the baselines?")
    bullet(doc, "Goal 2 (the important one) - Validate the AI judges. Every judge-based result "
                "in the results chapter currently rests on three AI models. If human raters "
                "agree with those judges, that entire evidence base becomes defensible. This is "
                "why every conversation below is drawn from the 64 conversations that all three "
                "AI judges scored.")

    # ---- 2. design
    h(doc, "2. Design decisions and the evidence behind them", 1)
    table(doc,
          ["Decision", "What we do", "Why"],
          [
              ["Comparison format", "Forced choice between replies shown side by side",
               "Both human-evaluation papers in our reading list (COLING 2025 main.355 and "
               "main.598) use A/B preference rather than solo rating. main.598 gives the "
               "reason: pairwise choice 'accounts for individual variations among annotators'."],
              ["Replies per item", "3",
               "A deviation from the papers' 2, forced by rater budget. With ~12 items a "
               "pairwise design would test only one comparison. Three-at-once tests every "
               "comparison on every item."],
              ["Dimensions", "Empathy only",
               "Fatigue control. main.355 used 4 aspects and main.598 used 3, but with only 3 "
               "and 7 expert annotators. We trade aspect breadth for rater count."],
              ["Item count", f"{N_SCORED_ITEMS + N_ATTENTION_CHECKS} "
                             f"({N_SCORED_ITEMS} scored + {N_ATTENTION_CHECKS} attention checks)",
               "Keeps completion near 13-15 minutes. Past roughly 15 minutes volunteers begin "
               "answering carelessly."],
              ["Item selection", "Random, fixed seed, one conversation per emotion",
               "Removes any possibility of cherry-picking favourable examples - the easiest "
               "attack on a human study."],
              ["Student situations", f"{len(PINNED_ITEMS)} items fixed to student scenarios",
               "Most raters are students. A study whose situations are all about spouses, "
               "children and jobs asks them to judge empathy for lives they have not lived. "
               "Only 4 of the 64 judged conversations are student-related and none from a "
               "student's own point of view, so these three come from the wider 300-conversation "
               "test set and are not AI-judge-scored - see the limitation on the last page."],
              ["Reply ordering", "Randomised independently for every item",
               "Controls position bias. If our system always appeared first the result would "
               "be worthless."],
              ["Attention checks", f"{N_ATTENTION_CHECKS} items contain an obviously poor reply",
               "Directly addresses the risk that bored raters click randomly. Anyone selecting "
               "the decoy is excluded from analysis."],
              ["Which items become checks",
               "The conversations where the AI judges scored the decoy lowest",
               "A check only works if the decoy is unmistakably poor. If it is merely mediocre, "
               "a careful rater may genuinely prefer it and then be wrongly excluded - worse "
               "than having no check. The two chosen here scored 2.0 and 2.7 out of 10 from the "
               "judge panel, against a decoy average of about 3.5."],
          ])

    p(doc, "Systems being compared:", bold=True)
    table(doc, ["Rater sees", "Actually is", "Why it is in the study"],
          [["A, B or C", "Our system (two-stage, Phi-4-mini)",
            "The system the thesis claims."],
           ["A, B or C", "Single model, same size, no detector",
            "The core claim: does the decoupled design beat a monolithic model of equal size?"],
           ["A, B or C", "Single bigger model, no detector",
            "The honest hard test. The AI judges say we lose this one; including it anyway is "
            "what makes the study credible."],
           ["A, B or C (2 items only)", "Fine-tuned monolithic model",
            "Attention-check decoy. Produces ~200-word replies full of questions and jokes."]])

    # ---- 3. form structure
    h(doc, "3. How to build the Google Form", 1)
    p(doc, "Settings to fix before adding any questions:", bold=True)
    bullet(doc, "Do NOT collect email addresses - the study is anonymous.")
    bullet(doc, "Turn OFF 'Shuffle question order'. Reply order is already randomised inside "
                "each item; shuffling items as well makes the answer key harder to track.")
    bullet(doc, "Mark every item Required, so there are no partial responses to clean up.")
    p(doc, "Never write a model's name anywhere in the form. Raters see only Response A, "
           "Response B and Response C.", bold=True)

    p(doc, "Section 1 - Introduction and consent (paste this text):", bold=True)
    p(doc, CONSENT_TEXT, italic=True, size=10)

    p(doc, "Section 2 - Two background questions (optional but useful):", bold=True)
    bullet(doc, "Are you a: Student / Faculty member / Other")
    bullet(doc, "Have you studied psychology or counselling? Yes / No")
    p(doc, "These let you report who your raters were, and check whether faculty and students "
           "answered differently.", size=10, italic=True)

    p(doc, f"Section 3 - The {N_TOTAL} items. Each is laid out exactly like this:", bold=True)
    p(doc, ITEM_LAYOUT, italic=True, size=10)

    p(doc, "Section 4 - Closing (optional free text):", bold=True)
    p(doc, "\"If anything stood out to you about the replies you read, feel free to tell us "
           "here.\"  (Paragraph answer, not required.)", italic=True, size=10)

    # ---- 4. the question
    h(doc, "4. The question wording", 1)
    p(doc, "Use exactly this wording on every item, with no variation:", bold=True)
    p(doc, THE_QUESTION, bold=True, size=12)
    p(doc, "Options: Response A / Response B / Response C. Required.", size=10)
    p(doc, "There is deliberately no 'they are the same' option. With three replies on screen a "
           "tie option absorbs a large share of answers and weakens the result; the papers used "
           "Tie only in two-way comparisons.", size=10)

    doc.add_page_break()

    # ---- 5. items
    h(doc, f"5. The {len(items)} items - copy these into the form", 1)
    p(doc, "Reply order below is already randomised. Paste exactly as shown: do not reorder, do "
           "not relabel, and do not include the word counts.", bold=True, size=10)

    for item in items:
        h(doc, f"Situation {item['number']} of {len(items)}", 2)
        para = doc.add_paragraph()
        lead = para.add_run("Someone wrote:  ")
        lead.bold = True
        lead.font.size = Pt(10)
        quoted = para.add_run('"' + item["user_text"] + '"')
        quoted.italic = True
        quoted.font.size = Pt(11)

        for reply in item["replies"]:
            rp = doc.add_paragraph()
            rp.paragraph_format.space_after = Pt(8)
            lab = rp.add_run("Response " + reply["label"] + ":  ")
            lab.bold = True
            lab.font.size = Pt(11)
            body = rp.add_run(reply["text"])
            body.font.size = Pt(11)

        qp = doc.add_paragraph()
        q = qp.add_run("Q: " + THE_QUESTION)
        q.bold = True
        q.font.size = Pt(10)
        doc.add_paragraph()

    doc.add_page_break()

    # ---- 6. analysis
    h(doc, "6. How to analyse the responses", 1)
    p(doc, "Run these in order. Step 1 comes first, because it decides whose data counts.",
      bold=True)
    bullet(doc, "Step 1 - Exclusion. Drop any rater who selected the decoy reply on either "
                "attention-check item. Report how many were dropped; this is evidence of "
                "quality control, not something to hide.")
    bullet(doc, "Step 2 - Selection rate per system. What share of all choices went to each "
                "system? Chance is 33.3%. Test our system against chance with a binomial test.")
    bullet(doc, "Step 3 - Head to head. For each rater, count how often they chose our system "
                "versus each baseline, then compare those counts across raters with a Wilcoxon "
                "signed-rank test. This is the number that answers the core claim.")
    bullet(doc, "Step 4 - Rater agreement. Compute Fleiss' kappa across raters, to show people "
                "were not guessing. Report it even if it is low - a low kappa on a subjective "
                "task is a normal and publishable finding.")
    bullet(doc, "Step 5 - The important one. For each item compare the humans' majority choice "
                "against whichever system the AI judges scored highest, and report the "
                "agreement rate. This is what justifies every AI-judge result in the thesis.")
    p(doc, "Step 5 needs the AI-judge scores for the two-stage system, which are still being "
           "collected. Everything else can run the moment responses arrive.",
      italic=True, size=10)

    # ---- 7. limitations
    h(doc, "7. Limitations to report honestly", 1)
    bullet(doc, "Raters are classmates, friends and faculty - not clinicians, and not people "
                "actually seeking mental-health support.")
    bullet(doc, f"{N_SCORED_ITEMS + N_ATTENTION_CHECKS} conversations is a small item set. The "
                "comparison papers used 100 and 200 conversations, though with far fewer raters "
                "(3 and 7) than we expect to recruit.")
    bullet(doc, "Raters may know the authors, which can bias them toward a favourable answer. "
                "Blinding the system names is the protection against this, which is why no "
                "model name may appear anywhere in the form.")
    bullet(doc, "Only empathy was measured. Fluency, coherence and informativity were not.")
    judged = [i["number"] for i in items if i["is_judged"] and not i["is_attention_check"]]
    unjudged = [i["number"] for i in items if not i["is_judged"]]
    bullet(doc, f"Situations {', '.join(str(n) for n in unjudged)} are student scenarios taken "
                "from the wider test set, and were not scored by the AI judges. They therefore "
                "count towards the preference result but not towards the human-vs-AI-judge "
                f"agreement analysis, which runs on the {len(judged)} scored items that were "
                "judged. That is enough for the agreement figure, but it should be stated "
                "rather than left for a reader to discover.")

    p(doc, "Emotions covered by the item set:", bold=True)
    p(doc, ", ".join(sorted({i["gold_emotion"] for i in items})) + ".", size=10)
    p(doc, f"{len(items)} of the 32 EmpatheticDialogues emotions, one conversation each. "
           f"Four families central to this application ({', '.join(REQUIRED_EMOTIONS)}) were "
           "guaranteed a place; the remaining items were drawn at random. Which conversation "
           "represents each emotion was random in every case, forced or not, so the set covers "
           "what a reader would expect without any scope to hand-pick favourable examples.",
      size=10, italic=True)

    # ---- 8. answer key
    doc.add_page_break()
    h(doc, "8. Answer key - which reply is ours", 1)
    p(doc, "DO NOT show this page to raters, and do not paste it into the form. It exists so "
           "you can score the responses after collection.",
      bold=True, color=RGBColor(0xC0, 0x00, 0x00))
    p(doc, "Cross-check against ANSWER_KEY_do_not_share_with_raters.csv, which additionally "
           "records what the other two responses were on every item.", size=10, italic=True)

    for item in items:
        ours = next(r["label"] for r in item["replies"] if r["system"] == OUR_SYSTEM)
        line = f"Situation {item['number']}:  Our model reply: Response {ours}"
        if item["is_attention_check"]:
            decoy = next(r["label"] for r in item["replies"] if r["system"] == DECOY)
            line += f"    [attention check - Response {decoy} is the decoy; "
            line += "exclude any rater who chose it]"
        para = doc.add_paragraph()
        run = para.add_run(line)
        run.font.size = Pt(11)
        run.bold = item["is_attention_check"]
        para.paragraph_format.space_after = Pt(4)

    p(doc, "")
    p(doc, "Full breakdown of every position:", bold=True, size=10)
    table(doc,
          ["Situation", "Emotion", "Response A", "Response B", "Response C"],
          [[item["number"], item["gold_emotion"]]
           + [DISPLAY_NAME[next(r["system"] for r in item["replies"] if r["label"] == lab)]
              .replace(" (ATTENTION-CHECK DECOY)", " [DECOY]")
              for lab in "ABC"]
           for item in items])

    return doc


def build_answer_key(items):
    rows = []
    for item in items:
        for reply in item["replies"]:
            rows.append({
                "item_number": item["number"],
                "conv_id": item["conv_id"],
                "gold_emotion": item["gold_emotion"],
                "is_attention_check": item["is_attention_check"],
                "ai_judge_scored": item["is_judged"],
                "student_situation": item["is_pinned"],
                "shown_as": "Response " + reply["label"],
                "actual_system": reply["system"],
                "system_description": DISPLAY_NAME[reply["system"]],
                "is_decoy": reply["system"] == DECOY,
                "word_count": reply["words"],
            })
    return pd.DataFrame(rows)


def main():
    items = select_items()

    doc = build_document(items)
    docx_path = OUT_DIR / "Human_Evaluation_Study_Plan.docx"
    doc.save(docx_path)
    print("Wrote " + str(docx_path))

    key = build_answer_key(items)
    key_path = OUT_DIR / "ANSWER_KEY_do_not_share_with_raters.csv"
    key.to_csv(key_path, index=False)
    print("Wrote " + str(key_path))

    total_words = sum(
        len(i["user_text"].split()) + sum(len(r["text"].split()) for r in i["replies"])
        for i in items
    )
    print()
    print(f"Items: {len(items)} ({N_SCORED_ITEMS} scored + {N_ATTENTION_CHECKS} attention checks)")
    print(f"Words a rater reads: {total_words}")
    print(f"Estimated completion: {total_words / 200:.0f}-{total_words / 150:.0f} minutes "
          "(brisk to careful reading)")
    print(f"Distinct emotions covered: {len({i['gold_emotion'] for i in items})}")
    print("Attention-check items: "
          + str([i["number"] for i in items if i["is_attention_check"]]))
    print()
    print("Position balance (times each system appeared in each slot):")
    print(pd.crosstab(key["actual_system"], key["shown_as"]).to_string())


if __name__ == "__main__":
    main()
