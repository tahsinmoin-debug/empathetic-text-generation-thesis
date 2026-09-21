# Human evaluation

## Use version 2

`Human_Evaluation_Form_v2.docx` is the current study, revised after the supervisor's feedback. Everything named `..._v2` belongs to it; the files without `v2` are the earlier forced-choice design, kept only as a record.

| File | What it is |
|---|---|
| `create_google_form_v2.gs` | **Start here.** Apps Script that builds the whole form in Google Forms |
| `Human_Evaluation_Form_v2.docx` | Every word the raters see, the design reasons, and the analysis plan |
| `ANSWER_KEY_v2_do_not_share.csv` | Which of Response A, B, C is which system — never send this to a rater |
| `form_items_v2.csv` | The same item set, flat, for the analysis script |
| `build_human_eval_form_v2.py` | Regenerates all of the above from the saved model outputs |

## Building the form in five minutes

1. Open [script.google.com](https://script.google.com) and choose **New project**.
2. Select everything in the editor (Ctrl+A), delete it, paste the whole of `create_google_form_v2.gs`, then press **Ctrl+S**. Saving matters: Apps Script runs the last saved version, and the function dropdown only lists functions from that saved version.
3. In the toolbar, set the function dropdown (the one next to **Debug**) to **createForm**.
4. Press **Run**. Google asks for permission the first time — allow it.
5. Open **Execution log**. It prints two links: one to edit the form, one to send to raters.

If the function dropdown does not list **createForm**, the file was not saved — press Ctrl+S and look again. If the log says "Execution completed" with no links, the dropdown was still on **myFunction** — select **createForm** and run again. Each run creates a *new* form, so run it once; if you run it twice, delete the extra form from Google Drive.

The script creates all 80 questions: 8 situations × (3 grids of three replies + 1 preference question), plus the consent question and an optional comment box. Nothing in the form names a model.

## What changed from version 1

- **Ratings, not a single choice.** Each reply is rated 1–5 on empathy/emotional validation, contextual relevance and supportiveness, exactly as the supervisor specified. One preference question per situation is kept as well, so the study still produces a head-to-head win rate.
- **No attention checks.** The two decoy replies from the fine-tuned model are gone; those situations now show three real replies like every other one.
- **8 situations, not 13.** Nine ratings per situation means 13 situations would take about 25 minutes. Eight keeps it near 15, which is where volunteers still read carefully. With 20–30 raters that is 160–240 paired observations for every comparison.

## The eight situations

Four core supportive emotions (sad, lonely, angry, anxious), three student situations, one positive emotion drawn at random with a fixed seed. Five of the eight were also scored by the AI judges, which is what lets us check whether human raters and the AI panel agree.

## Rules for running it

- Do not tell raters which reply comes from which system, and do not share the answer key.
- Do not turn on "Shuffle question order". The reply order is already balanced across situations (each system appears in each position two or three times); shuffling on top of that breaks the key.
- Aim for 20–30 people who will answer seriously. Two exclusion rules are fixed in advance: anyone who answers "No" to consent, and anyone who gives every reply in every situation the same rating.

## When the responses are in

Download the responses as CSV into this folder and ask for the analysis script. It will report, per system: the mean of each of the three ratings with paired Wilcoxon tests and Holm correction, the preference rate against the 33.3% chance level, agreement between raters, and how closely the human ranking matches the AI-judge ranking on the five judged situations. Those numbers fill the five PENDING spots in the thesis (Sections 5.1.9, 5.4.8, 5.5.2, 5.6.6 and 5.6.8).
