# Final writing — first draft of the thesis report

This folder holds the **first draft of the full report**, written to the section list in the CSE400 course-outcome sheet. One file per chapter, in Word and in Markdown.

| File | Contents |
|---|---|
| `Chapter_0_Front_Matter.docx` | Declaration, Approval, Abstract, Acknowledgment, Nomenclature |
| `Chapter_1_Introduction.docx` | 1.1 Background, 1.2 Rationale, 1.3 Problem Statement, 1.4 Objectives, 1.5 Methodology in Brief, 1.6 Scopes and Challenges, 1.7 Thesis Structure |
| `Chapter_2_Literature_Review.docx` | 2.1 Preliminaries, 2.2 Review of Existing Research, 2.3 Summary of Key Findings |
| `Chapter_3_Requirements_Impacts_Constraints.docx` | 3.1 Final Specifications and Requirements, 3.2 Societal Impact, 3.3 Environmental Impact, 3.4 Ethical Issues, 3.5 Standards, 3.6 Project Management Plan, 3.7 Risk Management, 3.8 Economic Analysis |
| `Chapter_4_Proposed_Methodology.docx` | 4.1 Design Process or Methodology Overview, 4.2 Preliminary Design or Design (Model) Specification, 4.3 Data Collection and Preparation, 4.4 Implementation of the Selected Design |
| `Chapter_5_Results_and_Analysis.docx` | 5.1 Performance Evaluation, 5.2 Analysis of Design Solutions, 5.3 Final Design Adjustments, 5.4 Statistical Analysis, 5.5 Comparisons and Relationships, 5.6 Discussion |
| `Chapter_6_Conclusion.docx` | 6.1 Conclusion, 6.2 Future Work |
| `Appendix.docx` | Tools and Technologies Used (CO8) + 10 supplementary tables |
| `References.docx` | 49 references, numbered in order of first citation |

Total: about 25,800 words, 36 tables and 10 figures in the chapters, 14 tables in the appendix. Chapter 3 is 9 pages and Chapter 4 is 14.

## Course outcome to section

| CO | Required section | Where it is |
|---|---|---|
| CO2 | Final specifications and requirements | 3.1 (+ Table 3.1, requirement-by-requirement verification) |
| CO3 | Societal impact | 3.2 |
| CO4 | Environmental impact | 3.3 |
| CO5 | Design process; preliminary design specification | 4.1, 4.2, 4.3, 4.4 |
| CO6 | Analysis of alternative design solutions | 5.1, 5.2, 5.3, 5.4, 5.5, 5.6 |
| CO7 | Final design adjustments after evaluation | 5.3 (+ Table 5.10, the adjustment log) |
| CO8 | Contemporary tools | Appendix, Tools and Technologies Used (+ presentation) |
| CO9 | Literature survey | 2.1, 2.2, 2.3 |
| CO10 | Resource management | 3.6, 3.7 |
| CO11 | Economic analysis | 3.8 |
| CO12 | Ethical considerations | 3.4 (+ 3.5 Standards) |

## Numbering of Chapter 5

The course-outcome sheet lists two sections as "5.2" and skips 5.5. This draft uses the order in the department's own report template, which the sample reports follow:

> 5.1 Performance Evaluation · 5.2 Analysis of Design Solutions · 5.3 Final Design Adjustments · 5.4 Statistical Analysis · 5.5 Comparisons and Relationships · 5.6 Discussion

Every cross-reference in the text has been renumbered to match. **If the supervisor wants the sheet's numbering instead, say so and it can be changed in one run of the build script.**

## What was left out, and why

The writing kit in `writing/` holds 70 table placements. A thesis cannot print all of them, so tables were kept only where the text needs the numbers in front of the reader. Ten were moved to the appendix:

| Moved to appendix | Reason |
|---|---|
| Requirement verification | the requirements themselves are listed in 3.1; the test evidence is supporting detail |
| Risk register | the three risks that mattered are described in 3.7; the full twelve-row table is detail |
| Licences of data and models | every licence is named in the text of 3.4.5 |
| Development cost breakdown | the text of 3.8 states what was spent and on what |
| Per-emotion precision/recall of the detector | 32 rows of detail behind one sentence in 5.1.2 |
| Human-evaluation item design | the design is described in the text of 5.1.9 |
| WASSA scores by length band | the length control is summarised in 5.2.3 and in Table 5.21 |
| Detector significance tests | the comparison itself is Table 4.3 |
| Length–empathy correlations per system | one sentence in 5.4.5 carries the result |
| WASSA validity checks | summarised in 5.4.6 |
| Detector-swap detail | the same result is in Table 5.12 (error propagation) |
| Detector-level parameter efficiency | the system-level table is what the claim rests on |
| Paired resource tests | the p-values are quoted in the text of 5.5.7 |
| AI-judge selection criteria | the criteria are argued in the text of 4.1.5.1 |

Four tables that exist in the kit are **not printed anywhere**, because every number in them already appears in a table that is printed: the per-judge dimension breakdown and the WASSA-by-system table (both inside Table 5.6 and Table 5.1), the final-system scorecard (inside Table 5.1) and the significance tests for the initial pipeline (superseded by Table 5.11).

## Still to fill in

41 markers remain, all of them things only the team or the pending study can supply.

- **Front matter (4):** member names and IDs, final title, supervisor/approval block, acknowledgements.
- **Human evaluation (14, marked PENDING):** 5.1.9 results and Table 5.7, 5.4.8 statistics, 5.5.2 comparison with published human studies, 5.6.6 participant count, 5.6.8 closing paragraph. Everything else is complete; paste the numbers and rebuild.
- **Team facts (23):** helpline number, ethics-review position, BRAC AI declaration, member roles, six dates, GPU price, electricity tariff, two unlogged GPU-hour rows, commercial API price with date, the source of the per-emotion guidance, case-study conversations (5.1.10) and the demo URL.

## Chapter 4 figures

The five Chapter 4 figures are drawn by `_build/make_ch4_figures.py`: the design loop, the system pipeline, the cue-extraction mechanism, the prompt assembly and the data pipeline. Two of them replace the writing-kit versions through `figures_override/`. Re-run that script before the main build if a figure needs changing.

## Table overrides

A table that needs fewer columns or shorter cells for this draft lives in `tables_override/` and wins over the writing kit's copy. Everything else is taken from the kit, so a rebuilt number flows through automatically.

## Rebuilding

Edit the text in `_src/`, not the chapter files — the chapter files are generated and will be overwritten.

```
python "final writing/_build/build_final_draft.py"
```

The script numbers every table and figure per chapter, rewrites every cross-reference, resolves citations to IEEE numbers from `writing/_scripts/references.py`, copies the tables it uses into `tables/`, and writes both the Markdown and the Word file for each chapter. It stops with an error if a reference points at a table that is not in the draft, so the numbering cannot silently break.

Chapters 2 to 5 came from `writing/_scripts/final_text_src/` and were copied here once. The copies here carry fewer citations than the kit: a source is cited where the claim is made rather than every time it is mentioned again, which brings the draft to about one citation per 190 words, in line with the sample reports in `Papers/Final Papers/`. If you copy a section from the kit again, trim its repeated citations the same way. If a number changes in the evidence, update the kit first (`python writing/_scripts/build_writing_tables.py`), then copy the affected section file into `_src/` and rebuild.
