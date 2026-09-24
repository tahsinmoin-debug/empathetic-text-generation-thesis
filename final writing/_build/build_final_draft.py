"""Assemble the first draft of the thesis report, one file per chapter.

Sources live in `final writing/_src/`. Chapters 2 to 5 are the text written for the
writing kit (`writing/_scripts/final_text_src/`), copied here once with Chapter 5
renumbered to the department's order:

    5.1 Performance Evaluation      5.4 Statistical Analysis
    5.2 Analysis of Design Solutions 5.5 Comparisons and Relationships
    5.3 Final Design Adjustments     5.6 Discussion

Chapter 1, Sections 3.1, 3.2, 3.3, 3.5, the front matter and Chapter 6 are written
for this draft and exist only here.

Tables are split into MAIN (printed where the text refers to them) and APPENDIX
(printed at the end, referred to as Table A.n). Both are renumbered per chapter, and
every "Table x.y" / "Figure x.y" reference in the text is rewritten to match.

Output, in `final writing/`:
    Chapter_0_Front_Matter.md/.docx ... Chapter_6_Conclusion.md/.docx
    Appendix.md/.docx, References.md/.docx
    tables/ (only the tables used), figures/

Usage:
    python "final writing/_build/build_final_draft.py"
"""

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SRC = OUT / "_src"
KIT = ROOT / "writing"
sys.path.insert(0, str(KIT / "_scripts"))
from references import REFS  # noqa: E402
from build_final_text import to_docx, CITE, TABLE, FIGURE, csv_as_markdown  # noqa: E402

# ---------------------------------------------------------------- structure
CHAPTERS = [
    ("Chapter_0_Front_Matter", None, ["ch0_front"]),
    ("Chapter_1_Introduction", "Chapter 1: Introduction", ["ch1"]),
    ("Chapter_2_Literature_Review", "Chapter 2: Literature Review", ["ch2_1", "ch2_2", "ch2_3"]),
    ("Chapter_3_Requirements_Impacts_Constraints", "Chapter 3: Requirements, Impacts and Constraints",
     ["ch3_1", "ch3_2", "ch3_3", "ch3_4", "ch3_5", "ch3_6", "ch3_7", "ch3_8"]),
    ("Chapter_4_Proposed_Methodology", "Chapter 4: Proposed Methodology",
     ["ch4_1", "ch4_2", "ch4_3", "ch4_4"]),
    ("Chapter_5_Results_and_Analysis", "Chapter 5: Results and Analysis",
     ["ch5_1", "ch5_2a", "ch5_2b", "ch5_3", "ch5_4", "ch5_6"]),
    ("Chapter_6_Conclusion", "Chapter 6: Conclusion and Future Work", ["ch6"]),
    ("Appendix", "Appendix", ["tools"]),
]

# Tables kept in the running text. Everything else with a marker goes to the appendix;
# a table with no marker at all is not used in this draft.
MAIN_TABLES = {
    "literature_review_matrix.csv",
    "ai_judge_panel.csv",
    "rag_negative_result.csv",
    "d1_candidates_design.csv", "d2_candidates_design.csv", "integration_variants_design.csv",
    "baseline_designs.csv", "detector_training_config.csv", "generation_config.csv",
    "preprocessed_data_summary.csv", "conversational_agent_design.csv",
    "key_parameters.csv",
    "context_window_long_scenarios.csv", "head_to_head_three_systems.csv",
    "rule_following.csv", "empathy_panel.csv",
    "human_eval_results_PENDING.csv",
    "emotion_accuracy_vs_literature.csv", "tools_and_versions.csv",
}
# Why each of these is not in the running text.
APPENDIX_REASON = {
    "configuration_and_hyperparameters.csv": "the full parameter listing behind Chapter 4",
    "cost_breakdown.csv": "item-by-item cost behind Section 3.8",
    "licences.csv": "licence of every dataset and model, named in Section 3.4.5",
    "requirements_verification.csv": "requirement-by-requirement verification behind Section 3.1",
    "risk_register.csv": "the full risk register behind Section 3.7",
    "d1_final_per_class.csv": "32 rows of per-emotion detail behind Section 5.1.2",
    "human_eval_item_design.csv": "item-by-item design of the human study described in Section 5.1.6",
    "gemma_length_band_analysis.csv": "length-band detail behind Section 5.2.3",
    "d1_backbone_significance.csv": "significance tests behind the detector comparison in Section 5.2.2",
    "length_vs_wassa_correlation.csv": "per-system correlations behind Section 5.4.5",
    "wassa_validity.csv": "validity checks on the empathy classifier, summarised in Section 5.4.4",
    "controlled_d1_swap_same_d2.csv": "detector-swap detail behind Section 5.5.2",
    "parameter_efficiency_detector.csv": "detector-level figures behind Section 5.5.2",
    "resource_paired_tests.csv": "paired tests behind the resource figures in Section 5.5.3",
    "ai_judge_selection_criteria.csv": "selection criteria behind Section 4.1.3.1",
    "tools_and_versions.csv": "software and hardware inventory for the tools appendix",
    "ai_judge_scores.csv": "the judges' relevance and safety scores, behind Section 5.1.5",
    "all_systems_master_comparison.csv": "every column measured for every system, behind Tables 5.1 and 5.2",
    "d2_bakeoff_candidates.csv": "the generator bake-off reported in Section 5.2.3",
    "integration_variant_ablation.csv": "the four integration variants compared in Section 5.2.4",
    "human_empathy_vs_literature.csv": "human empathy studies compared with ours, in Section 5.5.1",
    "ai_judge_agreement.csv": "the judge agreement checks behind Section 5.4.5",
    "d1_evolution_mismatch_fix.csv": "the training-input mismatch, quoted in Section 5.2.2",
    "two_stage_selection_accuracy.csv": "the stage-1 breakdown behind Section 5.1.4",
    "d1_final_overall.csv": "the detector's headline numbers, quoted in Section 5.1.2",
    "context_window_shift_scenarios.csv": "the ten short emotion-shift conversations behind Section 5.3.2",
    "design_adjustment_log.csv": "the full adjustment log narrated in Section 5.3",
    "d1_most_confused_pairs.csv": "counts behind the error chart in Section 5.1.2",
    "diversity_and_repetition.csv": "diversity and repetition detail behind Section 5.1.4",
    "d1_backbone_comparison.csv": "the numbers behind the detector chart in Section 5.2.2",
    "final_system_vs_baselines_significance.csv": "every test behind the forest plot in Section 5.4.2",
    "error_propagation.csv": "error-propagation detail behind Section 5.4.3",
    "detector_sensitivity_robertalarge.csv": "the RoBERTa-large substitution behind Section 5.4.3",
    "final_system_vs_baselines_judges.csv": "every judge comparison behind Section 5.4.5",
    "parameter_efficiency_systems.csv": "the numbers behind the cost chart in Section 5.5.3",
    "resource_efficiency.csv": "memory and timing measurements behind Section 5.5.3",
    "rules_validity.csv": "origin and measured effect of each rule, summarised in Section 5.5.4",
    "decoupled_vs_single_generator.csv": "the numbers behind the comparison in Section 5.5.5",
}


def find_asset(kind, name):
    """A table trimmed for this draft (fewer columns, shorter cells) lives in
    tables_override/ and wins; otherwise the file comes from the writing kit."""
    override = OUT / f"{kind}_override" / name
    if override.exists():
        return override
    hits = sorted(KIT.glob(f"*/{kind}/{name}")) + sorted((OUT / kind).glob(name))
    if not hits:
        raise SystemExit(f"Missing {kind[:-1]} file: {name}")
    return hits[0]


def copy_asset(kind, name):
    src = find_asset(kind, name)
    dst = OUT / kind / name
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
    return src


def load_sources():
    texts = {}
    for key, _, parts in CHAPTERS:
        for p in parts:
            f = SRC / f"{p}.md"
            if not f.exists():
                raise SystemExit(f"Missing source {f}")
            texts[p] = f.read_text(encoding="utf-8")
    return texts


def plan_numbers(texts):
    """Assign the final Table/Figure numbers and record the old -> new mapping.

    Two kinds of table appear in the sources: a [[TABLE ...]] marker backed by a CSV file,
    and a small table written directly in the text, which is recognised by its caption line.
    Both are numbered in the order they appear.
    """
    marker = re.compile(r"\[\[(TABLE|FIGURE) ([0-9.A-Z]+) \| ([^|\]]+?) \| ([^\]]+?)\]\]")
    inline = re.compile(r"^\*(Table|Figure) ([0-9]+\.[0-9]+): ", re.M)
    tmap, fmap, appendix, counters = {}, {}, [], {}

    # Numbers are the key of the old -> new mapping, so the same number must never be used
    # twice: a repeat silently gives one table the other's final number.
    seen = {}
    for p, text in texts.items():
        for m in marker.finditer(text):
            key = (m.group(1), m.group(2))
            if key in seen and seen[key] != m.group(3).strip():
                raise SystemExit(f"{m.group(1)} {m.group(2)} is used twice: "
                                 f"{seen[key]} and {m.group(3).strip()} (in {p})")
            seen[key] = m.group(3).strip()

    def take(prefix, kind, num):
        book = tmap if kind == "TABLE" else fmap
        if num in book:
            return
        counters.setdefault((prefix, kind), 0)
        counters[(prefix, kind)] += 1
        book[num] = f"{prefix}.{counters[(prefix, kind)]}"

    for key, title, parts in CHAPTERS:
        chapter = re.match(r"Chapter_(\d)", key)
        prefix = chapter.group(1) if chapter else "A"
        for p in parts:
            events = []
            for m in marker.finditer(texts[p]):
                events.append((m.start(), m.group(1), m.group(2), m.group(3).strip()))
            for m in inline.finditer(texts[p]):
                events.append((m.start(), m.group(1).upper(), m.group(2), None))
            for _, kind, num, fname in sorted(events):
                if kind == "TABLE" and fname is not None and fname not in MAIN_TABLES:
                    if fname not in [a[0] for a in appendix]:
                        cap = marker.search(texts[p][max(0, texts[p].find(fname) - 200):])
                        appendix.append((fname, next(m.group(4).strip() for m in marker.finditer(texts[p])
                                                     if m.group(3).strip() == fname)))
                    continue
                take(prefix, kind, num)

    old_by_file = {}
    for t in texts.values():
        for num, fname in re.findall(r"\[\[TABLE ([0-9.]+) \| ([^|\]]+?) \|", t):
            old_by_file.setdefault(fname.strip(), num)
    numbered = []
    start = counters.get(("A", "TABLE"), 0)  # continue after the tables printed with the tools section
    for i, (fname, cap) in enumerate(appendix, start=start + 1):
        new_num = f"A.{i}"
        numbered.append((fname, cap, new_num))
        if fname in old_by_file:
            tmap[old_by_file[fname]] = new_num
    return tmap, fmap, numbered


def renumber_refs(text, tmap, fmap):
    def repl(m):
        kind, num = m.group(1), m.group(2)
        book = tmap if kind == "Table" else fmap
        if num not in book:
            raise SystemExit(f"Reference to {kind} {num} but that {kind.lower()} is not in the draft")
        return f"{kind} {book[num]}"
    return re.sub(r"\b(Table|Figure) ([0-9]+\.[0-9]+)\b", repl, text)


def number_citations(texts):
    numbers = {}
    for key, _, parts in CHAPTERS:
        for p in parts:
            for group in CITE.findall(texts[p]):
                for k in [x.strip() for x in group.split(",") if x.strip()]:
                    if k not in REFS:
                        raise SystemExit(f"Unknown citation key: {k}")
                    numbers.setdefault(k, len(numbers) + 1)
    return numbers


def resolve_cites(text, numbers):
    return CITE.sub(lambda m: ", ".join(f"[{numbers[k.strip()]}]"
                                        for k in m.group(1).split(",") if k.strip()), text)


RENDERED = set()


def render(text, tmap, fmap, numbers):
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[[TABLE") or stripped.startswith("[[FIGURE"):
            if not (TABLE.fullmatch(stripped) or FIGURE.fullmatch(stripped)):
                raise SystemExit(f"Marker the renderer cannot read, so it would print raw: {stripped[:80]}")
        t = TABLE.fullmatch(line.strip())
        f = FIGURE.fullmatch(line.strip())
        if t:
            num, fname, cap = [x.strip() for x in t.groups()]
            if fname not in MAIN_TABLES or fname in RENDERED:
                continue  # printed in the appendix, or already printed earlier
            RENDERED.add(fname)
            src = copy_asset("tables", fname)
            out += ["", renumber_refs(csv_as_markdown(src), tmap, fmap), "", f"*Table {tmap[num]}: {cap}*", ""]
        elif f:
            num, fname, cap = [x.strip() for x in f.groups()]
            copy_asset("figures", fname)
            out += ["", f"![Figure {fmap[num]}](figures/{fname})", "", f"*Figure {fmap[num]}: {cap}*", ""]
        elif line.strip().startswith("> Note for the team"):
            continue  # guidance written for the writing kit, not part of the report
        else:
            out.append(resolve_cites(renumber_refs(line, tmap, fmap), numbers))
    return "\n".join(out)


def demote(text):
    """The chapter title is the H1, so every heading in a source file moves down one level."""
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + 1) + " ", text, flags=re.M)


def main():
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    texts = load_sources()
    tmap, fmap, appendix = plan_numbers(texts)
    numbers = number_citations(texts)

    used_any = set()
    for key, title, parts in CHAPTERS:
        body = []
        if title:
            body.append(f"# {title}\n")
        for p in parts:
            body.append(demote(texts[p]) if title else texts[p])
            used_any |= {k.strip() for g in CITE.findall(texts[p]) for k in g.split(",") if k.strip()}
        md = render("\n".join(body), tmap, fmap, numbers)

        if key == "Appendix":
            extra = ["", "## A. Supplementary Tables", "",
                     "These tables support statements in the main text and are placed here to keep the "
                     "chapters readable. Each caption states where the result is used.", ""]
            for fname, cap, num in appendix:
                if fname == "tools_and_versions.csv":
                    continue  # already printed with the tools section above
                src = copy_asset("tables", fname)
                extra += ["", renumber_refs(csv_as_markdown(src), tmap, fmap), "",
                          f"*Table {num}: {cap} ({APPENDIX_REASON.get(fname, 'supporting detail')}).*", ""]
            md += "\n".join(extra)

        (OUT / f"{key}.md").write_text(md + "\n", encoding="utf-8")
        to_docx(md, OUT, OUT / f"{key}.docx")
        print(f"wrote {key}")

    refs_md = "\n".join(["# References", ""] +
                        [f"[{n}] {REFS[k]}  " for k, n in sorted(numbers.items(), key=lambda kv: kv[1])])
    (OUT / "References.md").write_text(refs_md + "\n", encoding="utf-8")
    to_docx(refs_md, OUT, OUT / "References.docx")
    print(f"wrote References ({len(numbers)} entries)")
    print(f"tables in the chapters: {len([v for v in tmap.values() if not v.startswith('A.')])}, "
          f"in the appendix: {len(appendix)}; figures: {len(fmap)}")


if __name__ == "__main__":
    main()
