# -*- coding: utf-8 -*-
"""Build main.tex from the IEEE template in "Code for journal.txt".

The template's code is kept as it is: document class, packages, the author
blocks (left as placeholders for the authors to fill), \\maketitle, the
abstract/keyword environments, the table and figure style, \\section*{Acknowledgment}
and thebibliography. Only the template's instruction text is replaced by the
paper. Template lines that would otherwise print instructions are kept as
comments rather than deleted.

Citations in body.tex use readable keys (\\cite{rashkin2019}); they are renumbered
b1, b2, ... in order of first appearance, the template's own key style, and the
reference list is generated from writing/_scripts/references.py.

Usage:  python build.py
"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "writing" / "_scripts"))
from references import REFS  # noqa: E402

TEMPLATE = HERE / "Code for journal.txt"   # the IEEE template, copied unchanged
TITLE = "Context Aware Empathetic Text Generation for Mental Health Support"

ABSTRACT = (
    "Emotional distress such as depression and anxiety affects a large part of the global population, "
    "yet financial cost, social stigma and a shortage of mental health specialists keep many people from care. "
    "Advances in Natural Language Processing (NLP) have made dialogue support systems possible, and conversational "
    "agents are now widely used as a supplement to mental health support. Many systems built on Large Language "
    "Models (LLMs), however, respond in an emotionally generic and contextually indifferent way, and the emotional "
    "judgement behind a reply is never recorded, which limits their value in mental health care. This paper "
    "investigates the production of contextually thoughtful and emotionally responsive text on a single consumer "
    "graphics card. A RoBERTa-base emotion detector, fine-tuned on the first message of each EmpatheticDialogues "
    "conversation, passes its three most probable of 32 emotions and the words its attention selected to a 4-bit "
    "Phi-4-mini generator, which first chooses one emotion and then writes the reply under guidance written for that "
    "emotion. Twelve systems answer the same 300 test conversations and are compared on rule adherence, an "
    "independent empathy classifier, emotional fidelity, a blind panel of two LLM judges and a study with 40 human "
    "raters, using paired tests corrected for multiple comparisons. The detector reaches 50.3\\% top-1 and 84.3\\% "
    "top-5 accuracy. The decoupled system expresses the person's true emotion more often than any other system, "
    "is more empathetic than a single model of its own size and than its own generator used alone, and runs in "
    "3.40 GB of memory with 52\\% of the parameters of a 7.6-billion-parameter single model. Human raters scored it "
    "highest on empathy, relevance and supportiveness and preferred its replies in 61.3\\% of 320 blind choices. "
    "The research focuses on support and the absence of diagnosis."
)

KEYWORDS = ("Empathetic text generation, natural language processing, emotion-aware language modeling, "
            "computational empathy, contextual dialogue modeling, emotionally intelligent LLM")

ACKNOWLEDGMENT = (
    "The authors thank their supervisor, Mr.\\ Moin Mostakim, and co-supervisor, Mr.\\ Sheikh Araf Noshin, of the "
    "Department of Computer Science and Engineering, BRAC University, for their guidance and continuous support, and "
    "the 40 participants who took part in the human evaluation. The system, its training and every experiment were "
    "designed, implemented and run by the authors on their own hardware, and no experimental result was produced by "
    "an AI system. In accordance with IEEE policy on AI-generated text, the authors disclose that Claude (Anthropic) "
    "was used to help draft and edit the text from the authors' results and notes, and to write parts of the analysis "
    "code that computes the reported statistics and produces the tables and figures. The authors verified all "
    "results, references and claims and take full responsibility for the content of this paper."
)

FIGURES = {
    "fig_pipeline.png": ROOT / "latex writing" / "figures" / "ch4_fig2.png",
    "fig_same_generator.png": ROOT / "latex writing" / "figures" / "ch5_fig7.png",
    "fig_baselines.png": ROOT / "latex writing" / "figures" / "ch5_fig6.png",
}


# ------------------------------------------------------------ reference formatting
def tex_escape(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("<", r"$<$"), (">", r"$>$")]:
        s = s.replace(a, b)
    return s.replace(r"\textbackslash\{\}", r"\textbackslash{}")


def tex_quotes(s):
    out, opening = [], True
    for ch in s:
        if ch == '"':
            out.append("``" if opening else "''")
            opening = not opening
        else:
            out.append(ch)
    return "".join(out)


def tex_bib(entry):
    entry = tex_quotes(tex_escape(entry))
    return re.sub(r"(pp\.\s*\d+)-(\d+)", "\\1--\\2", entry)


# ------------------------------------------------------------------- the build
def comment(lines):
    return ["% " + ln if ln.strip() else ln for ln in lines]


def main():
    t = TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")

    def find(prefix, start=0):
        for i in range(start, len(t)):
            if t[i].startswith(prefix):
                return i
        raise SystemExit(f"template line not found: {prefix!r}")

    i_title = find(r"\title{")
    i_author = find(r"\author{")
    i_abs = find(r"\begin{abstract}")
    i_abs_end = find(r"\end{abstract}")
    i_kw = find(r"\begin{IEEEkeywords}")
    i_kw_end = find(r"\end{IEEEkeywords}")
    i_intro = find(r"\section{Introduction}")
    i_ack = find(r"\section*{Acknowledgment}")
    i_refs = find(r"\section*{References}")
    i_bib = find(r"\begin{thebibliography}")
    i_bib_end = find(r"\end{thebibliography}")
    i_end = find(r"\end{document}")

    body = (HERE / "body.tex").read_text(encoding="utf-8").rstrip("\n").split("\n")

    # 1) Citation numbering in order of first appearance (abstract and acknowledgment cite nothing).
    order = []
    for key_list in re.findall(r"\\cite\{([^}]*)\}", "\n".join(body)):
        for k in (x.strip() for x in key_list.split(",")):
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in REFS]
    if missing:
        raise SystemExit(f"citation keys not in references.py: {missing}")
    num = {k: f"b{i + 1}" for i, k in enumerate(order)}
    body = [re.sub(r"\\cite\{([^}]*)\}",
                   lambda m: r"\cite{" + ",".join(num[x.strip()] for x in m.group(1).split(",")) + "}", ln)
            for ln in body]

    out = []
    out += t[:i_title]                                                  # class, packages, \begin{document}
    out += [r"\title{" + TITLE]
    out += comment(t[i_title + 1:i_author - 2])                         # sub-title note and funding \thanks, kept as comments
    out += ["}", ""]
    out += t[i_author:i_abs]                                            # author blocks untouched, \maketitle
    out += [t[i_abs], ABSTRACT, t[i_abs_end], ""]
    out += [t[i_kw], KEYWORDS, t[i_kw_end], ""]
    out += body + [""]
    out += [t[i_ack], "", ACKNOWLEDGMENT, ""]
    out += ["% The bibliography below prints its own ``References'' heading, so the template's",
            "% separate heading and its citation instructions are kept only as comments."]
    out += comment(t[i_refs:i_bib]) + [""]
    out += [t[i_bib]]
    out += [r"\bibitem{" + num[k] + "} " + tex_bib(REFS[k]) for k in order]
    out += [t[i_bib_end]]
    out += ["% Template instruction text, kept as comments:"]
    out += comment(t[i_bib_end + 1:i_end])
    out += [t[i_end], ""]

    (OUT / "main.tex").write_text("\n".join(out), encoding="utf-8")
    for name, src in FIGURES.items():
        shutil.copyfile(src, OUT / name)
    words = len(re.sub(r"\\[a-zA-Z]+\*?|[{}$%\\]", " ", "\n".join(body)).split())
    print(f"wrote main.tex | {len(order)} references | ~{words} words in the body | "
          f"{len(FIGURES)} figures copied")


if __name__ == "__main__":
    main()
