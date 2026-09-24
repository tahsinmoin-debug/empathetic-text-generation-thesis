"""Convert the finished thesis chapters from .docx to LaTeX.

Citations are the delicate part. The chapters carry numbers like [26], but the
numbering in the chapter 4-6 file does not agree with our reference list (and
disagrees with itself in places), so numbers are never trusted on their own.
Every [n] is resolved by matching its paragraph against the writing-kit source,
which carries {cite:key} markers, and taking the key from there. Anything that
cannot be resolved that way is written as \\cite{CHECK-n} and listed at the end,
so nothing is silently guessed.

Usage:
    python docx_to_latex.py
"""

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(r"F:\Thesis")
SRCDOCX = ROOT / "Thesis final fixed writing"
KIT = ROOT / "final writing" / "_src"
OUT = ROOT / "latex writing"
FIGDIR = OUT / "figures"

FILES = [
    ("Thesis Final Writing (1).docx", [1, 2, 3]),
    ("Thesis Final Writing(4,5,6).docx", [4, 5, 6]),
]

# ----------------------------------------------------------------- citations
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cite_tables import FILE2  # noqa: E402

# chapters 1-3 number their references exactly as our list does, checked number by
# number; chapters 4-6 use their own numbering, recovered in cite_tables.py.
sys.path.insert(0, str(ROOT / "final writing" / "_build"))
import build_final_draft as B  # noqa: E402

_kit = {q: (B.SRC / f"{q}.md").read_text(encoding="utf-8")
        for _, _, parts in B.CHAPTERS for q in parts}
FILE1 = {str(n): k for k, n in B.number_citations(_kit).items()}

TABLES = {0: FILE1, 1: {n: v[0] for n, v in FILE2.items()}}
CURFILE = 0
UNRESOLVED = []


def resolve_citations(text, chapter):
    """Replace every [n], and runs like '[4] [5]' or '[7], [8]', with \\cite{}."""
    if "[" not in text:
        return text
    table = TABLES[CURFILE]

    def collapse(m):
        nums = re.findall(r"\d+", m.group(0))   # also catches the "[13, 14]" form
        keys = []
        for n in nums:
            if n in table:
                keys.append(table[n])
            else:
                UNRESOLVED.append((chapter, n, text[:90]))
                keys.append("CHECK-" + n)
        seen, uniq = set(), []
        for k in keys:                      # the same work cited twice in one place
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        return r"\cite{" + ",".join(uniq) + "}"

    return re.sub(r"\[\d+(?:\s*,\s*\d+)*\](?:[,]?\s*\[\d+(?:\s*,\s*\d+)*\])*", collapse, text)


# ----------------------------------------------------------------- escaping
SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
UNI = {
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2013": "--", "\u2014": "---", "\u2026": r"\ldots{}",
    "\u00a0": "~", "\u2212": "-", "\u00d7": r"$\times$", "\u2192": r"$\rightarrow$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00b1": r"$\pm$", "\u00b0": r"$^\circ$",
    "\u2011": "-", "\u2022": "", "\u25cf": "", "\u00ad": "",
}


def esc(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return "".join(SPECIAL.get(c, c) for c in s)


def runs_to_latex(p):
    """Keep bold and italic, escaping everything else."""
    out = []
    for r in p.runs:
        t = esc(r.text)
        if not t.strip():
            out.append(t)
            continue
        if r.bold and r.italic:
            t = r"\textbf{\textit{%s}}" % t
        elif r.bold:
            t = r"\textbf{%s}" % t
        elif r.italic:
            t = r"\textit{%s}" % t
        out.append(t)
    return "".join(out)


# ----------------------------------------------------------------- blocks
def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def image_rids(p):
    return [b.get(qn("r:embed")) for b in p._p.iter(qn("a:blip")) if b.get(qn("r:embed"))]


CAP = re.compile(r"^(Table|Figure)\s*(\d+\.\d+)\s*:?\s*(.*)$", re.S)


def label_of(kind, num):
    return ("tab:" if kind == "Table" else "fig:") + num


def find_caption(blocks, i, kind):
    """Look past blank paragraphs for 'Table 4.1: ...' after block i.

    Returns (number, escaped caption, index of the last block consumed).
    """
    j = i + 1
    while j < len(blocks) and isinstance(blocks[j], Paragraph) and not blocks[j].text.strip():
        j += 1
    if j < len(blocks) and isinstance(blocks[j], Paragraph):
        m = CAP.match(blocks[j].text.strip())
        if m and m.group(1) == kind:
            return m.group(2), esc(" ".join(m.group(3).split())), j
    return "", "", i


def table_to_latex(tbl, caption, num):
    rows = []
    for r in tbl.rows:
        cells, seen = [], set()
        for c in r.cells:
            if c._tc in seen:       # merged cell repeated by python-docx
                continue
            seen.add(c._tc)
            cells.append(" ".join(esc(c.text).split()))
        rows.append(cells)
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    size = r"\footnotesize" if ncol <= 4 else r"\scriptsize"
    colspec = "l" + "X" * (ncol - 1) if ncol > 1 else "X"
    out = [r"\begin{table}[htbp]", r"\centering", size]
    if caption:
        out.append(r"\caption{%s}" % caption)
    if num:
        out.append(r"\label{%s}" % label_of("Table", num))
    out.append(r"\begin{tabularx}{\textwidth}{@{}%s@{}}" % colspec)
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{%s}" % c for c in rows[0]) + r" \\")
    out.append(r"\midrule")
    for r in rows[1:]:
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(out)


def figure_to_latex(fname, caption, num):
    out = [r"\begin{figure}[htbp]", r"\centering",
           r"\includegraphics[width=0.95\linewidth]{figures/%s}" % fname]
    if caption:
        out.append(r"\caption{%s}" % caption)
    if num:
        out.append(r"\label{%s}" % label_of("Figure", num))
    out += [r"\end{figure}", ""]
    return "\n".join(out)


# ----------------------------------------------------------------- main pass
HEAD_CH = re.compile(r"^Chapter\s+(\d+)\s*(?:[:.\u2013-]\s*(.*))?$", re.I)
FALLBACK_TITLE = {
    1: "Introduction",
    2: "Literature Review",
    3: "Requirements, Impacts and Constraints",
    4: "Proposed Methodology",
    5: "Results and Analysis",
    6: "Conclusion and Future Work",
}
NUMHEAD = re.compile(r"^(\d+(?:\.\d+)+)\s+(\S.*)$")
BULLET = re.compile(r"^\s*[\u2022\u25cf\u00b7\-\u2013]\s+(.*)$")


def convert(path, chapters):
    doc = Document(str(path))
    blocks = list(iter_blocks(doc))
    out = {}           # chapter number -> list of latex lines
    cur = None
    pending_caption_for = None   # ("Table"/"Figure", latex-so-far)
    in_list = False
    img_count = Counter()
    labels = set()

    i = 0
    while i < len(blocks):
        b = blocks[i]

        # ---------- tables
        if isinstance(b, Table):
            num, cap_text, i = find_caption(blocks, i, "Table")
            if cur:
                if in_list:
                    out[cur].append(r"\end{itemize}")
                    in_list = False
                if ("Table", num) in labels:
                    num = ""
                elif num:
                    labels.add(("Table", num))
                out[cur].append(table_to_latex(b, cap_text, num))
            i += 1
            continue

        text = b.text.strip()
        rids = image_rids(b)

        # ---------- images
        if rids:
            num, cap_text, i = find_caption(blocks, i, "Figure")
            for rid in rids:
                part = doc.part.related_parts[rid]
                ext = Path(part.partname).suffix or ".png"
                img_count[cur] += 1
                fname = f"ch{cur}_fig{img_count[cur]}{ext}"
                (FIGDIR / fname).write_bytes(part.blob)
                if cur:
                    if in_list:
                        out[cur].append(r"\end{itemize}")
                        in_list = False
                    if ("Figure", num) in labels:
                        num = ""
                    elif num:
                        labels.add(("Figure", num))
                    out[cur].append(figure_to_latex(fname, cap_text, num))
            i += 1
            continue

        if not text:
            i += 1
            continue

        # ---------- headings
        mch = HEAD_CH.match(text)
        if mch and int(mch.group(1)) in chapters and len(text) < 60:
            if in_list and cur:
                out[cur].append(r"\end{itemize}")
                in_list = False
            cur = int(mch.group(1))
            out.setdefault(cur, [])
            title = esc((mch.group(2) or "").strip()) or esc(FALLBACK_TITLE.get(cur, f"Chapter {cur}"))
            out[cur] += [r"\chapter{%s}" % title, r"\label{ch:%d}" % cur, ""]
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        mnum = NUMHEAD.match(text)
        style = b.style.name
        is_head = style.startswith("Heading") or mnum
        if is_head and len(text) < 120:
            if in_list:
                out[cur].append(r"\end{itemize}")
                in_list = False
            if mnum:
                depth = mnum.group(1).count(".")
                title = esc(mnum.group(2).strip())
                key = mnum.group(1)
            else:
                depth = 1 if style == "Heading 1" else (2 if style == "Heading 2" else 3)
                title = esc(text)
                key = None
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(depth, "subsubsection")
            out[cur].append("\\%s{%s}" % (cmd, title))
            if key:
                out[cur].append(r"\label{sec:%s}" % key)
            out[cur].append("")
            i += 1
            continue

        # ---------- a caption that had no table/figure before it
        mcap = CAP.match(text)
        if mcap and len(text) < 300 and b.runs and b.runs[0].italic:
            i += 1
            continue

        # ---------- bullets
        mb = BULLET.match(text)
        if mb:
            if not in_list:
                out[cur].append(r"\begin{itemize}")
                in_list = True
            body = runs_to_latex(b)
            body = re.sub(r"^\s*(\\textbf\{)?\s*[\u2022\u25cf\u00b7]\s*", r"\1", body)
            body = re.sub(r"^\s*[-\u2013]\s+", "", body)
            out[cur].append(r"  \item " + resolve_citations(body, cur).strip())
            i += 1
            continue
        if in_list:
            out[cur].append(r"\end{itemize}")
            in_list = False

        # ---------- ordinary paragraph
        body = runs_to_latex(b)
        body = resolve_citations(body, cur)
        out[cur] += [body, ""]
        i += 1

    if in_list and cur:
        out[cur].append(r"\end{itemize}")
    return out, labels


def crossref(tex, labels):
    """Turn 'Table 4.1' / 'Figure 5.2' into \\ref{} when that label exists."""
    def r(m):
        kind, num = m.group(1), m.group(2)
        lab = label_of(kind, num)
        return "%s~\\ref{%s}" % (kind, lab) if (kind, num) in labels else m.group(0)
    # never touch the text inside \caption{} or \label{}
    parts = re.split(r"(\\(?:caption|label)\{[^{}]*\})", tex)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\b(Table|Figure)~?\s(\d+\.\d+)\b", r, parts[i])
    return "".join(parts)


def main():
    global CURFILE
    OUT.mkdir(exist_ok=True)
    if FIGDIR.exists():
        shutil.rmtree(FIGDIR)
    FIGDIR.mkdir(parents=True)
    all_labels = set()
    chapters = {}
    for idx, (fname, chs) in enumerate(FILES):
        CURFILE = idx
        got, labels = convert(SRCDOCX / fname, chs)
        chapters.update(got)
        all_labels |= labels

    def drop_repeated_title(tex):
        """A chapter that repeats its title as a bold line loses the repeat."""
        m = re.match(r"(\\chapter\{([^}]*)\}\n\\label\{ch:\d+\}\n\n)\\textbf\{([^}]*)\}\s*\n+", tex)
        if m and m.group(2).strip().lower() == m.group(3).strip().lower():
            return m.group(1) + tex[m.end():]
        return tex

    for n in sorted(chapters):
        tex = drop_repeated_title("\n".join(chapters[n]))
        tex = re.sub(r"\n{3,}", "\n\n", tex)
        tex = crossref(tex, all_labels)
        (OUT / f"chapter{n}.tex").write_text(tex, encoding="utf-8")
        print(f"chapter{n}.tex  {len(tex.split())} words")

    shutil.copy(ROOT / "final writing" / "references.bib", OUT / "references.bib")
    print(f"\nfigures written: {len(list(FIGDIR.glob('*')))}")
    print(f"labels available: {len(all_labels)}")
    if UNRESOLVED:
        print(f"\n!! {len(UNRESOLVED)} citations could not be resolved from the source:")
        for ch, n, ctx in UNRESOLVED:
            print(f"   ch{ch} [{n}] :: {ctx}")
    else:
        print("\nall citations resolved from the writing-kit source")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
