"""Turn the thesis reference list into a BibTeX file.

The report is built from IEEE-formatted strings in references.py, but the LaTeX version of
the thesis uses a .bib file, and a `thebibliography` block cannot be mixed with \\bibliography.
This parses each string back into fields and writes `final writing/references.bib`, choosing
the entry type from the shape of the reference: a paper "in Proc. ..." is @inproceedings, one
with a volume in a named journal is @article, a preprint is @misc with its eprint, and a page
or dataset is @misc with a url.

Usage:
    python writing/_scripts/make_bibtex.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import REFS  # noqa: E402

OUT = HERE.parents[1] / "final writing" / "references.bib"

# a few entries are books, theses or standards rather than papers
FORCED = {
    "efron1993": "book",
    "who2023suicide": "techreport",
    "who2022report": "techreport",
    "samhsa2019tip35": "techreport",
}


def split_parts(text):
    """authors, title, and everything after the title."""
    m = re.match(r'^(.*?),\s*"(.+?),?"\s*(.*)$', text, re.S)
    if not m:                                   # a page or dataset with no quoted title
        m2 = re.match(r'^(.*?),\s*(.*)$', text, re.S)
        return m2.group(1), "", m2.group(2)
    return m.group(1), m.group(2), m.group(3)


def field(pattern, text, group=1):
    m = re.search(pattern, text)
    return m.group(group).strip(" ,.") if m else ""


def bib_authors(raw):
    """IEEE lists initials first; BibTeX wants 'Last, First' separated by ' and '."""
    raw = raw.strip().rstrip(",")
    if not raw:
        return ""
    ORGS = r"(Organization|Initiative|Society|Google|NVIDIA|Administration|IEEE|Foundation|Research|Institute|Association|Ministry|University|Team|Inc\.)"
    if re.search(ORGS, raw) and not re.search(r"[A-Z]\.", raw):
        return "{" + raw + "}"                  # an organisation is one name, kept whole
    raw = re.sub(r"\bet al\.?", "and others", raw)
    people = re.split(r",\s*and\s+|\s+and\s+|,\s*(?![A-Z]\.)", raw)
    out = []
    for p in [p.strip() for p in people if p.strip()]:
        if p == "and others":
            out.append("others")
            continue
        bits = p.split()
        out.append(f"{bits[-1]}, {' '.join(bits[:-1])}" if len(bits) > 1 else p)
    return " and ".join(out)


def classify(rest, key):
    if key in FORCED:
        return FORCED[key]
    if "arXiv:" in rest:
        return "misc"
    if re.search(r"\bin Proc\.|\bin Findings|\bin Advances in|\bin \d+(st|nd|rd|th)|Workshop|Conference", rest):
        return "inproceedings"
    if re.search(r"vol\.\s*\d+", rest):
        return "article"
    return "misc"


# a handful of entries whose shape the parser cannot infer from an IEEE string
OVERRIDES = {
    "samhsa2019tip35": {"title": "Enhancing Motivation for Change in Substance Use Disorder "
                                 "Treatment, Treatment Improvement Protocol (TIP) Series No. 35",
                        "number": "SMA19-5271"},
    "efron1993": {"publisher": "Chapman and Hall/CRC", "address": "New York, NY, USA"},
    "nvidia_rtx4060": {"title": "GeForce RTX 4060 Ti and 4060 graphics cards: Specifications"},
}


def entry(key, text):
    authors, title, rest = split_parts(text)
    if not title:                               # a book or report: the title is not quoted
        head, _, tail = rest.partition(". ")
        title, rest = head.strip(" ,."), tail
    kind = classify(rest, key)
    f = {"author": bib_authors(authors), "title": title}

    year = field(r"\b(19|20)\d{2}\b", rest, 0)
    if year:
        f["year"] = year
    for name, pattern in (("volume", r"vol\.\s*([\w-]+)"), ("number", r"no\.\s*([\w-]+)"),
                          ("doi", r"doi:\s*(\S+)"), ("url", r"Available:\s*(\S+)")):
        got = field(pattern, rest)
        if got:
            f[name] = got
    pages = field(r"pp?\.\s*([\w-]+(?:\s*-\s*[\w-]+)?)", rest)
    if pages:
        f["pages"] = pages.replace("-", "--")

    if kind == "inproceedings":
        venue = field(r"\bin\s+(.+?)(?:,\s*(?:19|20)\d{2}|,\s*pp?\.)", rest)
        f["booktitle"] = venue or "Proceedings"
    elif kind == "article":
        journal = re.split(r",\s*vol\.", rest)[0].strip(" ,.")
        f["journal"] = journal
    else:
        eprint = field(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", rest)
        if eprint:
            f.update(eprint=eprint, archiveprefix="arXiv", howpublished=f"arXiv:{eprint}")
        elif "url" in f:
            f["howpublished"] = "[Online]"
        if kind == "techreport":
            f.setdefault("institution", f["author"].strip("{}"))

    f.update(OVERRIDES.get(key, {}))
    lines = [f"@{kind}{{{key},"]
    for name in ("author", "title", "booktitle", "journal", "institution", "publisher",
                 "address", "volume", "number", "pages", "year", "eprint", "archiveprefix",
                 "howpublished", "doi", "url"):
        if f.get(name):
            value = f[name] if name in ("doi", "url") else f[name].replace("&", "\&")
            lines.append(f"  {name:<14}= {{{value}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def main():
    order = list(REFS)
    out = ["% Bibliography for the thesis. Generated from writing/_scripts/references.py by",
           "% writing/_scripts/make_bibtex.py - edit the Python file, not this one.",
           "% Cite with the key, e.g. \\cite{rashkin2019}.", ""]
    kinds = {}
    for key in order:
        e = entry(key, REFS[key])
        kinds[e.split("{")[0]] = kinds.get(e.split("{")[0], 0) + 1
        out.append(e)
        out.append("")
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(order)} entries")
    for k, n in sorted(kinds.items()):
        print(f"  {k:<16}{n}")


if __name__ == "__main__":
    main()
