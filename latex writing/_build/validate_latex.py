"""Static checks on the generated LaTeX, since there is no compiler on this machine."""
import collections
import glob
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(r"F:\Thesis\latex writing")

files = sorted(OUT.glob("chapter*.tex"))
all_labels, all_refs, all_cites = set(), [], set()
problems = []

for f in files:
    s = f.read_text(encoding="utf-8")

    begins = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
    ends = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
    for k in set(begins) | set(ends):
        if begins[k] != ends[k]:
            problems.append(f"{f.name}: {k} begin={begins[k]} end={ends[k]}")

    depth = 0
    for i, c in enumerate(s):
        if c == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
    if depth != 0:
        problems.append(f"{f.name}: unbalanced braces ({depth:+d})")

    labs = re.findall(r"\\label\{([^}]*)\}", s)
    dup = [l for l, n in collections.Counter(labs).items() if n > 1]
    if dup:
        problems.append(f"{f.name}: duplicate labels {dup}")
    all_labels |= set(labs)
    all_refs += [(f.name, r) for r in re.findall(r"\\ref\{([^}]*)\}", s)]
    for g in re.findall(r"\\cite\{([^}]*)\}", s):
        all_cites |= {k.strip() for k in g.split(",")}

    # unescaped specials that would stop a compile
    for m in re.finditer(r"(?<!\\)&", s):
        line = s[:m.start()].count("\n") + 1
        ctx = s[max(0, m.start() - 40):m.start() + 20].replace("\n", " ")
        if r"\\" not in s[max(0, m.start() - 300):m.start()]:
            problems.append(f"{f.name}:{line}: bare & outside a table :: {ctx}")
    for sym in ["%", "$", "#", "_"]:
        for m in re.finditer(r"(?<!\\)\%s" % re.escape(sym), s):
            if sym == "$":     # $...$ math we inserted deliberately
                continue
            line = s[:m.start()].count("\n") + 1
            ctx = s[max(0, m.start() - 35):m.start() + 15].replace("\n", " ")
            problems.append(f"{f.name}:{line}: unescaped {sym} :: {ctx}")

    # leftovers from the source format
    for pat, what in [(r"\[\d+", "bracket citation"), (r"\{cite:", "kit marker"),
                      (r"CHECK-", "unresolved citation"), (r"\[\[", "kit table/figure marker")]:
        for m in re.finditer(pat, s):
            line = s[:m.start()].count("\n") + 1
            problems.append(f"{f.name}:{line}: leftover {what}")

dangling = [(fn, r) for fn, r in all_refs if r not in all_labels]

print(f"files: {len(files)}")
print(f"labels: {len(all_labels)}  refs: {len(all_refs)}  distinct cite keys: {len(all_cites)}")

bib = (OUT / "references.bib").read_text(encoding="utf-8")
bibkeys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
missing = sorted(all_cites - bibkeys)
print(f"cite keys missing from references.bib: {missing or 'none'}")
print(f"dangling \\ref: {dangling or 'none'}")

figs = {p.name for p in (OUT / "figures").glob("*")}
used = set()
for f in files:
    used |= set(re.findall(r"includegraphics\[[^\]]*\]\{figures/([^}]*)\}",
                           f.read_text(encoding="utf-8")))
print(f"figure files: {len(figs)}, referenced: {len(used)}, missing: {sorted(used - figs) or 'none'}")

print(f"\nproblems: {len(problems)}")
for p in problems[:40]:
    print("   ", p)
