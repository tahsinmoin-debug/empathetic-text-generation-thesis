# LaTeX source

The six chapters of the thesis as LaTeX, converted from the Word files in
`Thesis final fixed writing/`, with every bracketed citation turned into a
`\cite{}` command and every table and figure given a label.

## Files

| File | What it holds |
|---|---|
| `main.tex` | Preamble, title page placeholder, contents, and the six `\input` lines |
| `chapter1.tex` … `chapter6.tex` | One chapter each |
| `references.bib` | All 58 references, as BibTeX entries |
| `figures/` | The 15 figures, pulled out of the Word files |

## Compiling

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or `latexmk -pdf main`. The four passes are needed so that the citation
numbers and the cross-references settle.

## Citations

Every `[n]` in the Word files became a `\cite{key}`. The numbering had to be
handled carefully, because the two Word files do not number their references
the same way:

* **Chapters 1–3** number them exactly as our reference list does. All 46 of
  their numbers were checked against the list one at a time.
* **Chapters 4–6** use their own numbering, which differs. Each of its 35
  numbers was identified from the sentence that uses it rather than from its
  position, so, for example, its `[45]` is the bootstrap reference and not the
  45th entry of our list. That mapping is recorded in the conversion script.

Because the keys are now names rather than numbers, the order of the reference
list no longer matters: BibTeX renumbers everything on its own, and moving a
paragraph between chapters cannot break a citation.

Adjacent citations were merged, so `[8] [9] [10]` became a single
`\cite{li2020empdg,li2022kemp,sabour2022cem}`, which IEEEtran prints as a
range.

## Cross-references

Tables and figures carry `\label{tab:4.1}` and `\label{fig:5.2}`, and mentions
in the text were turned into `Table~\ref{tab:4.1}` and `Figure~\ref{fig:5.2}`.
Numbers are therefore produced by LaTeX, so inserting a figure renumbers
everything that follows it without any edits.

Sections carry `\label{sec:5.4.2}` if you want to reference them the same way;
the text still names sections in words, which was left alone.

## Things to do by hand

1. **Front matter.** `main.tex` has a placeholder title page. Replace it with
   the department's official title page, declaration, approval, ethics
   statement, acknowledgement and abstract.
2. **The appendix.** It is not part of this conversion. Chapters refer to
   tables A.6 to A.33, which live in `final writing/Appendix.docx`.
3. **Table widths.** Tables use `tabularx` at full text width with the first
   column plain and the rest ragged-right. A few of the wider ones may still
   need a smaller font or a landscape page.
4. **Check the figures.** They were extracted from Word at the resolution Word
   stored them. Where a sharper copy exists in `final writing/figures/`, use
   that instead; the file names there are more descriptive.

## Regenerating

The converter is `_build/docx_to_latex.py`. Running it overwrites every
`chapter*.tex` and the whole of `figures/`, so edit the Word files and convert
again, or edit the LaTeX and stop converting — not both.
