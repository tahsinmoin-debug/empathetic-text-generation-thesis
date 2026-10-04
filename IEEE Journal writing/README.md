# IEEE paper: Context Aware Empathetic Text Generation for Mental Health Support

`main.tex` is the paper, written into the IEEE template from `Code for journal.txt` (`\documentclass[conference]{IEEEtran}`). It is about 8 pages when compiled with pdfLaTeX, the Overleaf default.

## Compile

Upload `main.tex` and the three `fig_*.png` files to Overleaf, or run `pdflatex main.tex` twice.

## Still to fill in

- **Authors:** the six `\IEEEauthorblockN` / `\IEEEauthorblockA` blocks are the template's own placeholders, left unchanged. Fill in the four authors and delete the two unused blocks.
- **Funding:** the template's sub-title note and funding `\thanks{}` are kept as comments under `\title`. Uncomment the funding line only if there is a funder.

## Rebuilding after an edit

The text lives in `_build/body.tex`. Citations there use readable keys such as `\cite{rashkin2019}`.

```
python _build/build.py
```

The script rewrites `main.tex`. It keeps every line of the template's code, numbers citations `b1, b2, ...` in order of first use, and builds the reference list from `writing/_scripts/references.py`.
