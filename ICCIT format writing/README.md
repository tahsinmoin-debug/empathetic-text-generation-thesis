# ICCIT Format Writing: Conference Paper Draft

A full first draft of a **conference paper** built from the thesis, in the IEEE format used by ICCIT (IEEE Bangladesh Section, IEEE Xplore). It is **not** a copy of the thesis: a 6-page paper tells one focused story, so this draft keeps only the strongest, directly comparable results.

**Target:** ICCIT 2026 (29th ICCIT), Cox's Bazar, 18-20 Dec 2026. Full-paper deadline (extended): **20 September 2026**. Maximum **6 pages**, IEEE two-column template, **double-blind** review. <https://iccit.org.bd/2026/>

## Files

| File | What it is |
|---|---|
| `paper.tex` + `figures/` | The paper in the official IEEEtran conference class. **Recommended for submission:** upload both to Overleaf (New Project > Upload Project) and compile with pdfLaTeX. |
| `paper.docx` | The same paper as a two-column Word document (Word version measures **5 pages** before human-evaluation results are added). Use it if the team prefers Word, or to edit comfortably, then copy into the official IEEE Word template. |
| `PAPER_PREVIEW.md` | Plain preview for reading on GitHub. |
| `_build/paper_src.md` | **The single source text.** Edit here and rebuild so all three outputs stay identical. |
| `_build/build_paper.py` | Builds `paper.tex`, `paper.docx` and `PAPER_PREVIEW.md` (citations numbered automatically). |
| `_build/make_figure.py` | Draws `figures/fig_architecture.png`. |

```
python "ICCIT format writing/_build/make_figure.py"
python "ICCIT format writing/_build/build_paper.py"
```

If you edit `paper.docx` or `paper.tex` directly instead, do not run the build again, because it overwrites them.

## How the paper differs from the thesis

| Thesis | Paper |
|---|---|
| "Same-size single model" (Qwen2.5-1.5B) | Called **"1.5B single model"**. Our final system has **4.0B** parameters (Phi-4-mini 3.8B + RoBERTa 0.125B), so "same size" would be wrong in a paper and a reviewer would catch it |
| Two-stage selection presented as our idea | **HEF (Yang et al., 2024, arXiv 2402.11801)** already gave an LLM the top-k emotions and attention cause words of a small model. The paper cites it and states four differences. **The thesis does not cite HEF yet** |
| Many design iterations, project management, economics | Removed; only method, controlled comparisons and results |
| Main claim: decoupled beats same-size | Main claim: **decoupling improves structure (rules), emotion-specific guidance improves empathy, and detector accuracy is not the bottleneck** |
| - | New controlled comparison (computed from existing results, no new runs): GPT-2 + Qwen-1.5B vs Qwen-1.5B alone. Rules 63.7% vs 51.3% (p = 0.002), but WASSA (p = 0.69), emotion validation (p = 1.0) and judge overall (p = 0.14) unchanged |

Every number was taken from the regenerated tables in `writing/` (AI judges = final two-judge panel, all 7 systems).

## Checklist before submission

1. **Rewrite in your own words.** This draft was written with AI assistance. IEEE policy requires disclosure of AI-generated text; a disclosure sentence is already in the Acknowledgment. Keep it (and adjust it to what you actually did).
2. **Human evaluation.** Fill the three `PENDING` places (Section V-E, the ethics paragraph, Acknowledgment). If the form is not finished by the deadline, either remove human evaluation from the abstract, contributions and Section IV (5), or submit to a later venue.
3. **Anonymity.** No names, student IDs, university, supervisor or GitHub link in the submitted PDF. Add the author block only for the camera-ready version.
4. **Page limit.** Compile in Overleaf and confirm at most 6 pages including references. If over: shorten Related Work, or remove Table III and keep its numbers in the text.
5. **Chirpy Cardinal.** Not mentioned. The paper compares the detector only with *joint generation models*, which is true regardless. Decide as a team before adding any wider accuracy claim.
6. **Weakness to be ready for:** there is no Phi-4-mini single-model baseline, so the gain over the 1.5B single model mixes decoupling, guidance and generator size. The paper states this openly (Section VI). Adding that baseline would strengthen the paper but needs a new GPU generation run, WASSA scoring and judge runs.
7. **One venue at a time.** Do not submit this paper anywhere else while it is under review at ICCIT.
8. Check every reference once more against the publisher page (the entries are the verified thesis references plus HEF).
