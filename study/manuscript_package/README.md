# Digital Discovery manuscript package

`manuscript.tex` is the editable manuscript source. The five redesigned figures are supplied as PDF, SVG, and 300-dpi PNG with exact plotted data in `figure_data/`. The scientific scope and wording follow the revised package; the source's unresolved author, funding, conflict, data-access, and AI-disclosure fields remain placeholders until confirmed.

## Rebuild figures and check plotted data

Use Python with NumPy, pandas, matplotlib, and scikit-learn. Tested plotting versions are listed in `provenance/plot_environment.json`.

```bash
python code/verify_package.py
python code/make_figures.py
```

The figure scripts rebuild plotted artifacts from saved arrays and tables. The CuO benchmark panel uses the supplied source image; it does not provide trace-level numeric metrics. Figure 2's molecular cartoons are illustrative connectivity diagrams. Figure 5 shows starting coordinates and does not establish final OCCO retention or a measured coupling contrast.

The complete local cohort export supports full structural, population, and model reproduction through the repository's CLI. It is separate from this manuscript package. Figure-only reproduction and full scientific reproduction are distinct.

No SI source was present in the supplied revised package. The manuscript PDF supplied alongside the source predates edits to the data-availability and structural-design prose, so the editable source is authoritative until a TeX engine is available to rebuild the PDF. No authorship or other unconfirmed declarations have been filled in.
