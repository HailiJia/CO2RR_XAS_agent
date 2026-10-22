# Reference-v2

Run `python study/reference_benchmark_v2/process.py` from the repository, or execute every cell of `Cu_XAS_revised.ipynb`. `execute_notebook.py` provides a socket-free IPython execution route. The committed notebook is executed, not a template. Original data/notebook and reference-v1 are unchanged in `../reference_benchmark`.

`settings.json` defines the order, units, fixed grid and alignment degrees of freedom. `provenance_v2.csv` preserves all 18 material/series slots and raw hashes; missing slots remain missing. `local_output_search.json` records the repeat working-folder VASP/XML/HDF5 and PAW-lineage search (no potential contents). `before_after_processing_audit.csv` identifies where negatives first appear. Native arrays, broadening stages, and the recoverable original-notebook input route are frozen under `arrays/`.

The primary compares native-as-supplied curves with no unsupported local background or universal extra Lorentzian. Unknown remaining physical resolution prevents an unconditional accuracy ranking. The sigma=1 eV Gaussian is a separate diagnostic, not a measured resolution. Experimental raw baseline/noise is preserved. No absolute values/clipping conceal faulty absorption.

All primary curves and scores share `(I-min)/(max-min)` on 8980–9020 eV, 201 points. Each method has a separate Cu-fitted shift transferred to oxides; shape-aligned diagnostics are explicitly different. Shift is added to native energy. Each trial requires the entire fixed window. `calibration_objectives.csv` and `alignment.json` expose bounds, boundary hits and shift sensitivity (not statistical confidence). `normalization.csv` saves every offset/divisor. `plotted_arrays.csv` feeds the main figure and `similarity_metrics.csv` without additional display scaling. No correlation p-values are interpreted as independent energy-sample evidence.

Rebuild SI grids in PDF/SVG/450 dpi PNG with `process.py`; run `pytest -q tests/test_scientific_extensions_v2.py` to check common grids, identical normalization, exact metric reproduction, identity metrics, invalid inputs and positive convolution kernels.
