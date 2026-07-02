# CO2RR XAS Agent

A lightweight agent and web app for building CO2RR XAS workflows, generating FEFF/FDMNES/VASP inputs, converting completed runs into ISAAC AI-ready records, and using those records for XAS-focused machine learning.

The most useful entry point is now the Streamlit web app.

---

## Quick start: web app

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit pandas scikit-learn
streamlit run web_app/Home.py
```

Optional ISAAC Portal access:

```bash
export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
export ISAAC_KEY="..."
streamlit run web_app/Home.py
```

The web app can also work only from local ISAAC JSON files or ZIP archives; portal credentials are not required for local upload mode.

---

## Web app capabilities

### 1. XAS record loading and filtering

The ML page can load records from:

- local ISAAC JSON files
- ZIP archives of ISAAC records
- the ISAAC Portal, when `ISAAC_KEY` is set

It applies strict XAS spectral filtering before ML. A record must contain a numeric incident-energy axis with eV/keV units and an XAS-like signal channel. Records that look like XRD, XPS, PDF, Raman, STM/STS, etc. are excluded from XAS ML by default.

### 2. Simple XAS summary

For valid XAS spectra, the app summarizes:

- `record_id`
- `record_domain`
- material formula/name
- absorber
- edge
- technique
- number of spectral points
- classification reason

Absorber/edge metadata can be read from ISAAC fields such as `system.configuration.absorber` and `system.configuration.edge`, or inferred from the energy range when metadata are missing.

### 3. Spectrum plotting

The app plots spectra by absorber/edge group and supports:

- raw spectra
- min-max normalized spectra
- min-max normalized spectra with vertical offsets
- XAS-only, all plottable, or non-XAS diagnostic views

### 4. XAS ML target/label selection

The ML target selector is intentionally restricted to fields that are useful for XAS modeling:

- `sample.material.formula`
- non-null `descriptors.*` values
- optional parsed structural labels:
  - `structure.facet`
  - `structure.adsorption_site`

Other metadata such as record ID, source type, facility, instrument, timestamp, absorber, edge, technique, QC status, assets, links, and tags are not exposed as ML target labels. They can still be used for filtering or diagnosis, but they are not useful XAS ML targets and can introduce leakage/noise.

Parsed facet/site labels are extracted from record text and should be manually checked before training.

### 5. Auto-advisor for ML setup

The auto-advisor compares:

- feature sets: raw, derivative, CDF, peak-window features
- normalization: min-max, area, z-score
- dimension reduction choices, including no reduction and PCA
- simple models and baselines

It reports ranked configurations and a recommended starting point. Results should still be checked with grouped validation when enough records are available.

### 6. Manual ML run

Manual training supports:

- classification or regression, depending on the target field
- common-grid interpolation
- spectrum normalization
- optional dimension reduction with PCA
- simple baselines and standard models
- held-out prediction table
- training curve diagnostics when enough spectra are available

---

## Simulation workflow capabilities

The agent can generate and process XAS simulation workflows for CO2RR-relevant catalyst models:

- Cu/Au/Ni/Pt/Ir surfaces and interfaces
- adsorbates such as CO, CHO, OCCO, and related intermediates
- FEFF input generation
- FDMNES input generation
- VASP two-step SCF/XAS input generation
- NERSC submit/monitor/finalize workflow
- output parsing for FEFF, FDMNES, and VASP
- ISAAC simulation evidence record generation

Completed simulation folders can be converted into ISAAC records with:

```bash
bash XAS/convert_completed_simulation_xas_to_isaac.sh \
  --input-dir . \
  --output isaac_record_draft.json
```

The converter reads absorber and edge from the calculation inputs when possible:

- FEFF: `feff.inp`, `POTENTIALS`, `PARAMETERS`, `ATOMS`
- FDMNES: `*_in.txt`, `fdmfile.txt`
- VASP: `INCAR`, `POTCAR.spec`, and folder/file labels

Metadata and environment variables are used only as fallback.

---

## ISAAC validation and upload

For one record:

```bash
export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
export ISAAC_KEY="..."
python tools/convert_simulation_xas_run_to_isaac.py \
  --input-dir . \
  --output isaac_record_draft.json \
  --validate
```

For a batch of JSON records, use the helper generated during analysis or adapt the same pattern:

```bash
python batch_validate_upload_isaac_records.py \
  --input records.zip \
  --validate \
  --upload
```

The recommended workflow is to validate all records first, inspect the report, and upload only records that pass validation.

---

## Repository layout

```text
CO2RR_XAS_agent/
├── agent/                    # planner schemas, parser, and agent orchestration
├── tools/                    # XAS record utilities, portal client, converters, ML helpers
├── workflow/                 # NERSC submit/monitor/finalize helpers
├── generators/               # slab/input generator utilities
├── web_app/                  # Streamlit web app
│   ├── Home.py
│   └── pages/
│       └── 2_Machine_learning_for_XAS.py
├── XAS/                      # reusable XAS shell helpers
├── examples/                 # runnable examples
├── tests/                    # regression tests
└── requirements.txt
```

---

## Notes

- Use the web app for most interactive work.
- Use `tools/convert_simulation_xas_run_to_isaac.py` for repeated simulation-folder conversion.
- Keep experimental/literature one-off conversions outside the repository unless they are intended to become reusable workflows.
- For XAS ML, use metadata for filtering and descriptors/chemically meaningful labels for targets.
