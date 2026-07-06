# CO2RR XAS Agent

A lightweight agent and web app for building CO2RR XAS workflows, generating FEFF/FDMNES/VASP inputs, submitting and monitoring NERSC jobs through the NERSC Superfacility API, converting completed runs into ISAAC AI-ready records, and using those records for XAS-focused machine learning.

The main interactive entry point is the Streamlit web app at `web_app/main.py`.

---

## Quick start: web app

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit pandas scikit-learn
streamlit run web_app/main.py
```

From inside `web_app/`:

```bash
streamlit run main.py
```

---

## ISAAC Portal API usage

For ISAAC Portal validation/upload from the web app or command line, set:

```bash
export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
export ISAAC_KEY="..."
streamlit run web_app/main.py
```

For one record from the command line:

```bash
export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
export ISAAC_KEY="..."
python tools/convert_simulation_xas_run_to_isaac.py \
  --input-dir . \
  --output isaac_record_draft.json \
  --validate
```

For a batch of JSON records, validate first, inspect the report, and upload only records that pass validation:

```bash
python batch_validate_upload_isaac_records.py \
  --input records.zip \
  --validate \
  --upload
```

The web app can also work from local ISAAC JSON files or ZIP archives without portal credentials.

---

## NERSC Superfacility API usage

The **NERSC agent** panel in the web app uses the NERSC Superfacility API when the app is running on your laptop. It expects these environment variables:

```bash
export NERSC_SFAPI_CLIENT_ID="your_client_id_from_iris"
export NERSC_SFAPI_PRIVATE_KEY_PATH="$HOME/Downloads/priv_key.pem"
streamlit run web_app/main.py
```

The private key must be the PEM key downloaded from Iris. Protect it before launching the app:

```bash
chmod 400 "$HOME/Downloads/priv_key.pem"
```

The app also supports a literal PEM key through `NERSC_SFAPI_PRIVATE_KEY`, but the file-path form is recommended:

```bash
export NERSC_SFAPI_PRIVATE_KEY="$(cat $HOME/Downloads/priv_key.pem)"
```

The variable names used by this web app are **not** `SFAPI_CLIENT_ID` and `SFAPI_SECRET`; those names are used by the standalone `sfapi_client` package examples. For this repository, use:

```text
NERSC_SFAPI_CLIENT_ID
NERSC_SFAPI_PRIVATE_KEY_PATH
```

To test the key with the app backend:

```bash
python - <<'PY'
from tools.nersc_job_manager import NERSCSFAPIClient
client = NERSCSFAPIClient()
print(client.status())
PY
```

If you create the client in Iris for a laptop-run web app, use your laptop's current public IPv4 address in the API client IP allowlist:

```bash
curl -4 https://ifconfig.me
```

Use the same VPN/network state that you will use when running Streamlit.

---

## Structure, XAS, and NERSC workflow capabilities

The web app can:

- build Cu/Au/Ni/Pt/Ir surfaces and interfaces;
- add adsorbates such as CO, CHO, OCCO, COCO, CO2, OH, and H;
- generate VASP relaxation folders;
- generate FEFF, FDMNES, and VASP XAS input folders;
- upload calculation packages to NERSC through the Superfacility API;
- submit Slurm jobs from the web app;
- check SF API task status and Slurm job status;
- list remote folders and preview remote files;
- download selected outputs;
- convert completed outputs into ISAAC draft records;
- validate/upload ISAAC records through the ISAAC Portal API.

The chat box can run the stepwise NERSC workflow, for example:

```text
Generate relaxation inputs
Upload to NERSC under test04
change to 8 hrs and submit the job
check slurm job status
cancel slurm job 55602116
```

---

## XAS ML capabilities

The ML page can load records from:

- local ISAAC JSON files;
- ZIP archives of ISAAC records;
- the ISAAC Portal, when `ISAAC_KEY` is set.

It applies strict XAS spectral filtering before ML. A record must contain a numeric incident-energy axis with eV/keV units and an XAS-like signal channel. Records that look like XRD, XPS, PDF, Raman, STM/STS, etc. are excluded from XAS ML by default.

For valid XAS spectra, the app summarizes:

- `record_id`;
- `record_domain`;
- material formula/name;
- absorber;
- edge;
- technique;
- number of spectral points;
- classification reason.

The ML target selector is intentionally restricted to fields useful for XAS modeling:

- `sample.material.formula`;
- non-null `descriptors.*` values;
- optional parsed structural labels such as `structure.facet` and `structure.adsorption_site`.

Other metadata such as record ID, source type, facility, instrument, timestamp, absorber, edge, technique, QC status, assets, links, and tags are not exposed as ML target labels. They can still be used for filtering or diagnosis, but they are not useful XAS ML targets and can introduce leakage/noise.

The auto-advisor compares feature sets, normalization, dimension reduction choices, simple baselines, and standard models. Manual training supports classification or regression, common-grid interpolation, normalization, optional PCA, baseline/model comparison, held-out prediction tables, and training-curve diagnostics.

---

## Simulation-folder ISAAC conversion

Completed simulation folders can be converted into ISAAC records with:

```bash
bash XAS/convert_completed_simulation_xas_to_isaac.sh \
  --input-dir . \
  --output isaac_record_draft.json
```

The converter reads absorber and edge from calculation inputs when possible:

- FEFF: `feff.inp`, `POTENTIALS`, `PARAMETERS`, `ATOMS`;
- FDMNES: `*_in.txt`, `fdmfile.txt`;
- VASP: `INCAR`, `POTCAR.spec`, and folder/file labels.

Metadata and environment variables are used only as fallback.

---

## Repository layout

```text
CO2RR_XAS_agent/
├── agent/                    # planner schemas, parser, and agent orchestration
├── tools/                    # XAS utilities, portal client, NERSC client, converters, ML helpers
├── workflow/                 # NERSC submit/monitor/finalize helpers
├── generators/               # slab/input generator utilities
├── web_app/                  # Streamlit web app
│   ├── main.py
│   └── pages/
│       └── 2_Machine_learning_for_XAS.py
├── XAS/                      # reusable XAS shell helpers
├── examples/                 # runnable examples
├── tests/                    # regression tests
└── requirements.txt
```

---

## Notes

- Use `web_app/main.py` for most interactive work.
- Keep private API keys out of the repository.
- Use `tools/convert_simulation_xas_run_to_isaac.py` for repeated simulation-folder conversion.
- Keep experimental/literature one-off conversions outside the repository unless they are intended to become reusable workflows.
- For XAS ML, use metadata for filtering and descriptors/chemically meaningful labels for targets.
