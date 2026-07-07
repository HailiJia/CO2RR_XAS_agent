# CO2RR XAS Agent

CO2RR XAS Agent is a Streamlit web app and workflow toolkit for building CO2 reduction catalyst structures, generating relaxation and XAS inputs, running restartable NERSC Slurm workflows, and converting completed simulations into ISAAC AI-ready records.

The main interactive entry point is:

```bash
cd CO2RR_XAS_Agent
streamlit run web_app/CO2RR_XAS_Agent.py
```


---

## Main capabilities

The web app supports:

- building Cu/Au/Ni/Pt/Ir surfaces and interfaces;
- adding CO2RR reaction pathway adsorbates such as CO, CHO, OCCO, COCO, CO2, OH, and H;
- generating VASP relaxation input folders;
- generating XAS calculation folders (supported software: FEFF, FDMNES, and VASP);
- preparing a restartable full NERSC workflow package;
- uploading workflow packages to NERSC through the NERSC Superfacility API;
- submitting Slurm jobs through a dependency-driven relax -> XAS workflow;
- checking workflow, Slurm, and SF API task status from the web app;
- listing remote folders and previewing remote files;
- downloading selected outputs;
- converting completed outputs into ISAAC draft records;
- validating/uploading ISAAC records through the ISAAC Portal API;
- loading ISAAC XAS records;
- training machine learning models for XAS from ISAAC records;
- comparing XAS machine learning features and models.

---

## Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit pandas scikit-learn
streamlit run web_app/CO2RR_XAS_Agent.py
```

If you already use a conda environment:

```bash
conda activate co2rr_agent
streamlit run web_app/CO2RR_XAS_Agent.py
```

Generated local packages are written under `generated_outputs/`. This directory is intended to stay local and should not be committed.

---

## NERSC Superfacility API setup

The web app can run on a laptop and communicate with NERSC through the NERSC Superfacility API. Set:

```bash
export NERSC_SFAPI_CLIENT_ID="your_client_id_from_iris"
export NERSC_SFAPI_PRIVATE_KEY_PATH="$HOME/Downloads/priv_key.pem"
chmod 400 "$NERSC_SFAPI_PRIVATE_KEY_PATH"
streamlit run web_app/CO2RR_XAS_Agent.py
```

The app expects these variable names:

```text
NERSC_SFAPI_CLIENT_ID
NERSC_SFAPI_PRIVATE_KEY_PATH
```

The names `SFAPI_CLIENT_ID` and `SFAPI_SECRET` are used by some standalone SF API examples, but not by this app.

To test the key with the app backend:

```bash
python - <<'PY'
from tools.nersc_job_manager import NERSCSFAPIClient
client = NERSCSFAPIClient()
print(client.status())
PY
```

If the API client in Iris uses an IP allowlist, add the public IPv4 address of the machine/network where Streamlit is running:

```bash
curl -4 https://ifconfig.me
```

---

## ISAAC Portal API usage

For ISAAC Portal validation/upload from the web app or command line, set:

```bash
export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
export ISAAC_KEY="..."
streamlit run web_app/CO2RR_XAS_Agent.py
```

Convert one completed simulation folder into an ISAAC draft record:

```bash
python tools/convert_simulation_xas_run_to_isaac.py \
  --input-dir . \
  --output isaac_record_draft.json \
  --validate
```

For batch validation/upload:

```bash
python batch_validate_upload_isaac_records.py \
  --input records.zip \
  --validate \
  --upload
```

The web app can also load local ISAAC JSON files or ZIP archives without portal credentials.

---

## Full NERSC relax -> XAS workflow

The recommended NERSC path is the full workflow package, not manual one-off job submission. The package contains:

```text
01_structure/
  POSCAR
  INCAR
  KPOINTS
  make_potcar.sh
  submit_relax.sh
  structure_info.json
workflow_submit.sh
workflow_xas.sh
workflow_status.sh
workflow_restart.sh
workflow_cancel.sh
workflow_state.py
workflow_state.json
workflow_manifest.json
```

The workflow is Slurm-driven and does not require the web app to stay open after submission. `workflow_submit.sh` submits:

1. the VASP relaxation job, and
2. the `workflow_xas.sh` driver job with `--dependency=afterok:<relax_job_id>`.

`workflow_xas.sh` starts only after relaxation succeeds. It reads `01_structure/CONTCAR`, regenerates XAS inputs from the relaxed structure, then submits the VASP/FDMNES/FEFF XAS jobs.

Example chat commands in the web app:

```text
prepare full workflow and upload to NERSC under test05
start full workflow
check workflow status in /pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test05
restart full workflow
cancel full workflow
```

Expected status after starting:

```text
Relaxation   job_id=<relax_job_id>       state=PENDING/RUNNING
XAS driver   job_id=<workflow_xas_job_id> state=PENDING, dependency=afterok:<relax_job_id>
```

A dependency-held `workflow_xas` job does not consume compute time while waiting for relaxation to finish.

---

## Manual NERSC checks

On Perlmutter:

```bash
cd /path/to/your/workflow/folder
bash workflow_status.sh
squeue -u username -o "%.18i %.35j %.10T %.20R"
sacct -u username -S now-2hours -o JobID,JobName,State,ExitCode,Elapsed,Submit,Start,End%20
```

To inspect the workflow state directly:

```bash
cat workflow_state.json
```

To restart a failed or incomplete stage:

```bash
bash workflow_restart.sh
```

To cancel active workflow jobs recorded in `workflow_state.json`:

```bash
bash workflow_cancel.sh
```

---

## Python on NERSC

Generated workflow scripts select a valid Python executable automatically. They try, in order:

```text
CO2RR_NERSC_PYTHON
python3.11
python3.10
python3.9
python3
python
```

The scripts also attempt:

```bash
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
```

If needed, override the executable explicitly before running a workflow script:

```bash
export CO2RR_NERSC_PYTHON=$(command -v python3.11)
```

---

## XAS ML page

The ML page loads records from:

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

## Repository layout

```text
CO2RR_XAS_agent/
├── agent/                    # planner schemas, parser, and agent orchestration
├── tools/                    # XAS utilities, portal client, NERSC client, converters, ML helpers
├── workflow/                 # NERSC submit/monitor/finalize helpers
├── generators/               # slab/input generator utilities
├── web_app/                  # Streamlit web app
│   ├── CO2RR_XAS_Agent.py    # recommended Streamlit entry point
│   ├── main.py               # main app implementation
│   └── pages/
│       └── 2_Machine_learning_for_XAS.py
├── XAS/                      # reusable XAS shell helpers
├── examples/                 # runnable examples
├── tests/                    # regression tests
└── requirements.txt
```

---


## Notes

- Use `web_app/CO2RR_XAS_Agent.py` for interactive work.
- Keep API keys and private NERSC keys out of the repository.
- Keep generated outputs under `generated_outputs/`; do not commit them.
- The full NERSC workflow should regenerate XAS inputs from relaxed `01_structure/CONTCAR`, not the original POSCAR.
- For XAS ML, use metadata for filtering and descriptors/chemically meaningful labels for targets.
