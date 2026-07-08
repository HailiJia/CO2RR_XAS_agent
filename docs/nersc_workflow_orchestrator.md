# Restartable NERSC relax-to-XAS workflow

This branch introduces a Slurm-side workflow layer for the Structure/XAS web app. The goal is to make the relax -> post-relax XAS workflow independent of the Streamlit process.

## Correct web-app order

The intended user-facing workflow is:

```text
Web app:
1. Generate structure.
2. Prepare the full NERSC workflow package.
   - This internally generates 01_structure/submit_relax.sh.
   - The user should not manually run a separate relaxation-input step.
3. Generate workflow_submit.sh / workflow_xas.sh / workflow_status.sh / workflow_restart.sh / workflow_cancel.sh.
4. Upload the whole package to NERSC.
5. Remotely run: bash workflow_submit.sh.
6. Later read workflow_state.json or run workflow_status.sh.
```

The web-app integration helper is implemented in:

```text
tools/web_nersc_workflow_integration.py
```

To wire the existing Streamlit app to this helper, run once from the repository root:

```bash
python scripts/apply_webapp_nersc_workflow_orchestrator_patch.py
python -m py_compile web_app/main.py tools/web_nersc_workflow_integration.py
```

After patching, the chat box should support deterministic commands such as:

```text
prepare full workflow
upload full workflow to NERSC under test03
prepare, upload, and start full workflow under test03
refresh full workflow status
restart full workflow
cancel full workflow
```

## Design

The web app should prepare and upload a package. NERSC then owns execution through scripts in the uploaded run directory:

```text
workflow_submit.sh          # first submission entry point
workflow_xas.sh             # post-relaxation XAS continuation step
workflow_restart.sh         # checks completed stages and continues safely
workflow_status.sh          # refreshes workflow_state.json from Slurm
workflow_cancel.sh          # cancels all known unfinished jobs
workflow_state.py           # persistent state helper
workflow_manifest.json      # static workflow settings
workflow_state.json         # live persistent workflow state
01_structure/submit_relax.sh
01_structure/CONTCAR        # produced by relaxation
02_XAS/{VASP,FDMNES,FEFF}   # regenerated from post-relax CONTCAR
```

The web app can be closed and reopened. It only needs to read `workflow_state.json` and optionally call `workflow_status.sh`, `workflow_restart.sh`, or `workflow_cancel.sh` through the NERSC SF API.

## Why this is better than Streamlit session state

- Slurm dependencies, not the web app, control stage ordering.
- Workflow state is stored on NERSC in `workflow_state.json`.
- Restart logic checks what is already completed and skips completed jobs.
- `workflow_xas.sh` regenerates XAS from `01_structure/CONTCAR`, not from the unrelaxed starting POSCAR.
- `workflow_cancel.sh` cancels all known active jobs from the state file.

## Script behavior

### `workflow_submit.sh`

1. Initializes `workflow_state.json`.
2. Submits `01_structure/submit_relax.sh` if no active/completed relaxation job exists.
3. Submits `workflow_xas.sh` with `--dependency=afterok:<relax_job_id>`.
4. Refreshes workflow state.

### `workflow_xas.sh`

1. Runs after relaxation succeeds.
2. Requires `01_structure/CONTCAR`.
3. Regenerates `02_XAS` from the relaxed CONTCAR using `tools/remote_xas_from_contcar.py`.
4. Submits VASP, FDMNES, and FEFF XAS jobs if they are missing, failed, or cancelled.
5. Skips jobs already pending, running, or completed.
6. Updates `workflow_state.json`.

### `workflow_restart.sh`

1. Refreshes Slurm states.
2. If relaxation is missing or failed, resubmits relaxation and a dependent `workflow_xas.sh`.
3. If relaxation is completed, submits `workflow_xas.sh` immediately.
4. If relaxation is still pending/running, ensures the dependent XAS continuation exists.
5. Does not resubmit completed XAS jobs.

### `workflow_status.sh`

Refreshes `workflow_state.json` from `squeue`/`sacct` and prints the JSON state.

### `workflow_cancel.sh`

Reads all known job IDs from `workflow_state.json` and calls `scancel` for unfinished jobs.

## Local package generation

The old manual backend-only way is still available. After the app generates `01_structure/submit_relax.sh`, add the workflow layer with:

```bash
python tools/nersc_workflow_package.py \
  --package-root generated_outputs/web_xas_agent \
  --repo-root /path/to/CO2RR_XAS_agent \
  --absorber Cu \
  --edge K \
  --account m5268 \
  --queue regular \
  --nodes 1 \
  --walltime 05:00:00
```

For the corrected web-app flow, use `tools/web_nersc_workflow_integration.py` instead; it generates `01_structure/submit_relax.sh` and then calls the workflow package generator in one action.

On NERSC, `--repo-root` should point to a clone of this repository. The generated `workflow_xas.sh` calls:

```bash
python3 <repo-root>/tools/remote_xas_from_contcar.py ...
```

## NERSC usage

Start from inside the uploaded run directory:

```bash
bash workflow_submit.sh
```

Refresh:

```bash
bash workflow_status.sh
```

Restart/continue:

```bash
bash workflow_restart.sh
```

Cancel:

```bash
bash workflow_cancel.sh
```

## Web app integration target

The intended deterministic web-app actions are:

- `Prepare full workflow`: generate structure if needed, generate `01_structure/submit_relax.sh`, then generate `workflow_*.sh`.
- `Upload full workflow package`: upload the package root to NERSC.
- `Start full workflow`: run `bash workflow_submit.sh` remotely through the SF API.
- `Refresh workflow status`: run `bash workflow_status.sh` remotely or read `workflow_state.json`.
- `Restart workflow`: run `bash workflow_restart.sh` remotely.
- `Cancel workflow`: run `bash workflow_cancel.sh` remotely.
- `Download workflow outputs`: download outputs and state files from the remote run folder.

The chat box should route to these deterministic operations instead of acting as the workflow engine.
