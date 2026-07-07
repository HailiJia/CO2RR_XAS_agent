# Restartable NERSC relax-to-XAS workflow

This branch introduces a Slurm-side workflow layer for the Structure/XAS web app. The goal is to make the relax -> post-relax XAS workflow independent of the Streamlit process.

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

After the app generates `01_structure/submit_relax.sh`, add the workflow layer with:

```bash
python tools/nersc_workflow_package.py \
  --package-root generated_outputs/web_xas_agent \
  --repo-root /path/to/CO2RR_XAS_agent \
  --absorber Cu \
  --edge K \
  --account m5268 \
  --queue regular \
  --nodes 1 \
  --walltime 02:00:00
```

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

The next integration step is to add web-app buttons and chat commands that call these scripts through the NERSC SF API:

- `Generate workflow package`
- `Upload workflow package`
- `Start workflow`
- `Refresh workflow status`
- `Restart workflow`
- `Cancel workflow`
- `Download workflow outputs`

The chat box should route to these deterministic operations instead of acting as the workflow engine.
