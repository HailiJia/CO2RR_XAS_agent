# NERSC end-to-end workflow

The deterministic agent can run on NERSC login nodes.  It does not require the
LLM planner to generate inputs, submit jobs, monitor jobs, or write ISAAC
records.  The optional LLM planner is compatible with any OpenAI-compatible
endpoint that is reachable from the NERSC environment.

## LLM compatibility

For local deterministic parsing, use `use_llm=False` and no model credentials are
needed.  For LLM planning, install `openai` and point the planner at an
OpenAI-compatible endpoint:

```bash
export CO2RR_LLM_MODEL="your-model"
export CO2RR_LLM_API_KEY="your-token"
export CO2RR_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

The planner also honors `OPENAI_*` and `ALCF_*` aliases described in
`README_LLM_SETUP.md`.  If the endpoint is unavailable, the agent falls back to
its local regex parser.

## Generate, submit, monitor, and write records

On a NERSC login node, from the repository root:

```bash
python - <<'PY'
from agent.co2rr_xas_agent import create_agent

agent = create_agent(use_llm=False)
result = agent.run_nersc_workflow(
    structure_file="example_ouput/VASP/XAS/POSCAR",
    output_dir="generated_outputs/nersc_cu_xas",
    software="FEFF",          # FEFF, FDMNES, VASP, or all
    submit=True,              # call sbatch
    monitor=True,             # poll squeue/sacct
    parse_when_complete=True, # write isaac_record.json after COMPLETED
    nersc_account="mXXXX",
    nersc_queue="regular",
    nersc_nodes=1,
    nersc_walltime="02:00:00",
    edge_override="K",
    poll_interval=60,
)
print(result["status"])
print(result["submitted_jobs"])
print(result["records"])
PY
```

To only generate inputs and SLURM submit scripts without submitting jobs:

```python
agent.run_nersc_workflow(
    structure_file="POSCAR",
    output_dir="generated_outputs/dry_run",
    software="all",
    submit=False,
)
```

## What is monitored

`workflow.nersc_workflow.NERSCJobManager` submits with `sbatch --parsable`, checks
active jobs with `squeue`, and checks finished jobs with `sacct`.  Terminal
states include `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`, `NODE_FAIL`, and
other SLURM terminal states.  ISAAC records are only parsed automatically for
jobs that reach `COMPLETED`.

The generated `submit.sh` scripts already write `isaac_run_metadata.json` with
start/end timestamps, command, exit code, hostname, SLURM job fields, cluster,
and assets.  The final parser combines that run metadata with the parsed FEFF,
FDMNES, or VASP spectra to write `isaac_record.json`.
