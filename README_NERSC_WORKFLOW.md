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

## Structured catalyst/adsorbate metadata for XAS ML

When the NERSC workflow starts from generated structures, `structure_info.json`
contains nested metadata that is propagated through input generation and copied
into the final `isaac_record.json` under `sample.catalyst` and
`sample.adsorbate`.  This supports downstream XAS ML workflows without relying on
path-name parsing.

Example prompt fragment accepted by the local parser and LLM planner:

```yaml
catalyst:
  elements: [Cu, Au]
  composition: CuAu
  surface_facet: "111"
  site_type: interface
  structure_id: "CuAu_111_OCCO_bridge_001"
adsorbate:
  identity: "OCCO"
  formula: "C2O2"
  intermediate_class: "C2"
  binding_mode: "bridge"
  adsorption_site: "Cu-Au interface"
  binding_atom: "C"
```

## Agentic recovery behavior

The workflow phases (`generate_xas_inputs`, `submit_jobs`, `monitor_jobs`, and `parse_completed_jobs`) are run as custodian jobs, so `custodian.json` records the job-management log in the workflow output directory. If a NERSC workflow is requested from a machine that does not expose `sbatch` and
does not set `NERSC_HOST`, the high-level agent converts the request to a dry run
instead of failing immediately.  The returned result includes generated inputs,
`submitted_jobs` with `state="DRY_RUN"`, and `agent.warnings` /
`agent.recovery_actions` explaining what happened.  Missing structure/catalyst
information returns `status="needs_input"` with concrete suggestions.



## Structure relaxation and absorber edges

Generated bimetallic `Cu(111)-Au(111)`-style structures now use a lateral
interface rather than a vertical Cu/Au stack, and OCCO is placed across the
interface carbons for bridge-interface prompts.  Relaxation POSCARs include
selective dynamics with the bottom two substrate layers fixed.  Production XAS
inputs should be regenerated from `relax/CONTCAR` after relaxation; when no
`CONTCAR` is present, dry-run input generation records that XAS files are
provisional and based on the unrelaxed structure.  Heavy-metal absorbers keep
their default L3 edge even if a prompt asks for a generic K-edge calculation, so
Cu uses K while Au uses L3 in Cu/Au workflows.

## Output consistency checks

Completed XAS jobs are now passed through a code-level Custodian validation
step before ISAAC parsing.  The validation job parses the candidate FEFF,
FDMNES, or VASP output, compares the output timestamp against nearby input files
to flag stale data, and checks energy-window coverage when an expected window is
available.  FDMNES `Range` cards in `*_in.txt` are converted to absolute eV
windows automatically; other workflows can pass `expected_energy_window_eV` or
`expected_energy_range_eV`.  Validation results are returned in the workflow
record under `validation_result` and are also copied into
`measurement.qc.output_consistency` in the ISAAC record.  Pass
`parameters={"strict_output_validation": True}` to stop ISAAC record creation
when Custodian detects an output/input mismatch.

## Configurable defaults

The direct API still accepts explicit `nersc_account`, `nersc_queue`,
`nersc_nodes`, and `nersc_walltime` arguments.  If they are omitted, the workflow
uses these environment variables before falling back to safe placeholders:

```bash
export CO2RR_NERSC_ACCOUNT=mXXXX
export CO2RR_NERSC_QUEUE=regular
export CO2RR_NERSC_NODES=2
export CO2RR_NERSC_WALLTIME=8:00:00
export CO2RR_NERSC_CLUSTER=Perlmutter
export CO2RR_NERSC_FACILITY=NERSC
export CO2RR_NERSC_ORGANIZATION=LBNL
```
