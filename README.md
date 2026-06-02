# CO2RR XAS Agent

A lightweight Python agent for generating CO2RR surface/adsorbate structures, writing XAS simulation inputs, and converting outputs into ISAAC-style AI-ready records.

The current workflow supports:

- CO2RR adsorbate/surface structure generation
- FDMNES input generation
- FEFF input generation
- VASP two-step XAS input generation
- Output parsing for FEFF, FDMNES, and VASP
- ISAAC record generation
- Local regex parsing for offline tests
- Optional LLM planning through an OpenAI-compatible endpoint, including ALCF/AskSage-style gateways
- NERSC submit/monitor/finalize workflow; see `README_NERSC_WORKFLOW.md`

---

## 1. Repository structure

```text
CO2RR_XAS_agent/
├── agent/                    # planner schemas, local parser, agent orchestration
├── tools/                    # structure generation, XAS input generation, result parsing
├── workflow/                 # NERSC submit/monitor/finalize helpers
├── generators/               # lower-level slab/input generator utilities
├── skills/                   # LLM planner skill context and legacy skill descriptors
├── examples/                 # runnable examples
├── tests/                    # regression tests for local, NERSC, ALCF, ISAAC records
├── README.md
├── README_NERSC_WORKFLOW.md
├── README_LLM_SETUP.md
└── requirements.txt
```

Notes:

- `skills/co2rr-xas/` is the only skill folder used by the Python runtime.
- The Python LLM planner loads skill instructions from `skills/co2rr-xas/SKILL.md` when `use_llm=True`.
- The YAML files in `skills/co2rr-xas/references/` provide detailed workflow context for structure generation, XAS input generation, and result parsing.
- Local regex mode (`use_llm=False`) does not use skill files; it only uses the deterministic parser in `agent/co2rr_xas_agent.py`.

---

## 2. Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pydantic openai custodian
```

Optional dependencies:

```bash
pip install pymatgen ase
```

`pymatgen` or `ase` is needed for CIF input support. POSCAR and CONTCAR can be read directly by the built-in parser.

---

## 3. Local parser versus LLM planner

The agent has two parsing modes.

### Local mode

```python
use_llm=False
```

This is good for quick tests and offline use. It supports common phrasing but is still regex-based, so use explicit prompts.

### LLM mode

```python
use_llm=True
```

This uses `agent/planner.py` to call an OpenAI-compatible chat-completions endpoint and convert flexible natural language into a structured `XASPlan`. The deterministic Python tools still write the actual files.

The intended architecture is:

```text
natural language
→ LLM planner or LocalIntentParser
→ ParsedIntent / XASPlan
→ deterministic Python file generators
→ input folders / output parsers / ISAAC records
```

---

## 4. Quick local tests

Run from the repository root.

### CHO on Cu(111), default XAS inputs

```bash
python - <<'PY'
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate XAS inputs for CHO adsorbate on Cu(111)",
    output_dir="generated_outputs/local_tests/cho_cu111_default",
    use_llm=False,
)

print(result["status"])
PY
```

### CHO on Cu(111), FDMNES Green, no SCF

```bash
python - <<'PY'
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate FDMNES Green multiple scattering XAS inputs without SCF, with Quadrupole and Spinorbit "
    "for CHO adsorbate on Cu(111), radius 7 Angstrom, energy range -5 0.2 50, K edge, "
    "NERSC account m5268",
    output_dir="generated_outputs/local_tests/cho_cu111_fdmnes_green",
    use_llm=False,
)

print(result["status"])
PY
```

Check FDMNES cards:

```bash
grep -R -n "Range\|Radius\|Green\|SCF\|Quadrupole\|Spinorbit\|Edge" \
  generated_outputs/local_tests/cho_cu111_fdmnes_green
```

Expected key cards:

```text
Range
  -5. 0.2 50.
Radius
  7.0
Quadrupole
Spinorbit
Green
Edge
  K
```

There should be no standalone `SCF` card if the prompt says `without SCF`.

### OCCO on Cu(111)-Au(111) interface

```bash
python - <<'PY'
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate FDMNES Green multiple scattering XAS inputs without SCF, with Quadrupole and Spinorbit "
    "for OCCO adsorbate on Cu(111)-Au(111) interface, radius 7 Angstrom, "
    "energy range -5 0.2 50, K edge, NERSC account m5268",
    output_dir="generated_outputs/local_tests/occo_cu111_au111_fdmnes_green",
    use_llm=False,
)

print(result["status"])
PY
```

Inspect generated files:

```bash
find generated_outputs/local_tests/occo_cu111_au111_fdmnes_green -maxdepth 6 -type f | sort
```

---

## 5. LLM setup

The LLM planner uses the `openai` Python package but can point to any OpenAI-compatible chat-completions endpoint.

Set environment variables:

```bash
export CO2RR_LLM_MODEL="your-model-name"
export CO2RR_LLM_API_KEY="your-token"
export CO2RR_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

Aliases also supported:

```bash
export ALCF_MODEL="your-model-name"
export ALCF_API_KEY="your-token"
export ALCF_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

Then run:

```bash
python - <<'PY'
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate FDMNES Green multiple scattering XAS inputs without SCF, "
    "with Quadrupole and Spinorbit for CO on Cu(111), radius 7 Angstrom, "
    "energy range start from -5 eV to 50 eV with step size 0.2 eV, "
    "K edge, NERSC account m5268",
    output_dir="generated_outputs/llm_tests/co_cu111_green",
    use_llm=True,
)

print(result["status"])
PY
```

In LLM mode, the planner loads the repo skill instructions from:

```text
skills/co2rr-xas/SKILL.md
```

and also loads YAML reference files from:

```text
skills/co2rr-xas/references/
```

---

## 6. Structure-file input

The agent can generate XAS inputs from an existing structure file:

```bash
python - <<'PY'
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate VASP PBE XAS inputs from /path/to/CONTCAR, K edge",
    output_dir="generated_outputs/from_contcar",
    use_llm=False,
)

print(result["status"])
PY
```

Supported structure files:

```text
POSCAR
CONTCAR
*.cif
```

CIF support requires `pymatgen` or `ase`.

---

## 7. XAS input generation

### FDMNES

Each FDMNES folder contains:

```text
FDMNES_fdm/ or FDMNES_green/
├── fdmfile.txt
├── <absorber>_in.txt
└── submit.sh
```

The `Range` card uses FDMNES order:

```text
emin step emax
```

For example:

```text
Range
  -5. 0.2 50.
```

Natural-language LLM prompts should map to the semantic form:

```text
emin = -5 eV, emax = 50 eV, step = 0.2 eV
```

which is then converted to FDMNES order internally.

### FEFF

Each FEFF calculation folder contains one required FEFF input:

```text
FEFF/
├── feff.inp
└── submit.sh
```

The FEFF generator now writes an absorber-centered periodic cluster:

```text
ipot = 0   absorber atom
ipot > 0   scattering potentials
```

For a Cu K-edge calculation, same-element Cu scatterers are assigned a positive `ipot`, separate from the absorbing Cu atom.

### VASP

VASP XAS is generated as a two-step workflow:

```text
VASP/
├── 01_scf/
│   ├── POSCAR
│   ├── INCAR
│   ├── KPOINTS
│   ├── POTCAR.spec
│   └── make_potcar.sh
├── 02_xas/
│   ├── POSCAR
│   ├── INCAR
│   ├── KPOINTS
│   ├── POTCAR.spec
│   └── make_potcar.sh
└── submit.sh
```

The first step is SCF/static:

```text
LWAVE = .TRUE.
LCHARG = .TRUE.
```

The second step reads the converged charge/wavefunction when available:

```text
ISTART = 1
ICHARG = 11
LOPTICS = .TRUE.
ICORELEVEL = 2
```

For GW/GW0 mode, the XAS step includes:

```text
ALGO = GW0
NOMEGA = 100
```

The folder name remains simply:

```text
VASP/
```

The method is controlled inside the generated INCAR and metadata, not by creating separate `VASP_PBE` and `VASP_GW` folders.

---

## 8. POTCAR handling

The agent writes:

```text
POTCAR.spec
make_potcar.sh
```

If a valid POTCAR root is available, it also tries to write a real `POTCAR` directly.

Set one of these environment variables:

```bash
export CO2RR_POTCAR_DIR=/path/to/potpaw_PBE
# or
export VASP_POTCAR_DIR=/path/to/potpaw_PBE
```

Expected structure:

```text
/path/to/potpaw_PBE/
├── Cu_pv/POTCAR
├── C/POTCAR
├── O/POTCAR
└── H/POTCAR
```

You can also pass a path in the prompt:

```text
POTCAR path /path/to/potpaw_PBE
```

---

## 9. Result parsing and ISAAC records

The result parser supports:

```text
FEFF    xmu.dat
FDMNES  *_conv.txt
VASP    OUTCAR dielectric-function output
```

A parsed record is converted into an ISAAC-style simulation record with:

```text
record metadata
structure metadata
calculation parameters
spectral arrays
provenance fields
```

Generated structures now carry nested catalyst and adsorbate metadata in `structure_info.json`, and the result parser copies those fields into the ISAAC record under `sample.catalyst` and `sample.adsorbate`. This is intended to make later XAS ML featurization easier. For example, an OCCO bridge site on a Cu-Au interface records metadata shaped like:

```yaml
catalyst:
  elements: [Cu, Au]
  composition: CuAu
  surface_facet: "111"
  site_type: interface
  structure_id: CuAu_111_OCCO_bridge_001
adsorbate:
  identity: OCCO
  formula: C2O2
  intermediate_class: C2
  binding_mode: bridge
  adsorption_site: Cu-Au interface
  binding_atom: C
```

The local parser and LLM planner also accept this YAML-like metadata in prompts. Direct API callers can pass metadata overrides through `metadata_overrides={"catalyst": {...}, "adsorbate": {...}, "structure_id": "..."}`.

---

## 10. Agentic recovery and missing-information handling

The Python entry points now do a preflight pass before launching deterministic tools, and the NERSC orchestration layer runs through `custodian` when it is installed. If a request is missing required context, the agent returns `status="needs_input"` with `missing_information`, `suggestions`, and the parsed intent instead of a bare script/tool error. Examples include missing catalyst metals for generated structures or a missing structure file for result parsing.

The agent also attempts safe recovery when enough context is available:

- result parsing searches the result directory for nearby `POSCAR` or `CONTCAR` files when `structure_file` was omitted;
- NERSC requests made off-NERSC automatically fall back to `submit=False` dry-run input generation when `sbatch`/`NERSC_HOST` is unavailable;
- recovered choices and warnings are reported under the returned `agent.recovery_actions` and `agent.warnings` fields.

---

## 11. Configurable NERSC defaults

Hard-coded NERSC values are minimized. If you do not pass explicit values, the workflow checks these environment variables before using safe placeholders/defaults:

```bash
export CO2RR_NERSC_ACCOUNT=mXXXX
export CO2RR_NERSC_QUEUE=regular
export CO2RR_NERSC_NODES=2
export CO2RR_NERSC_WALLTIME=8:00:00
export CO2RR_NERSC_CLUSTER=Perlmutter
export CO2RR_NERSC_FACILITY=NERSC
export CO2RR_NERSC_ORGANIZATION=LBNL
```

---

## 12. Common issues

### Are the skills used?

Yes, but only in LLM mode. When you run with:

```python
use_llm=True
```

`agent/planner.py` loads:

```text
skills/co2rr-xas/SKILL.md
skills/co2rr-xas/references/*.yaml
```

When you run with:

```python
use_llm=False
```

the skills are not used. Local mode uses the regex parser only.

### Local parser does not understand a flexible prompt

Use more explicit phrasing in local mode, or switch to LLM mode:

```python
use_llm=True
```

### CIF input fails

Install `pymatgen` or `ase`:

```bash
pip install pymatgen ase
```

### POTCAR is missing

Set:

```bash
export CO2RR_POTCAR_DIR=/path/to/potpaw_PBE
```

or run the generated `make_potcar.sh` after loading the correct VASP/POTCAR environment.

---

## 13. Minimal Python API

```python
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate XAS inputs for CHO adsorbate on Cu(111)",
    output_dir="generated_outputs/example",
    use_llm=False,
)

print(result)
```

For LLM mode:

```python
result = process_request(
    "Generate FDMNES Green XAS input for CHO on Cu(111), from -5 to 50 eV with 0.2 eV step",
    output_dir="generated_outputs/example_llm",
    use_llm=True,
)
```

---

## 12. Development notes

- `agent/schemas.py` contains shared schema and normalization logic.
- `agent/planner.py` contains the optional LLM planner.
- `agent/co2rr_xas_agent.py` contains orchestration and local fallback parsing.
- `tools/structure_generator.py` writes structures.
- `tools/xas_input_generator.py` writes FDMNES, FEFF, and VASP inputs.
- `tools/result_parser.py` parses outputs and writes ISAAC records.
- `tools/utils.py` contains shared constants, POSCAR/CONTCAR/CIF reading, and helper utilities.
