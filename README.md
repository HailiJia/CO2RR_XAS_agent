# CO2RR XAS Agent

AI-assisted workflow tools for generating CO2RR catalyst–adsorbate structures and spectroscopy input files for XAS simulations.

## Current Capabilities

- Generate simple metal slab structures for CO2RR model systems.
- Add adsorbates such as `CO`, `CHO`, `COCO`, and `OCCO`.
- Generate VASP relaxation inputs.
- Generate XAS inputs for:
  - FEFF
  - FDMNES
  - VASP PBE/GW-style XAS templates
- Parse selected XAS outputs into AI-ready records.
- Use natural-language prompts through the local parser or an ALCF-hosted LLM planner.
- Write generated files into local output folders.

## Project Layout

```text
CO2RR_XAS_agent/
├── agent/
│   ├── __init__.py
│   └── co2rr_xas_agent.py
├── tools/
│   ├── __init__.py
│   ├── structure_generator.py
│   ├── xas_input_generator.py
│   ├── result_parser.py
│   └── utils.py
├── skills/
│   ├── skill_structure_generation.yaml
│   ├── skill_xas_input_generation.yaml
│   └── skill_result_parsing.yaml
├── examples/
├── alcf_co2rr_runner.py
├── README.md
└── .gitignore

