---
name: co2rr-xas
description: Use this skill for CO2RR catalyst structure generation, XAS input generation with FDMNES/FEFF/VASP, parsing XAS outputs, and creating ISAAC records.
---

# CO2RR XAS Agent Skill

Use this skill when the user asks for any part of the CO2RR XAS workflow:

1. Generate CO2RR metal surface or interface structures.
2. Add CO2RR adsorbates such as CO, CHO, CHOH, COCO, or OCCO.
3. Generate XAS inputs for FDMNES, FEFF, and/or VASP.
4. Collect or parse XAS outputs.
5. Convert parsed spectra into ISAAC records.

The workflow should use deterministic Python tools for file creation and parsing. The LLM should only convert the user request into a structured plan and select validated parameters.

## Execution model

Use the following division of responsibility:

- LLM/planner: interpret flexible natural language and produce a structured XASPlan.
- Python tools: validate, normalize, and generate files.
- Parsers: read calculation outputs and create ISAAC-compatible records.

Do not directly invent POSCAR, FDMNES, FEFF, VASP, or ISAAC file contents in free text if a Python tool can generate them.

## Structure generation

Extract these fields from the user request:

- metals: one metal for a pure surface, two metals for an interface.
- facets: common values are 111, 100, 110, and 0001.
- adsorbates: CO, CH, CH2, CHO, CHOH, CH3, CH4, COCO, OCCO.
- full_pathway: true when the user asks for all adsorbates, full CO2RR pathway, or complete pathway.
- site: top, bridge, fcc, or hcp.
- supercell: e.g. 3x3 or 4x4.
- layers: number of slab layers.

Preserve chemical capitalization. CO is carbon monoxide. Co is cobalt.

## XAS input generation

Extract these fields:

- software: fdmnes, feff, vasp, or all.
- absorber_elements: requested absorbing atoms/elements, if specified.
- edge: K, L2, L3, or L23.
- cluster_radius: radius in Angstrom.
- nersc_account, nersc_queue, nersc_nodes, nersc_walltime, and email if provided.

For FDMNES, map language to options:

- "Green", "Green function", or "multiple scattering" -> method = green.
- "finite difference" or "FDM" -> method = fdm.
- "without SCF", "no SCF", or "disable SCF" -> SCF = false.
- "Quadrupole" -> Quadrupole = true.
- "Spinorbit", "spin orbit", or "spin-orbit" -> Spinorbit = true.
- "Relativism" or "relativistic" -> Relativism = true.
- "SCFexc" -> SCFexc = true.
- "Screening" -> Screening = true.
- "Full_atom", "full atom", or "full-atom" -> Full_atom = true.
- "TDDFT" or "TD-DFT" -> TDDFT = true.

## Energy range normalization

Always represent energy ranges semantically first:

```json
{"emin": -5.0, "emax": 50.0, "step": 0.2}
```

Then convert to the FDMNES `Range` card order:

```text
-5. 0.2 50.
```

Examples:

- "energy range -5 0.2 50" -> emin = -5, emax = 50, step = 0.2.
- "from -5 eV to 50 eV with step size 0.2 eV" -> emin = -5, emax = 50, step = 0.2.
- "50 eV to -5 eV with interval 0.2 eV" -> emin = -5, emax = 50, step = 0.2.
- "start from -5 eV, end at 50 eV, step 0.2 eV" -> emin = -5, emax = 50, step = 0.2.

## Result parsing and ISAAC records

For output parsing, extract:

- software: FEFF, FDMNES, VASP, VASP_GW, or VASP_PBE.
- output_dir or results_dir.
- structure_file.
- absorber.
- edge.
- vasp_mode: trace, parallel, normal, or tauc.

Expected output files:

- FEFF: `xmu.dat`.
- FDMNES: `*_conv.txt`.
- VASP: `OUTCAR`, with extracted dielectric data written to `CORE_DIELECTRIC_IMAG.dat` when needed.

ISAAC records should preserve structure, parameters, parsed spectra, provenance, tags, and description.

## Additional references

The original YAML skill definitions are kept as reference material in:

- `references/skill_structure_generation.yaml`
- `references/skill_xas_input_generation.yaml`
- `references/skill_result_parsing.yaml`
