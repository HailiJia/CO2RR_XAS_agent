---
name: co2rr-isaac-xas-ml
description: Use this skill to build CO2RR catalyst XAS machine-learning workflows from ISAAC AI-ready records, including simulation-only training, simulation-to-experiment classification, XAS preprocessing/featurization, grouped validation, domain-of-applicability scoring, and ISAAC-compatible ML training/prediction records.
---

# CO2RR ISAAC XAS ML Skill

Use this skill when the user asks to train, evaluate, compare, or apply ML models for CO2RR XAS/XANES records stored as ISAAC JSON. The deterministic helper module is `tools.ml_xas_workflow`.

## Scope

Supported catalyst elements/edges:

- Cu K, Ni K
- Au L3, Ir L3, Pt L3
- adsorbate C K

Supported task families:

1. `adsorbate_classification`: CO2RR adsorbate/intermediate labels such as clean, CO*, CHO*, OCCO*, COCO*.
2. `adsorption_motif_classification`: atop, bridge, hollow, interface, C-bound/O-bound, C1/C2 motifs.
3. `oxidation_coordination_prediction`: oxidation/local coordination classes or descriptors.

Supported modes:

- `simulation_only_training`: train/validate/test using simulation records with grouped validation.
- `simulation_to_experiment`: train/validate on simulation records and classify experimental records externally. Do **not** randomly mix experimental spectra into simulation train/test splits unless explicitly requested.

## Deterministic helper

For dataset assembly and ISAAC ML record scaffolding, call:

```python
from tools.ml_xas_workflow import execute_ml_dataset_assembly

result = execute_ml_dataset_assembly(
    isaac_record_paths=["example_ouput/FEFF/isaac_record.json"],
    output_dir="generated_outputs/ml_dataset",
    mode="simulation_only_training",
    task_name="adsorbate_classification",
)
```

Outputs:

- `spectrum_table.json`: one row per `measurement.series` spectrum.
- `condition_table.json`: spectra merged by condition/structure.
- `ml_training_record.json`: ISAAC-compatible ML training record scaffold.
- `ml_prediction_record.json`: prediction scaffold when using `simulation_to_experiment`.

## Workflow

1. Load ISAAC simulation and experimental records.
2. Extract spectra from `measurement.series`.
3. Build spectrum-level and condition-level tables.
4. Extract catalyst, adsorbate, experiment, descriptor, and ML-label metadata.
5. Validate spectra and flag missing fields.
6. Interpolate spectra to common edge-specific grids.
7. Normalize, compute first derivative, and compute CDF representation.
8. Generate edge-specific features (raw, derivative, CDF, peak/window descriptors).
9. Build single-edge, early-fusion, and late-fusion datasets.
10. Compare interpretable baselines and high-capacity models.
11. Use grouped validation to prevent leakage.
12. For simulation-to-experiment mode, classify experimental records as external predictions and compute domain-of-applicability status.
13. Write human-readable reports and ISAAC-compatible ML training/prediction records.

## Required scientific rules

- Descriptors are features unless explicitly selected as labels.
- Default classification metric: balanced accuracy or macro F1.
- Default regression metric: RMSE or MAE.
- Do not use random search as the default HPO method.
- Use grid search for simple models/PCA/PLS and Bayesian optimization for high-dimensional or neural models.
- Always report edge coverage, edge ablation, and domain-of-applicability for experimental predictions.
- Do not claim definitive experimental identification solely from model probability.
- Outside-domain predictions should be described as closest simulated candidates, not confident assignments.

## References

Load these only as needed:

- `references/metadata_schema.yaml`: expected CO2RR metadata blocks and ISAAC ML record templates.
- `references/workflow_plan.yaml`: detailed workflow modes, model families, validation, HPO, and reporting rules.
