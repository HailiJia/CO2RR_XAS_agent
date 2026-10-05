# Geometry checks, dataset labels, and grouped XAS learning

This workflow supports simulation-trained identification of CO2RR adsorbates and adsorption geometry. Cu–Au is the demonstration system; the lateral interface builder supports distinct fcc metals with matching (111) facets. FEFF, FDMNES, and VASP remain the three XAS backends. Building an interface does not establish its equilibrium stability or ML transfer to another catalyst.

## 1. Chemical checks and general interfaces

`tools.contact_checks.validate_geometry` checks all atom pairs using periodic minimum images, including skew cells, with x/y periodicity and a nonperiodic vacuum direction by default. It also checks self images, declared molecule formulas, intended bonds, bond angles, possible intermolecular coupling, and adsorbates below the selected top face. A short normal C–H bond cannot hide a metal contact elsewhere in the cell.

Cutoffs in `pair_cutoffs` and `BOND_RANGES` are configurable screening rules in Angstrom. Angle tolerances are derived from the reference templates, with a wider allowed bending range for CO2. These checks flag structures for review; they do not calculate stability, bond order, or relaxation convergence. Changed or unmapped chemistry receives no verified ML identity/geometry target.

The stripe generator accepts the following generic regions. M1 and M2 are the ordered `element1` and `element2` metals. Existing Cu/Au names remain supported when their named metals exist.

| Region | Supported placement |
| --- | --- |
| `M1_side_interface`, `M2_side_interface` | Single-binding adsorbates on a boundary row |
| `M1_near_interface_row_1/2`, `M2_near_interface_row_1/2` | Neighboring row classes, when available |
| `M1_terrace`, `M2_terrace` | Rows furthest from a boundary |
| `M1_M2_boundary_bridge` | Coupled OCCO across the interface |
| `M1_side_interface_dimer_along_x`, `M2_side_interface_dimer_along_x` | Same-side OCCO along the interface |
| `M1_side_interface_dimer_row0_row1`, `M2_side_interface_dimer_row0_row1` | OCCO spanning two same-side row classes |

Unsupported facets, missing region metals, unknown regions, and unavailable sites raise errors. OCCO placement preserves its coupled C–C geometry rather than stretching it to metal-site separation. `COCO` remains a legacy input alias for OCCO in the dictionary generator. The methyl template has tetrahedral H–C–H geometry.

```python
from tools.structure_generator import StructureGenerator
from tools.stripe_interface_patch import install

install()
generator = StructureGenerator()
parent = generator.generate_interface(
    "Ni", "Pt", element1_rows=4, element2_rows=4, layers1=4, layers2=4
)
structure = generator.add_adsorbate(parent, "CO", adsorption_region="M1_terrace")
generator.save_structure(structure, "generated_outputs/NiPt_CO")
```

Coverage scenario sweeps also use the actual ordered metal pair for OCCO and spectator CO. Asymmetric bidentate COOH sweeps retain their existing Cu/Au-specific parameters and explicitly reject other pairs. Generic monodentate placements remain available through the structure generator.

## 2. Labels, coverage, and relaxed structures

Saved `structure_info.json` files use label schema 2.0. Atom and molecule mappings are remapped into POSCAR's element-grouped order. Atom IDs persist from the initial decorated configuration through CONTCAR, while geometry measurements and metal layer assignments are refreshed. Sequential coadsorbates are placed on substrate atoms rather than on an existing adsorbate.

| Field | Meaning |
| --- | --- |
| `configuration_id` | Decorated configuration linkage shared by its derived spectra |
| `parent_structure_id` | Clean-parent provenance; optional stricter family holdout |
| `geometry_hash_initial`, `geometry_hash` | Initial and current order-independent geometry fingerprints |
| `molecules` | Intended species, molecule IDs, global atom indices, anchors, bonds, angles |
| `atom_ids`, `atom_layers`, `surface_atom_ids` | Atom correspondence and measured metal layers/top face |
| `species_requested`, `geometry_requested`, `distribution_requested` | Starting choices retained as provenance |
| `species_observed`, `geometry_observed`, `observed_geometry` | Connectivity-screened identity and measured contact motifs/distances |
| `coverage_labels` | Requested coverage, realized numerator/denominator, reference basis, per-species counts and molecular coverage, occupied surface sites |
| `distribution_observed` | Binding-anchor distances to both periodic boundaries and occupied sites |
| `geometry_qc`, `label_status` | Machine-readable issues and whether targets require review |
| `ml_targets` | Verified adsorbate identity, local geometry, and realized coverage; withheld when review is required |

Local geometry uses metal contacts within 3 Angstrom and within 0.25 Angstrom of the nearest top-face metal contact. The target includes metal composition and the resulting atop/bridge/hollow/multicoordinated motif. Continuous contact distances and interface distances are retained. It is a disclosed geometric heuristic, not an electronic bonding assignment.

The existing OCCO coverage convention is retained: one OCCO core plus spectator CO, with carbon-equivalent coverage. Molecular coverage and occupied-site counts are reported separately. Other scenarios use molecular count divided by the declared surface reference. The denominator is preserved through relaxation; the observed occupied sites are recomputed. Distribution selection remains deterministic; no random placement seed is implied.

The CONTCAR regeneration and synchronization routes refresh these labels before writing XAS inputs. ISAAC record builders and generated job metadata carry the same configuration, targets, and structure descriptors, so different backends and processing variants preserve their linkage. Absorber/edge selection continues to follow each backend's input deck; a common configuration ID alone does not establish equivalent absorber sites or numerical agreement across backends.

Audit an existing saved dataset and export an inventory:

```bash
python -m tools.dataset_labels generated_outputs/dataset
```

This command refreshes `structure_info.json` from CONTCAR where present (otherwise POSCAR) and writes `dataset_manifest.json` and `.csv`. Legacy datasets without explicit molecule mappings are marked for review, rather than assigning relaxed species from requested names. Add or verify mappings before treating those labels as scientific training targets. The audit does not modify coordinates or start calculations.

## 3. Executed grouped learning

The Streamlit ML builder and automatic advisor use the same configuration grouping and fixed outer holdout. Simulation rows require configuration linkage or an explicitly selected group field; record IDs alone are not silently treated as independent configurations. Identical current geometry hashes link configurations that converge to the same structure. The clean parent is not the default group, because distinct decorated configurations on one parent are separate samples; users can select it for a stricter family holdout.

1. Filter valid labeled spectra within one absorber/edge representation.
2. Reserve whole test groups using seed 7, without consulting spectra or targets.
3. Fit interpolation grids, scaling, PCA, and estimators only within the relevant training fold.
4. Rank advisor candidates using inner GroupKFold validation on outer training groups only.
5. Fit the selected model on outer training groups and evaluate that winner once on the reserved test groups.

At least three configuration groups are required. Infeasible classification folds or PCA settings are skipped with their reasons. Training scores are never relabeled as validation scores. Test class counts and unseen test classes are reported; a small or class-incomplete test set cannot establish general performance. Learning curves use whole training groups and exclude the outer test set.

```python
from tools.xas_ml_utils import train_manual_rows, run_auto_advisor, export_model

# rows come from tools.xas_record_utils.rows_from_records(...)
result = train_manual_rows(
    rows, target="sample.ml_labels.adsorbate_identity",
    task="classification", model_name="Logistic regression",
    kinds=["raw", "peak_descriptors"], normalization="minmax", n_grid=256,
)
with open("xas_ml_model.joblib", "wb") as handle:
    handle.write(export_model(result))

advisor = run_auto_advisor(
    rows, "sample.ml_labels.adsorbate_identity", "classification", n_grid=128
)
```

The download contains the fitted spectral grid/recipe, scaler, optional PCA, estimator/classes, target, split manifest, dataset digest, metrics, and Python/NumPy/scikit-learn/workflow versions. Predictions on new valid spectra use the saved pipeline; spectra outside its fitted energy interval are rejected instead of extrapolated. The split manifest lists row IDs, configuration groups, indices, and the split ID. Advisor recommendations additionally expose the inner fold groups.

```python
import joblib
bundle = joblib.load("xas_ml_model.joblib")
predictions = bundle["pipeline"].predict(new_spectral_rows)
```

Tests exercise periodic clashes, molecule connectivity/angles, generic Cu–Au/Cu–Ni/Cu–Pt/Ni–Pt controls, POSCAR remapping, relaxation handoff to all three backends, coadsorbate counts, group leakage, holdout-independent advisor ranking, training-only preprocessing, and exported prediction replay. No production NERSC calculations, experimental inference, or manuscript ML performance claims are produced by these code changes.
