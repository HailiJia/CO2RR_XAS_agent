# Adsorbate coverage and distribution generation

The structure generator supports dataset-oriented adsorbate placement in addition
to the original single-adsorbate mode.

## Natural-language examples

```text
Generate CO on Cu Au (111) at 0.25 ML, uniform, coverage based on Cu.

Generate CO on Cu Au (111) at coverages 0.056, 0.111 and 0.25 ML, uniform.

Generate OH on Cu Au (111) at 0.111 ML, interface-biased, coverage based on Cu.

Generate CO at 0.25 ML, interior-biased, coverage based on Cu.

Generate OCCO across the interface at 0.056 ML on Cu Au (111).

Generate all reasonable scenarios for OCCO on Cu Au (111) at 0.056 ML.

Generate all reasonable scenarios for OCCO on Cu Au (111).
# With no coverage supplied, use the species-specific recommended coverage grid.

Generate all reasonable scenarios for COOH on Cu Au (111) at 0.056 ML,
coverage based on Cu, boundary margin 0.15.
```

## Supported distributions

- `uniform`: maximin selection over eligible sites, maximizing spatial separation.
- `interface_biased`: keeps adsorbates separated while preferring sites near the
  internal lateral interface.
- `interior_biased`: keeps adsorbates separated while preferring sites farther
  from the interface.
- `cross_interface`: pair-based motif spanning the two metals, currently used
  for species such as OCCO and COOH.

The generator deliberately avoids simply packing all adsorbates into the highest
bias region. Spatial separation is the primary objective; interface/interior bias
breaks ties among physically separated placements.

## Coverage

Coverage is converted to an integer adsorbate count from the selected top-layer
coverage basis.

For Cu/Au datasets, use:

```text
coverage based on Cu
```

to reproduce the Cu-referenced convention used in the training structures. For
example, with 36 top-layer Cu atoms:

- 1 adsorbate = 0.028 ML
- 2 adsorbates = 0.056 ML
- 4 adsorbates = 0.111 ML
- 6 adsorbates = 0.167 ML
- 8 adsorbates = 0.222 ML
- 12 adsorbates = 0.333 ML

For OCCO, coverage is reported as **total CO-equivalent carbon coverage**.
One OCCO contains two carbon centers and therefore contributes 2/36 = 0.056 ML
on the 36-Cu basis. At higher total coverage, the generator keeps **one OCCO
core and adds spectator CO** rather than filling the surface with multiple OCCO
dimers. For example:

- 0.056 ML = 1 OCCO
- 0.111 ML = 1 OCCO + 2 spectator CO
- 0.167 ML = 1 OCCO + 4 spectator CO
- 0.222 ML = 1 OCCO + 6 spectator CO
- 0.333 ML = 1 OCCO + 10 spectator CO

## "All reasonable scenarios"

This is species-specific rather than a blind Cartesian product. If the user
supplies a coverage, all reasonable binding/distribution scenarios are generated
at that coverage. If the user omits coverage, the agent also uses a conservative
species-specific coverage grid:

- CO: 0.028, 0.056, 0.111, 0.25, 0.50, 0.75 ML
- H / OH: 0.111, 0.25, 0.50 ML
- CHO / COH: 0.028, 0.056, 0.111, 0.167 ML
- COOH: 0.028, 0.056, 0.111 ML
- OCCO total C-equivalent coverage: 0.056, 0.111, 0.167, 0.222, 0.333 ML


For an internal Cu/Au interface:

### OCCO

- Cu-Cu, uniformly placed
- Cu-Cu, interface-biased
- Cu-Au cross-interface: one carbon associated with Cu and the other with Au
- at higher total C coverage: the same single OCCO core + spatially distributed
  spectator CO

### High-coverage OCCO

OCCO is treated as a transient C-C-coupled intermediate, not as an overlayer.
For requested total C-equivalent coverage above the dilute 2/36 case, the
generator keeps **one OCCO core** and fills the remaining coverage with
spectator CO:

- 0.056 ML = one OCCO, no spectator CO
- 0.111 ML = one OCCO + 2 CO
- 0.167 ML = one OCCO + 4 CO
- 0.222 ML = one OCCO + 6 CO
- 0.333 ML = one OCCO + 10 CO

For the Cu-Au cross-interface scenario, the OCCO core spans an internal Cu-Au
pair. Spectator CO is mostly Cu-bound, with a small Au-bound fraction at higher
coverage. `2CO` is not treated as a separate adsorbate class; adjacent uncoupled
CO belongs to the CO distribution space.

`COCO` is accepted only as a backward-compatible input alias and is canonicalized
to the `OCCO` label.

### COOH

- Cu-bound, uniform
- Cu-bound, interface-biased
- Au-top
- Au-Cu asymmetric interface motif

### CO / CHO / COH and other single-center intermediates

- uniform
- interface-biased
- interior-biased

The list can be extended with additional species-specific rules without changing
the natural-language API.

## Geometry conventions

These are **initial structures for relaxation**, not fixed final bond lengths.

- CO: C-down top binding; C-O about 1.15 Å.
- CHO: true formyl, surface-C(H)=O; H is bonded to C.
- COH: surface-C-O-H; H is bonded to O.
- COOH:
  - Cu-rich motif: C anchored to one Cu and carbonyl O directed toward a
    neighboring Cu / bridge environment.
  - Au-Cu motif: C associated with Au and carbonyl O directed toward Cu.
  - Au-top motif is also available.
- OCCO: C-C-coupled C2O2 with initial C-C about 1.45 Å and C-O about 1.25 Å.
  A Cu-Au cross-interface case explicitly places one carbon on the Cu side and
  the other on the Au side. Higher total coverage is represented as OCCO + CO.
- H and OH: use threefold fcc hollow candidates when available.

## Boundary and spacing safeguards

The scenario generator:

1. keeps lateral multi-atom motifs such as OCCO and COOH away from the periodic
   x/y cell edge so the molecule is not visually/structurally split across PBC;
2. places cross-interface motifs at the **internal** Cu/Au interface, not the
   periodic edge;
3. uses maximin selection as the primary criterion so interface- or
   interior-biased high-coverage structures do not artificially cluster;
4. allows upright PBC-safe single-site CO/H/OH to use edge-equivalent sites,
   which is necessary for physically meaningful dense coverages such as
   0.5-0.75 ML CO;
5. rejects severe inter-adsorbate overlaps;
6. records requested/actual coverage, selected sites, distribution, and boundary
   settings in `structure_info.json`.

Example:

```text
boundary margin 0.15
```

applies the internal-cell safeguard to lateral multi-atom adsorbate motifs.

## Python API

```python
from tools.adsorbate_scenarios import generate_adsorbate_scenarios

structures = generate_adsorbate_scenarios(
    interface_structure,
    adsorbate="OCCO",
    coverage=0.056,
    coverage_basis="Cu",
    all_scenarios=True,
    boundary_margin=0.12,
)
```

The legacy single-adsorbate generator is unchanged unless coverage, coverages,
distribution, or `all_scenarios` is requested. Scenario folders include both
coverage and distribution, for example `CO/cov_0p250_interface_biased/structure`.
