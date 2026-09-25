# Adsorbate coverage and distribution generation

The structure generator supports dataset-oriented adsorbate placement in addition
to the original single-adsorbate mode.

## Natural-language examples

```text
Generate CO on Cu Au (111) at 0.25 ML, uniform, coverage based on Cu.

Generate OH on Cu Au (111) at 0.111 ML, interface-biased, coverage based on Cu.

Generate CO at 0.25 ML, interior-biased, coverage based on Cu.

Generate OCCO across the interface at 0.056 ML on Cu Au (111).

Generate all reasonable scenarios for OCCO on Cu Au (111) at 0.056 ML.

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

For OCCO, coverage is reported as **CO-equivalent carbon coverage**: one OCCO
contains two carbon centers and therefore contributes 2/36 = 0.056 ML on the
36-Cu basis.

## "All reasonable scenarios"

This is species-specific rather than a blind Cartesian product.

For an internal Cu/Au interface:

### OCCO

- Cu-Cu, uniformly placed
- Cu-Cu, interface-biased
- Cu-Au cross-interface: one carbon associated with Cu and the other with Au

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
- H and OH: use threefold fcc hollow candidates when available.

## Boundary and spacing safeguards

The scenario generator:

1. rejects adsorption sites within a configurable fractional x/y boundary margin;
2. places interface motifs at the internal interface rather than the periodic
   edge when eligible internal sites exist;
3. uses maximin selection to avoid artificial clustering;
4. rejects severe inter-adsorbate overlaps;
5. records actual coverage, selected sites, distribution, and boundary margin in
   `structure_info.json`.

Example:

```text
boundary margin 0.15
```

requires all generated adsorbate atoms to remain at least 0.15 fractional
coordinate from x/y cell boundaries.

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

The legacy single-adsorbate generator is unchanged unless coverage, distribution,
or `all_scenarios` is requested.
