Realistic OH training set for Cu/Au stripe slab.

Coverage naming is referenced ONLY to 36 surface-Cu reference sites.

Nominal 0.125 ML:
Exact 0.125 ML is impossible with 36 Cu sites. Two uniform representatives are
provided:
  4/36 = 0.111 ML
  5/36 = 0.139 ML
The filename keeps 0125ML to identify the intended low-coverage regime, while
manifest.csv and each POSCAR header record the actual discrete coverage.

Main coverage range:
  nominal ~0.125 ML (4 or 5 Cu OH)
  0.25 ML (9 Cu OH)
  0.50 ML (18 Cu OH; upper edge of the main set)
0.75 and 1.00 ML were removed from the realistic production set.

Uniform distributions:
0.25 and 0.50 ML use exactly the validated uniform subsets from the corrected H
dataset. The ~0.125 ML 4- and 5-site uniform structures use maximin spacing on the
same validated 36-site Cu-reference lattice.

Au occupation:
Pure-Au OH is included only in selected interface_Au structures and is NOT counted
toward the Cu coverage:
  ~0.125 ML: +1 Au OH
  0.25 ML:   +1 Au OH
  0.50 ML:   +2 Au OH
Pure-Au OH starts at Au-Au bridge sites near the Cu/Au boundary.

Initial geometry:
  Cu3 fcc:                 O ~1.15 A above local relaxed surface
  Cu-side interface fcc:   O ~1.25 A above local relaxed surface
  Au-Au bridge extra:      O ~1.45 A above local bridge
  O-H:                     0.98 A, initially outward along +z
