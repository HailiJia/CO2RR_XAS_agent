Final realistic CHO-labelled / surface-C-O-H dataset

12 STRUCTURES
-------------
1 x 0.028 ML = 1/36
5 x 0.056 ML = 2/36
5 x 0.111 ML = 4/36
1 x 0.167 ML = 6/36

COVERAGE
--------
The denominator is exactly the 36 ACTUAL top-layer Cu atoms.

For Au-Cu mixed-hollow variants, one Cu3 adsorbate is replaced by one mixed-site
adsorbate. No extra adsorbate is added, so coverage remains exactly 2/36 or 4/36.

0.056 and 0.111 ML each sample five environments:
- uniform Cu3
- Cu-interior Cu3
- Cu3 nearest the Cu/Au interface
- one Au-Cu-Cu fcc hollow + remaining adsorbates on Cu3
- one Au-Au-Cu fcc hollow + remaining adsorbates on Cu3

The 0.167 ML case is maximum-dispersion Cu3 only.

SPATIAL REALISM
---------------
The uniform selections are exact maximum-separation solutions over the available
pure Cu3 fcc hollows. Interface/interior variants are required to remain spatially
separated; they change local chemistry without intentionally creating clusters.

Minimum C-center design constraints:
- 0.056 ML variants: >= 7.0 A
- 0.111 ML variants: >= 5.0 A
- 0.167 ML: exact maximum-dispersion solution (about 4.47 A minimum C-C)

MOTIF
-----
Generated connectivity:
  surface-C-O-H

This is conventionally *COH; the CHO label is retained for compatibility with the
existing CO2RR agent dataset.

Initial geometry:
- Cu3 fcc:      C ~1.20 A above local relaxed plane
- Au-Cu-Cu:     C ~1.26 A above local relaxed plane
- Au-Au-Cu:     C ~1.32 A above local relaxed plane
- C-O = 1.33 A
- O-H = 0.98 A
- C-O tilt = 18 degrees
- C-O-H = 110 degrees

Every folder contains POSCAR, INCAR, KPOINTS, make_potcar.sh, submit_relax.sh,
and sites.csv. Bottom three metal layers are fixed; top two and all adsorbate
atoms are relaxed.
