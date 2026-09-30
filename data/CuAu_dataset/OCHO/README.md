# OCHO Cu/Au 8-case dataset

All OCHO species are connected bidentate formate-like intermediates.

Geometry used for initial structures:
- C-O ~1.27 A
- C-H ~1.10 A
- O-O ~2.25 A
- Cu-O ~1.9 A initial distance
- Au-O is started slightly longer (~2.05 A) in the mixed Au-Cu bridge.

Site definitions:
- CuCu_interface: neighboring Cu atoms along the Cu side of the Cu/Au boundary.
- AuCu_across: one Cu and one Au directly across the boundary.
- CuCu_interior: neighboring Cu atoms near the middle of the Cu stripe.

Each case contains POSCAR, INCAR, KPOINTS, make_potcar.sh, submit_relax.sh.
All eight cases use the CPU submit template because they contain only 1-3 OCHO adsorbates.
These are starting geometries and should be relaxed.
