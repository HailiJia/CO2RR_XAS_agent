OH VASP relaxation package

Each subfolder contains:
  POSCAR
  INCAR
  KPOINTS
  make_potcar.sh
  submit_relax.sh

Folder names use Cu-only nominal coverage:
  OH_0p125_...
  OH_0p25_...
  OH_0p50_...

Important:
- 0.125 ML is a nominal low-coverage label. The low4Cu and high5Cu uniform
  structures correspond to 4/36 = 0.111 ML and 5/36 = 0.139 ML Cu coverage.
- Au OH in interface_Au cases is extra and is not counted in the Cu coverage label.
- POTCAR order follows POSCAR element order Cu Au O H.
- make_potcar.sh therefore maps Cu->Cu, Au->Au, O->O, H->H.
