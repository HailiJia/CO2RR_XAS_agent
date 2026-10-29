# CO2RR_1029 scientific data upload

The manifest maps every selected current source path to exact bytes: either a new lossless archive here or an identical member of the already-uploaded differential_v1 source archives pinned at db1efb575773f341e3fa726aa0d312848fb3019c. Reused bytes are not absent data. New archives were verified member by member after compression.

The selection preserves current original ISAAC records, CONTCAR/POSCAR, settings, absorber tables, every numeric raw per-site electronic spectrum, canonical convolved spectra and final convolution logs. Current reference/unfinished-case sources are retained when relevant and under the stated selection limit; unfinished geometries are not completed ML examples. Original Cu100 ASE snapshots remain in the pinned Cu100_original_structures archive, also linked/hash-checked in the geometry audit.

Large wavefunctions, charge densities, licensed potentials, executables, hidden files and redundant verbose raw-run/BAV/relaxation history are excluded from this ML refresh upload. Source author files remain untouched locally. Full original source roots are not overwritten by conversions or reconvolution. The selected-file counts/bytes and exact reused/new storage mapping are in inventory_summary.json and manifest.json. This is not an upload of all37GB of simulation scratch.
