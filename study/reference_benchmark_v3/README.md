# Numeric reference benchmark v3

Cu2O PBE and GW-PAW inputs/OUTCAR/HDF5 responses were recovered from the current supplied folder, with independent diagonal-tensor checks. CuO PBE is retained as a missing numeric slot: the current HDF5 is unreadable and OUTCAR lacks a complete response table. Original sources, including incomplete files, are uploaded in the differential source package.

All curves use fixed-window minmax normalization on the same 8980–9015 eV grid for display and cosine/Pearson/Spearman scores. Axes preserve 8975–9015 eV, including the unextrapolated pre-window margin. Main early-feature diagnostic is separate from Cu-calibrated transferable offsets. Cu2O anchors its real first peak at 8981.2919 eV; CuO uses its first rising-edge inflection at 8983.2926 eV rather than a fictional peak at 8982 or the later white line. A dotted Cu2O FEFF curve explicitly retains Cu calibration because the early peak is unresolved.

VASP retains native CH_SIGMA and adds the author-notebook Gaussian sigma1 eV (FWHM2.35482 eV), not a fitted lifetime width. Other already convolved backends are not convolved twice. Native/broadened and aligned arrays, normalization offsets/divisors, shifts and both comparison types are saved. No unsupported pre-edge subtraction or clipping of negative absorption is used. Raw experimental baseline/noise values remain unchanged.

The current Cu2O inputs use PBE+U (Ueff6 eV on Cu) and independent-particle core-hole response. Cu_GW/O_GW and IALGO labels do not establish quasiparticle GW or BSE. Method titles use GW-PAW with these settings disclosed.

CuO FDMNES-fdm CPU time129600 s is retained as supplied; an older archived log value91018.9 s remains separately noted, without a claimed matched run. FEFF module entries are CPU times. VASP SCF and XAS stages remain separate from allocated core-time estimates. No measured speedup is claimed.
