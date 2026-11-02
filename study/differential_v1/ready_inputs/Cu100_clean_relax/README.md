# Missing clean Cu100 reference

No completed bare Cu100 geometry/XAS record was found in the current CO2RR export, original Cu100 ASE snapshot tree, CO2RR_XAS_agent/data export or Downloads/vasp_inputs. Existing Cu100 adsorbate configurations are not clean references. This directory is a proposed clean relaxation starting point, not a frozen-metal spectrum and not a completed relaxed clean reference. No jobs were submitted.

After a verified successful relaxation, call `tools.prepare_clean_Cu100.prepare_xas(relax_dir, out_dir)`. It refuses to generate complete XAS decks without a successful relaxation record and the expected clean Cu16 cell. Do not replace this check with the starting POSCAR. Review PAW/mesh convergence against the original ASE calculation metadata, which lacks those detailed parameters.
