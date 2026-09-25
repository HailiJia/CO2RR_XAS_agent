#!/bin/bash
set -euo pipefail

POTCAR_DIR="${CO2RR_POTCAR_DIR:-/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE/potpaw_PBE}"

[ -d "$POTCAR_DIR" ] || { echo "ERROR: POTCAR directory not found: $POTCAR_DIR" >&2; exit 1; }
[ -f POSCAR ] || { echo "ERROR: POSCAR not found" >&2; exit 1; }

# VASP 5/6 POSCAR element symbols are on line 6.
read -ra elements <<< "$(sed -n '6p' POSCAR)"

rm -f POTCAR

for el in "${elements[@]}"; do
  case "$el" in
    Cu) pot="Cu" ;;
    Au) pot="Au" ;;
    O)  pot="O" ;;
    H)  pot="H" ;;
    *) echo "ERROR: No POTCAR mapping defined for element: $el" >&2; exit 1 ;;
  esac

  src="$POTCAR_DIR/$pot/POTCAR"
  [ -f "$src" ] || { echo "ERROR: Missing POTCAR: $src" >&2; exit 1; }
  cat "$src" >> POTCAR
done

echo "Wrote POTCAR for: ${elements[*]}"
