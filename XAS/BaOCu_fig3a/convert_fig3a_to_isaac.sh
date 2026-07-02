#!/bin/bash
# Convert Xu et al. Nature Catalysis 2022 Fig. 3a BaO/Cu operando XANES data
# to an ISAAC XAS record.
#
# Run from the CO2RR_XAS_agent repository root:
#   bash XAS/BaOCu_fig3a/convert_fig3a_to_isaac.sh
#
# Optional local source file:
#   FIG3A_DAT=/path/to/fig3a.dat bash XAS/BaOCu_fig3a/convert_fig3a_to_isaac.sh
#
# Optional ISAAC validation/upload:
#   export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
#   export ISAAC_KEY="..."
#   bash XAS/BaOCu_fig3a/convert_fig3a_to_isaac.sh --validate
#   bash XAS/BaOCu_fig3a/convert_fig3a_to_isaac.sh --upload

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${FIG3A_DAT:-https://raw.githubusercontent.com/onealshu/CO2_alcohol_BaOCu/main/exp_Data/fig3a.dat}"
OUTPUT="${OUTPUT_JSON:-$ROOT_DIR/XAS/BaOCu_fig3a/baocu_fig3a_isaac_xas_record.json}"

python "$ROOT_DIR/tools/convert_fig3a_baocu_to_isaac.py" \
  --input "$SOURCE" \
  --output "$OUTPUT" \
  "$@"
