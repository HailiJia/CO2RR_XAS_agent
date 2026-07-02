#!/bin/bash
# Reusable ISAAC simulation-XAS finalizer for completed FEFF/FDMNES/VASP runs.
#
# This writes a real ISAAC 1.05 simulation evidence record:
#   record_type= evidence
#   record_domain= simulation
#   source_type= computation
#
# Run inside a completed XAS folder:
#   bash /path/to/CO2RR_XAS_agent/XAS/convert_completed_simulation_xas_to_isaac.sh
#
# Or pass the completed folder explicitly:
#   bash XAS/convert_completed_simulation_xas_to_isaac.sh --input-dir /path/to/FEFF
#
# Optional portal validation/upload:
#   export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
#   export ISAAC_KEY="..."
#   bash XAS/convert_completed_simulation_xas_to_isaac.sh --validate
#   bash XAS/convert_completed_simulation_xas_to_isaac.sh --upload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ISAAC_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

python "$REPO_ROOT/tools/convert_simulation_xas_run_to_isaac.py" "$@"
