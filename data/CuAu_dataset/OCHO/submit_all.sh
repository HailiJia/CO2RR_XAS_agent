#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
(cd "$ROOT/01_1OCHO_CuCu_interface" && sbatch submit_relax.sh)
(cd "$ROOT/02_1OCHO_AuCu_across" && sbatch submit_relax.sh)
(cd "$ROOT/03_1OCHO_CuCu_interior" && sbatch submit_relax.sh)
(cd "$ROOT/04_2OCHO_CuCu_interface_plus_interior" && sbatch submit_relax.sh)
(cd "$ROOT/05_2OCHO_CuCu_interface_plus_AuCu" && sbatch submit_relax.sh)
(cd "$ROOT/06_2OCHO_2CuCu_interface" && sbatch submit_relax.sh)
(cd "$ROOT/07_3OCHO_2CuCu_interface_plus_interior" && sbatch submit_relax.sh)
(cd "$ROOT/08_3OCHO_1CuCu_interface_plus_2interior" && sbatch submit_relax.sh)
