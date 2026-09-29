#!/bin/bash
#SBATCH -J CO2_0p028_linear_Cu_top
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -q regular
#SBATCH -A m5268
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 24:00:00
#SBATCH --mail-user=beaver.jhl@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "$(dirname "$0")"
module load vasp/6.6.1-cpu
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ ! -s POTCAR ]; then bash make_potcar.sh; fi
srun -n 128 -c 2 --cpu-bind=cores vasp_std > vasp.out
