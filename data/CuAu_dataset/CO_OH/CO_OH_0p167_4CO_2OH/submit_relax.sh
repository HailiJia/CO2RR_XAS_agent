#!/bin/bash
#SBATCH -J CO_OH_0p167_4CO_2OH_distributed
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -q regular
#SBATCH -A m5268
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH --mail-user=beaver.jhl@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "$(dirname "$0")"
if module load vasp/6.6.1-gpu 2>/dev/null; then
  :
else
  module load vasp/6.6.0-gpu
fi
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ ! -s POTCAR ]; then bash make_potcar.sh; fi
srun -n 4 -c 32 -G 4 --cpu-bind=cores --gpu-bind=none vasp_std > vasp.out
