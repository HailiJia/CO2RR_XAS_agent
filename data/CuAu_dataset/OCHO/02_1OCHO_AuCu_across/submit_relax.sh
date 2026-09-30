#!/bin/bash
#SBATCH -J 02_1OCHO_AuCu_across
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -q regular
#SBATCH -A m5268
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 30:00
#SBATCH --mail-user=beaver.jhl@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
cd "$(dirname "$0")"
module load vasp/6.6.1-gpu
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ ! -s POTCAR ]; then bash make_potcar.sh; fi

srun -n 8 -c 32 -G 8 --cpu-bind=cores --gpu-bind=none vasp_gam > vasp.out
