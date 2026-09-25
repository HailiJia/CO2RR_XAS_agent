#!/bin/bash
#SBATCH -J 2CO_adjacent_0p167_uniform
#SBATCH -o 2CO_adjacent_0p167_uniform.out
#SBATCH -e 2CO_adjacent_0p167_uniform.err
#SBATCH -q regular
#SBATCH -A m5268
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH --mail-user=beaver.jhl@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load vasp/6.6.1-gpu
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ ! -s POTCAR ]; then bash make_potcar.sh; fi
srun -n 4 -c 32 --cpu-bind=cores --gpu-bind=none vasp_std > vasp.out
