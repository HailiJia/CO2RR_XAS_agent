#!/bin/bash
#SBATCH -J COOH_0p056_AuCu_interface
#SBATCH -o COOH_0p056_AuCu_interface.out
#SBATCH -e COOH_0p056_AuCu_interface.err
#SBATCH -q regular
#SBATCH -A m5268
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH --mail-user=beaver.jhl@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load vasp/6.6.1-cpu
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ ! -s POTCAR ]; then bash make_potcar.sh; fi
srun --cpu-bind=cores vasp_gam > vasp.out
