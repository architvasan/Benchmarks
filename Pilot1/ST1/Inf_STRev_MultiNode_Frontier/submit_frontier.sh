#!/bin/bash
#SBATCH -A chm155_001
#SBATCH -J ST_Scale
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 10

### Change to working path
cd /lustre/orion/chm155/scratch/avasan/ST_Code/Benchmarks/Pilot1/ST1/Inf_STRev_MultiNode_Frontier

module load cray-python
module load rocm/5.4.0
export TF_FORCE_GPU_ALLOW_GROWTH=true

### change to current environment path
source /autofs/nccs-svm1_proj/chm155/avasan/envs/st_env/bin/activate
srun -A chm155_001 -t 01:00:00 -p batch -N 10 -n 80 --ntasks-per-node=8 python smiles_regress_transformer_run_large.py
