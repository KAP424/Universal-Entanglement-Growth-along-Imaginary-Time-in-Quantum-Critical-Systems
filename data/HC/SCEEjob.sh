#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=fat6348

#SBATCH --output=out.log
#SBATCH --error=err.log

current_path=$1
t=$2
U=$3
size=$4
Theta=$5
Delta_t=$6
BS=$7
Sweeps=$8
N=$9
lambda=${10}
Pt=${11}

export JULIA_NUM_PROCS=$SLURM_CPUS_PER_TASK

julia /home/zxli_1/KAP/tUdata/HC/SCEErun.jl $t $U HoneyComb120 $size $Delta_t $Theta $BS $Sweeps $N $lambda $current_path $Pt

