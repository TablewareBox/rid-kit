#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=cpu
#SBATCH -t 34:0:0
#SBATCH --mem 6G
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 4

conf_dir="../../../../conf000"

for i in `seq 0 7`;do
cd 00$i;
gmx rms -s $conf_dir/npt.gro -f traj_comp.xtc -n $conf_dir/index.ndx << EOF
MainChain
Loop
EOF
cd ..
done