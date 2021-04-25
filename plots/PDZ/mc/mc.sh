#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 72:0:0
#SBATCH --mem 32G 
#SBATCH --ntasks-per-node 4
#SBATCH --gres=gpu:1

module load cudatoolkit/9.1
module load cudnn/cuda-9.1/7.1.2

source /home/dongdong/software/GMX20192plumed/bin/GMXRC.bash
source /home/linfengz/SCR/softwares/tf_venv_r1.8/bin/activate

python mc1d_2d_pdz3.py -m graph.pb
if test $? -eq 0; then
    touch tag_finished
fi
sleep 1
