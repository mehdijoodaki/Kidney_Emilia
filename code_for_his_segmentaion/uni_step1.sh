#!/bin/bash
#SBATCH --partition=gpu_node
#SBATCH --gpus-per-node=1
#SBATCH --mem=400G
#SBATCH --output=logs/output.%J.%x.txt
#SBATCH --error=logs/error.%J.%x.txt
#SBATCH --job-name=adenin_uni
#SBATCH --mail-type=END
#SBATCH --mail-user=judakimehdi@gmail.com
#SBATCH --time=2-10:00:00
source /beegfs/home/mjoodaki/miniconda3/bin/activate

conda activate UNI

python uni_step1.py 

