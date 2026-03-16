#!/bin/bash
#SBATCH -c 80
#SBATCH --mem=500G
#SBATCH --output=logs/output.%J.%x.txt
#SBATCH --error=logs/error.%J.%x.txt
#SBATCH --job-name=human_mice
#SBATCH --mail-type=END
#SBATCH --mail-user=judakimehdi@gmail.com
#SBATCH --time=01-10:00:00


source ~/miniconda3/bin/activate

conda activate PILOT-GM-VAE 
################################################################

echo date
python combination_code.py
echo date
