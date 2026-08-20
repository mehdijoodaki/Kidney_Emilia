#!/bin/bash
#SBATCH -c 50
#SBATCH --mem=400G
#SBATCH --output=logs/output.%J.%x.txt
#SBATCH --error=logs/error.%J.%x.txt
#SBATCH --job-name=integration
#SBATCH --mail-type=END
#SBATCH --mail-user=judakimehdi@gmail.com
#SBATCH --time=02-10:00:00


source ~/miniconda3/bin/activate

conda activate PILOT-GM-VAE 
################################################################

echo date

python combination_code.py

echo date
