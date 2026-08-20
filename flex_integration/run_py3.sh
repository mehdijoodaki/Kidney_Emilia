#!/bin/bash
#SBATCH -c 50
#SBATCH --mem=500G
#SBATCH --output=logs/output.%J.%x.txt
#SBATCH --error=logs/error.%J.%x.txt
#SBATCH --job-name=cellhint_env
#SBATCH --mail-type=END
#SBATCH --mail-user=judakimehdi@gmail.com
#SBATCH --time=01-10:00:00


source ~/miniconda3/bin/activate

conda activate cellhint_env 
################################################################

echo date
python cellhint_env_hvg.py
echo date
