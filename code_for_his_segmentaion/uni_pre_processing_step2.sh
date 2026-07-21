#!/bin/bash
#!/bin/bash
#SBATCH --partition=gpu_node
#SBATCH --gpus-per-node=1
#SBATCH --mem=300G
#SBATCH --output=logs/output.%J.%x.txt
#SBATCH --error=logs/error.%J.%x.txt
#SBATCH --job-name=adenin
#SBATCH --mail-type=END
#SBATCH --mail-user=judakimehdi@gmail.com
#SBATCH --time=9-00:00:00
source /beegfs/home/mjoodaki/miniconda3/bin/activate
############################spatialpilot#######################  
conda activate  scvi-env
################################################################
echo date
python uni_pre_processing_step2.py 
echo date
