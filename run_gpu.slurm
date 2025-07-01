#!/bin/bash
#SBATCH --job-name="Daily PM 2D sweep 4GPU"
#SBATCH --partition=general
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
# SBATCH --exclude=c2-gpu-[009]
#SBATCH --output=job_output/job-%j-output.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.siyuan@wustl.edu
#SBATCH --account=compute-rvmartin
#SBATCH -vvv


# If you're using containers with Slurm
# Uncomment this line if needed:
#SBATCH --container-image=syword/python3-pytorch:2025.06
#SBATCH --container-mounts=/storage1/fs1/rvmartin/Active/s.siyuan:/my-projects,/storage1/fs1/rvmartin2/Active/s.siyuan:/my-projects2
#SBATCH --container-workdir=/my-projects2/Projects/MLCNN_PM25_2021/code/Training_Evaluation_Estimation/PM25/V6.02.03-test

# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/v0.0.0

# Random pause

pause_time=$((RANDOM % 30 + 30))
echo "Pausing for $pause_time seconds..."
sleep $pause_time

# Buffer_size=[80]
# sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml

cd /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/v0.0.0
python3 main.py

