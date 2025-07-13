#!/bin/bash
#SBATCH --job-name="Daily PM 3D sweep 1GPU testR2 examination"
#SBATCH --partition=general-gpu
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
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
#SBATCH --container-workdir=/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25


# Redirect W&B and tmp paths inside the container
export WANDB_DIR=/my-projects2/tmp/wandb_dir
export TMPDIR=/my-projects2/tmp/tmpdir
export XDG_CONFIG_HOME=/my-projects2/tmp/config_home
export WANDB_CACHE_DIR=/my-projects2/tmp/wandb_cache
mkdir -p $WANDB_DIR $TMPDIR $XDG_CONFIG_HOME $WANDB_CACHE_DIR

# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/v0.1.0

# Random pause

pause_time=$((RANDOM % 3 + 1))
echo "Pausing for $pause_time seconds..."
sleep $pause_time

# Buffer_size=[80]
# sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml

cd /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/v0.1.0
python3 main.py

