#!/bin/bash
#SBATCH --job-name="Daily 3DCNN 19-23 Estimation Training 4GPU"
#SBATCH --partition=general-gpu
#SBATCH --mem=800G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
# SBATCH --exclude=c2-gpu-[004,005]
#SBATCH --output=job_output/job-%j-output.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.siyuan@wustl.edu
#SBATCH -vvv


# If you're using containers with Slurm
# Uncomment this line if needed:
#SBATCH --container-image=syword/python3-pytorch:2025.06
#SBATCH --container-mounts=/storage1/fs1/rvmartin/Active/s.siyuan:/my-projects,/storage1/fs1/rvmartin2/Active/s.siyuan:/my-projects2
#SBATCH --container-workdir=/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25

VERSION=v0.2.1

# Redirect W&B and tmp paths inside the container
export WANDB_DIR=/my-projects2/tmp/wandb_dir
export TMPDIR=/my-projects2/tmp/tmpdir
export XDG_CONFIG_HOME=/my-projects2/tmp/config_home
export WANDB_CACHE_DIR=/my-projects2/tmp/wandb_cache
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
mkdir -p $WANDB_DIR $TMPDIR $XDG_CONFIG_HOME $WANDB_CACHE_DIR

# Create a frozen snapshot of current code
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_BASE=/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/snapshots
SNAPSHOT_DIR=$SNAPSHOT_BASE/$VERSION'_'$TIMESTAMP/

echo "Creating snapshot at: $SNAPSHOT_DIR"
mkdir -p $SNAPSHOT_DIR
rsync -av --exclude=Results --exclude=job_output --exclude=Figures --exclude=wandb  /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/$VERSION/ $SNAPSHOT_DIR 

# Run from the snapshot
cd $SNAPSHOT_DIR
echo "Running from snapshot: $SNAPSHOT_DIR"


# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd $SNAPSHOT_DIR

# Random pause

pause_time=$((RANDOM % 3 + 1))
echo "Pausing for $pause_time seconds..."
sleep $pause_time

# Buffer_size=[80]
# sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml
nvidia-smi
python3 main.py

