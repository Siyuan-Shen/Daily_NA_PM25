#!/bin/bash
#SBATCH --job-name="Daily HalfYearModel Test AVD_d20250804 True True 5x5 sweep test 8GPU"
#SBATCH --partition=cpuqueue 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --output=job_output/job-%j-output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.siyuan@wustl.edu
#SBATCH --time=14-00:00:00
#SBATCH -vvv


# If you're using containers with Slurm
# Uncomment this line if needed:

VERSION=v0.4.0

# Redirect W&B and tmp paths inside the container
export WANDB_DIR=/s.siyuan/my-projects2/tmp/wandb_dir
export TMPDIR=/s.siyuan/my-projects2/tmp/tmpdir
export XDG_CONFIG_HOME=/s.siyuan/my-projects2/tmp/config_home
export WANDB_CACHE_DIR=/s.siyuan/my-projects2/tmp/wandb_cache
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
mkdir -p $WANDB_DIR $TMPDIR $XDG_CONFIG_HOME $WANDB_CACHE_DIR

# Create a frozen snapshot of current code
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_BASE=/s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/snapshots
SNAPSHOT_DIR=$SNAPSHOT_BASE/$VERSION'_'$TIMESTAMP/

echo "Creating snapshot at: $SNAPSHOT_DIR"
mkdir -p $SNAPSHOT_DIR
rsync -av --exclude=Results --exclude=job_output --exclude=Figures --exclude=wandb  /s.siyuan/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/$VERSION/ $SNAPSHOT_DIR 

# Run from the snapshot
cd $SNAPSHOT_DIR
echo "Running from snapshot: $SNAPSHOT_DIR"


# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd $SNAPSHOT_DIR

# Random pause

# pause_time=$((RANDOM % 3 + 1))
# echo "Pausing for $pause_time seconds..."
# sleep $pause_time

# sed -i '/section_start/,/section_end/{/key/s/:.*/: value/}'

Buffer_size=[80]
sed -i "/'BLISCO-CrossValidation'/,/'Temporal-CrossValidation'/{/'BLISCO_CV_buffer_radius_km'/s/:.*/: ${Buffer_size},/}" config.py

nvidia-smi

MM=/s.siyuan/shared/tools/bin/micromamba
ENV=/s.siyuan/shared/envs/myenv

# load micromamba’s shell function
eval "$($MM shell hook -s bash)"

micromamba activate "$ENV"
python3 main.py

