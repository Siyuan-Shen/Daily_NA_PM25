#!/bin/bash

# Define the range for the loop
start_radius=0
end_radius=110
radius_bin=20

radii=(1 5)
# Job script file
job_script="run_aws_gpu.slurm"
count=0
# Loop through the years
#for (( radius=$start_radius; radius<=$end_radius; radius+=$radius_bin )); do
for radius in "${radii[@]}"; do
    
    # Update beginyears_endyears and Estimation_years dynamically
    Buffer_size="[$radius]"

    # Create a temporary modified script
    modified_script="modified_job_script_${Buffer_size}.slurm"
    cp $job_script $modified_script

    # Use sed to replace variables in the script
    sed -i "s/^Buffer_size=.*/Buffer_size=${Buffer_size}/" $modified_script
    sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"V0.4.0 BLISCO ${radius}\"/" $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % .*/pause_time=\$((RANDOM % 50 + (${count}) * 120))/" $modified_script

    # Submit the modified script using sbatch
    echo "Submitting job for radius $radius..."
    sbatch < $modified_script

    # Optional: Clean up temporary script after submission
    rm $modified_script

    # pause for 3 second before the next submission
    sleep 3
    count=$((count + 1))

done
