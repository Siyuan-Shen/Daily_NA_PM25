
TBO_buffers=(0 1 3 5 7 15)
job_script="run_aws_gpu.slurm"
for TBO_buffer in ${TBO_buffers[@]}
    TBO_running_buffer="[$TBO_buffer]"
    modified_script="modified_job_script_TBO_${TBO_running_buffer}.slurm"
    cp $job_script $modified_script

    sed -i "s/^Temporal_Buffer_size=.*/Temporal_Buffer_size=${TBO_running_buffer}/" $modified_script
    sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"V0.4.0 TBO ${TBO_buffer}\"/" $modified_script

    sed -i "s/^pause_time=\$((RANDOM % .*/pause_time=\$((RANDOM % 50 + (${TBO_buffer}) * 10))/" $modified_script

    echo "Submitting job for TBO buffer $TBO_buffer..."
    sbatch < $modified_script
    
done