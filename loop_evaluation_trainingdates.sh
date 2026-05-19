
Evaluation_trained_begin_list=(
   "[20190101]"  "[20200101]" "[20210101]" "[20220101]" "[20230101]" # "[20220101]" "[20230101]" "[20230101]" "[20230101]" "[20230101]"
    # add more if needed, e.g.
    # "[20200101,20210101]"
)

Evaluation_trained_end_list=(
    "[20191231]"  "[20201231]" "[20211231]" "[20221231]" "[20231231]" # "[20221231]" "[20231231]" "[20231231]" "[20231231]" "[20231231]"
    # must match length and order of Evaluation_trained_begin_list
    # "[20201231,20211231]"
)

# ================================
# Script settings
# ================================
job_script="run_aws_gpu.slurm"
count=0

# Safety: check all arrays same length
if [[ ${#Evaluation_trained_begin_list[@]} -ne ${#Evaluation_trained_end_list[@]} ]]; then
    echo "Error: arrays for date configs are not the same length!"
    exit 1
fi

# ================================
# Main loop
# ================================
for i in "${!Evaluation_trained_begin_list[@]}"; do
    Evaluation_trained_begin_dates="${Evaluation_trained_begin_list[$i]}"
    Evaluation_trained_end_dates="${Evaluation_trained_end_list[$i]}"

    echo "======================================================="
    echo "Job index: $i"
    echo "  Evaluation_trained_begin_dates = $Evaluation_trained_begin_dates"
    echo "  Evaluation_trained_end_dates   = $Evaluation_trained_end_dates"
    echo "======================================================="

    # Create a temporary modified script
    modified_script="modified_job_script_${i}.slurm"
    cp "$job_script" "$modified_script"

    # Replace the four variables in the slurm script
    # Assumes lines like:
    #   Estimation_trained_begin_dates=[20220101,20230101]
    #   Estimation_trained_end_dates=[20221231,20231231]
    #   Estimation_begindates=[[20190101],...]
    #   Estimation_enddates=[[20191231],...]
    sed -i "s/^Training_begin_dates=.*/Training_begin_dates=${Evaluation_trained_begin_dates}/" "$modified_script"
    sed -i "s/^Training_end_dates=.*/Training_end_dates=${Evaluation_trained_end_dates}/" "$modified_script"

    # Optional: update job name to encode index or dates
    sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"DailyEvaluation_${i}\"/" "$modified_script"

    # Optional: update pause_time like your example (if that line exists)
    # Original pattern: pause_time=$((RANDOM % ...))
    sed -i "s/^pause_time=\$((RANDOM % .*/pause_time=\$((RANDOM % 50 + (${count}) * 120))/" "$modified_script"

    # Submit the modified script using sbatch
    echo "Submitting job index $i ..."
    sbatch < "$modified_script"

    # Clean up temporary script after submission
    #rm "$modified_script"

    # pause 3 seconds before the next submission
    sleep 3
    count=$((count + 1))
done
