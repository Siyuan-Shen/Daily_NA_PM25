#!/bin/bash

# ================================
# Config: edit these lists as you like
# Each index i corresponds to one job config
# ================================

Est_trained_begin_list=(
   "[20190101]"  "[20200101]" "[20210101]" "[20220101]" "[20230101]" # "[20220101]" "[20230101]" "[20230101]" "[20230101]" "[20230101]"
    # add more if needed, e.g.
    # "[20200101,20210101]"
)

Est_trained_end_list=(
    "[20191231]"  "[20201231]" "[20211231]" "[20221231]" "[20231231]" # "[20221231]" "[20231231]" "[20231231]" "[20231231]" "[20231231]"
    # must match length and order of Est_trained_begin_list
    # "[20201231,20211231]"
)

Est_begindates_list=(
    "[[20190914]]"  "[[20200913]]" "[[20210914]]" "[[20220912]]" "[[20230918]]" #"[[20221001]]" "[[20230101]]" "[[20230401]]" "[[20230701]]" "[[20231001]]"
    # e.g. another set:
    # "[[20180101],[20190101],[20200101],[20210101]]"
)

Est_enddates_list=(
    "[[20191231]]"  "[[20201231]]" "[[20211231]]" "[[20221231]]" "[[20231231]]" #"[[20221031]]" "[[20230131]]" "[[20230430]]" "[[20230731]]" "[[20231031]]"
    # e.g. another set:
    # "[[20181231],[20191231],[20201231],[20211231]]"
)

# ================================
# Script settings
# ================================
job_script="run_aws_gpu.slurm"
count=0

# Safety: check all arrays same length
if [[ ${#Est_trained_begin_list[@]} -ne ${#Est_trained_end_list[@]} ]] || \
   [[ ${#Est_trained_begin_list[@]} -ne ${#Est_begindates_list[@]} ]] || \
   [[ ${#Est_trained_begin_list[@]} -ne ${#Est_enddates_list[@]} ]]; then
    echo "Error: arrays for date configs are not the same length!"
    exit 1
fi

# ================================
# Main loop
# ================================
for i in "${!Est_trained_begin_list[@]}"; do
    Estimation_trained_begin_dates="${Est_trained_begin_list[$i]}"
    Estimation_trained_end_dates="${Est_trained_end_list[$i]}"
    Estimation_begindates="${Est_begindates_list[$i]}"
    Estimation_enddates="${Est_enddates_list[$i]}"

    echo "======================================================="
    echo "Job index: $i"
    echo "  Estimation_trained_begin_dates = $Estimation_trained_begin_dates"
    echo "  Estimation_trained_end_dates   = $Estimation_trained_end_dates"
    echo "  Estimation_begindates          = $Estimation_begindates"
    echo "  Estimation_enddates            = $Estimation_enddates"
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
    sed -i "s/^Estimation_trained_begin_dates=.*/Estimation_trained_begin_dates=${Estimation_trained_begin_dates}/" "$modified_script"
    sed -i "s/^Estimation_trained_end_dates=.*/Estimation_trained_end_dates=${Estimation_trained_end_dates}/" "$modified_script"
    sed -i "s/^Estimation_begindates=.*/Estimation_begindates=${Estimation_begindates}/" "$modified_script"
    sed -i "s/^Estimation_enddates=.*/Estimation_enddates=${Estimation_enddates}/" "$modified_script"

    # Optional: update job name to encode index or dates
    sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"DailyEst_${i}\"/" "$modified_script"

    # Optional: update pause_time like your example (if that line exists)
    # Original pattern: pause_time=$((RANDOM % ...))
    sed -i "s/^pause_time=\$((RANDOM % .*/pause_time=\$((RANDOM % 50 + (${count}) * 120))/" "$modified_script"

    # Submit the modified script using sbatch
    echo "Submitting job index $i ..."
    sbatch < "$modified_script"

    # Clean up temporary script after submission
    rm "$modified_script"

    # pause 3 seconds before the next submission
    sleep 3
    count=$((count + 1))
done
