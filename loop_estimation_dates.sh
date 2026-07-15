#!/bin/bash

# ================================
# Config: edit these lists as you like
# Each index i corresponds to one job config
# ================================

Est_trained_begin_list=(
   # "[20190101]"  
   # "[20190101]" 
   # "[20200101]"  
   # "[20200101]" 
   # "[20210101]"  
   #  "[20210101]" 
   "[20220101]"  
   # "[20220101]" 
   "[20230101]"  
   # "[20230101]" 
    
)

Est_trained_end_list=(
    # "[20191231]"  
    # "[20191231]" 
    # "[20201231]"  
    # "[20201231]"
    # "[20211231]" 
    # "[20211231]"
    "[20221231]"
    # "[20221231]"
    "[20231231]"
    # "[20231231]"  
)

Est_begindates_list=(
       # "[[20190101]]"  
       #  "[[20190701]]"  
       # "[[20200101]]"  
       #  "[[20200701]]"  
       # "[[20210101]]"  
       #  "[[20210701]]"  
       "[[20220101]]"  
       # "[[20220701]]"
       "[[20230101]]"
       # "[[20230701]]"
)

Est_enddates_list=(
    # "[[20190630]]"  
    # "[[20191231]]"
    # "[[20200630]]"
    # "[[20201231]]"
    # "[[20210630]]"
    # "[[20211231]]"
    "[[20220630]]"
    # "[[20221231]]"
    "[[20230630]]"
    # "[[20231231]]"
)

# ================================
# Script settings
# ================================
job_script="run_gpu.bsub"
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
    modified_script="modified_job_script_${i}.bsub"
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
    sed -i "s/^#BSUB -J=.*/#BSUB -J \"DailyEst_${i}\"/" "$modified_script"

    # Optional: update pause_time like your example (if that line exists)
    # Original pattern: pause_time=$((RANDOM % ...))
    sed -i "s/^pause_time=\$((RANDOM % .*/pause_time=\$((RANDOM % 50 + (${count}) * 120))/" "$modified_script"

    # Submit the modified script using bsub
    echo "Submitting job index $i ..."
    bsub < "$modified_script"

    # Clean up temporary script after submission
    rm "$modified_script"

    # pause 3 seconds before the next submission
    sleep 3
    count=$((count + 1))
done
