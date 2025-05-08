#!/bin/bash
#SBATCH --partition=rss-gpu       # Your GPU partition
#SBATCH -N 1                      # Number of nodes
#SBATCH -c 16                     # Number of CPU cores per task (adjust based on dataloader/analysis needs)
#SBATCH --mem=60G                # Memory per node (adjust based on model size/data)
#SBATCH --gres=gpu:A100:1         # Request 1 A100 GPU (adjust type/count if needed)
#SBATCH --export=all
#SBATCH --out=OSPA_Pipeline-%j.out
#SBATCH --job-name=OSPA_Pipeline  # Job name reflecting the pipeline
#SBATCH --output=OSPA_Pipeline_%j.txt # Combined output log
#SBATCH --error=OSPA_Pipeline_%j.txt  # Combined error log
#SBATCH --time=12:00:00           # Max runtime (adjust as needed)
#SBATCH --mail-user=khurram.khalil@missouri.edu # Your email
#SBATCH --mail-type=ALL      # Notify on failure or completion

# --- Environment Setup ---
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.
set -o pipefail # Pipelines fail if any command fails, not just the last one.

# Load required modules (adjust path/version if needed)
module load miniconda3/4.10.3_gcc_9.5.0
# Activate your conda environment
source activate deepseek # Replace 'deepseek' with your actual environment name
# Explicitly handle tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# --- Configuration ---
# Base directory for all outputs of this job run
BASE_OUTPUT_DIR="Experiments_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${BASE_OUTPUT_DIR}/logs"
# Main log file for the entire job
MAIN_LOG="${BASE_OUTPUT_DIR}/logs/pipeline_progress_${SLURM_JOB_ID}.log"

ANALYSIS_DIR="$BASE_OUTPUT_DIR/analysis" 
mkdir -p "$ANALYSIS_DIR"
# Scripts
TRAIN_SCRIPT="train_ospa.py"
ANALYZE_SCRIPT="analyze_head_diversity.py"

# Function to log messages
log_message() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $1" | tee -a "$MAIN_LOG"
}

# --- Initial Checks ---
log_message "==================================================="
log_message "        STARTING OSPA TRAINING & ANALYSIS PIPELINE     "
log_message "==================================================="
log_message "Job ID: $SLURM_JOB_ID"
log_message "Host: $(hostname)"
log_message "Output Directory: $BASE_OUTPUT_DIR"
log_message "Started at: $(date)"
log_message "Checking for Python scripts..."

scripts_ok=true
if [ ! -f "$TRAIN_SCRIPT" ]; then log_message "✗ ERROR: Training script '$TRAIN_SCRIPT' not found!"; scripts_ok=false; fi
if [ ! -f "$ANALYZE_SCRIPT" ]; then log_message "✗ ERROR: Analysis script '$ANALYZE_SCRIPT' not found!"; scripts_ok=false; fi
if [ "$scripts_ok" = false ]; then exit 1; fi
log_message "✓ Found required Python scripts."
log_message "==================================================="

# --- Experiment Configurations ---
# Define configurations for different sizes
# Format: size_name;d_model;nhead;nlayers;dim_feedforward;epochs;batch_size;max_seq_len_lm;max_seq_len_glue;lr_lm;lr_glue
# Adjust epochs, batch size, LR per size/task as needed
CONFIGS=(
    "small;256;4;4;1024;15;64;128;64;5e-4;3e-5"  # Smaller batch/seq for small model maybe? Adjusted epochs/LR
    "medium;512;8;6;2048;20;32;128;128;5e-4;2e-5" # Used 512/8/6 in previous results, increased epochs
    "large;768;12;12;3072;25;32;128;128;3e-4;1e-5" # Large model needs more epochs, maybe smaller LR
)

# Define tasks to run
TASKS=(
    "lm"
    # "glue"
)

# Define GLUE tasks (if task='glue')
GLUE_TASKS=(
    "sst2"
    "mrpc"
    "cola"
    "mnli"
    "qqp"
    # Add other GLUE tasks here e.g., "cola" "mnli" "qqp"
)

# Define Model Variants to run
VARIANTS=(
    # "vanilla;init;0.0"
    "ospa;init;0.0"
    "ospa;regularize;0.001" # Use the lambda that worked well
    "ospa;strict;0.0"
)

# --- Function Definitions ---

# Function to run a single training instance
run_training_instance() {
    local config_str=$1
    local task=$2
    local glue_task_name=$3 # Can be empty if task is 'lm'
    local variant_str=$4

    # Parse configurations
    IFS=';' read -r size_name d_model nhead nlayers dim_ff epochs batch_size max_len_lm max_len_glue lr_lm lr_glue <<< "$config_str"
    IFS=';' read -r transformer_type orth_mode orth_weight <<< "$variant_str"

    # Determine task-specific parameters
    local task_args=""
    local current_lr=""
    local current_max_seq_len=""
    local model_name_suffix=""
    local data_args=""

    if [ "$task" == "lm" ]; then
        task_args="--task lm --lm_dataset_name wikitext --lm_dataset_config wikitext-103-raw-v1"
        current_lr=$lr_lm
        current_max_seq_len=$max_len_lm
        model_name_suffix="lm"
        data_args="--max_seq_len $current_max_seq_len" # Pass max_seq_len for LM blocking
    elif [ "$task" == "glue" ]; then
        task_args="--task glue --glue_task $glue_task_name"
        current_lr=$lr_glue
        current_max_seq_len=$max_len_glue
        model_name_suffix="glue_${glue_task_name}"
        data_args="--max_seq_len $current_max_seq_len"
    else
        log_message "✗ ERROR: Unknown task '$task' in run_training_instance."
        return 1
    fi

    # Construct experiment name and output directory
    local orth_mode_label="${orth_mode}"
    if [ "$orth_mode" == "regularize" ]; then orth_mode_label="reg${orth_weight}"; fi
    local experiment_name="${size_name}_${transformer_type}_${orth_mode_label}_${model_name_suffix}"
    local output_dir="${BASE_OUTPUT_DIR}/${task}/${size_name}/${experiment_name}"
    local model_save_name="best_model.pt" # Consistent save name
    local model_save_path="${output_dir}/${model_save_name}"
    local exp_log="${BASE_OUTPUT_DIR}/logs/train_${experiment_name}.log"

    mkdir -p "$output_dir"
    echo "--- Training Run: $experiment_name --- $(date)" > "$exp_log"
    log_message "--- Starting Training: $experiment_name ---"
    log_message "    Config: Size=$size_name, Task=$task${glue_task_name:+/$glue_task_name}, Type=$transformer_type, OrthMode=$orth_mode, Lambda=$orth_weight"
    log_message "    Output to: $output_dir"

    # Define base command arguments (using variables defined above)
    local base_cmd="python \"$TRAIN_SCRIPT\" \
        $task_args \
        $data_args \
        --transformer_type $transformer_type \
        --tokenizer_name bert-base-uncased \
        --d_model $d_model \
        --nhead $nhead \
        --nlayers $nlayers \
        --dim_feedforward $dim_ff \
        --dropout 0.1 \
        --orth_mode $orth_mode \
        --orth_penalty_weight $orth_weight \
        --epochs $epochs \
        --batch_size $batch_size \
        --gradient_accumulation_steps 4 \
        --lr $current_lr \
        --weight_decay 0.01 \
        --clip 1.0 \
        --scheduler_update_every_step \
        --seed 42 \
        --num_workers 0 \
        --log_interval 100 \
        --output_dir \"$output_dir\" \
        --save \"$model_save_name\" \
        --device cuda" # Assume CUDA is available based on SLURM request

    log_message "Executing Training Command:"
    echo "$base_cmd" | tee -a "$MAIN_LOG" # Log command clearly

    # Execute command, redirect stdout/stderr to specific log file
    if eval $base_cmd >> "$exp_log" 2>&1; then
        if [ -f "$model_save_path" ]; then
            log_message "✓ Training $experiment_name completed. Model saved."
            return 0 # Success
        else
            log_message "✗ Training $experiment_name finished, BUT model file NOT FOUND at $model_save_path"
            return 1 # Failure
        fi
    else
        log_message "✗ Training $experiment_name FAILED (Command exited with error)."
        echo "--- Last 50 lines of $exp_log ---" >> "$MAIN_LOG"
        tail -n 50 "$exp_log" >> "$MAIN_LOG"
        echo "--- End of log tail ---" >> "$MAIN_LOG"
        return 1 # Failure
    fi
}

# Function to run analysis for a specific config (size/task)
run_analysis_for_config() {
    local config_str=$1
    local task=$2
    local glue_task_name=$3 # Optional

    IFS=';' read -r size_name d_model nhead nlayers dim_ff epochs batch_size max_len_lm max_len_glue lr_lm lr_glue <<< "$config_str"

    local task_label="$task"
    if [ "$task" == "glue" ]; then task_label="glue_${glue_task_name}"; fi

    local config_analysis_dir="${ANALYSIS_DIR}/${task_label}/${size_name}"
    mkdir -p "$config_analysis_dir"
    local config_analysis_log="${BASE_OUTPUT_DIR}/logs/analyze_${task_label}_${size_name}.log"
    echo "--- Analysis Run for ${task_label}_${size_name} --- $(date)" > "$config_analysis_log"

    log_message "--- Starting Analysis for Config: Size=$size_name, Task=$task_label ---"
    log_message "    Analysis Output to: $config_analysis_dir"

    local model_paths_for_config=()
    local analysis_success_count=0
    local analysis_fail_count=0

    # Find all model files for this specific config
    for variant_str in "${VARIANTS[@]}"; do
        IFS=';' read -r transformer_type orth_mode orth_weight <<< "$variant_str"
        local orth_mode_label="${orth_mode}"
        if [ "$orth_mode" == "regularize" ]; then orth_mode_label="reg${orth_weight}"; fi
        local experiment_name="${size_name}_${transformer_type}_${orth_mode_label}_${task_label}"
        local model_path="${BASE_OUTPUT_DIR}/${task}/${size_name}/${experiment_name}/best_model.pt"

        if [ -f "$model_path" ]; then
            model_paths_for_config+=("\"$model_path\"") # Quote for safety
            log_message "    Found model for analysis: $model_path"
        else
            log_message "    WARNING: Model not found for analysis: $model_path"
        fi
    done

    if [ ${#model_paths_for_config[@]} -lt 2 ]; then
        log_message "✗ Not enough models found (${#model_paths_for_config[@]}) for comparison in this config. Skipping comparison."
        return 1
    fi

    # Run comparison analysis for this specific config
    local compare_cmd="python \"$ANALYZE_SCRIPT\" \
        --model_paths ${model_paths_for_config[*]} \
        --output_dir \"$config_analysis_dir\" \
        --d_model $d_model \
        --nhead $nhead \
        --nlayers $nlayers"

    log_message "Executing Comparison Command for ${task_label}_${size_name}:"
    echo "$compare_cmd" | tee -a "$MAIN_LOG"

    if eval $compare_cmd >> "$config_analysis_log" 2>&1; then
        log_message "✓ Comparison analysis for ${task_label}_${size_name} completed successfully."
        analysis_success_count=$((analysis_success_count + 1))
    else
        log_message "✗ Comparison analysis for ${task_label}_${size_name} FAILED."
        echo "--- Last 50 lines of $config_analysis_log ---" >> "$MAIN_LOG"
        tail -n 50 "$config_analysis_log" >> "$MAIN_LOG"
        echo "--- End of log tail ---" >> "$MAIN_LOG"
        analysis_fail_count=$((analysis_fail_count + 1))
    fi

    if [ $analysis_fail_count -gt 0 ]; then return 1; else return 0; fi
}


# =================================================================
#                       MAIN EXECUTION PIPELINE
# =================================================================

overall_start_time=$(date +%s)

# Loop through each configuration size
for config_data in "${CONFIGS[@]}"; do
    IFS=';' read -r size_name d_model nhead nlayers _ <<< "$config_data" # Extract size name for logging
    log_message "========== PROCESSING CONFIGURATION: ${size_name} =========="

    # Loop through each task (lm, glue)
    for task_type in "${TASKS[@]}"; do

        if [ "$task_type" == "lm" ]; then
            log_message "------ Starting Language Modeling Task (Size: ${size_name}) ------"
            task_success=true
            # Loop through each model variant for LM
            for variant_data in "${VARIANTS[@]}"; do
                run_training_instance "$config_data" "lm" "" "$variant_data"
                if [ $? -ne 0 ]; then task_success=false; fi # Mark task as failed if any variant fails
            done

            # Run analysis for this LM config if training was successful
            if $task_success; then
                 run_analysis_for_config "$config_data" "lm" ""
            else
                 log_message "✗ Skipping analysis for LM (Size: ${size_name}) due to training failures."
            fi
            log_message "------ Finished Language Modeling Task (Size: ${size_name}) ------"

        elif [ "$task_type" == "glue" ]; then
            # Loop through specified GLUE tasks
            for glue_name in "${GLUE_TASKS[@]}"; do
                log_message "------ Starting GLUE Task: ${glue_name} (Size: ${size_name}) ------"
                task_success=true
                # Loop through each model variant for this GLUE task
                for variant_data in "${VARIANTS[@]}"; do
                    run_training_instance "$config_data" "glue" "$glue_name" "$variant_data"
                     if [ $? -ne 0 ]; then task_success=false; fi
                done

                # Run analysis for this GLUE config if training was successful
                if $task_success; then
                    run_analysis_for_config "$config_data" "glue" "$glue_name"
                else
                    log_message "✗ Skipping analysis for GLUE/${glue_name} (Size: ${size_name}) due to training failures."
                fi
                log_message "------ Finished GLUE Task: ${glue_name} (Size: ${size_name}) ------"
            done
        fi
    done # End task loop
    log_message "========== FINISHED CONFIGURATION: ${size_name} =========="
done # End config loop

overall_end_time=$(date +%s)
total_duration=$((overall_end_time - overall_start_time))

log_message "==================================================="
log_message "           PIPELINE COMPLETED                   "
log_message "Total Duration: $(printf '%dh:%dm:%ds\n' $((total_duration/3600)) $((total_duration%3600/60)) $((total_duration%60)))"
log_message "Results, Models, Logs saved in: $BASE_OUTPUT_DIR"
log_message "Completed at: $(date)"
echo "==================================================="

exit 0