#!/bin/bash
#SBATCH --partition=rss-gpu       # Or your GPU partition
#SBATCH -N 1                      # Number of nodes
#SBATCH -c 32                     # Number of CPU cores per task
#SBATCH --mem 100G                # Memory per node
#SBATCH --gres=gpu:A100:1         # Request 1 A100 GPU (adjust as needed)
#SBATCH --export=all
#SBATCH --job-name=OSPA_Train_Analyze
#SBATCH --output=logs/ospa_train_analyze_%J.out # Combined output log
#SBATCH --error=logs/ospa_train_analyze_%J.err  # Combined error log
#SBATCH --time=72:00:00           # Max runtime
#SBATCH --mail-user=khurram.khalil@missouri.edu # Your email
#SBATCH --mail-type=ALL           # Email notifications

# --- Environment Setup ---
# Load required modules (adjust path/version if needed)
module load miniconda3/4.10.3_gcc_9.5.0
# Activate your conda environment
source activate deepseek # Replace 'deepseek' with your actual environment name

echo "==================================================="
echo "          OSPA TRAINING & ANALYSIS JOB             "
echo "==================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted from: $(pwd)"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "==================================================="

# --- Directory Setup ---
# Define base experiment directory
BASE_EXP_DIR="experiments_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}" # Unique dir for each run
VANILLA_DIR="$BASE_EXP_DIR/vanilla"
OSPA_INIT_DIR="$BASE_EXP_DIR/ospa_init"
OSPA_REG_DIR="$BASE_EXP_DIR/ospa_regularize"
OSPA_STRICT_DIR="$BASE_EXP_DIR/ospa_strict"
ANALYSIS_DIR="$BASE_EXP_DIR/analysis"
LOG_DIR="$BASE_EXP_DIR/logs" # Logs inside the main experiment dir

mkdir -p "$VANILLA_DIR" "$OSPA_INIT_DIR" "$OSPA_REG_DIR" "$OSPA_STRICT_DIR" "$ANALYSIS_DIR" "$LOG_DIR"
echo "Created experiment directory: $BASE_EXP_DIR"

# Log file for overall progress
MAIN_LOG="$LOG_DIR/experiment_progress_${SLURM_JOB_ID}.log"
echo "OSPA Experiment Run - Job ID: $SLURM_JOB_ID - $(date)" > "$MAIN_LOG"

# Function to log messages to both console (SLURM output) and log file
log_message() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $1" | tee -a "$MAIN_LOG"
}

# Function to run training experiments
run_training() {
    local experiment_name=$1
    local transformer_type=$2
    local orth_mode=$3
    local orth_weight=$4
    local output_dir=$5 # Specific subdir for this run's model/results
    local specific_params=$6 # Any extra params for this run

    log_message "--- Starting Training: $experiment_name ---"
    log_message "Config: Type=$transformer_type, OrthMode=$orth_mode, Lambda=$orth_weight"
    log_message "Output Dir: $output_dir"

    # Create output directory for this specific run
    mkdir -p "$output_dir"

    # Log file for this specific experiment run
    local exp_log="$LOG_DIR/train_${experiment_name}.log"
    echo "Training Run: $experiment_name - $(date)" > "$exp_log"

    # --- Define Base Parameters ---
    # Consistent parameters across runs (matching your results)
    local d_model=512
    local nhead=8
    local nlayers=6
    local dim_feedforward=2048
    local epochs=15
    local batch_size=32
    local accumulation=4
    local lr=5e-4
    local seed=42

    # Define consistent base command arguments
    local base_cmd="python train_new.py \
        --task lm \
        --d_model $d_model \
        --nhead $nhead \
        --nlayers $nlayers \
        --dim_feedforward $dim_feedforward \
        --dropout 0.1 \
        --bptt 70 \
        --vocab_cutoff 30000 \
        --epochs $epochs \
        --batch_size $batch_size \
        --gradient_accumulation_steps $accumulation \
        --lr $lr \
        --weight_decay 0.01 \
        --clip 0.25 \
        --log_interval 50 \
        --scheduler_update_every_step \
        --seed $seed \
        --num_workers 2 \
        --transformer_type $transformer_type \
        --orth_mode $orth_mode \
        --orth_penalty_weight $orth_weight \
        --output_dir $output_dir \
        --save ${experiment_name}.pt \
        $specific_params" # Add any extra specific params

    log_message "Executing Command:"
    echo "$base_cmd" | tee -a "$MAIN_LOG" # Log command clearly

    # Execute command, redirect stdout/stderr to specific log file
    if eval $base_cmd >> "$exp_log" 2>&1; then
        # Check if model file was actually created
        if [ -f "$output_dir/${experiment_name}.pt" ]; then
            log_message "✓ Training $experiment_name completed successfully."
            echo "Model saved: $output_dir/${experiment_name}.pt" >> "$exp_log"
            return 0 # Success
        else
            log_message "✗ Training $experiment_name finished, BUT model file NOT FOUND at $output_dir/${experiment_name}.pt"
            return 1 # Failure
        fi
    else
        log_message "✗ Training $experiment_name FAILED (Command exited with error)."
        # Tail the log file for quick error view
        echo "--- Last 20 lines of $exp_log ---" >> "$MAIN_LOG"
        tail -n 20 "$exp_log" >> "$MAIN_LOG"
        echo "--- End of log tail ---" >> "$MAIN_LOG"
        return 1 # Failure
    fi
}

# Function to run analysis script
run_analysis() {
    local model_path=$1
    local analysis_output_dir=$2
    local model_name=$3
    local d_model=$4
    local nhead=$5
    local nlayers=$6

    log_message "--- Starting Analysis: $model_name ---"
    log_message "Model Path: $model_path"
    log_message "Analysis Output Dir: $analysis_output_dir"

    if [ ! -f "$model_path" ]; then
        log_message "✗ Cannot analyze $model_name - model file not found at $model_path"
        return 1
    fi

    mkdir -p "$analysis_output_dir"
    local analysis_log="$LOG_DIR/analyze_${model_name}.log"
    echo "Analysis Run: $model_name - $(date)" > "$analysis_log"

    # --- Define Analysis Command ---
    # Pass required model architecture arguments
    local analysis_cmd="python analyze_head_diversity.py \
        --model_path \"$model_path\" \
        --output_dir \"$analysis_output_dir\" \
        --d_model $d_model \
        --nhead $nhead \
        --nlayers $nlayers \
        --device cpu" # Run analysis on CPU typically

    log_message "Executing Command:"
    echo "$analysis_cmd" | tee -a "$MAIN_LOG"

    if eval $analysis_cmd >> "$analysis_log" 2>&1; then
        log_message "✓ Analysis for $model_name completed successfully."
        return 0 # Success
    else
        log_message "✗ Analysis for $model_name FAILED."
        echo "--- Last 20 lines of $analysis_log ---" >> "$MAIN_LOG"
        tail -n 20 "$analysis_log" >> "$MAIN_LOG"
        echo "--- End of log tail ---" >> "$MAIN_LOG"
        return 1 # Failure
    fi
}

# Function to run comparison analysis
run_comparison() {
    local model_paths_array=("$@") # Pass array of model paths
    local comparison_output_dir=$1 # First element is output dir now
    shift # Remove output dir from array
    local d_model=$1
    local nhead=$2
    local nlayers=$3
    shift 3 # Remove model args

    log_message "--- Starting Comparison Analysis ---"
    log_message "Comparison Output Dir: $comparison_output_dir"

    # Check if enough models exist
    local available_models=()
    for model_path in "${model_paths_array[@]}"; do
        if [ -f "$model_path" ]; then
            available_models+=("\"$model_path\"") # Quote paths with spaces
        else
            log_message "Warning: Model path not found for comparison: $model_path"
        fi
    done

    if [ ${#available_models[@]} -lt 2 ]; then
        log_message "✗ Not enough models found (${#available_models[@]}) for comparison. Need at least 2."
        return 1
    fi

    log_message "Found ${#available_models[@]} models for comparison."
    mkdir -p "$comparison_output_dir"
    local comparison_log="$LOG_DIR/analysis_comparison.log"
    echo "Comparison Analysis Run - $(date)" > "$comparison_log"

    # --- Define Comparison Command ---
    # Pass model paths correctly
    local comparison_cmd="python analyze_head_diversity.py \
        --model_paths ${available_models[*]} \
        --output_dir \"$comparison_output_dir\" \
        --d_model $d_model \
        --nhead $nhead \
        --nlayers $nlayers \
        --device cpu" # Run analysis on CPU

    log_message "Executing Command:"
    echo "$comparison_cmd" | tee -a "$MAIN_LOG"

    if eval $comparison_cmd >> "$comparison_log" 2>&1; then
        log_message "✓ Model comparison completed successfully."
        return 0 # Success
    else
        log_message "✗ Model comparison FAILED."
        echo "--- Last 20 lines of $comparison_log ---" >> "$MAIN_LOG"
        tail -n 20 "$comparison_log" >> "$MAIN_LOG"
        echo "--- End of log tail ---" >> "$MAIN_LOG"
        return 1 # Failure
    fi
}


# --- Verify Scripts ---
log_message "Checking for required Python scripts..."
scripts_ok=true
if [ ! -f "train_new.py" ]; then
    log_message "✗ ERROR: Training script 'train_new.py' not found!"
    scripts_ok=false
fi
if [ ! -f "analyze_head_diversity.py" ]; then
    log_message "✗ ERROR: Analysis script 'analyze_head_diversity.py' not found!"
    scripts_ok=false
fi
if [ "$scripts_ok" = false ]; then
    exit 1
fi
log_message "✓ Found required Python scripts."


# =================================================================
# PART 1: TRAIN MODELS
# Define model parameters used for training AND analysis
D_MODEL=512
NHEAD=8
NLAYERS=6
# =================================================================
log_message "========== PART 1: TRAINING MODELS =========="

# Train vanilla transformer (baseline)
run_training "vanilla_transformer" "vanilla" "init" 0.0 "$VANILLA_DIR" ""
VANILLA_MODEL_PATH="$VANILLA_DIR/vanilla_transformer.pt"

# Train OSPA with initialization only
run_training "ospa_init" "ospa" "init" 0.0 "$OSPA_INIT_DIR" ""
OSPA_INIT_MODEL_PATH="$OSPA_INIT_DIR/ospa_init.pt"

# Train OSPA with regularization (medium strength)
run_training "ospa_regularize_medium" "ospa" "regularize" 0.001 "$OSPA_REG_DIR" ""
OSPA_REG_MODEL_PATH="$OSPA_REG_DIR/ospa_regularize_medium.pt"

# Train OSPA with strict orthogonality
run_training "ospa_strict" "ospa" "strict" 0.0 "$OSPA_STRICT_DIR" ""
OSPA_STRICT_MODEL_PATH="$OSPA_STRICT_DIR/ospa_strict.pt"


# =================================================================
# PART 2: ANALYZE MODELS
# =================================================================
log_message "========== PART 2: ANALYZING MODELS =========="

# Analyze each model individually
# Pass required architecture parameters D_MODEL, NHEAD, NLAYERS
run_analysis "$VANILLA_MODEL_PATH" "$ANALYSIS_DIR/vanilla" "vanilla" $D_MODEL $NHEAD $NLAYERS
run_analysis "$OSPA_INIT_MODEL_PATH" "$ANALYSIS_DIR/ospa_init" "ospa_init" $D_MODEL $NHEAD $NLAYERS
run_analysis "$OSPA_REG_MODEL_PATH" "$ANALYSIS_DIR/ospa_regularize" "ospa_regularize" $D_MODEL $NHEAD $NLAYERS
run_analysis "$OSPA_STRICT_MODEL_PATH" "$ANALYSIS_DIR/ospa_strict" "ospa_strict" $D_MODEL $NHEAD $NLAYERS

# Run comparison analysis
COMPARISON_DIR="$ANALYSIS_DIR/comparison"
# Pass the comparison output directory first, then model parameters, then the list of model paths
run_comparison "$COMPARISON_DIR" $D_MODEL $NHEAD $NLAYERS "$VANILLA_MODEL_PATH" "$OSPA_INIT_MODEL_PATH" "$OSPA_REG_MODEL_PATH" "$OSPA_STRICT_MODEL_PATH"


log_message "==================================================="
log_message "Job Completed at: $(date)"
echo "Results and logs saved in: $BASE_EXP_DIR"
echo "==================================================="

exit 0