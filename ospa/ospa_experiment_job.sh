#!/bin/bash
#SBATCH --partition=rss-gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 0G
#SBATCH --gres=gpu:A100:1
#SBATCH --export=all
#SBATCH --out=OSPA_Fixed-%j.out
#SBATCH --output=ospa_new_train.%J_stdout.txt
#SBATCH --error=ospa_new_train.%J_stderr.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=OSPA_Fixed
#SBATCH --mail-user=khurram.khalil@missouri.edu
#SBATCH --mail-type=ALL

# Load required modules
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

echo "================ OSPA EXPERIMENT JOB (FIXED) ================"
echo "Starting at: $(date)"
echo "Running on host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "==================================================="

# Create all necessary directories upfront
mkdir -p experiments
mkdir -p experiments/vanilla
mkdir -p experiments/ospa_init
mkdir -p experiments/ospa_regularize
mkdir -p experiments/ospa_strict
mkdir -p experiments/analysis
mkdir -p experiments/logs

# Log file for tracking experiment progress
MAIN_LOG="experiments/logs/experiment_progress.log"
echo "OSPA Experiment Run (FIXED) - $(date)" > $MAIN_LOG

# Function to log messages to both console and log file
log_message() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a $MAIN_LOG
}

# Modified function to run training with specific parameters and log outputs
# The key fix is ensuring the save path is constructed correctly
run_experiment() {
    local experiment_name=$1
    local transformer_type=$2
    local orth_mode=$3
    local orth_weight=$4
    local output_dir=$5
    local specific_params=$6
    
    log_message "Starting experiment: $experiment_name"
    log_message "Model type: $transformer_type, Orth mode: $orth_mode, Lambda: $orth_weight"
    
    # Create experiment-specific log
    local exp_log="experiments/logs/${experiment_name}.log"
    echo "Experiment: $experiment_name - $(date)" > $exp_log
    
    # Base parameters from the paper draft
    local base_params="--task lm --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048 --dropout 0.1 --bptt 70 --vocab_cutoff 30000 --epochs 15 --batch_size 32 --gradient_accumulation_steps 4 --lr 5e-4 --weight_decay 0.01 --clip 0.25 --log_interval 50 --scheduler_update_every_step"
    
    # IMPORTANT FIX: Ensure correct path construction for saved model
    # This uses the explicit output_dir and saves the model with the experiment name
    mkdir -p "$output_dir"  # Ensure output directory exists
    
    # Construct full command with explicit output path
    local cmd="python train_new.py $base_params --transformer_type $transformer_type --orth_mode $orth_mode --orth_penalty_weight $orth_weight --output_dir $output_dir --save ${experiment_name}.pt --seed 42 $specific_params"
    
    # Log full command
    echo "Running command: $cmd" >> $exp_log
    
    # Execute the training command, capturing all output
    eval $cmd 2>&1 | tee -a $exp_log
    
    # Check if model was saved successfully
    if [ -f "$output_dir/${experiment_name}.pt" ]; then
        log_message "✓ Experiment $experiment_name completed successfully"
        echo "Model saved at: $output_dir/${experiment_name}.pt" >> $exp_log
        return 0
    else
        log_message "✗ Experiment $experiment_name FAILED - model not saved"
        # Print directory contents for debugging
        echo "Contents of $output_dir:" >> $exp_log
        ls -la "$output_dir" >> $exp_log
        return 1
    fi
}

# Add debug code to verify that both training and analysis scripts exist
log_message "Checking if required scripts exist..."
if [ -f "train.py" ]; then
    log_message "✓ Found train.py"
else
    log_message "✗ ERROR: train.py not found!"
    exit 1
fi

if [ -f "analyze_head_diversity.py" ]; then
    log_message "✓ Found analyze_head_diversity.py"
else
    log_message "✗ ERROR: analyze_head_diversity.py not found!"
    exit 1
fi

# =================================================================
# PART 1: TRAIN BASELINE AND OSPA MODELS (FIXED SAVING)
# =================================================================

log_message "========== TRAINING BASELINE AND OSPA MODELS =========="

# Train vanilla transformer (baseline)
run_experiment "vanilla_transformer" "vanilla" "init" "0.0" "experiments/vanilla" ""

# Train OSPA with initialization only 
run_experiment "ospa_init" "ospa" "init" "0.0" "experiments/ospa_init" ""

# Train OSPA with regularization (medium strength only to save time)
run_experiment "ospa_regularize_medium" "ospa" "regularize" "0.001" "experiments/ospa_regularize" ""

# Train OSPA with strict orthogonality
run_experiment "ospa_strict" "ospa" "strict" "0.0" "experiments/ospa_strict" ""

# =================================================================
# PART 2: MODEL ANALYSIS - HEAD DIVERSITY AND ORTHOGONALITY
# =================================================================

log_message "========== ANALYZING MODEL HEAD DIVERSITY =========="

# Function to analyze trained models
analyze_model() {
    local model_path=$1
    local output_dir=$2
    local model_name=$3
    
    if [ ! -f "$model_path" ]; then
        log_message "✗ Cannot analyze $model_name - model file not found at $model_path"
        return 1
    fi
    
    mkdir -p "$output_dir"
    log_message "Analyzing model: $model_name (from $model_path)"
    
    # Run head diversity analysis
    python analyze_head_diversity.py \
        --model_path "$model_path" \
        --output_dir "$output_dir" \
        --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048 \
        --task lm \
        --prefix "$model_name" \
        --device cpu
        
    if [ $? -eq 0 ]; then
        log_message "✓ Analysis completed for $model_name"
        return 0
    else
        log_message "✗ Analysis failed for $model_name"
        return 1
    fi
}

# Analyze each model individually
analyze_model "experiments/vanilla/vanilla_transformer.pt" "experiments/analysis/vanilla" "vanilla"
analyze_model "experiments/ospa_init/ospa_init.pt" "experiments/analysis/ospa_init" "ospa_init"
analyze_model "experiments/ospa_regularize/ospa_regularize_medium.pt" "experiments/analysis/ospa_regularize" "ospa_regularize"
analyze_model "experiments/ospa_strict/ospa_strict.pt" "experiments/analysis/ospa_strict" "ospa_strict"

# Compare all models
log_message "Running comparative analysis of all models"

# Get all available model paths
available_models=()
for model_path in "experiments/vanilla/vanilla_transformer.pt" "experiments/ospa_init/ospa_init.pt" "experiments/ospa_regularize/ospa_regularize_medium.pt" "experiments/ospa_strict/ospa_strict.pt"; do
    if [ -f "$model_path" ]; then
        available_models+=("$model_path")
    fi
done

# Only run comparison if we have at least 2 models
if [ ${#available_models[@]} -ge 2 ]; then
    log_message "Found ${#available_models[@]} models for comparison"
    
    # Convert array to space-separated string
    model_paths_str="${available_models[*]}"
    
    # Run comparison
    python analyze_head_diversity.py --compare \
        --model_paths $model_paths_str \
        --output_dir experiments/analysis/comparison \
        --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048 \
        --task lm \
        --device cpu
        
    if [ $? -eq 0 ]; then
        log_message "✓ Model comparison completed successfully"
    else
        log_message "✗ Model comparison failed"
    fi
else
    log_message "Not enough models for comparison, need at least 2"
fi

log_message "OSPA experiment (FIXED) completed at $(date)"