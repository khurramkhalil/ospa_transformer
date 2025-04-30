#!/bin/bash
#SBATCH --partition=rss-gpu       # Or your GPU partition
#SBATCH -N 1                      # Number of nodes
#SBATCH -c 16                     # Number of CPU cores per task (adjust if needed)
#SBATCH --mem 50G                 # Memory per node
#SBATCH --gres=gpu:A100:1         # Request 1 A100 GPU (REMOVE if analysis is CPU-only)
#SBATCH --export=all
#SBATCH --job-name=OSPA_Diversity_Analysis # Clearer job name
#SBATCH --output=logs/OSPA_Diversity_Analysis_%j_output.txt # Corrected typo, added logs/ subdir
#SBATCH --error=logs/OSPA_Diversity_Analysis_%j_error.txt  # Added logs/ subdir
#SBATCH --time=12:00:00           # Reduced time if analysis is faster than training
#SBATCH --mail-user=khurram.khalil@missouri.edu # Your email
#SBATCH --mail-type=FAIL,END      # Notify on failure or completion

# --- Environment Setup ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines fail if any command fails, not just the last one.
set -o pipefail

# Load required modules (adjust path/version if needed)
module load miniconda3/4.10.3_gcc_9.5.0
# Activate your conda environment
source activate deepseek # Replace 'deepseek' with your actual environment name

echo "==================================================="
echo "        OSPA HEAD DIVERSITY ANALYSIS JOB           "
echo "==================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "==================================================="

# --- Configuration ---
# ** IMPORTANT: Set these to match the models you are analyzing **
D_MODEL=512
NHEAD=8
NLAYERS=6
# Analysis script filename (VERIFY THIS NAME)
ANALYSIS_SCRIPT="analyze_head_diversity.py"
# Base directory for analysis outputs
ANALYSIS_BASE_DIR="diversity_analysis_run_${SLURM_JOB_ID}"
# Directories to search for model checkpoints
SEARCH_DIRS=(
    "experiments" # Assumes models are in subdirs like experiments/vanilla, experiments/ospa_init etc.
    # Add other parent directories if needed, e.g., "outputs"
)

# --- Directory Setup ---
mkdir -p "$ANALYSIS_BASE_DIR"
mkdir -p "$ANALYSIS_BASE_DIR/models" # Subdir for individual model results
mkdir -p "$ANALYSIS_BASE_DIR/comparison" # Subdir for comparison results
mkdir -p logs # Ensure logs directory exists for SLURM output

# Log file for this analysis run
LOG_FILE="$ANALYSIS_BASE_DIR/analysis_progress.log"
echo "OSPA Diversity Analysis - Job ID: $SLURM_JOB_ID - $(date)" > $LOG_FILE

# Function to log messages
log_message() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $1" | tee -a $LOG_FILE
}

# --- Script Verification ---
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    log_message "ERROR: Analysis script '$ANALYSIS_SCRIPT' not found! Please ensure it's in the current directory or provide the correct path."
    exit 1
fi
log_message "✓ Found analysis script: $ANALYSIS_SCRIPT"

# --- Find Model Files ---
log_message "Searching for model files (*.pt) in specified directories..."
MODEL_FILES=()
for search_dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$search_dir" ]; then
        log_message "Searching in: $search_dir"
        # Use find -print0 and read -d $'\0' for safer handling of filenames with special characters
        while IFS= read -r -d $'\0' file; do
            # Basic filtering (can be made more specific)
            if [[ "$file" == *.pt ]] && [[ "$file" != *"error_checkpoint"* ]] && [[ "$file" != *"backup"* ]]; then
                MODEL_FILES+=("$file")
                log_message "  Found model: $file"
            fi
        done < <(find "$search_dir" -name "*.pt" -type f -print0)
    else
        log_message "Warning: Search directory not found: $search_dir"
    fi
done

# Check if any models were found
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    log_message "ERROR: No model files (.pt) found in the specified search directories!"
    exit 1
fi
log_message "Found ${#MODEL_FILES[@]} model files for analysis."

# --- Individual Model Analysis ---
log_message "--- Starting Individual Model Analysis ---"
ANALYSIS_RESULTS_DIR="$ANALYSIS_BASE_DIR/models"
SUCCESS_COUNT=0
FAIL_COUNT=0

for model_path in "${MODEL_FILES[@]}"; do
    model_name=$(basename "$model_path" .pt)
    # Sanitize model name for directory usage if needed (e.g., replace special chars)
    safe_model_name=$(echo "$model_name" | tr -cd '[:alnum:]_-')
    model_output_dir="$ANALYSIS_RESULTS_DIR/$safe_model_name"
    mkdir -p "$model_output_dir"

    log_message "Analyzing: $model_name"
    log_message "  Outputting to: $model_output_dir"

    # Define the command for individual analysis
    cmd="python \"$ANALYSIS_SCRIPT\" \
        --model_path \"$model_path\" \
        --output_dir \"$model_output_dir\" \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --nlayers $NLAYERS \
        --device cpu" # Explicitly run on CPU unless GPU is needed

    # Execute and log
    echo "Executing: $cmd" >> $LOG_FILE
    if eval $cmd >> $LOG_FILE 2>&1; then
        log_message "✓ Successfully analyzed $model_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log_message "✗ Failed to analyze $model_name (check $LOG_FILE and $ANALYSIS_BASE_DIR/logs/ for details)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done
log_message "Individual analysis complete. Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT."

# --- Comparison Analysis ---
if [ $SUCCESS_COUNT -lt 2 ]; then
    log_message "Skipping comparison analysis: Fewer than 2 models successfully analyzed."
else
    log_message "--- Starting Comparison Analysis ---"
    COMPARISON_OUTPUT_DIR="$ANALYSIS_BASE_DIR/comparison"
    mkdir -p "$COMPARISON_OUTPUT_DIR"

    # Pass the list of successfully analyzed model paths directly to the script
    # The "${MODEL_FILES[@]}" syntax handles spaces in paths correctly.
    cmd_compare="python \"$ANALYSIS_SCRIPT\" \
        --model_paths \"${MODEL_FILES[@]}\" \
        --output_dir \"$COMPARISON_OUTPUT_DIR\" \
        --d_model $D_MODEL \
        --nhead $NHEAD \
        --nlayers $NLAYERS \
        --device cpu" # Explicitly run on CPU

    log_message "Executing Comparison Command:"
    echo "$cmd_compare" >> $LOG_FILE # Log the command

    if eval $cmd_compare >> $LOG_FILE 2>&1; then
        log_message "✓ Comparison analysis completed successfully."
    else
        log_message "✗ Comparison analysis failed."
    fi
fi

log_message "==================================================="
log_message "Analysis Job Completed at: $(date)"
log_message "Results saved in: $ANALYSIS_BASE_DIR"
echo "==================================================="

exit 0 # Explicitly exit with success code