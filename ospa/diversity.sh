#!/bin/bash
#SBATCH --partition=rss-gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 50G
#SBATCH --gres=gpu:A100:1
#SBATCH --export=all
#SBATCH --out=OSPA_Fixed-%j.out
#SBATCH --output=OSPA_Diversity_Analysis-%j_outout.txt
#SBATCH --error=OSPA_Diversity_Analysis-%j_err.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=OSPA_Diversity
#SBATCH --mail-user=khurram.khalil@missouri.edu
#SBATCH --mail-type=ALL

# Load required modules
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

echo "================ OSPA HEAD DIVERSITY ANALYSIS ================"
echo "Starting at: $(date)"
echo "Running on host: $(hostname)"

# Create output directory
ANALYSIS_DIR="diversity_analysis"
mkdir -p $ANALYSIS_DIR

# Create a log file
LOG_FILE="$ANALYSIS_DIR/analysis.log"
echo "OSPA Diversity Analysis - $(date)" > $LOG_FILE

# Function to log messages to both console and log file
log_message() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a $LOG_FILE
}

# Ensure the diversity analysis script exists
if [ ! -f "analyze_head_diversity.py" ]; then
    log_message "ERROR: analyze_head_diversity.py not found! Please make sure it's in the current directory."
    exit 1
fi

# Find all available model files
log_message "Searching for model files..."

# Add all directories to search for models
SEARCH_DIRS=(
    # "experiments_old"
    "experiments"
    # "outputs"
)

# Find all .pt files in the search directories
MODEL_FILES=()
for dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        for file in $(find "$dir" -name "*.pt" -type f); do
            # Skip files that might be temp files or backups
            if [[ "$file" != *"temp"* ]] && [[ "$file" != *"backup"* ]]; then
                MODEL_FILES+=("$file")
                log_message "Found model: $file"
            fi
        done
    fi
done

# Check if we found any models
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    log_message "ERROR: No model files found!"
    exit 1
fi

log_message "Found ${#MODEL_FILES[@]} model files for analysis"

# Create individual analysis directories
for model in "${MODEL_FILES[@]}"; do
    model_name=$(basename "$model" .pt)
    model_dir="$ANALYSIS_DIR/models/$model_name"
    mkdir -p "$model_dir"
    
    # Analyze each model individually
    log_message "Analyzing model: $model"
    python analyze_head_diversity.py \
        --model_path "$model" \
        --output_dir "$model_dir" \
        --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048
    
    if [ $? -eq 0 ]; then
        log_message "✓ Successfully analyzed $model_name"
    else
        log_message "✗ Failed to analyze $model_name"
    fi
done

# Create comparison directory
COMPARE_DIR="$ANALYSIS_DIR/comparison"
mkdir -p "$COMPARE_DIR"

# Run comparison analysis on all models
log_message "Running comparison analysis on all models"

# Convert array to space-separated string for command line
MODEL_PATHS=""
for model in "${MODEL_FILES[@]}"; do
    MODEL_PATHS="$MODEL_PATHS $model"
done

# Run comparison
python analyze_head_diversity.py \
    --compare \
    --model_paths $MODEL_PATHS \
    --output_dir "$COMPARE_DIR" \
    --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048

if [ $? -eq 0 ]; then
    log_message "✓ Comparison analysis completed successfully"
else
    log_message "✗ Comparison analysis failed"
fi

# Create a summary of the diversity scores
log_message "Creating diversity score summary"

python -c "
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Collect all diversity metrics
metrics = []
model_names = []
diversity_scores = []
model_types = []

for root, dirs, files in os.walk('$ANALYSIS_DIR/models'):
    for file in files:
        if file == 'diversity_metrics.json':
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    model_name = os.path.basename(root)
                    model_names.append(model_name)
                    diversity_scores.append(data.get('diversity_score', 0))
                    model_types.append(data.get('model_type', 'unknown'))
                    metrics.append(data)
            except Exception as e:
                print(f'Error processing {file}: {e}')

# Sort models by type and diversity score
sorted_indices = sorted(range(len(model_names)), 
                       key=lambda i: (model_types[i] == 'vanilla', diversity_scores[i]))
sorted_names = [model_names[i] for i in sorted_indices]
sorted_scores = [diversity_scores[i] for i in sorted_indices]
sorted_types = [model_types[i] for i in sorted_indices]

# Save as CSV
with open('$ANALYSIS_DIR/diversity_scores.csv', 'w') as f:
    f.write('Model,Type,Diversity Score (lower is better)\n')
    for i in range(len(sorted_names)):
        f.write(f'{sorted_names[i]},{sorted_types[i]},{sorted_scores[i]:.6f}\n')

print(f'Saved diversity scores to $ANALYSIS_DIR/diversity_scores.csv')

# Create summary plot
plt.figure(figsize=(12, 8))
colors = ['#ff9999' if t == 'vanilla' else '#66b3ff' for t in sorted_types]
plt.bar(range(len(sorted_names)), sorted_scores, color=colors)
plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
plt.grid(axis='y', alpha=0.3)
plt.xlabel('Model')
plt.ylabel('Diversity Score (Lower is Better)')
plt.title('Comparison of Attention Head Diversity Across All Models')
plt.tight_layout()
plt.savefig('$ANALYSIS_DIR/all_models_diversity.png', dpi=300)
plt.close()

print(f'Created summary plot at $ANALYSIS_DIR/all_models_diversity.png')
"

log_message "Analysis complete. Results saved to $ANALYSIS_DIR/"
log_message "Summary CSV file: $ANALYSIS_DIR/diversity_scores.csv"
log_message "Summary plot: $ANALYSIS_DIR/all_models_diversity.png"

echo "================ ANALYSIS COMPLETE ================"
echo "Finished at: $(date)"