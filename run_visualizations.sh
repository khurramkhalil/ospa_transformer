#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base experiment directory
EXPERIMENT_DIR="experiments/sst2"

# Output visualization directory
VIZ_DIR="$EXPERIMENT_DIR/visualizations"
mkdir -p $VIZ_DIR

# Run the visualization script
echo "Generating OSPA visualizations..."
python src/scripts/visualize_results.py \
    --experiment_dir $EXPERIMENT_DIR \
    --output_dir $VIZ_DIR \
    --dataset "SST-2" \
    --tokenizer "bert-base-uncased"

echo "Visualizations complete! Results saved to $VIZ_DIR"
echo "Open $VIZ_DIR/SST-2_visualization_report.html to view the report"