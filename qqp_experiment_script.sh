#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
OUTPUT_DIR="experiments/qqp"
mkdir -p $OUTPUT_DIR

echo "Starting OSPA experiments on QQP dataset..."

# Run baseline experiment
echo "Running baseline experiment..."
python src/scripts/fixed_train_glue.py \
    --task qqp \
    --data_dir ./data \
    --output_dir $OUTPUT_DIR/baseline \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --enforce_mode init \
    --ortho_penalty_weight 0.0 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --epochs 3 \
    --seed 42 \
    --num_workers 0

# Run OSPA with regularization (different penalty weights)
for penalty in 0.001 0.005 0.01; do
    echo "Running OSPA with regularization (penalty = $penalty)..."
    python src/scripts/fixed_train_glue.py \
        --task qqp \
        --data_dir ./data \
        --output_dir $OUTPUT_DIR/ospa_regularize_${penalty} \
        --d_model 512 \
        --nhead 8 \
        --num_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight $penalty \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --epochs 3 \
        --seed 42 \
        --num_workers 0
done

echo "QQP experiments complete!"
echo "Run visualization script to analyze results"