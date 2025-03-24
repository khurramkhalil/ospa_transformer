#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
OUTPUT_DIR="experiments/sst2"
mkdir -p $OUTPUT_DIR

# Run baseline experiment (standard attention)
echo "Running baseline experiment..."
python src/scripts/fixed_train_glue.py \
    --task sst2 \
    --data_dir ./data \
    --output_dir $OUTPUT_DIR/baseline \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --enforce_mode init \
    --ortho_penalty_weight 0.0 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --seed 42 \
    --num_workers 0

# Run OSPA with regularization (different penalty weights)
for penalty in 0.001 0.01 0.1; do
    echo "Running OSPA with regularization (penalty = $penalty)..."
    python src/scripts/fixed_train_glue.py \
        --task sst2 \
        --data_dir ./data \
        --output_dir $OUTPUT_DIR/ospa_regularize_${penalty} \
        --d_model 512 \
        --nhead 8 \
        --num_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight $penalty \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --epochs 3 \
        --seed 42 \
        --num_workers 0
done

# Run OSPA with strict orthogonalization
echo "Running OSPA with strict orthogonalization..."
python src/scripts/fixed_train_glue.py \
    --task sst2 \
    --data_dir ./data \
    --output_dir $OUTPUT_DIR/ospa_strict \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --enforce_mode strict \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --seed 42 \
    --num_workers 0