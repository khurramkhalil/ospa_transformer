#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
OUTPUT_DIR="experiments/iwslt"
mkdir -p $OUTPUT_DIR

echo "Starting OSPA experiments on IWSLT machine translation dataset..."

# Run baseline experiment
echo "Running baseline experiment..."
python src/scripts/train_translation.py \
    --data_dir ./data/iwslt \
    --output_dir $OUTPUT_DIR/baseline \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --enforce_mode init \
    --ortho_penalty_weight 0.0 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --epochs 5 \
    --seed 42 \
    --num_workers 0

# Run OSPA with regularization (different penalty weights)
for penalty in 0.001 0.005 0.01; do
    echo "Running OSPA with regularization (penalty = $penalty)..."
    python src/scripts/train_translation.py \
        --data_dir ./data/iwslt \
        --output_dir $OUTPUT_DIR/ospa_regularize_${penalty} \
        --d_model 512 \
        --nhead 8 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight $penalty \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --epochs 5 \
        --seed 42 \
        --num_workers 0
done

echo "IWSLT experiments complete!"
echo "Run visualization script to analyze results"
