#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
LARGE_DIR="experiments/large_scale"
mkdir -p $LARGE_DIR

echo "Starting large-scale OSPA experiments using all 4 GPUs with DataParallel..."

# Large model configurations
MODEL_DIM=768
HEADS=12
LAYERS=12
FFN_DIM=3072

# Best penalty weight based on SST-2 tuning
PENALTY=0.0005

# If we're continuing after a failure, we can comment out completed experiments
# Run MNLI experiment with OSPA
if [ ! -d "$LARGE_DIR/large_mnli" ]; then
    echo "Running large-scale MNLI experiment..."
    python src/scripts/fixed_train_glue.py \
        --task mnli \
        --data_dir ./data \
        --output_dir $LARGE_DIR/large_mnli \
        --d_model $MODEL_DIM \
        --nhead $HEADS \
        --num_layers $LAYERS \
        --dim_feedforward $FFN_DIM \
        --enforce_mode regularize \
        --ortho_penalty_weight $PENALTY \
        --batch_size 24 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --epochs 8 \
        --seed 42 \
        --num_workers 2
else
    echo "Skipping large-scale MNLI experiment (already completed)"
fi

# Run MNLI baseline
if [ ! -d "$LARGE_DIR/large_mnli_baseline" ]; then
    echo "Running large-scale MNLI baseline experiment..."
    python src/scripts/fixed_train_glue.py \
        --task mnli \
        --data_dir ./data \
        --output_dir $LARGE_DIR/large_mnli_baseline \
        --d_model $MODEL_DIM \
        --nhead $HEADS \
        --num_layers $LAYERS \
        --dim_feedforward $FFN_DIM \
        --enforce_mode init \
        --ortho_penalty_weight 0.0 \
        --batch_size 24 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --epochs 8 \
        --seed 42 \
        --num_workers 2
else
    echo "Skipping large-scale MNLI baseline experiment (already completed)"
fi

# Run WMT16 experiment with OSPA
if [ ! -d "$LARGE_DIR/large_wmt16" ]; then
    echo "Running large-scale WMT16 translation experiment..."
    python src/scripts/update_translation.py \
        --data_dir ./data \
        --dataset_name "wmt16" \
        --language_pair "de-en" \
        --output_dir $LARGE_DIR/large_wmt16 \
        --d_model $MODEL_DIM \
        --nhead $HEADS \
        --num_encoder_layers $LAYERS \
        --num_decoder_layers $LAYERS \
        --dim_feedforward $FFN_DIM \
        --enforce_mode regularize \
        --ortho_penalty_weight $PENALTY \
        --batch_size 8 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --epochs 6 \
        --seed 42 \
        --num_workers 2
else
    echo "Skipping large-scale WMT16 translation experiment (already completed)"
fi

# Run WMT16 baseline
if [ ! -d "$LARGE_DIR/large_wmt16_baseline" ]; then
    echo "Running large-scale WMT16 translation baseline experiment..."
    python src/scripts/update_translation.py \
        --data_dir ./data \
        --dataset_name "wmt16" \
        --language_pair "de-en" \
        --output_dir $LARGE_DIR/large_wmt16_baseline \
        --d_model $MODEL_DIM \
        --nhead $HEADS \
        --num_encoder_layers $LAYERS \
        --num_decoder_layers $LAYERS \
        --dim_feedforward $FFN_DIM \
        --enforce_mode init \
        --ortho_penalty_weight 0.0 \
        --batch_size 8 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --epochs 6 \
        --seed 42 \
        --num_workers 2
else
    echo "Skipping large-scale WMT16 translation baseline experiment (already completed)"
fi

echo "Large-scale experiments complete!"