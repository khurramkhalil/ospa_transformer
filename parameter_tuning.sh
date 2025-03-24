#!/bin/bash

# Set CUDA visible devices based on availability
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
TUNING_DIR="experiments/sst2_tuning"
mkdir -p $TUNING_DIR

echo "Starting OSPA parameter tuning experiments..."

# Run baseline as reference point
echo "Running baseline experiment..."
python src/scripts/fixed_train_glue.py \
    --task sst2 \
    --data_dir ./data \
    --output_dir $TUNING_DIR/baseline \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --enforce_mode init \
    --ortho_penalty_weight 0.0 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --epochs 5 \
    --seed 42 \
    --num_workers 0

# Try a range of penalty weights
for penalty in 0.0005 0.001 0.003 0.005 0.01 0.03 0.05; do
    echo "Running OSPA with regularization (penalty = $penalty)..."
    python src/scripts/fixed_train_glue.py \
        --task sst2 \
        --data_dir ./data \
        --output_dir $TUNING_DIR/ospa_regularize_${penalty} \
        --d_model 512 \
        --nhead 8 \
        --num_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight $penalty \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --weight_decay 0.02 \
        --epochs 5 \
        --seed 42 \
        --num_workers 0
done

# Try different model depths with the best penalty weight
for layers in 4 6 8; do
    echo "Running OSPA with ${layers} layers..."
    python src/scripts/fixed_train_glue.py \
        --task sst2 \
        --data_dir ./data \
        --output_dir $TUNING_DIR/ospa_layers_${layers} \
        --d_model 512 \
        --nhead 8 \
        --num_layers $layers \
        --enforce_mode regularize \
        --ortho_penalty_weight 0.005 \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --weight_decay 0.02 \
        --epochs 5 \
        --seed 42 \
        --num_workers 0
done

# Try different numbers of attention heads
for heads in 4 8 16; do
    echo "Running OSPA with ${heads} attention heads..."
    python src/scripts/fixed_train_glue.py \
        --task sst2 \
        --data_dir ./data \
        --output_dir $TUNING_DIR/ospa_heads_${heads} \
        --d_model 512 \
        --nhead $heads \
        --num_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight 0.005 \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --weight_decay 0.02 \
        --epochs 5 \
        --seed 42 \
        --num_workers 0
done

# Try different learning rates
for lr in 1e-5 3e-5 7e-5; do
    echo "Running OSPA with learning rate ${lr}..."
    python src/scripts/fixed_train_glue.py \
        --task sst2 \
        --data_dir ./data \
        --output_dir $TUNING_DIR/ospa_lr_${lr} \
        --d_model 512 \
        --nhead 8 \
        --num_layers 6 \
        --enforce_mode regularize \
        --ortho_penalty_weight 0.005 \
        --batch_size 32 \
        --learning_rate $lr \
        --weight_decay 0.02 \
        --epochs 5 \
        --seed 42 \
        --num_workers 0
done

echo "Parameter tuning experiments complete!"
echo "Run visualization script to analyze results"