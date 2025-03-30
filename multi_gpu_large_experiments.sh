#!/bin/bash

# Define all GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Add the project root to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Base output directory
LARGE_DIR="experiments/large_scale"
mkdir -p $LARGE_DIR

echo "Starting large-scale OSPA experiments using all 4 GPUs..."

# Large model configurations
MODEL_DIM=768
HEADS=12
LAYERS=12
FFN_DIM=3072

# Best penalty weight based on SST-2 tuning
PENALTY=0.0005

# Create a launcher function for distributed training
function run_distributed {
    NUM_GPUS=4
    MASTER_PORT=$(( 10000 + RANDOM % 50000 ))
    SCRIPT=$1
    shift
    
    echo "Running distributed training on $NUM_GPUS GPUs with arguments: $@"
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $SCRIPT --distributed $@
}

# Run on GLUE MNLI with large model (distributed training)
echo "Running large-scale MNLI experiment..."
run_distributed src/scripts/fixed_train_glue.py \
    --task mnli \
    --data_dir ./data \
    --output_dir $LARGE_DIR/large_mnli \
    --d_model $MODEL_DIM \
    --nhead $HEADS \
    --num_layers $LAYERS \
    --dim_feedforward $FFN_DIM \
    --enforce_mode regularize \
    --ortho_penalty_weight $PENALTY \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --epochs 8 \
    --seed 42 \
    --num_workers 2

# Run baseline for comparison (distributed training)
echo "Running large-scale MNLI baseline experiment..."
run_distributed src/scripts/fixed_train_glue.py \
    --task mnli \
    --data_dir ./data \
    --output_dir $LARGE_DIR/large_mnli_baseline \
    --d_model $MODEL_DIM \
    --nhead $HEADS \
    --num_layers $LAYERS \
    --dim_feedforward $FFN_DIM \
    --enforce_mode init \
    --ortho_penalty_weight 0.0 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --epochs 8 \
    --seed 42 \
    --num_workers 2

# Run on WMT16 translation dataset (distributed training)
echo "Running large-scale WMT16 translation experiment..."
run_distributed src/scripts/update_translation.py \
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
    --batch_size 4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --epochs 6 \
    --seed 42 \
    --num_workers 2

# Run WMT16 baseline for comparison (distributed training)
echo "Running large-scale WMT16 translation baseline experiment..."
run_distributed src/scripts/update_translation.py \
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
    --batch_size 4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --epochs 6 \
    --seed 42 \
    --num_workers 2

echo "Large-scale experiments complete!"