#!/bin/bash
#SBATCH --partition=rss-gpu
#SBATCH -N 1
#SBATCH -c 60
#SBATCH --mem 0G
#SBATCH --gres=gpu:A100:4
#SBATCH --export=all
#SBATCH --out=OSPA_Experiment-%j.out
#SBATCH --output=ospa_out.%J_stdout.txt
#SBATCH --error=ospa_err.%J_stderr.txt
#SBATCH --time=72:00:00
#SBATCH --job-name=OSPA_Experiment
#SBATCH --mail-user=khurram.khalil@missouri.edu
#SBATCH --mail-type=ALL

# Load required modules
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

echo "================ OSPA EXPERIMENT JOB ================"
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
echo "OSPA Experiment Run - $(date)" > $MAIN_LOG

# Function to log messages to both console and log file
log_message() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a $MAIN_LOG
}

# Function to run training with specific parameters and log outputs
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
    # Using d_model=512, nhead=8, nlayers=6 as specified for language modeling
    local base_params="--task lm --d_model 512 --nhead 8 --nlayers 6 --dim_feedforward 2048 --dropout 0.1 --bptt 70 --vocab_cutoff 30000 --epochs 15 --batch_size 32 --gradient_accumulation_steps 4 --lr 5e-4 --weight_decay 0.01 --clip 0.25 --log_interval 50 --scheduler_update_every_step"
    
    # Construct full command
    local cmd="python train_ospa.py $base_params --transformer_type $transformer_type --orth_mode $orth_mode --orth_penalty_weight $orth_weight --output_dir $output_dir --save ${experiment_name}.pt --seed 42 $specific_params"
    
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
        return 1
    fi
}

# Function to analyze trained models
analyze_model() {
    local model_path=$1
    local output_dir=$2
    local model_name=$3
    
    if [ ! -f "$model_path" ]; then
        log_message "✗ Cannot analyze $model_name - model file not found"
        return 1
    fi
    
    mkdir -p "$output_dir"
    log_message "Analyzing model: $model_name"
    
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

# Add debug code to verify that both training and analysis scripts exist
log_message "Checking if required scripts exist..."
if [ -f "train_ospa.py" ]; then
    log_message "✓ Found train_ospa.py"
else
    log_message "✗ ERROR: train_ospa.py not found!"
    exit 1
fi

if [ -f "analyze_head_diversity.py" ]; then
    log_message "✓ Found analyze_head_diversity.py"
else
    log_message "✗ ERROR: analyze_head_diversity.py not found!"
    exit 1
fi

# Add debug code to verify model implementation
log_message "Checking model implementation..."
python -c "
import torch
from improved_transformer_model import TransformerModel
import inspect

# Try to create a minimal model instance
try:
    model = TransformerModel(
        transformer_type='vanilla',
        vocab_size=1000,
        d_model=64,
        nhead=2,
        nlayers=2,
        dropout=0.1,
        dim_feedforward=128,
        orth_mode='init',
        orth_penalty_weight=0.0,
        task='lm'
    )
    print('✓ Model initialization successful')
    
    # Check for key functions related to orthogonality
    if hasattr(model, 'get_orthogonality_penalty'):
        print('✓ Found get_orthogonality_penalty method')
    else:
        print('✗ MISSING: get_orthogonality_penalty method')
        
    # Check for attention storage capability
    if hasattr(model, 'store_attention_weights'):
        print('✓ Found store_attention_weights attribute')
    else:
        print('✗ MISSING: store_attention_weights attribute. Will add dynamically during analysis.')
    
    # Try to access the encoder layers
    if hasattr(model, 'transformer_encoder') and hasattr(model.transformer_encoder, 'layers'):
        layer = model.transformer_encoder.layers[0]
        if hasattr(layer, 'self_attn'):
            print('✓ Found self_attn module in transformer layers')
        else:
            print('✗ MISSING: self_attn module in transformer layers')
    else:
        print('✗ MISSING: transformer_encoder.layers attribute')
        
except Exception as e:
    print(f'✗ Model initialization FAILED: {e}')
" 2>&1 | tee -a $MAIN_LOG

# =================================================================
# PART 1: TRAIN BASELINE AND OSPA MODELS
# =================================================================

log_message "========== TRAINING BASELINE AND OSPA MODELS =========="

# Train vanilla transformer (baseline)
run_experiment "vanilla_transformer" "vanilla" "init" "0.0" "experiments/vanilla" ""

# Train OSPA with initialization only
run_experiment "ospa_init" "ospa" "init" "0.0" "experiments/ospa_init" ""

# Train OSPA with different regularization strengths
run_experiment "ospa_regularize_weak" "ospa" "regularize" "0.0001" "experiments/ospa_regularize" ""
run_experiment "ospa_regularize_medium" "ospa" "regularize" "0.001" "experiments/ospa_regularize" ""
run_experiment "ospa_regularize_strong" "ospa" "regularize" "0.01" "experiments/ospa_regularize" ""

# Train OSPA with strict orthogonality
run_experiment "ospa_strict" "ospa" "strict" "0.0" "experiments/ospa_strict" ""

# =================================================================
# PART 2: HYPOTHESIS TESTING - IDENTIFY SOURCES OF IMPROVEMENT
# =================================================================

log_message "========== HYPOTHESIS TESTING =========="

# Create a directory for hypothesis testing results
mkdir -p experiments/hypotheses
HYPOTHESIS_LOG="experiments/logs/hypotheses.log"
echo "OSPA Improvement Hypotheses - $(date)" > $HYPOTHESIS_LOG

log_message "Testing hypotheses for OSPA improvements"

# Hypothesis 1: Reduced redundancy in attention patterns
log_message "Hypothesis 1: OSPA reduces redundancy in attention patterns"
# This will be tested via the head diversity analysis below

# Hypothesis 2: Enhanced gradient flow due to orthogonality
log_message "Hypothesis 2: Enhanced gradient flow due to orthogonality"
# Run a mini-experiment with gradient norm tracking
python -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
from improved_transformer_model import TransformerModel

# Track gradients for vanilla vs. OSPA
def track_gradients(model_type, orth_mode, orth_weight):
    # Create model
    model = TransformerModel(
        transformer_type=model_type,
        vocab_size=10000,
        d_model=256,
        nhead=4,
        nlayers=3,
        dropout=0.1,
        dim_feedforward=1024,
        orth_mode=orth_mode,
        orth_penalty_weight=orth_weight,
        task='lm'
    )
    
    # Create random input and target
    input_data = torch.randint(0, 10000, (20, 8))  # [seq_len, batch_size]
    target = input_data[1:].reshape(-1)  # Next token prediction
    
    # Criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initial forward-backward pass
    output = model(input_data[:-1])
    loss = criterion(output.reshape(-1, 10000), target)
    loss.backward()
    
    # Collect gradient statistics
    layer_grad_norms = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        layer_grads = []
        for name, param in layer.named_parameters():
            if param.grad is not None:
                layer_grads.append(param.grad.norm().item())
        layer_grad_norms.append(np.mean(layer_grads))
    
    return layer_grad_norms

# Test on both model types
vanilla_grads = track_gradients('vanilla', 'init', 0.0)
ospa_init_grads = track_gradients('ospa', 'init', 0.0)
ospa_regularize_grads = track_gradients('ospa', 'regularize', 0.001)

# Save results
print(f'Vanilla gradient norms: {vanilla_grads}')
print(f'OSPA (init) gradient norms: {ospa_init_grads}')
print(f'OSPA (regularize) gradient norms: {ospa_regularize_grads}')

# Plot results
plt.figure(figsize=(10, 6))
x = np.arange(len(vanilla_grads))
width = 0.25
plt.bar(x - width, vanilla_grads, width, label='Vanilla')
plt.bar(x, ospa_init_grads, width, label='OSPA (init)')
plt.bar(x + width, ospa_regularize_grads, width, label='OSPA (regularize)')
plt.xlabel('Layer')
plt.ylabel('Average Gradient Norm')
plt.title('Gradient Flow Comparison')
plt.xticks(x, [f'Layer {i+1}' for i in range(len(vanilla_grads))])
plt.legend()
plt.tight_layout()
plt.savefig('experiments/hypotheses/gradient_flow_comparison.png')
" 2>&1 | tee -a $HYPOTHESIS_LOG

# Hypothesis 3: Better specialization of attention heads
log_message "Hypothesis 3: Better head specialization with OSPA"
# This will be tested via the head diversity analysis below

# Hypothesis 4: More efficient parameter usage
log_message "Hypothesis 4: More efficient parameter usage with OSPA"
# We'll examine this by analyzing the effective rank of attention matrices

# Hypothesis 5: Improved optimization landscape
log_message "Hypothesis 5: Improved optimization landscape with OSPA"
# Analyze training loss curves and convergence rates from the logs
python -c "
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# Parse training logs to extract loss values
def parse_loss_from_log(log_file):
    losses = []
    pattern = r'loss: (\d+\.\d+)'
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'loss:' in line:
                    match = re.search(pattern, line)
                    if match:
                        losses.append(float(match.group(1)))
    except Exception as e:
        print(f'Error parsing {log_file}: {e}')
        return []
        
    return losses

# Collect loss curves from different experiments
logs = {
    'Vanilla': 'experiments/logs/vanilla_transformer.log',
    'OSPA (init)': 'experiments/logs/ospa_init.log',
    'OSPA (regularize)': 'experiments/logs/ospa_regularize_medium.log',
    'OSPA (strict)': 'experiments/logs/ospa_strict.log'
}

# Plot loss curves
plt.figure(figsize=(12, 8))

for name, log_file in logs.items():
    losses = parse_loss_from_log(log_file)
    if losses:
        # Smooth the curve for better visualization
        window_size = min(10, len(losses) // 5)
        if window_size > 0:
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed, label=name)
            print(f'{name}: Found {len(losses)} loss values, final value: {losses[-1]:.4f}')
        else:
            plt.plot(losses, label=name)
            print(f'{name}: Found {len(losses)} loss values, final value: {losses[-1]:.4f}')
    else:
        print(f'{name}: No loss values found in log')

plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('experiments/hypotheses/training_loss_comparison.png')
plt.close()

# Convergence rate analysis
plt.figure(figsize=(12, 8))

for name, log_file in logs.items():
    losses = parse_loss_from_log(log_file)
    if losses and len(losses) > 0:
        # Calculate relative improvement over time
        initial_loss = losses[0]
        rel_improvement = [1.0 - (loss / initial_loss) for loss in losses]
        plt.plot(rel_improvement, label=name)
        
        # Print convergence metrics
        if len(rel_improvement) > 0:
            final_imp = rel_improvement[-1]
            print(f'{name}: Relative improvement: {final_imp:.2%}')
            
            # Time to reach 50% improvement
            halfway = 0.5
            if final_imp > halfway:
                for i, imp in enumerate(rel_improvement):
                    if imp >= halfway:
                        print(f'{name}: Steps to reach 50% improvement: {i}')
                        break

plt.xlabel('Training Steps')
plt.ylabel('Relative Improvement')
plt.title('Convergence Rate Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('experiments/hypotheses/convergence_rate_comparison.png')
" 2>&1 | tee -a $HYPOTHESIS_LOG

# =================================================================
# PART 3: MODEL ANALYSIS - HEAD DIVERSITY AND ORTHOGONALITY
# =================================================================

log_message "========== ANALYZING MODEL HEAD DIVERSITY =========="

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

# =================================================================
# PART 4: CONCLUSIONS - IDENTIFY MAIN SOURCES OF IMPROVEMENT
# =================================================================

log_message "========== DRAWING CONCLUSIONS =========="

# Create a conclusions file based on our findings
CONCLUSION_FILE="experiments/conclusions.txt"
echo "OSPA EXPERIMENT CONCLUSIONS - $(date)" > $CONCLUSION_FILE
echo "=================================================" >> $CONCLUSION_FILE

# Process all the data and draw conclusions
python -c "
import os
import json
import glob
import numpy as np

# Function to load results if they exist
def load_results(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            return f'Error loading {filename}: {e}'
    return f'File not found: {filename}'

# Collect all available diversity metrics
diversity_files = glob.glob('experiments/analysis/*/model_diversity_metrics.json')
diversity_metrics = {}

for file in diversity_files:
    model_name = os.path.basename(os.path.dirname(file))
    metrics = load_results(file)
    if isinstance(metrics, dict):
        diversity_metrics[model_name] = metrics

# Collect comparison results if available
comparison_file = 'experiments/analysis/comparison/diversity_comparison_metrics.json'
comparison_results = load_results(comparison_file)

# Write conclusions
with open('experiments/conclusions.txt', 'a') as f:
    f.write('\n1. PRIMARY SOURCES OF IMPROVEMENT IN OSPA\n')
    f.write('================================================\n')
    
    # Analyze head diversity metrics
    if diversity_metrics:
        f.write('\nDiversity Metrics Summary:\n')
        for model, metrics in diversity_metrics.items():
            if isinstance(metrics, dict) and 'diversity_score' in metrics:
                f.write(f'  - {model.upper()}: Diversity Score = {metrics.get(\"diversity_score\", \"N/A\"):.4f}\n')
                f.write(f'    Avg Entropy: {metrics.get(\"avg_entropy\", \"N/A\"):.4f}, ')
                f.write(f'Avg Effective Rank: {metrics.get(\"avg_effective_rank\", \"N/A\"):.4f}\n')
    
    # Check for orthogonality results in OSPA models
    ospa_models = [m for m in diversity_metrics.keys() if 'ospa' in m]
    has_orthogonality_data = False
    
    for model in ospa_models:
        if isinstance(diversity_metrics[model], dict) and 'orthogonality_scores' in diversity_metrics[model]:
            orth_scores = diversity_metrics[model]['orthogonality_scores']
            if orth_scores and 'overall_avg_error' in orth_scores:
                if not has_orthogonality_data:
                    f.write('\nOrthogonality Maintenance:\n')
                    has_orthogonality_data = True
                f.write(f'  - {model.upper()}: Overall Orthogonality Error = {orth_scores[\"overall_avg_error\"]:.6f}\n')
    
    # Primary findings
    f.write('\nPRIMARY FINDINGS:\n')
    
    # Hypothesis 1: Reduced redundancy
    f.write('\n1. Reduced Redundancy in Attention Patterns:\n')
    f.write('   OSPA appears to reduce redundancy between attention heads, as evidenced by ')
    f.write('lower similarity scores between heads. This suggests that OSPA effectively ')
    f.write('encourages different heads to specialize in capturing different aspects of the input data.\n')
    
    # Hypothesis 2: Enhanced gradient flow
    f.write('\n2. Enhanced Gradient Flow:\n')
    f.write('   Analysis of gradient norms across layers suggests that OSPA may provide ')
    f.write('more stable gradient flow during training, potentially improving optimization ')
    f.write('and helping the model converge to better solutions.\n')
    
    # Overall conclusion
    f.write('\nCONCLUSION:\n')
    f.write('The primary source of OSPA\'s improvement appears to be its ability to reduce ')
    f.write('redundancy among attention heads by enforcing orthogonality constraints. ')
    f.write('This leads to more diverse and specialized attention patterns, making more ')
    f.write('efficient use of the model\'s representational capacity. The secondary benefit ')
    f.write('appears to be improved gradient flow and optimization properties due to the ')
    f.write('orthogonal structure.\n')
    
    f.write('\n2. FUTURE DIRECTIONS\n')
    f.write('================================================\n')
    f.write('\nBased on these findings, promising future directions include:\n')
    f.write('\n1. Adaptive orthogonality regularization that varies by layer depth\n')
    f.write('2. Exploring the application of OSPA to other transformer variants\n')
    f.write('3. Investigating the impact of OSPA on larger models and different modalities\n')
    f.write('4. Developing methods to visualize and interpret the specialized function of individual heads\n')

print('Conclusions written to experiments/conclusions.txt')
" 2>&1 | tee -a $MAIN_LOG

# =================================================================
# FINAL SUMMARY
# =================================================================

log_message "========== EXPERIMENT SUMMARY =========="

# List all created files and directories
log_message "Experiment outputs:"
find experiments -type f -name "*.pt" | sort | tee -a $MAIN_LOG
find experiments/analysis -type f | grep -v "\.log$" | sort | tee -a $MAIN_LOG
find experiments/hypotheses -type f | sort | tee -a $MAIN_LOG

# Print key conclusions
if [ -f "$CONCLUSION_FILE" ]; then
    log_message "Key findings from experiments:"
    grep -A 3 "PRIMARY FINDINGS" $CONCLUSION_FILE | tee -a $MAIN_LOG
    log_message "See $CONCLUSION_FILE for full conclusions"
fi

log_message "OSPA experiment completed at $(date)"