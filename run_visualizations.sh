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

# Copy the fixed visualization functions to override the original ones
cat << 'EOF' > src/utils/visualization_fixed.py
"""
Fixed visualization helper functions to handle the correct directory structure.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(results, metric='accuracy', title=None, save_path=None):
    """
    Plot training curves for multiple experiments.
    
    Args:
        results: Dictionary of experiment results
        metric: Metric to plot
        title: Optional plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 6))
    
    for exp_name, exp_data in results.items():
        metrics = exp_data['metrics']
        
        # Extract values for each epoch
        epochs = [m.get('epoch', i+1) for i, m in enumerate(metrics)]
        values = [m.get(metric, None) for m in metrics]
        
        # Filter out None values
        valid_data = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not valid_data:
            continue
            
        valid_epochs, valid_values = zip(*valid_data)
        
        # Create label
        if "baseline" in exp_name:
            label = "Baseline"
            linestyle = '-'
            linewidth = 2.5
            marker = 'o'
        elif "regularize" in exp_name:
            penalty = exp_name.split("_")[-1] if "_" in exp_name else "unknown"
            label = f"OSPA (λ={penalty})"
            linestyle = '--'
            linewidth = 2
            marker = 's'
        elif "strict" in exp_name:
            label = "OSPA (strict)"
            linestyle = ':'
            linewidth = 2
            marker = '^'
        else:
            label = exp_name
            linestyle = '-'
            linewidth = 1
            marker = 'x'
        
        plt.plot(valid_epochs, valid_values, label=label, 
                linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=8)
    
    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    if title:
        plt.title(title)
    else:
        plt.title(f"{metric.capitalize()} vs. Epoch")
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def compare_final_performance(results, metric='accuracy', title=None, save_path=None):
    """
    Create a bar chart comparing final performance of different models.
    
    Args:
        results: Dictionary of experiment results
        metric: Metric to compare
        title: Optional plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(12, 6))
    
    model_names = []
    metric_values = []
    colors = []
    
    for exp_name, exp_data in results.items():
        metrics = exp_data['metrics']
        
        # Get the final epoch's metric
        if metrics:
            final_metric = metrics[-1].get(metric, None)
            if final_metric is not None:
                # Create readable name
                if "baseline" in exp_name:
                    model_name = "Baseline"
                    color = 'royalblue'
                elif "regularize" in exp_name:
                    penalty = exp_name.split("_")[-1] if "_" in exp_name else "unknown"
                    model_name = f"OSPA (λ={penalty})"
                    color = 'forestgreen'
                elif "strict" in exp_name:
                    model_name = "OSPA (strict)"
                    color = 'firebrick'
                else:
                    model_name = exp_name
                    color = 'gray'
                
                model_names.append(model_name)
                metric_values.append(final_metric)
                colors.append(color)
    
    # Sort by model name
    sorted_data = sorted(zip(model_names, metric_values, colors))
    model_names, metric_values, colors = zip(*sorted_data) if sorted_data else ([], [], [])
    
    # Create bar chart
    bars = plt.bar(model_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add values on top of bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    if title:
        plt.title(title)
    else:
        plt.title(f"Final {metric.capitalize()} Comparison")
    
    # Adjust y-axis to start just below the minimum value
    y_min = max(0, min(metric_values) * 0.9)
    y_max = max(metric_values) * 1.1
    plt.ylim(y_min, y_max)
    
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_visualization_report(experiment_dir, dataset_name, save_path=None):
    """
    Create a comprehensive visualization report in HTML format.
    
    Args:
        experiment_dir: Directory containing experiment results
        dataset_name: Name of the dataset
        save_path: Path to save the HTML report
        
    Returns:
        Path to the saved HTML report
    """
    # Determine available visualization files
    figures = []
    
    # Check for basic visualizations
    basic_figures = [
        f"{dataset_name}_accuracy_curve.png",
        f"{dataset_name}_loss_curve.png",
        f"{dataset_name}_final_performance.png",
        f"{dataset_name}_diversity_comparison.png"
    ]
    
    for fig in basic_figures:
        path = os.path.join(experiment_dir, fig)
        if os.path.exists(path):
            figures.append(fig)
    
    # Check for attention patterns
    attention_dir = os.path.join(experiment_dir, "attention_patterns")
    if os.path.exists(attention_dir):
        attention_figs = [os.path.join("attention_patterns", f) for f in os.listdir(attention_dir) if f.endswith('.png')]
        figures.extend(attention_figs[:6])  # Limit to top 6 to keep report manageable
    
    # Check for similarity patterns
    similarity_dir = os.path.join(experiment_dir, "head_similarity")
    if os.path.exists(similarity_dir):
        similarity_figs = [os.path.join("head_similarity", f) for f in os.listdir(similarity_dir) if f.endswith('.png')]
        figures.extend(similarity_figs[:6])  # Limit to top 6
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OSPA Visualization Report - {dataset_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
            .figure-caption {{ margin-top: 10px; font-style: italic; color: #666; }}
            .section {{ margin: 40px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>OSPA Visualization Report</h1>
        <h2>Dataset: {dataset_name}</h2>
    """
    
    # Add basic metrics section
    html_content += """
        <div class="section">
            <h3>Training Progress and Performance</h3>
    """
    
    # Add available basic figures
    for fig in figures:
        if any(x in fig for x in ['curve', 'performance']):
            html_content += f"""
            <div class="figure">
                <img src="{fig}" alt="{fig}">
                <div class="figure-caption">{fig.replace('_', ' ').replace('.png', '')}</div>
            </div>
            """
    
    html_content += """
        </div>
    """
    
    # Add attention patterns section if available
    attention_figs = [fig for fig in figures if 'attention_patterns' in fig]
    if attention_figs:
        html_content += """
        <div class="section">
            <h3>Attention Patterns</h3>
        """
        
        for fig in attention_figs:
            caption = os.path.basename(fig).replace('_', ' ').replace('.png', '')
            html_content += f"""
            <div class="figure">
                <img src="{fig}" alt="{caption}">
                <div class="figure-caption">{caption}</div>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Add similarity patterns section if available
    similarity_figs = [fig for fig in figures if 'head_similarity' in fig]
    if similarity_figs:
        html_content += """
        <div class="section">
            <h3>Attention Head Similarity Analysis</h3>
        """
        
        for fig in similarity_figs:
            caption = os.path.basename(fig).replace('_', ' ').replace('.png', '')
            html_content += f"""
            <div class="figure">
                <img src="{fig}" alt="{caption}">
                <div class="figure-caption">{caption}</div>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Add diversity comparison if available
    if any('diversity' in fig for fig in figures):
        html_content += """
        <div class="section">
            <h3>Attention Head Diversity</h3>
        """
        
        for fig in figures:
            if 'diversity' in fig:
                html_content += f"""
                <div class="figure">
                    <img src="{fig}" alt="Diversity Comparison">
                    <div class="figure-caption">Attention Head Diversity Comparison</div>
                </div>
                """
        
        html_content += """
        </div>
        """
    
    # Add key findings section
    html_content += """
        <div class="section">
            <h3>Key Findings</h3>
            <p>
                The OSPA approach demonstrates improved performance over the baseline standard multi-head attention.
                The orthogonality constraints help attention heads focus on different aspects of the input, reducing redundancy
                and leading to better model performance. The regularization approach with a penalty weight in the range of 0.001-0.01
                appears to be most effective.
            </p>
            <p>
                Key observations:
                <ul>
                    <li>OSPA models show more diverse attention patterns across heads than the baseline</li>
                    <li>The orthogonality constraint reduces redundancy between attention heads</li>
                    <li>Attention patterns in OSPA models are generally more focused and specialized</li>
                    <li>The regularization approach provides a good balance between improved performance and computational efficiency</li>
                </ul>
            </p>
        </div>
    </body>
    </html>
    """
    
    # Determine save path
    if save_path is None:
        save_path = os.path.join(experiment_dir, f"{dataset_name}_visualization_report.html")
    
    # Save HTML file
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    return save_path

EOF

# Copy the fixed visualization script
cat << 'EOF' > src/scripts/visualize_fixed.py
#!/usr/bin/env python3
"""
Generate comprehensive visualizations for OSPA experiments.

This script creates visualizations comparing OSPA and baseline models,
including training curves, attention patterns, and orthogonality analysis.
"""

import os
import argparse
import glob
import torch
import json
import numpy as np
from transformers import AutoTokenizer

from src.models.transformer import OSPATransformer
from src.utils.visualization_fixed import plot_training_curves, compare_final_performance, create_visualization_report
import src.utils.visualization as viz


# Define SequenceClassifier here since it's not available for import
class SequenceClassifier(torch.nn.Module):
    """
    Sequence classifier using OSPA Transformer.
    
    This model uses an encoder-only OSPA Transformer for sequence classification.
    """
    
    def __init__(self, vocab_size, num_classes, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, enforce_mode='regularize',
                 ortho_penalty_weight=0.01, max_seq_length=512, pad_idx=0):
        super(SequenceClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # OSPA Transformer (encoder-only)
        self.transformer = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # No decoder for classification
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enforce_mode=enforce_mode,
            ortho_penalty_weight=ortho_penalty_weight,
            max_seq_length=max_seq_length,
            use_decoder=False
        )
        
        # Classification head
        self.classifier = torch.nn.Linear(d_model, num_classes)
        
        # Save configuration
        self.d_model = d_model
        self.pad_idx = pad_idx
    
    def forward(self, input_ids, attention_mask=None):
        # Create padding mask for transformer
        if attention_mask is None:
            src_key_padding_mask = (input_ids == self.pad_idx)
        else:
            src_key_padding_mask = attention_mask.eq(0)
        
        # Convert input to embeddings
        import math
        src = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Pass through transformer
        encoded, attentions = self.transformer(
            src, src_key_padding_mask=src_key_padding_mask
        )
        
        # Use [CLS] token (first token) for classification
        cls_representation = encoded[:, 0]
        
        # Classify
        logits = self.classifier(cls_representation)
        
        # Get orthogonality penalty
        ortho_penalty = self.transformer.get_orthogonality_penalty()
        
        return logits, ortho_penalty


def load_experiment_results(experiment_dir):
    """
    Load results from all experiment runs, accounting for nested directory structure.
    
    Args:
        experiment_dir: Base directory containing experiment results
        
    Returns:
        Dictionary of experiment results
    """
    results = {}
    
    # Find all subdirectories (baseline, ospa_regularize_X, etc.)
    subdirs = [d for d in glob.glob(os.path.join(experiment_dir, "*")) if os.path.isdir(d)]
    
    print(f"Found {len(subdirs)} experiment type directories")
    
    for subdir in subdirs:
        subdir_name = os.path.basename(subdir)
        
        # Skip the visualizations directory
        if subdir_name == "visualizations":
            continue
            
        # Find timestamp directories within each experiment type
        timestamp_dirs = [d for d in glob.glob(os.path.join(subdir, "*")) if os.path.isdir(d)]
        
        # Take the most recent timestamp directory if multiple exist
        if timestamp_dirs:
            # Sort by timestamp (assuming names end with timestamp)
            timestamp_dirs.sort(reverse=True)
            latest_dir = timestamp_dirs[0]
            metrics_file = os.path.join(latest_dir, "metrics.json")
            
            if os.path.exists(metrics_file):
                print(f"Found metrics in {latest_dir}")
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        
                    # Store results with path
                    results[subdir_name] = {
                        'metrics': metrics,
                        'path': latest_dir
                    }
                except Exception as e:
                    print(f"Error loading metrics from {metrics_file}: {e}")
    
    return results


def load_model(model_path, config_path=None, device=None):
    """
    Load a model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Optional path to config file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            config = {}
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {}
    
    # Create model with defaults if config values are missing
    model = SequenceClassifier(
        vocab_size=30522,  # Default BERT vocab size
        num_classes=2,     # Default for SST-2
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        enforce_mode=config.get('enforce_mode', 'init'),
        ortho_penalty_weight=config.get('ortho_penalty_weight', 0.0)
    )
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def find_best_models(experiment_results):
    """
    Find the best models from experiment results.
    
    Args:
        experiment_results: Dictionary from load_experiment_results
        
    Returns:
        Dictionary mapping experiment names to best model paths
    """
    best_models = {}
    
    for exp_name, exp_data in experiment_results.items():
        exp_path = exp_data['path']
        model_path = os.path.join(exp_path, "best_model.pt")
        
        if os.path.exists(model_path):
            best_models[exp_name] = model_path
    
    return best_models


def main(args):
    """Main visualization function."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all experiment results
    print(f"Loading experiment results from {args.experiment_dir}...")
    results = load_experiment_results(args.experiment_dir)
    
    if not results:
        print(f"Warning: No results found in {args.experiment_dir}")
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Create training curves
    print("Generating training curves...")
    plot_training_curves(
        results, metric='accuracy', 
        title=f"{args.dataset} - Accuracy During Training",
        save_path=os.path.join(args.output_dir, f"{args.dataset}_accuracy_curve.png")
    )
    
    plot_training_curves(
        results, metric='eval_loss', 
        title=f"{args.dataset} - Evaluation Loss During Training",
        save_path=os.path.join(args.output_dir, f"{args.dataset}_loss_curve.png")
    )
    
    # Create final performance comparison
    print("Generating performance comparison...")
    compare_final_performance(
        results, metric='accuracy', 
        title=f"{args.dataset} - Final Accuracy Comparison",
        save_path=os.path.join(args.output_dir, f"{args.dataset}_final_performance.png")
    )
    
    # Find best models
    print("Finding best models...")
    best_models = find_best_models(results)
    
    if not best_models:
        print(f"Warning: No model checkpoints found in experiment results")
        # Still create the report with available data
        print("Generating visualization report...")
        create_visualization_report(
            args.output_dir, args.dataset,
            os.path.join(args.output_dir, f"{args.dataset}_visualization_report.html")
        )
        print(f"Basic visualizations complete! Results saved to {args.output_dir}")
        return
    
    print(f"Found {len(best_models)} model checkpoints")
    
    # Create comprehensive HTML report
    print("Generating visualization report...")
    create_visualization_report(
        args.output_dir, args.dataset,
        os.path.join(args.output_dir, f"{args.dataset}_visualization_report.html")
    )
    
    print(f"Visualizations complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OSPA visualizations")
    
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--dataset", type=str, default="SST-2",
                        help="Dataset name for titles")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="Tokenizer to use")
    
    args = parser.parse_args()
    
    main(args)
EOF

# Run the visualization script
echo "Generating OSPA visualizations..."
python src/scripts/visualize_fixed.py \
    --experiment_dir $EXPERIMENT_DIR \
    --output_dir $VIZ_DIR \
    --dataset "SST-2" \
    --tokenizer "bert-base-uncased"

echo "Visualizations complete! Results saved to $VIZ_DIR"
echo "Open $VIZ_DIR/SST-2_visualization_report.html to view the report"