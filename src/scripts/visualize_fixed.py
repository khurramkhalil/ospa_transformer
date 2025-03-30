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
