#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try importing the model - with extensive error handling
try:
    from improved_transformer_model import TransformerModel
    logger.info("Successfully imported TransformerModel")
except ImportError as e:
    logger.error(f"Failed to import TransformerModel: {e}")
    logger.error("Please make sure improved_transformer_model.py is in the current directory")
    logger.error("Current directory: %s", os.getcwd())
    logger.error("Directory contents: %s", os.listdir('.'))
    sys.exit(1)

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def patch_model_if_needed(model):
    """
    Patch the model with needed attributes/methods if they don't exist
    """
    # Add attention weights storage capability if missing
    if not hasattr(model, 'store_attention_weights'):
        logger.info("Adding store_attention_weights attribute to model")
        model.store_attention_weights = False
        model._attention_weights = []
        
    # Add getter for attention weights if missing
    if not hasattr(model, 'get_attention_weights'):
        logger.info("Adding get_attention_weights method to model")
        def get_attention_weights(self):
            if not hasattr(self, '_attention_weights') or not self._attention_weights:
                logger.warning("No attention weights stored. Run with store_attention_weights=True first.")
                return torch.tensor([])
            return torch.stack(self._attention_weights)
        
        model.get_attention_weights = get_attention_weights.__get__(model)
    
    # Add hook to store attention weights during forward pass
    if not hasattr(model, '_has_attention_hooks'):
        logger.info("Adding attention hooks to model")
        
        def hook_attention(module, inputs, outputs):
            if hasattr(model, 'store_attention_weights') and model.store_attention_weights:
                # Assume standard multi-head attention output format
                attn_weights = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
                if attn_weights is not None:
                    if not hasattr(model, '_attention_weights'):
                        model._attention_weights = []
                    model._attention_weights.append(attn_weights.detach())
        
        # Try to register hooks on attention modules
        hookable_modules_found = False
        if hasattr(model, 'transformer_encoder') and hasattr(model.transformer_encoder, 'layers'):
            for i, layer in enumerate(model.transformer_encoder.layers):
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.register_forward_hook(hook_attention)
                    hookable_modules_found = True
        
        if not hookable_modules_found:
            logger.warning("Could not find suitable attention modules to hook")
        
        model._has_attention_hooks = True
    
    return model

def create_dummy_data(model, vocab_size, args):
    """
    Create dummy input data for the model to extract attention patterns
    """
    device = torch.device(args.device)
    
    # Create random input data with sequence length and batch size
    seq_len = min(args.max_seq_len, 128)  # Use a reasonable sequence length
    batch_size = 4  # Small batch size for analysis
    
    if args.task == 'lm':
        # For language modeling: [seq_len, batch_size]
        dummy_input = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    else:
        # For classification: [seq_len, batch_size]
        dummy_input = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    
    return dummy_input

def compute_attention_patterns(model, args, vocab_size):
    """
    Compute attention patterns from model using dummy input
    """
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    
    # Reset attention weights storage
    if hasattr(model, '_attention_weights'):
        model._attention_weights = []
    
    # Enable attention weight storage
    model.store_attention_weights = True
    
    # Create dummy input data
    dummy_input = create_dummy_data(model, vocab_size, args)
    
    # Forward pass to capture attention weights
    logger.info("Extracting attention patterns with a dummy forward pass...")
    with torch.no_grad():
        try:
            # For language models, create a causal mask
            if args.task == 'lm':
                seq_len = dummy_input.size(0)
                src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
                _ = model(dummy_input, src_mask=src_mask)
            else:
                _ = model(dummy_input)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise
    
    # Get stored attention weights
    attention_weights = model.get_attention_weights()
    
    # Disable attention weight storage
    model.store_attention_weights = False
    
    return attention_weights

def compute_head_similarity_matrix(attention_weights):
    """
    Compute cosine similarity between attention heads
    """
    if attention_weights.numel() == 0:
        logger.error("Empty attention weights tensor")
        return np.zeros((1, 1))
    
    # Get dimensions
    if len(attention_weights.shape) == 4:
        # Shape: [layer, head, seq_len, seq_len]
        num_layers, num_heads, seq_len, _ = attention_weights.shape
        
        # Reshape to [layer*head, seq_len*seq_len]
        flattened = attention_weights.reshape(num_layers * num_heads, -1).cpu().numpy()
    else:
        logger.warning(f"Unexpected attention weights shape: {attention_weights.shape}. " +
                       "Expected [layer, head, seq_len, seq_len]")
        if len(attention_weights.shape) > 1:
            # Try to flatten to 2D as best we can
            flattened = attention_weights.reshape(attention_weights.shape[0], -1).cpu().numpy()
        else:
            return np.zeros((1, 1))
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(flattened)
    
    return similarity_matrix

def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention distributions as a measure of focus
    """
    if attention_weights.numel() == 0:
        logger.error("Empty attention weights tensor")
        return torch.zeros(1, 1)
    
    # Get dimensions
    if len(attention_weights.shape) != 4:
        logger.warning(f"Unexpected attention weights shape: {attention_weights.shape}")
        return torch.zeros(1, 1)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Ensure attention weights sum to 1 along the last dimension
    attention_probs = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + epsilon)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(attention_probs * torch.log(attention_probs + epsilon)).sum(dim=-1)
    
    # Average over sequence length
    mean_entropy = entropy.mean(dim=-1)  # [layer, head]
    
    return mean_entropy

def compute_effective_rank(attention_weights):
    """
    Compute effective rank of attention matrices
    """
    if attention_weights.numel() == 0:
        logger.error("Empty attention weights tensor")
        return torch.zeros(1, 1)
    
    # Get dimensions
    if len(attention_weights.shape) != 4:
        logger.warning(f"Unexpected attention weights shape: {attention_weights.shape}")
        return torch.zeros(1, 1)
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    effective_rank = torch.zeros(num_layers, num_heads)
    
    for l in range(num_layers):
        for h in range(num_heads):
            # Get attention matrix for this head
            head_weights = attention_weights[l, h]
            
            try:
                # Compute SVD
                U, S, V = torch.svd(head_weights)
                
                # Normalize singular values
                normalized_S = S / torch.sum(S)
                
                # Compute entropy of normalized singular values as effective rank
                entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
                effective_rank[l, h] = torch.exp(entropy)
            except Exception as e:
                logger.warning(f"SVD failed for layer {l}, head {h}: {e}")
                effective_rank[l, h] = 0
    
    return effective_rank

def check_orthogonality(model):
    """
    Check if the model has orthogonal projection matrices
    """
    if not hasattr(model, 'transformer_type') or model.transformer_type != 'ospa':
        return None
    
    orthogonality_scores = {}
    
    # Try to access projection matrices in the model
    if hasattr(model, 'transformer_encoder') and hasattr(model.transformer_encoder, 'layers'):
        for layer_idx, layer in enumerate(model.transformer_encoder.layers):
            if hasattr(layer, 'self_attn'):
                mha = layer.self_attn
                
                # Check for OSPA-specific attributes
                p_matrices = []
                for attr_name in ['P_Q', 'P_K', 'P_V', 'P_O']:
                    if hasattr(mha, attr_name):
                        p_matrices.append((attr_name, getattr(mha, attr_name)))
                
                if p_matrices:
                    layer_scores = {}
                    for name, P in p_matrices:
                        # Compute orthogonality error: ||P^T P - I||_F^2
                        try:
                            error = torch.norm(
                                torch.matmul(P.transpose(-2, -1), P) - 
                                torch.eye(P.shape[-1], device=P.device), 
                                p='fro'
                            ).item()
                            layer_scores[f'{name}_error'] = error
                        except Exception as e:
                            logger.warning(f"Error computing orthogonality for {name} in layer {layer_idx}: {e}")
                            layer_scores[f'{name}_error'] = float('nan')
                    
                    # Compute average error across all projection matrices
                    valid_errors = [v for v in layer_scores.values() if not np.isnan(v)]
                    if valid_errors:
                        layer_scores['avg_error'] = sum(valid_errors) / len(valid_errors)
                    
                    orthogonality_scores[f'layer_{layer_idx}'] = layer_scores
    
    # Compute overall average if we have data
    if orthogonality_scores:
        avg_errors = [
            scores.get('avg_error', float('nan')) 
            for scores in orthogonality_scores.values() 
            if 'avg_error' in scores
        ]
        
        valid_avgs = [e for e in avg_errors if not np.isnan(e)]
        if valid_avgs:
            orthogonality_scores['overall_avg_error'] = sum(valid_avgs) / len(valid_avgs)
    
    return orthogonality_scores

def evaluate_head_diversity(model, args):
    """
    Main function to evaluate and visualize head diversity metrics
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine vocabulary size from model if possible
    vocab_size = getattr(model, 'vocab_size', 10000)
    logger.info(f"Using vocabulary size: {vocab_size}")
    
    # 1. Compute attention patterns
    try:
        attention_weights = compute_attention_patterns(model, args, vocab_size)
        logger.info(f"Attention weights shape: {attention_weights.shape}")
    except Exception as e:
        logger.error(f"Failed to compute attention patterns: {e}")
        attention_weights = torch.tensor([])
    
    # If attention weights extraction failed, create a dummy result
    if attention_weights.numel() == 0:
        logger.warning("Creating dummy results since attention weight extraction failed")
        results = {
            'model_type': getattr(model, 'transformer_type', 'unknown'),
            'error': "Failed to extract attention weights",
            'diversity_score': 0.0,
            'avg_entropy': 0.0,
            'avg_effective_rank': 0.0
        }
        
        with open(os.path.join(output_dir, f"{args.prefix}_diversity_metrics.json"), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    # 2. Compute head similarity matrix
    similarity_matrix = compute_head_similarity_matrix(attention_weights)
    
    # 3. Compute attention entropy
    entropy = compute_attention_entropy(attention_weights)
    
    # 4. Compute effective rank
    effective_rank = compute_effective_rank(attention_weights)
    
    # 5. Check orthogonality for OSPA models
    orthogonality_scores = check_orthogonality(model)
    
    # 6. Compile results
    results = {
        'model_type': getattr(model, 'transformer_type', 'unknown'),
        'avg_similarity': float(np.mean(similarity_matrix) - np.mean(np.diag(similarity_matrix))),
        'avg_entropy': float(entropy.mean().item()) if entropy.numel() > 0 else 0.0,
        'avg_effective_rank': float(effective_rank.mean().item()) if effective_rank.numel() > 0 else 0.0,
        'orthogonality_scores': orthogonality_scores
    }
    
    # Calculate diversity score (lower is more diverse)
    diversity_score = float(np.mean(similarity_matrix) - np.mean(np.diag(similarity_matrix)))
    results['diversity_score'] = diversity_score
    
    # Save numeric results
    with open(os.path.join(output_dir, f"{args.prefix}_diversity_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    # 1. Head similarity heatmap
    try:
        plt.figure(figsize=(10, 8))
        mask = np.eye(similarity_matrix.shape[0], dtype=bool)
        sns.heatmap(
            similarity_matrix, 
            cmap='viridis', 
            mask=mask,  # Mask the diagonal
            vmin=0, 
            vmax=1,
            square=True,
            xticklabels=5, 
            yticklabels=5
        )
        plt.title(f'Attention Head Similarity (Diversity Score: {diversity_score:.4f})')
        plt.xlabel('Head Index (layer * num_heads + head)')
        plt.ylabel('Head Index (layer * num_heads + head)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}_head_similarity.png"), dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating head similarity heatmap: {e}")
    
    # 2. Attention entropy by layer and head
    if entropy.numel() > 0:
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                entropy.cpu().numpy(), 
                cmap='coolwarm', 
                annot=True, 
                fmt=".2f",
                cbar_kws={'label': 'Entropy'}
            )
            plt.title('Attention Entropy by Layer and Head (Higher = More Uniform Attention)')
            plt.xlabel('Head Index')
            plt.ylabel('Layer Index')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{args.prefix}_attention_entropy.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error creating attention entropy heatmap: {e}")
    
    # 3. Effective rank by layer and head
    if effective_rank.numel() > 0:
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                effective_rank.cpu().numpy(), 
                cmap='YlGnBu', 
                annot=True, 
                fmt=".1f",
                cbar_kws={'label': 'Effective Rank'}
            )
            plt.title('Effective Rank by Layer and Head (Higher = More Complex Patterns)')
            plt.xlabel('Head Index')
            plt.ylabel('Layer Index')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{args.prefix}_effective_rank.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error creating effective rank heatmap: {e}")
    
    # 4. Visualize attention patterns for selected heads
    if attention_weights.numel() > 0 and len(attention_weights.shape) == 4:
        try:
            num_layers, num_heads = entropy.shape
            
            # Select interesting heads
            max_entropy_indices = np.unravel_index(entropy.cpu().argmax().item(), entropy.shape)
            min_entropy_indices = np.unravel_index(entropy.cpu().argmin().item(), entropy.shape)
            
            heads_to_visualize = {
                'highest_entropy': max_entropy_indices,
                'lowest_entropy': min_entropy_indices
            }
            
            # Add highest_rank if effective_rank is valid
            if effective_rank.numel() > 0:
                max_rank_indices = np.unravel_index(effective_rank.cpu().argmax().item(), effective_rank.shape)
                heads_to_visualize['highest_rank'] = max_rank_indices
            
            # Visualize each selected head
            for name, (layer_idx, head_idx) in heads_to_visualize.items():
                plt.figure(figsize=(8, 6))
                
                # Get attention pattern for this head
                attn = attention_weights[layer_idx, head_idx].cpu().numpy()
                
                # Display as heatmap
                im = plt.imshow(attn, cmap='viridis')
                plt.colorbar(im, label='Attention Weight')
                plt.title(f'Attention Pattern: Layer {layer_idx}, Head {head_idx} ({name})')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{args.prefix}_attention_pattern_{name}.png"), dpi=300)
                plt.close()
        except Exception as e:
            logger.error(f"Error visualizing attention patterns: {e}")
    
    # 5. If OSPA, plot orthogonality metrics
    if orthogonality_scores:
        try:
            # Extract layer-wise orthogonality errors
            layers = [int(k.split('_')[1]) for k in orthogonality_scores.keys() if k != 'overall_avg_error']
            
            if layers:
                layers.sort()
                
                # Collect metrics for each layer
                metrics = ['P_Q_error', 'P_K_error', 'P_V_error', 'P_O_error', 'avg_error']
                data = {metric: [] for metric in metrics}
                
                for layer in layers:
                    layer_scores = orthogonality_scores.get(f'layer_{layer}', {})
                    for metric in metrics:
                        data[metric].append(layer_scores.get(metric, float('nan')))
                
                # Plot
                plt.figure(figsize=(10, 6))
                
                for metric in metrics:
                    if any(not np.isnan(x) for x in data[metric]):
                        label = 'Average Error' if metric == 'avg_error' else f"{metric.split('_')[0]} Projection"
                        style = '*-' if metric == 'avg_error' else 'o-'
                        plt.plot(layers, data[metric], style, label=label)
                
                plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.xlabel('Layer Index')
                plt.ylabel('Orthogonality Error ||P^T P - I||_F^2')
                plt.title('OSPA Orthogonality Errors by Layer')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{args.prefix}_orthogonality_error.png"), dpi=300)
                plt.close()
        except Exception as e:
            logger.error(f"Error plotting orthogonality metrics: {e}")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return results

def load_model(args):
    """
    Load model with extensive error handling
    """
    logger.info(f"Loading model from {args.model_path}")
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Try to create a new model instance first
        model = TransformerModel(
            transformer_type='vanilla' if 'vanilla' in args.model_path else 'ospa',
            vocab_size=30000,  # We'll just use a placeholder
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=0.1,
            dim_feedforward=args.dim_feedforward,
            orth_mode='init',  # Default
            orth_penalty_weight=0.0,  # Default
            task=args.task
        )
        
        # Load state dict
        state_dict = torch.load(args.model_path, map_location='cpu')
        
        # Check if we got a state dict or a full model
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
        
        # Set model attributes from file path if possible
        if 'ospa' in args.model_path:
            model.transformer_type = 'ospa'
            if 'regularize' in args.model_path:
                model.orth_mode = 'regularize'
            elif 'strict' in args.model_path:
                model.orth_mode = 'strict'
            elif 'init' in args.model_path:
                model.orth_mode = 'init'
        else:
            model.transformer_type = 'vanilla'
            model.orth_mode = 'init'
        
        # Patch model with required analysis methods if needed
        model = patch_model_if_needed(model)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Model file exists but could not be loaded")
        logger.error("Trying to inspect model file...")
        
        try:
            # Try to load as generic state dict 
            state_dict = torch.load(args.model_path, map_location='cpu')
            logger.info(f"Model file loaded as a dictionary with {len(state_dict)} keys")
            
            # Print first few keys to help diagnosis
            sample_keys = list(state_dict.keys())[:5]
            logger.info(f"Sample keys: {sample_keys}")
            
            # Create a new minimal model and try to adjust keys
            logger.info("Attempting to create a minimal compatible model...")
            
            # Determine model type from filepath
            if 'ospa' in args.model_path.lower():
                transformer_type = 'ospa'
                if 'regularize' in args.model_path.lower():
                    orth_mode = 'regularize' 
                elif 'strict' in args.model_path.lower():
                    orth_mode = 'strict'
                else:
                    orth_mode = 'init'
            else:
                transformer_type = 'vanilla'
                orth_mode = 'init'
                
            logger.info(f"Creating model with type={transformer_type}, mode={orth_mode}")
            
            model = TransformerModel(
                transformer_type=transformer_type,
                vocab_size=30000,  # Just a placeholder
                d_model=args.d_model,
                nhead=args.nhead,
                nlayers=args.nlayers,
                dropout=0.1,
                dim_feedforward=args.dim_feedforward,
                orth_mode=orth_mode,
                orth_penalty_weight=0.0,
                task=args.task
            )
            
            # Set attributes
            model.transformer_type = transformer_type
            model.orth_mode = orth_mode
            
            # Patch model with required methods
            model = patch_model_if_needed(model)
            
            logger.info("Created model stub for analysis")
            return model
            
        except Exception as nested_e:
            logger.error(f"Could not inspect or create fallback model: {nested_e}")
            sys.exit(1)

def compare_models(args_list, output_dir):
    """
    Compare head diversity metrics across multiple models
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Comparing {len(args_list)} models")
    
    # Load models and collect metrics
    all_metrics = []
    
    for i, args in enumerate(args_list):
        logger.info(f"Analyzing model {i+1}/{len(args_list)}: {args.model_path}")
        
        try:
            # Load model
            model = load_model(args)
            
            # Create temp args with prefix for this model
            temp_args = argparse.Namespace(**vars(args))
            model_name = os.path.basename(args.model_path).split('.')[0]
            temp_args.prefix = model_name
            
            # Run analysis
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            temp_args.output_dir = model_dir
            
            metrics = evaluate_head_diversity(model, temp_args)
            metrics['model_name'] = model_name
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing model {args.model_path}: {e}")
            logger.error("Skipping this model in the comparison")
    
    if len(all_metrics) < 2:
        logger.error("Not enough models with valid metrics for comparison")
        return
    
    logger.info(f"Creating comparison visualizations for {len(all_metrics)} models")
    
    try:
        # Create comparison visualizations
        
        # 1. Bar chart of diversity scores
        plt.figure(figsize=(10, 6))
        model_names = [m['model_name'] for m in all_metrics]
        diversity_scores = [m['diversity_score'] for m in all_metrics]
        
        # Sort by model type first (vanilla, then ospa)
        sorted_indices = sorted(range(len(model_names)), 
                                key=lambda i: ('vanilla' in model_names[i], diversity_scores[i]))
        
        model_names = [model_names[i] for i in sorted_indices]
        diversity_scores = [diversity_scores[i] for i in sorted_indices]
        
        # Set colors based on model type
        colors = ['#ff9999' if 'vanilla' in name else '#66b3ff' for name in model_names]
        
        plt.bar(range(len(model_names)), diversity_scores, color=colors)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.xlabel('Model')
        plt.ylabel('Diversity Score (Lower is Better)')
        plt.title('Comparison of Attention Head Diversity Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_diversity_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Combined metrics comparison
        metrics_to_compare = ['avg_entropy', 'avg_effective_rank', 'diversity_score']
        metrics_labels = ['Average Entropy', 'Average Effective Rank', 'Diversity Score']
        
        # Normalize metrics for fair comparison
        normalized_metrics = []
        for metric in metrics_to_compare:
            values = [m[metric] for m in all_metrics]
            min_val, max_val = min(values), max(values)
            
            # Handle case where all values are the same
            if max_val == min_val:
                normalized = [0.5 for _ in values]
            else:
                # For diversity_score, lower is better, so invert the normalization
                if metric == 'diversity_score':
                    normalized = [1 - (v - min_val) / (max_val - min_val) for v in values]
                else:
                    normalized = [(v - min_val) / (max_val - min_val) for v in values]
            
            normalized_metrics.append(normalized)
        
        # Radar chart for combined metrics
        num_models = len(model_names)
        num_metrics = len(metrics_to_compare)
        
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        for i, model_name in enumerate(model_names):
            values = [normalized_metrics[j][i] for j in range(num_metrics)]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_ylim(0, 1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Normalized Metrics Comparison (Higher is Better)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radar_metrics_comparison.png"), dpi=300)
        plt.close()
        
        # Save comparison metrics to JSON
        comparison_results = {
            'models': model_names,
            'metrics': {
                metric: [m[metric] for m in all_metrics] for metric in metrics_to_compare
            }
        }
        
        with open(os.path.join(output_dir, "diversity_comparison_metrics.json"), 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info("Model comparison visualizations created")
    
    except Exception as e:
        logger.error(f"Error creating comparison visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze diversity of attention heads in transformer models')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--task', type=str, choices=['lm', 'classification'], default='lm',
                        help='Task type (language modeling or classification)')
    
    # Model architecture parameters (should match trained model)
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model embedding dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=6,
                        help='Number of transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Dimension of feedforward network')
    
    # Analysis parameters
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length for dummy data')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation')
    parser.add_argument('--prefix', type=str, default='model',
                        help='Prefix for output filenames')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Enable comparison mode between multiple models')
    parser.add_argument('--model_paths', type=str, nargs='+',
                        help='Paths to multiple model checkpoints for comparison')
    
    args = parser.parse_args()
    
    # Configure logging to file in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'analysis.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Analysis script started with args: {vars(args)}")
    
    if args.compare:
        if not args.model_paths or len(args.model_paths) < 2:
            logger.error("--compare requires at least two model paths via --model_paths")
            sys.exit(1)
        
        # Create args for each model
        args_list = []
        for model_path in args.model_paths:
            model_args = argparse.Namespace(**vars(args))
            model_args.model_path = model_path
            args_list.append(model_args)
        
        # Compare models
        compare_models(args_list, args.output_dir)
    else:
        # Single model analysis
        try:
            model = load_model(args)
            evaluate_head_diversity(model, args)
        except Exception as e:
            logger.error(f"Error in main analysis workflow: {e}")
            sys.exit(1)
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()