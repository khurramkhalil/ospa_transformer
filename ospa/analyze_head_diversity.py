#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def extract_weights_from_state_dict(state_dict, args):
    """
    Extract attention weights directly from state dict without loading the model
    """
    # Determine model structure based on state dict keys
    keys = list(state_dict.keys())
    
    # Identify key prefixes to understand structure
    encoder_prefix = None
    for potential_prefix in ['transformer.encoder', 'encoder', 'transformer_encoder']:
        pattern = potential_prefix + ".*self_attn"
        matching_keys = [k for k in keys if re.search(pattern, k)]
        if matching_keys:
            encoder_prefix = potential_prefix
            logger.info(f"Detected encoder prefix: {encoder_prefix}")
            break
    
    if not encoder_prefix:
        logger.error("Could not identify encoder prefix in state dict")
        return []
    
    # Find layer pattern
    layer_pattern = None
    patterns_to_try = [
        encoder_prefix + r"\.layers\.(\d+)",
        encoder_prefix + r"\.layer\.(\d+)",
        encoder_prefix + r"\.(\d+)"
    ]
    
    for pattern in patterns_to_try:
        for key in keys:
            match = re.search(pattern, key)
            if match:
                layer_pattern = pattern
                logger.info(f"Detected layer pattern: {layer_pattern}")
                break
        if layer_pattern:
            break
    
    if not layer_pattern:
        logger.error("Could not identify layer pattern in state dict")
        return []
    
    # Identify attention projection keys
    attn_q_keys = [k for k in keys if 'q_proj.weight' in k or 'query.weight' in k]
    attn_k_keys = [k for k in keys if 'k_proj.weight' in k or 'key.weight' in k]
    attn_v_keys = [k for k in keys if 'v_proj.weight' in k or 'value.weight' in k]
    
    # Check if this is OSPA (has P_Q, P_K, etc.)
    ospa_keys = [k for k in keys if any(p in k for p in ['P_Q', 'P_K', 'P_V', 'P_O'])]
    is_ospa = len(ospa_keys) > 0
    
    if is_ospa:
        logger.info("Detected OSPA model")
        # Find OSPA projection keys
        p_q_keys = [k for k in keys if 'P_Q' in k]
        p_k_keys = [k for k in keys if 'P_K' in k]
        p_v_keys = [k for k in keys if 'P_V' in k]
        
        if not (p_q_keys and p_k_keys and p_v_keys):
            logger.error("Missing OSPA projection keys")
            return []
    
    # Group keys by layer
    layer_weights = []
    
    # Determine number of layers
    num_layers = 0
    for key in keys:
        match = re.search(layer_pattern, key)
        if match:
            layer_idx = int(match.group(1))
            num_layers = max(num_layers, layer_idx + 1)
    
    if not num_layers:
        logger.error("Could not determine number of layers")
        return []
    
    logger.info(f"Detected {num_layers} layers")
    
    # Process each layer
    for layer_idx in range(num_layers):
        if is_ospa:
            # Find OSPA projections for this layer
            p_q_key = None
            p_k_key = None
            p_v_key = None
            
            layer_pattern_str = layer_pattern.replace(r'(\d+)', str(layer_idx))
            
            for key in p_q_keys:
                if re.search(layer_pattern_str, key):
                    p_q_key = key
                    break
            
            for key in p_k_keys:
                if re.search(layer_pattern_str, key):
                    p_k_key = key
                    break
            
            for key in p_v_keys:
                if re.search(layer_pattern_str, key):
                    p_v_key = key
                    break
            
            if p_q_key and p_k_key and p_v_key:
                try:
                    p_q = state_dict[p_q_key]
                    p_k = state_dict[p_k_key]
                    p_v = state_dict[p_v_key]
                    
                    layer_weights.append((p_q, p_k, p_v, True))
                    logger.info(f"Extracted OSPA weights for layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Error extracting OSPA weights for layer {layer_idx}: {e}")
            else:
                logger.warning(f"Missing OSPA projection keys for layer {layer_idx}")
        else:
            # Find attention projections for this layer
            q_key = None
            k_key = None
            v_key = None
            
            layer_pattern_str = layer_pattern.replace(r'(\d+)', str(layer_idx))
            
            for key in attn_q_keys:
                if re.search(layer_pattern_str, key):
                    q_key = key
                    break
            
            for key in attn_k_keys:
                if re.search(layer_pattern_str, key):
                    k_key = key
                    break
            
            for key in attn_v_keys:
                if re.search(layer_pattern_str, key):
                    v_key = key
                    break
            
            if q_key and k_key and v_key:
                try:
                    q_weight = state_dict[q_key]
                    k_weight = state_dict[k_key]
                    v_weight = state_dict[v_key]
                    
                    layer_weights.append((q_weight, k_weight, v_weight, False))
                    logger.info(f"Extracted vanilla weights for layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Error extracting vanilla weights for layer {layer_idx}: {e}")
            else:
                logger.warning(f"Missing projection keys for layer {layer_idx}")
    
    return layer_weights

def compute_head_similarity(qkv_weights, nhead):
    """
    Compute similarity between attention heads
    """
    if not qkv_weights:
        logger.error("No weights to compute similarity from")
        return np.zeros((1, 1)), 0
    
    # Process weights into head representations
    flattened_heads = []
    
    for layer_idx, (q, k, v, is_ospa) in enumerate(qkv_weights):
        try:
            if is_ospa:
                # For OSPA: P_Q, P_K, P_V are orthogonal matrices
                # Reshape based on number of heads
                d_model = q.size(0)
                head_dim = d_model // nhead
                
                # Split projection matrices by head dimensions
                for head_idx in range(nhead):
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    
                    # Get slice of projection matrices for this head
                    q_head = q[start_idx:end_idx].flatten()
                    k_head = k[start_idx:end_idx].flatten()
                    v_head = v[start_idx:end_idx].flatten()
                    
                    # Concatenate to get head representation
                    head_vector = torch.cat([q_head, k_head, v_head])
                    flattened_heads.append(head_vector.cpu().numpy())
            else:
                # For vanilla transformer with shape [nhead*head_dim, d_model]
                d_model = q.size(1) if q.dim() > 1 else int(np.sqrt(q.size(0)))
                
                # Try to figure out head dimension
                if q.dim() > 1:
                    head_dim = q.size(0) // nhead
                else:
                    head_dim = d_model // nhead
                
                # Try to reshape
                try:
                    if q.dim() > 1:
                        # If already in form [nhead*head_dim, d_model]
                        q_reshaped = q.view(nhead, head_dim, d_model)
                        k_reshaped = k.view(nhead, head_dim, d_model)
                        v_reshaped = v.view(nhead, head_dim, d_model)
                    else:
                        # If flattened
                        q_reshaped = q.view(nhead, head_dim, d_model)
                        k_reshaped = k.view(nhead, head_dim, d_model)
                        v_reshaped = v.view(nhead, head_dim, d_model)
                except:
                    # Try alternate dimension
                    logger.warning(f"Reshape failed, trying alternate dimensions for layer {layer_idx}")
                    total_size = q.numel()
                    head_dim = total_size // (nhead * d_model)
                    
                    q_reshaped = q.view(nhead, head_dim, d_model)
                    k_reshaped = k.view(nhead, head_dim, d_model)
                    v_reshaped = v.view(nhead, head_dim, d_model)
                
                # Process each head
                for head_idx in range(nhead):
                    # Concatenate q, k, v for this head and flatten
                    head_vector = torch.cat([
                        q_reshaped[head_idx].flatten(), 
                        k_reshaped[head_idx].flatten(), 
                        v_reshaped[head_idx].flatten()
                    ])
                    flattened_heads.append(head_vector.cpu().numpy())
        except Exception as e:
            logger.error(f"Error processing layer {layer_idx}: {e}")
    
    if not flattened_heads:
        logger.error("Failed to process any heads")
        return np.zeros((1, 1)), 0
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(flattened_heads)
    
    # Compute diversity score (average off-diagonal similarity)
    diversity_score = float(np.mean(similarity_matrix) - np.mean(np.diag(similarity_matrix)))
    
    return similarity_matrix, diversity_score

def check_orthogonality(qkv_weights):
    """
    Check orthogonality of OSPA projection matrices
    """
    if not qkv_weights:
        return None
    
    orthogonality_scores = {}
    
    for layer_idx, (q, k, v, is_ospa) in enumerate(qkv_weights):
        if not is_ospa:
            continue
            
        try:
            # Compute orthogonality error: ||P^T P - I||_F^2
            q_error = torch.norm(
                torch.matmul(q.transpose(-2, -1), q) - 
                torch.eye(q.shape[-1], device=q.device), 
                p='fro'
            ).item()
            
            k_error = torch.norm(
                torch.matmul(k.transpose(-2, -1), k) - 
                torch.eye(k.shape[-1], device=k.device), 
                p='fro'
            ).item()
            
            v_error = torch.norm(
                torch.matmul(v.transpose(-2, -1), v) - 
                torch.eye(v.shape[-1], device=v.device), 
                p='fro'
            ).item()
            
            # Add to results
            orthogonality_scores[f'layer_{layer_idx}'] = {
                'P_Q_error': q_error,
                'P_K_error': k_error,
                'P_V_error': v_error,
                'avg_error': (q_error + k_error + v_error) / 3
            }
            
        except Exception as e:
            logger.error(f"Error computing orthogonality for layer {layer_idx}: {e}")
    
    # Compute overall average
    if orthogonality_scores:
        avg_errors = [scores['avg_error'] for scores in orthogonality_scores.values()]
        orthogonality_scores['overall_avg_error'] = sum(avg_errors) / len(avg_errors)
        
    return orthogonality_scores

def analyze_model(model_path, output_dir, args):
    """
    Analyze head diversity for a single model
    """
    logger.info(f"Analyzing model: {model_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        logger.info(f"Successfully loaded state dict from {model_path}")
        
        # If state_dict has a 'state_dict' key, use that
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            logger.info("Extracted nested state_dict")
            
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        return None
    
    # Extract weights directly from state dict
    qkv_weights = extract_weights_from_state_dict(state_dict, args)
    
    if not qkv_weights:
        logger.error("Failed to extract weights from model")
        return None
    
    # Check if this is an OSPA model
    is_ospa = any(is_ospa for _, _, _, is_ospa in qkv_weights)
    model_type = 'ospa' if is_ospa else 'vanilla'
    logger.info(f"Detected model type: {model_type}")
    
    # Compute similarity
    similarity_matrix, diversity_score = compute_head_similarity(qkv_weights, args.nhead)
    
    # Check orthogonality (for OSPA models)
    orthogonality_scores = check_orthogonality(qkv_weights) if is_ospa else None
    
    # Compile results
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'diversity_score': diversity_score,
        'num_layers': len(qkv_weights),
        'num_heads': args.nhead,
        'orthogonality_scores': orthogonality_scores
    }
    
    # Save results as JSON
    with open(os.path.join(output_dir, "diversity_metrics.json"), 'w') as f:
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
        plt.savefig(os.path.join(output_dir, "head_similarity.png"), dpi=300)
        plt.close()
        logger.info(f"Created head similarity visualization")
    except Exception as e:
        logger.error(f"Error creating head similarity visualization: {e}")
    
    # 2. Plot orthogonality metrics (for OSPA models)
    if orthogonality_scores and len(orthogonality_scores) > 1:  # More than just overall_avg_error
        try:
            layers = [int(k.split('_')[1]) for k in orthogonality_scores.keys() if k != 'overall_avg_error']
            layers.sort()
            
            metrics = ['P_Q_error', 'P_K_error', 'P_V_error', 'avg_error']
            data = {metric: [] for metric in metrics}
            
            for layer in layers:
                scores = orthogonality_scores[f'layer_{layer}']
                for metric in metrics:
                    data[metric].append(scores[metric])
            
            plt.figure(figsize=(10, 6))
            
            for metric, values in data.items():
                label = 'Average Error' if metric == 'avg_error' else f"{metric.split('_')[0]} Projection"
                style = '*-' if metric == 'avg_error' else 'o-'
                plt.plot(layers, values, style, label=label)
            
            plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Layer Index')
            plt.ylabel('Orthogonality Error ||P^T P - I||_F^2')
            plt.title('OSPA Orthogonality Errors by Layer')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "orthogonality_error.png"), dpi=300)
            plt.close()
            logger.info("Created orthogonality visualization")
        except Exception as e:
            logger.error(f"Error creating orthogonality visualization: {e}")
    
    logger.info(f"Analysis complete for {model_path}")
    logger.info(f"Diversity score: {diversity_score:.4f} (lower is better)")
    
    return results

def compare_models(model_results, output_dir):
    """
    Create comparison visualizations for multiple models
    """
    if len(model_results) < 2:
        logger.error("Need at least 2 models to compare")
        return
    
    logger.info(f"Creating comparison visualizations for {len(model_results)} models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and diversity scores
    model_names = []
    diversity_scores = []
    model_types = []
    
    for model_path, results in model_results.items():
        if results is None:
            continue
            
        model_name = os.path.basename(model_path).split('.')[0]
        model_names.append(model_name)
        diversity_scores.append(results['diversity_score'])
        model_types.append(results['model_type'])
    
    if len(model_names) < 2:
        logger.error("Not enough models with valid results to compare")
        return
    
    # Sort by model type first (vanilla, then ospa), then by diversity score
    sorted_indices = sorted(range(len(model_names)), 
                          key=lambda i: (model_types[i] == 'vanilla', diversity_scores[i]))
    
    model_names = [model_names[i] for i in sorted_indices]
    diversity_scores = [diversity_scores[i] for i in sorted_indices]
    model_types = [model_types[i] for i in sorted_indices]
    
    # Create bar chart of diversity scores
    try:
        plt.figure(figsize=(10, 6))
        
        # Set colors based on model type
        colors = ['#ff9999' if t == 'vanilla' else '#66b3ff' for t in model_types]
        
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
        logger.info("Created diversity comparison visualization")
    except Exception as e:
        logger.error(f"Error creating diversity comparison: {e}")
    
    # Save comparison data
    comparison_data = {
        'models': model_names,
        'diversity_scores': diversity_scores,
        'model_types': model_types
    }
    
    with open(os.path.join(output_dir, "diversity_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info("Model comparison complete")

def main():
    parser = argparse.ArgumentParser(description='Analyze attention head diversity in transformer models')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'], help='Task type')
    
    # Compare mode
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--model_paths', nargs='+', help='Paths to multiple models for comparison')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare:
        if not args.model_paths or len(args.model_paths) < 2:
            logger.error("--compare requires at least two model paths via --model_paths")
            return
        
        # Analyze multiple models
        results = {}
        for model_path in args.model_paths:
            model_name = os.path.basename(model_path).split('.')[0]
            model_dir = os.path.join(args.output_dir, model_name)
            results[model_path] = analyze_model(model_path, model_dir, args)
        
        # Create comparison visualizations
        compare_dir = os.path.join(args.output_dir, "comparison")
        compare_models(results, compare_dir)
        
    else:
        if not args.model_path:
            logger.error("--model_path is required when not in compare mode")
            return
            
        # Analyze single model
        analyze_model(args.model_path, args.output_dir, args)
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()