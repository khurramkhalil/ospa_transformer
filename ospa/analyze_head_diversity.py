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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

# Import model definitions
from improved_transformer_model import TransformerModel

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def compute_attention_patterns(model, data_loader, device, max_samples=100):
    """
    Compute attention patterns from model for a set of input sequences.
    Returns: attention patterns for each head in shape [layer, head, sample, seq_len, seq_len]
    """
    model.eval()
    
    # Initialize lists to store attention weights
    attention_weights = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Computing attention patterns")):
            if i >= max_samples:
                break
                
            # Handle different data formats
            if isinstance(batch, tuple) and len(batch) == 2:
                # Classification data loader returns (data, targets)
                data, _ = batch
                if data.dim() == 2:  # [batch_size, seq_len]
                    data = data.t()  # Convert to [seq_len, batch_size]
            else:
                # Language modeling data is already in [seq_len, batch_size]
                data = batch
                
            data = data.to(device)
            
            # Ensure model returns attention weights
            model.store_attention_weights = True
            
            # Forward pass
            _ = model(data)
            
            # Get stored attention weights
            batch_attention_weights = model.get_attention_weights()
            attention_weights.append(batch_attention_weights)
            
            # Reset attention storage
            model.store_attention_weights = False
    
    # Convert list of attention weights to a single tensor
    # Each element in attention_weights is [layer, head, seq_len, seq_len]
    all_attention_weights = torch.stack(attention_weights, dim=2)  # [layer, head, sample, seq_len, seq_len]
    
    return all_attention_weights

def compute_head_similarity_matrix(attention_weights):
    """
    Compute cosine similarity between attention heads.
    
    Args:
        attention_weights: Tensor of shape [layer, head, sample, seq_len, seq_len]
        
    Returns:
        similarity_matrix: Tensor of shape [layer*head, layer*head]
    """
    num_layers, num_heads, num_samples, seq_len, _ = attention_weights.shape
    
    # Reshape to [layer*head, sample*seq_len*seq_len]
    flattened_weights = attention_weights.reshape(num_layers * num_heads, -1)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(flattened_weights.cpu().numpy())
    
    return similarity_matrix

def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention distributions as a measure of focus.
    
    Args:
        attention_weights: Tensor of shape [layer, head, sample, seq_len, seq_len]
        
    Returns:
        entropy: Tensor of shape [layer, head]
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Ensure attention weights sum to 1 along the last dimension
    attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + epsilon)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(attention_weights * torch.log(attention_weights + epsilon)).sum(dim=-1)
    
    # Average over sequence length and samples
    mean_entropy = entropy.mean(dim=-1).mean(dim=-1)  # [layer, head]
    
    return mean_entropy

def compute_effective_rank(attention_weights):
    """
    Compute effective rank of attention matrices to measure dimensionality of features.
    
    Args:
        attention_weights: Tensor of shape [layer, head, sample, seq_len, seq_len]
        
    Returns:
        effective_rank: Tensor of shape [layer, head]
    """
    num_layers, num_heads, num_samples, seq_len, _ = attention_weights.shape
    effective_rank = torch.zeros(num_layers, num_heads)
    
    for l in range(num_layers):
        for h in range(num_heads):
            # Reshape to [sample*seq_len, seq_len]
            head_weights = attention_weights[l, h].reshape(-1, seq_len)
            
            # Compute SVD
            try:
                U, S, V = torch.svd(head_weights)
                
                # Normalize singular values
                normalized_S = S / torch.sum(S)
                
                # Compute entropy of normalized singular values as effective rank
                entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
                effective_rank[l, h] = torch.exp(entropy)
            except Exception as e:
                print(f"SVD failed for layer {l}, head {h}: {e}")
                effective_rank[l, h] = 0
    
    return effective_rank

def compute_orthogonality_metric(model):
    """
    Compute orthogonality metrics for OSPA projections.
    For vanilla transformer, returns None.
    
    Returns:
        orthogonality_scores: Dictionary of orthogonality scores
    """
    if not hasattr(model, 'transformer_type') or model.transformer_type != 'ospa':
        return None
    
    orthogonality_scores = {}
    
    # Get projection matrices from model
    for layer_idx, layer in enumerate(model.transformer_encoder.layers):
        # Access attention module
        mha = layer.self_attn
        
        if hasattr(mha, 'P_Q') and hasattr(mha, 'P_K') and hasattr(mha, 'P_V') and hasattr(mha, 'P_O'):
            # Compute orthogonality errors: ||P^T P - I||_F^2
            p_q_error = torch.norm(torch.matmul(mha.P_Q.transpose(-2, -1), mha.P_Q) - torch.eye(mha.P_Q.shape[-1], device=mha.P_Q.device), p='fro').item()
            p_k_error = torch.norm(torch.matmul(mha.P_K.transpose(-2, -1), mha.P_K) - torch.eye(mha.P_K.shape[-1], device=mha.P_K.device), p='fro').item()
            p_v_error = torch.norm(torch.matmul(mha.P_V.transpose(-2, -1), mha.P_V) - torch.eye(mha.P_V.shape[-1], device=mha.P_V.device), p='fro').item()
            p_o_error = torch.norm(torch.matmul(mha.P_O.transpose(-2, -1), mha.P_O) - torch.eye(mha.P_O.shape[-1], device=mha.P_O.device), p='fro').item()
            
            orthogonality_scores[f'layer_{layer_idx}'] = {
                'P_Q_error': p_q_error,
                'P_K_error': p_k_error,
                'P_V_error': p_v_error,
                'P_O_error': p_o_error,
                'avg_error': (p_q_error + p_k_error + p_v_error + p_o_error) / 4
            }
    
    # Add average across layers
    if orthogonality_scores:
        avg_error = np.mean([scores['avg_error'] for scores in orthogonality_scores.values()])
        orthogonality_scores['overall_avg_error'] = avg_error
    
    return orthogonality_scores

def compute_weight_correlation(model):
    """
    Compute correlations between Q/K/V/O weight matrices to check for redundancy.
    
    Returns:
        correlation_data: Dictionary of correlation metrics
    """
    correlation_data = {}
    
    # Access attention modules from each transformer layer
    for layer_idx, layer in enumerate(model.transformer_encoder.layers):
        # Access attention module
        mha = layer.self_attn
        
        # Get weight matrices
        if hasattr(mha, 'q_proj') and hasattr(mha, 'k_proj') and hasattr(mha, 'v_proj') and hasattr(mha, 'out_proj'):
            # Standard Transformer
            q_weight = mha.q_proj.weight.detach().cpu().numpy()
            k_weight = mha.k_proj.weight.detach().cpu().numpy()
            v_weight = mha.v_proj.weight.detach().cpu().numpy()
            o_weight = mha.out_proj.weight.detach().cpu().numpy()
            
            # Compute correlations between flattened weights
            weights = {
                'Q': q_weight.flatten(),
                'K': k_weight.flatten(),
                'V': v_weight.flatten(),
                'O': o_weight.flatten()
            }
            
            # Compute correlation matrix
            corr_matrix = {}
            for name1, w1 in weights.items():
                corr_matrix[name1] = {}
                for name2, w2 in weights.items():
                    if name1 != name2:
                        corr, _ = scipy.stats.pearsonr(w1, w2)
                        corr_matrix[name1][name2] = corr
            
            correlation_data[f'layer_{layer_idx}'] = corr_matrix
    
    return correlation_data

def evaluate_head_diversity(model, data_loader, output_dir, args):
    """
    Main function to evaluate and visualize head diversity metrics.
    """
    device = torch.device(args.device)
    model.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute attention patterns
    attention_weights = compute_attention_patterns(
        model, data_loader, device, max_samples=args.max_samples
    )
    
    # 2. Compute head similarity matrix
    similarity_matrix = compute_head_similarity_matrix(attention_weights)
    
    # 3. Compute attention entropy
    entropy = compute_attention_entropy(attention_weights)
    
    # 4. Compute effective rank
    effective_rank = compute_effective_rank(attention_weights)
    
    # 5. Compute orthogonality metrics (only for OSPA)
    orthogonality_scores = compute_orthogonality_metric(model)
    
    # 6. Compute weight correlations
    weight_correlations = compute_weight_correlation(model)
    
    # Compile results into a dictionary
    results = {
        'model_type': getattr(model, 'transformer_type', 'unknown'),
        'avg_similarity': float(np.mean(similarity_matrix) - np.mean(np.diag(similarity_matrix))),
        'avg_entropy': float(entropy.mean().item()),
        'avg_effective_rank': float(effective_rank.mean().item()),
        'orthogonality_scores': orthogonality_scores,
        'weight_correlations': weight_correlations
    }
    
    # Calculate diversity score (lower is more diverse)
    diversity_score = float(np.mean(similarity_matrix) - np.mean(np.diag(similarity_matrix)))
    results['diversity_score'] = diversity_score
    
    # Save numeric results
    with open(os.path.join(output_dir, f"{args.prefix}_diversity_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # ===============================
    # Create visualizations
    # ===============================
    
    # 1. Head similarity heatmap
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
    
    print(f"Model comparison complete. Results saved to {output_dir}")

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
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to process for attention patterns')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--prefix', type=str, default='model',
                        help='Prefix for output filenames')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Enable comparison mode between multiple models')
    parser.add_argument('--model_paths', type=str, nargs='+',
                        help='Paths to multiple model checkpoints for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.model_paths or len(args.model_paths) < 2:
            parser.error("--compare requires at least two model paths via --model_paths")
        
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
        model, data_loader, task = load_and_configure_model(args)
        evaluate_head_diversity(model, data_loader, args.output_dir, args)

if __name__ == "__main__":
    main()figure(figsize=(10, 8))
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
    
    # 2. Attention entropy by layer and head
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
    
    # 3. Effective rank by layer and head
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
    
    # 4. Visualize attention patterns for selected heads
    num_layers, num_heads = entropy.shape
    
    # Select a few interesting heads:
    # 1. Head with highest entropy
    # 2. Head with lowest entropy
    # 3. Head with highest effective rank
    max_entropy_indices = np.unravel_index(entropy.cpu().argmax(), entropy.shape)
    min_entropy_indices = np.unravel_index(entropy.cpu().argmin(), entropy.shape)
    max_rank_indices = np.unravel_index(effective_rank.cpu().argmax(), effective_rank.shape)
    
    heads_to_visualize = {
        'highest_entropy': max_entropy_indices,
        'lowest_entropy': min_entropy_indices,
        'highest_rank': max_rank_indices
    }
    
    # Sample a few sequence samples for visualization
    num_samples = min(5, attention_weights.shape[2])
    for name, (layer_idx, head_idx) in heads_to_visualize.items():
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
        for i in range(num_samples):
            # Get attention pattern for this head and sample
            attn = attention_weights[layer_idx, head_idx, i].cpu().numpy()
            
            # Display as heatmap
            if num_samples > 1:
                ax = axes[i]
            else:
                ax = axes
            
            im = ax.imshow(attn, cmap='viridis')
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
            
            # Add colorbar to last plot
            if i == num_samples - 1:
                plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Attention Patterns: Layer {layer_idx}, Head {head_idx} ({name})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}_attention_pattern_{name}.png"), dpi=300)
        plt.close()
    
    # 5. If OSPA, plot orthogonality metrics
    if orthogonality_scores:
        # Extract layer-wise orthogonality errors
        layers = [int(k.split('_')[1]) for k in orthogonality_scores.keys() if k != 'overall_avg_error']
        layers.sort()
        
        avg_errors = [orthogonality_scores[f'layer_{layer}']['avg_error'] for layer in layers]
        q_errors = [orthogonality_scores[f'layer_{layer}']['P_Q_error'] for layer in layers]
        k_errors = [orthogonality_scores[f'layer_{layer}']['P_K_error'] for layer in layers]
        v_errors = [orthogonality_scores[f'layer_{layer}']['P_V_error'] for layer in layers]
        o_errors = [orthogonality_scores[f'layer_{layer}']['P_O_error'] for layer in layers]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, q_errors, 'o-', label='Query Projection')
        plt.plot(layers, k_errors, 's-', label='Key Projection')
        plt.plot(layers, v_errors, '^-', label='Value Projection')
        plt.plot(layers, o_errors, 'D-', label='Output Projection')
        plt.plot(layers, avg_errors, '*-', label='Average Error', linewidth=2)
        
        plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Layer Index')
        plt.ylabel('Orthogonality Error ||P^T P - I||_F^2')
        plt.title('OSPA Orthogonality Errors by Layer')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}_orthogonality_error.png"), dpi=300)
        plt.close()
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return results

def load_dataset_for_analysis(args):
    """
    Load a small portion of dataset for analysis.
    """
    if args.task == 'lm':
        # For language modeling, use WikiText-2
        from torch.utils.data import DataLoader
        from datasets import load_dataset
        
        print("Loading WikiText-2 dataset sample for analysis...")
        wikitext = load_dataset("wikitext", "wikitext-2-v1")
        
        # Simple tokenizer
        def tokenize(text):
            return text.split()
        
        # Create a small vocabulary
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Process a sample of text for analysis
        data = []
        for i, text in enumerate(wikitext['validation']['text']):
            if i >= 50:  # Limit to 50 samples
                break
            if text.strip():
                tokens = [vocab.get(token, 0) for token in tokenize(text)]
                if tokens:
                    data.append(torch.tensor(tokens, dtype=torch.long))
        
        # Batch the data
        def collate_fn(batch):
            # Pad sequences to same length
            max_len = max(len(x) for x in batch)
            batch_padded = []
            for x in batch:
                if len(x) < max_len:
                    padded = torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
                else:
                    padded = x
                batch_padded.append(padded)
            return torch.stack(batch_padded, dim=1)  # [seq_len, batch_size]
        
        # Create a dataloader
        dataloader = DataLoader(data, batch_size=4, collate_fn=collate_fn)
        return dataloader, len(vocab)
    
    else:  # classification
        # For classification, use IMDB
        from torch.utils.data import DataLoader
        from datasets import load_dataset
        
        print("Loading IMDB dataset sample for analysis...")
        imdb = load_dataset("imdb")
        
        # Take a small sample
        sample = imdb['test'].select(range(min(50, len(imdb['test']))))
        
        # Simple tokenizer
        def tokenize(text):
            return text.split()
        
        # Create a small vocabulary
        special_tokens = ['<unk>', '<pad>']
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        pad_idx = vocab['<pad>']
        
        # Define collate function
        def collate_batch(batch):
            label_list, text_list = [], []
            max_len = 512  # Max sequence length
            
            for example in batch:
                label_list.append(example['label'])
                # Tokenize and convert to indices
                processed_text = [vocab.get(token, 0) for token in tokenize(example['text'])]
                processed_text_tensor = torch.tensor(processed_text, dtype=torch.long)
                
                # Truncate or pad
                if len(processed_text_tensor) > max_len:
                    processed_text_tensor = processed_text_tensor[:max_len]
                else:
                    padding = torch.full((max_len - len(processed_text_tensor),), pad_idx, dtype=torch.long)
                    processed_text_tensor = torch.cat([processed_text_tensor, padding])
                
                text_list.append(processed_text_tensor)
            
            label_tensor = torch.tensor(label_list, dtype=torch.long)
            text_tensor = torch.stack(text_list, dim=0)  # [batch_size, seq_len]
            return text_tensor.t().contiguous(), label_tensor  # [seq_len, batch_size], [batch_size]
        
        # Create a dataloader
        dataloader = DataLoader(
            sample,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_batch
        )
        
        return dataloader, len(vocab)

def load_and_configure_model(args):
    """
    Load the transformer model for analysis.
    """
    # Determine model parameters from the checkpoint filename
    model_filename = os.path.basename(args.model_path)
    model_params = model_filename.split('.')[0].split('_')
    
    # Parse parameters
    task = model_params[0]  # 'lm' or 'classification'
    transformer_type = model_params[1]  # 'ospa' or 'vanilla'
    orth_mode = model_params[2]  # 'init', 'regularize', or 'strict'
    
    # Determine lambda value if applicable
    if transformer_type == 'ospa' and orth_mode == 'regularize':
        lambda_val = float(model_params[3].replace('lambda', ''))
    else:
        lambda_val = 0.0
    
    print(f"Analyzing model: {transformer_type.upper()} transformer")
    print(f"Task: {task}, Orthogonality mode: {orth_mode}")
    if transformer_type == 'ospa' and orth_mode == 'regularize':
        print(f"Orthogonality penalty weight (lambda): {lambda_val}")
    
    # Create a minimal dataset to infer model parameters
    data_loader, vocab_size = load_dataset_for_analysis(args)
    
    # Create model instance
    model = TransformerModel(
        transformer_type=transformer_type,
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=0.1,  # Not important for analysis
        dim_feedforward=args.dim_feedforward,
        orth_mode=orth_mode,
        orth_penalty_weight=lambda_val,
        task=task
    )
    
    # Load trained weights
    print(f"Loading model weights from {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise e
    
    # Set model to eval mode
    model.eval()
    
    return model, data_loader, task

def load_multiple_models_for_comparison(args_list):
    """
    Load multiple models for comparison of head diversity.
    
    Args:
        args_list: List of argument objects for different models
        
    Returns:
        models_data: List of tuples (model, data_loader, name)
    """
    models_data = []
    
    for args in args_list:
        model, data_loader, task = load_and_configure_model(args)
        model_name = os.path.basename(args.model_path).split('.')[0]
        models_data.append((model, data_loader, model_name))
    
    return models_data

def compare_models(args_list, output_dir):
    """
    Compare head diversity metrics across multiple models.
    
    Args:
        args_list: List of argument objects for different models
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    models_data = load_multiple_models_for_comparison(args_list)
    
    # Collect diversity metrics for each model
    all_metrics = []
    
    for model, data_loader, model_name in models_data:
        print(f"\nAnalyzing model: {model_name}")
        
        # Create temporary args object with prefix set to model_name
        temp_args = argparse.Namespace()
        temp_args.__dict__.update(args_list[0].__dict__)
        temp_args.prefix = model_name
        
        # Evaluate diversity metrics
        metrics = evaluate_head_diversity(model, data_loader, output_dir, temp_args)
        metrics['model_name'] = model_name
        all_metrics.append(metrics)
    
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
    plt.show()