"""
Utility functions for OSPA analysis and visualization.

This module provides tools for analyzing orthogonality and
visualizing attention patterns in OSPA transformer models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Union


def measure_orthogonality(weight_matrix: torch.Tensor) -> float:
    """
    Measure how orthogonal a weight matrix is.
    
    For a perfectly orthogonal matrix W, W*W^T should be the identity matrix.
    This function calculates the deviation from identity.
    
    Args:
        weight_matrix: Weight matrix to analyze
        
    Returns:
        orthogonality_score: Lower is more orthogonal (0 is perfectly orthogonal)
    """
    # If matrix is not square, use the minimum dimension
    rows, cols = weight_matrix.shape
    min_dim = min(rows, cols)
    
    if rows > cols:
        # Tall matrix: W^T * W should approximate identity
        product = torch.matmul(weight_matrix.t(), weight_matrix)
    else:
        # Wide or square matrix: W * W^T should approximate identity
        product = torch.matmul(weight_matrix, weight_matrix.t())
    
    # Get identity matrix of appropriate size
    identity = torch.eye(product.size(0), device=weight_matrix.device)
    
    # Calculate Frobenius norm of difference
    orthogonality_score = torch.norm(product - identity, p='fro').item()
    
    # Normalize by dimensions
    orthogonality_score /= min_dim
    
    return orthogonality_score


def analyze_orthogonality(model, detailed=False) -> Dict:
    """
    Analyze orthogonality of projection matrices in OSPA model.
    
    Args:
        model: OSPA transformer model
        detailed: Whether to return detailed per-layer metrics
        
    Returns:
        metrics: Dictionary of orthogonality metrics
    """
    metrics = {}
    layer_metrics = []
    
    # Iterate through encoder layers
    for i, layer in enumerate(model.encoder.layers):
        layer_metric = {}
        
        # Q, K, V projections
        q_ortho = measure_orthogonality(layer.self_attn.q_proj.weight)
        k_ortho = measure_orthogonality(layer.self_attn.k_proj.weight)
        v_ortho = measure_orthogonality(layer.self_attn.v_proj.weight)
        o_ortho = measure_orthogonality(layer.self_attn.o_proj.weight)
        
        layer_metric['q_proj'] = q_ortho
        layer_metric['k_proj'] = k_ortho
        layer_metric['v_proj'] = v_ortho
        layer_metric['o_proj'] = o_ortho
        layer_metric['avg'] = (q_ortho + k_ortho + v_ortho + o_ortho) / 4
        
        layer_metrics.append(layer_metric)
    
    # Calculate average metrics across layers
    avg_q = np.mean([m['q_proj'] for m in layer_metrics])
    avg_k = np.mean([m['k_proj'] for m in layer_metrics])
    avg_v = np.mean([m['v_proj'] for m in layer_metrics])
    avg_o = np.mean([m['o_proj'] for m in layer_metrics])
    avg_all = np.mean([m['avg'] for m in layer_metrics])
    
    metrics['avg_q_ortho'] = avg_q
    metrics['avg_k_ortho'] = avg_k
    metrics['avg_v_ortho'] = avg_v
    metrics['avg_o_ortho'] = avg_o
    metrics['avg_overall'] = avg_all
    
    if detailed:
        metrics['layers'] = layer_metrics
    
    return metrics


def analyze_head_specialization(attention_weights: torch.Tensor) -> float:
    """
    Analyze how specialized each attention head is.
    
    Specialized heads have more focused (peaky) attention patterns.
    
    Args:
        attention_weights: Attention weights from an OSPA layer,
                          shape [batch_size, num_heads, seq_len, seq_len]
        
    Returns:
        specialization_score: Average entropy-based specialization (lower is more specialized)
    """
    # Average across batch dimension
    avg_attention = attention_weights.mean(dim=0)  # [num_heads, seq_len, seq_len]
    
    # Calculate entropy for each head
    entropy_per_head = []
    for head_idx in range(avg_attention.shape[0]):
        head_attention = avg_attention[head_idx]  # [seq_len, seq_len]
        
        # Calculate entropy per query position
        entropy_per_query = []
        for query_idx in range(head_attention.shape[0]):
            query_attn = head_attention[query_idx]  # [seq_len]
            
            # Skip if all attention is masked
            if torch.all(query_attn == 0):
                continue
                
            # Calculate entropy: -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            entropy = -torch.sum(query_attn * torch.log(query_attn + eps))
            entropy_per_query.append(entropy.item())
        
        # Average entropy across query positions
        if entropy_per_query:
            avg_entropy = np.mean(entropy_per_query)
            entropy_per_head.append(avg_entropy)
    
    # Average across heads
    if entropy_per_head:
        avg_specialization = np.mean(entropy_per_head)
        return avg_specialization
    else:
        return 0.0


def analyze_head_diversity(attention_weights: torch.Tensor) -> float:
    """
    Analyze how diverse the attention patterns are across heads.
    
    Diverse heads attend to different parts of the input.
    
    Args:
        attention_weights: Attention weights from an OSPA layer,
                          shape [batch_size, num_heads, seq_len, seq_len]
        
    Returns:
        diversity_score: Average cosine similarity between heads (lower is more diverse)
    """
    # Average across batch dimension
    avg_attention = attention_weights.mean(dim=0)  # [num_heads, seq_len, seq_len]
    
    num_heads = avg_attention.shape[0]
    
    # Flatten spatial dimensions for comparison
    flat_attention = avg_attention.reshape(num_heads, -1)  # [num_heads, seq_len*seq_len]
    
    # Normalize for cosine similarity
    norm_attention = flat_attention / (torch.norm(flat_attention, dim=1, keepdim=True) + 1e-10)
    
    # Calculate pairwise cosine similarities
    similarities = torch.matmul(norm_attention, norm_attention.t())  # [num_heads, num_heads]
    
    # Remove self-similarities (diagonal)
    mask = ~torch.eye(num_heads, dtype=torch.bool, device=similarities.device)
    similarities = similarities[mask].reshape(num_heads, num_heads - 1)
    
    # Average similarity (lower is more diverse)
    avg_similarity = similarities.mean().item()
    
    return avg_similarity


def visualize_attention(attention_weights: torch.Tensor, tokens: List[str] = None,
                       layer_idx: int = 0, head_idx: Optional[int] = None,
                       cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize attention patterns.
    
    Args:
        attention_weights: Attention weights, can be:
                          - From a single head: [seq_len, seq_len]
                          - From all heads in a layer: [num_heads, seq_len, seq_len]
                          - From all layers and heads: [num_layers, num_heads, seq_len, seq_len]
        tokens: Optional list of tokens for axis labels
        layer_idx: Layer index to visualize if attention_weights includes multiple layers
        head_idx: Head index to visualize (None to show all heads)
        cmap: Colormap name
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Handle different input formats
    if len(attention_weights.shape) == 4:
        # [num_layers, num_heads, seq_len, seq_len]
        attn = attention_weights[layer_idx]
    elif len(attention_weights.shape) == 3:
        # [num_heads, seq_len, seq_len]
        attn = attention_weights
    elif len(attention_weights.shape) == 2:
        # [seq_len, seq_len]
        attn = attention_weights.unsqueeze(0)  # Add head dimension
    else:
        raise ValueError(f"Unexpected attention weight shape: {attention_weights.shape}")
    
    # Convert to numpy for plotting
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    
    # Get dimensions
    num_heads, seq_len, _ = attn.shape
    
    # Prepare tokens for axis labels
    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]
    else:
        # Truncate token list if necessary
        tokens = tokens[:seq_len]
    
    # Create plot
    if head_idx is not None:
        # Plot single head
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn[head_idx], cmap=cmap)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set axis labels
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        
        ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
    else:
        # Plot all heads in a grid
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
        # Flatten axes for easy indexing
        axes = axes.flatten()
        
        for h in range(num_heads):
            if h < len(axes):
                im = axes[h].imshow(attn[h], cmap=cmap)
                axes[h].set_title(f"Head {h}")
                
                # Set axis labels for border plots only
                if h >= len(axes) - grid_size:  # Bottom row
                    axes[h].set_xticks(np.arange(len(tokens)))
                    axes[h].set_xticklabels(tokens, rotation=90)
                else:
                    axes[h].set_xticks([])
                
                if h % grid_size == 0:  # Leftmost column
                    axes[h].set_yticks(np.arange(len(tokens)))
                    axes[h].set_yticklabels(tokens)
                else:
                    axes[h].set_yticks([])
        
        # Hide unused subplots
        for h in range(num_heads, len(axes)):
            axes[h].axis('off')
        
        fig.suptitle(f"Attention Patterns - Layer {layer_idx}")
        plt.tight_layout()
    
    return fig


def visualize_orthogonality(model, figsize=(12, 8)):
    """
    Visualize orthogonality of projection matrices across layers.
    
    Args:
        model: OSPA transformer model
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Analyze orthogonality
    metrics = analyze_orthogonality(model, detailed=True)
    layers_metrics = metrics['layers']
    num_layers = len(layers_metrics)
    
    # Extract per-layer metrics
    q_metrics = [m['q_proj'] for m in layers_metrics]
    k_metrics = [m['k_proj'] for m in layers_metrics]
    v_metrics = [m['v_proj'] for m in layers_metrics]
    o_metrics = [m['o_proj'] for m in layers_metrics]
    avg_metrics = [m['avg'] for m in layers_metrics]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = np.arange(num_layers)
    width = 0.15
    
    # Plot bars for each projection type
    ax.bar(layers - 2*width, q_metrics, width, label='Query Proj')
    ax.bar(layers - width, k_metrics, width, label='Key Proj')
    ax.bar(layers, v_metrics, width, label='Value Proj')
    ax.bar(layers + width, o_metrics, width, label='Output Proj')
    ax.bar(layers + 2*width, avg_metrics, width, label='Average')
    
    # Add labels and legend
    ax.set_xlabel('Layer')
    ax.set_ylabel('Orthogonality Score (lower is better)')
    ax.set_title('Orthogonality Across Layers')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'Layer {i}' for i in range(num_layers)])
    ax.legend()
    
    plt.tight_layout()
    return fig


def compare_attention_patterns(ospa_attn, standard_attn, tokens=None, layer_idx=0, head_idx=0, figsize=(18, 8)):
    """
    Compare attention patterns between OSPA and standard transformer.
    
    Args:
        ospa_attn: Attention weights from OSPA model
        standard_attn: Attention weights from standard transformer
        tokens: Optional list of tokens for axis labels
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract attention weights for specified layer and head
    if len(ospa_attn.shape) == 4:
        ospa_attn = ospa_attn[layer_idx, head_idx]
    elif len(ospa_attn.shape) == 3:
        ospa_attn = ospa_attn[head_idx]
    
    if len(standard_attn.shape) == 4:
        standard_attn = standard_attn[layer_idx, head_idx]
    elif len(standard_attn.shape) == 3:
        standard_attn = standard_attn[head_idx]
    
    # Convert to numpy for plotting
    if isinstance(ospa_attn, torch.Tensor):
        ospa_attn = ospa_attn.detach().cpu().numpy()
    if isinstance(standard_attn, torch.Tensor):
        standard_attn = standard_attn.detach().cpu().numpy()
    
    # Get sequence length
    seq_len = ospa_attn.shape[0]
    
    # Prepare tokens for axis labels
    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]
    else:
        tokens = tokens[:seq_len]
    
    # Create difference map
    diff_attn = ospa_attn - standard_attn
    
    # Set up colormap for difference
    # Red-white-blue diverging colormap
    cmap_div = LinearSegmentedColormap.from_list(
        'rwb', [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    )
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot standard attention
    im0 = axes[0].imshow(standard_attn, cmap='viridis')
    axes[0].set_title(f"Standard Attention\nLayer {layer_idx}, Head {head_idx}")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot OSPA attention
    im1 = axes[1].imshow(ospa_attn, cmap='viridis')
    axes[1].set_title(f"OSPA Attention\nLayer {layer_idx}, Head {head_idx}")
    plt.colorbar(im1, ax=axes[1])
    
    # Plot difference
    max_diff = max(abs(diff_attn.min()), abs(diff_attn.max()))
    im2 = axes[2].imshow(diff_attn, cmap=cmap_div, vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f"Difference (OSPA - Standard)")
    plt.colorbar(im2, ax=axes[2])
    
    # Set axis labels
    for ax in axes:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
    
    plt.tight_layout()
    return fig


def analyze_attention_subspaces(model, input_ids, attention_mask=None):
    """
    Analyze attention subspaces to measure orthogonality between heads.
    
    This function passes inputs through the model and examines whether
    different attention heads are operating in orthogonal subspaces.
    
    Args:
        model: OSPA transformer model
        input_ids: Input token IDs
        attention_mask: Attention mask
    
    Returns:
        results: Dictionary of analysis results
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Move inputs to model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    results = {}
    all_layer_results = []
    
    with torch.no_grad():
        # Forward pass to get all attention weights
        outputs = model(input_ids, attention_mask, output_attentions=True)
        attention_outputs = outputs.attentions  # [layers, batch, heads, seq, seq]
        
        # Analyze each layer
        for layer_idx, layer_attention in enumerate(attention_outputs):
            layer_results = {}
            
            # Average over batch dimension
            layer_attention = layer_attention.mean(dim=0)  # [heads, seq, seq]
            
            # Reshape attention patterns for analysis
            num_heads = layer_attention.shape[0]
            flat_attention = layer_attention.reshape(num_heads, -1)  # [heads, seq*seq]
            
            # Normalize attention vectors
            norm_attention = flat_attention / (torch.norm(flat_attention, dim=1, keepdim=True) + 1e-10)
            
            # Calculate head cosine similarity matrix
            sim_matrix = torch.matmul(norm_attention, norm_attention.t())  # [heads, heads]
            
            # Calculate head orthogonality (diagonality of sim_matrix)
            # Perfect orthogonality would have 1s on diagonal and 0s elsewhere
            identity = torch.eye(num_heads, device=device)
            ortho_score = torch.norm(sim_matrix - identity, p='fro').item() / num_heads
            
            # Calculate attention entropy (specialization)
            entropy = analyze_head_specialization(layer_attention.unsqueeze(0))
            
            # Store results
            layer_results['ortho_score'] = ortho_score
            layer_results['entropy'] = entropy
            layer_results['sim_matrix'] = sim_matrix.cpu().numpy()
            
            all_layer_results.append(layer_results)
    
    results['layers'] = all_layer_results
    results['avg_ortho_score'] = np.mean([l['ortho_score'] for l in all_layer_results])
    results['avg_entropy'] = np.mean([l['entropy'] for l in all_layer_results])
    
    return results


def plot_attention_subspaces(subspace_results, layer_idx=0, figsize=(10, 8)):
    """
    Plot attention subspace analysis results.
    
    Args:
        subspace_results: Results from analyze_attention_subspaces
        layer_idx: Layer index to visualize
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    layer_results = subspace_results['layers'][layer_idx]
    sim_matrix = layer_results['sim_matrix']
    ortho_score = layer_results['ortho_score']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot similarity matrix as heatmap
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set axis labels
    num_heads = sim_matrix.shape[0]
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(num_heads))
    ax.set_xticklabels([f'Head {i}' for i in range(num_heads)])
    ax.set_yticklabels([f'Head {i}' for i in range(num_heads)])
    
    # Add title with orthogonality score
    ax.set_title(f"Attention Head Similarity - Layer {layer_idx}\nOrthogonality Score: {ortho_score:.4f} (lower is better)")
    
    # Add values in cells
    for i in range(num_heads):
        for j in range(num_heads):
            text = ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                          ha="center", va="center", color="w" if abs(sim_matrix[i, j]) > 0.5 else "k")
    
    plt.tight_layout()
    return fig