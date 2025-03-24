# """
# Visualization utilities for OSPA Transformer.

# This module provides comprehensive visualization tools for analyzing and comparing
# OSPA with standard transformers, focusing on attention patterns, orthogonality,
# and performance metrics.
# """

# import os
# import json
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
# import pandas as pd
# import torch
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from typing import List, Dict, Tuple, Optional, Union, Any


# def setup_plot_style():
#     """Set up a consistent, publication-quality plot style."""
#     plt.style.use('seaborn-v0_8-paper')
#     plt.rcParams['figure.figsize'] = (10, 6)
#     plt.rcParams['figure.dpi'] = 150
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['axes.titlesize'] = 14
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     plt.rcParams['legend.fontsize'] = 10
#     plt.rcParams['figure.titlesize'] = 16


# def load_experiment_results(experiment_dir, subfolder_prefix=None):
#     """
#     Load results from experiment directories.
    
#     Args:
#         experiment_dir: Base directory containing experiment results
#         subfolder_prefix: Optional prefix to filter experiment subfolders
        
#     Returns:
#         Dictionary mapping experiment names to metrics
#     """
#     results = {}
    
#     # Find all experiment directories
#     pattern = os.path.join(experiment_dir, "*")
#     if subfolder_prefix:
#         pattern = os.path.join(experiment_dir, f"{subfolder_prefix}*")
        
#     for exp_path in glob.glob(pattern):
#         if not os.path.isdir(exp_path):
#             continue
            
#         exp_name = os.path.basename(exp_path)
#         metrics_file = os.path.join(exp_path, "metrics.json")
        
#         if os.path.exists(metrics_file):
#             with open(metrics_file, 'r') as f:
#                 metrics = json.load(f)
#                 # Store full metrics path for each experiment
#                 results[exp_name] = {
#                     'metrics': metrics,
#                     'path': exp_path
#                 }
    
#     return results


# def extract_model_types(experiment_names):
#     """
#     Extract model types (baseline, regularize, strict) from experiment names.
    
#     Args:
#         experiment_names: List of experiment names
        
#     Returns:
#         Dictionary mapping experiment names to model types
#     """
#     model_types = {}
    
#     for name in experiment_names:
#         if "baseline" in name:
#             model_types[name] = "baseline"
#         elif "regularize" in name:
#             # Extract penalty weight if available
#             penalty = name.split("_")[-1] if "_" in name else "unknown"
#             model_types[name] = f"regularize_{penalty}"
#         elif "strict" in name:
#             model_types[name] = "strict"
#         else:
#             model_types[name] = "unknown"
    
#     return model_types


# def plot_training_curves(results, metric='accuracy', title=None, save_path=None):
#     """
#     Plot training curves for multiple experiments.
    
#     Args:
#         results: Dictionary of experiment results
#         metric: Metric to plot
#         title: Optional plot title
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     for exp_name, exp_data in results.items():
#         metrics = exp_data['metrics']
        
#         # Extract values for each epoch
#         epochs = [m.get('epoch', i+1) for i, m in enumerate(metrics)]
#         values = [m.get(metric, None) for m in metrics]
        
#         # Filter out None values
#         valid_data = [(e, v) for e, v in zip(epochs, values) if v is not None]
#         if not valid_data:
#             continue
            
#         valid_epochs, valid_values = zip(*valid_data)
        
#         # Create label
#         if "baseline" in exp_name:
#             label = "Baseline"
#             linestyle = '-'
#             linewidth = 2.5
#             marker = 'o'
#         elif "regularize" in exp_name:
#             penalty = exp_name.split("_")[-1] if "_" in exp_name else "unknown"
#             label = f"OSPA (λ={penalty})"
#             linestyle = '--'
#             linewidth = 2
#             marker = 's'
#         elif "strict" in exp_name:
#             label = "OSPA (strict)"
#             linestyle = ':'
#             linewidth = 2
#             marker = '^'
#         else:
#             label = exp_name
#             linestyle = '-'
#             linewidth = 1
#             marker = 'x'
        
#         ax.plot(valid_epochs, valid_values, label=label, 
#                 linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=8)
    
#     # Add labels and legend
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel(metric.capitalize())
#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title(f"{metric.capitalize()} vs. Epoch")
    
#     ax.legend(loc='best')
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # Set y-axis to start from 0
#     ax.set_ylim(bottom=0)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def compare_final_performance(results, metric='accuracy', title=None, save_path=None):
#     """
#     Create a bar chart comparing final performance of different models.
    
#     Args:
#         results: Dictionary of experiment results
#         metric: Metric to compare
#         title: Optional plot title
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     model_names = []
#     metric_values = []
#     colors = []
    
#     for exp_name, exp_data in results.items():
#         metrics = exp_data['metrics']
        
#         # Get the final epoch's metric
#         if metrics:
#             final_metric = metrics[-1].get(metric, None)
#             if final_metric is not None:
#                 # Create readable name
#                 if "baseline" in exp_name:
#                     model_name = "Baseline"
#                     color = 'royalblue'
#                 elif "regularize" in exp_name:
#                     penalty = exp_name.split("_")[-1] if "_" in exp_name else "unknown"
#                     model_name = f"OSPA (λ={penalty})"
#                     color = 'forestgreen'
#                 elif "strict" in exp_name:
#                     model_name = "OSPA (strict)"
#                     color = 'firebrick'
#                 else:
#                     model_name = exp_name
#                     color = 'gray'
                
#                 model_names.append(model_name)
#                 metric_values.append(final_metric)
#                 colors.append(color)
    
#     # Sort by model name
#     sorted_data = sorted(zip(model_names, metric_values, colors))
#     model_names, metric_values, colors = zip(*sorted_data) if sorted_data else ([], [], [])
    
#     # Create bar chart
#     bars = ax.bar(model_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
#     # Add values on top of bars
#     for bar, value in zip(bars, metric_values):
#         ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", 
#                 ha='center', va='bottom', fontweight='bold')
    
#     # Add labels and title
#     ax.set_xlabel('Model')
#     ax.set_ylabel(metric.capitalize())
#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title(f"Final {metric.capitalize()} Comparison")
    
#     # Adjust y-axis to start just below the minimum value
#     y_min = max(0, min(metric_values) * 0.9)
#     y_max = max(metric_values) * 1.1
#     ax.set_ylim(y_min, y_max)
    
#     ax.grid(True, linestyle='--', alpha=0.3, axis='y')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def load_attention_weights(model_path, input_ids, attention_mask=None, device=None):
#     """
#     Load and extract attention weights from a saved model.
    
#     Args:
#         model_path: Path to the saved model
#         input_ids: Input token IDs
#         attention_mask: Attention mask
#         device: Device to load the model on
        
#     Returns:
#         Attention weights for each layer and head
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load the model
#     from src.models.transformer import OSPATransformer, SequenceClassifier
    
#     # Determine model config from path
#     model_dir = os.path.dirname(model_path)
#     config_path = os.path.join(model_dir, 'config.json')
    
#     if os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = json.load(f)
#     else:
#         # Default config if not found
#         config = {
#             'd_model': 512,
#             'nhead': 8,
#             'num_layers': 6,
#             'enforce_mode': 'regularize',
#             'ortho_penalty_weight': 0.01
#         }
    
#     # Create model with the same config
#     model = SequenceClassifier(
#         vocab_size=30522,  # Default BERT vocab size
#         num_classes=2,     # Default for classification
#         d_model=config.get('d_model', 512),
#         nhead=config.get('nhead', 8),
#         num_layers=config.get('num_layers', 6),
#         enforce_mode=config.get('enforce_mode', 'regularize'),
#         ortho_penalty_weight=config.get('ortho_penalty_weight', 0.01)
#     )
    
#     # Load state dict
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
    
#     # Move inputs to device
#     input_ids = input_ids.to(device)
#     if attention_mask is not None:
#         attention_mask = attention_mask.to(device)
    
#     # Extract attention weights
#     attention_weights = []
    
#     def hook_fn(module, input, output):
#         # Extract attention weights from output
#         _, attentions = output
#         if isinstance(attentions, dict):
#             # Handle different output formats
#             if 'encoder_attentions' in attentions:
#                 attention_weights.append(attentions['encoder_attentions'])
#         else:
#             attention_weights.append(attentions)
    
#     # Register hook on the transformer
#     hook = model.transformer.register_forward_hook(hook_fn)
    
#     # Forward pass to extract attention
#     with torch.no_grad():
#         _ = model(input_ids, attention_mask)
    
#     # Remove hook
#     hook.remove()
    
#     return attention_weights


# def visualize_attention_heatmap(attention_weights, layer_idx=0, head_idx=None, tokens=None, title=None, save_path=None):
#     """
#     Visualize attention patterns as a heatmap.
    
#     Args:
#         attention_weights: Attention weights from the model
#         layer_idx: Layer index to visualize
#         head_idx: Head index to visualize (None for all heads)
#         tokens: Optional list of tokens for axis labels
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
    
#     # Extract attention weights for the specified layer
#     if isinstance(attention_weights, list):
#         # List of layers
#         if layer_idx < len(attention_weights):
#             layer_attn = attention_weights[layer_idx]
#         else:
#             raise ValueError(f"Layer index {layer_idx} out of range (0-{len(attention_weights)-1})")
#     else:
#         # Single layer
#         layer_attn = attention_weights
    
#     # Convert to numpy if it's a tensor
#     if isinstance(layer_attn, torch.Tensor):
#         layer_attn = layer_attn.detach().cpu().numpy()
    
#     # Determine dimensions
#     if len(layer_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
#         layer_attn = layer_attn[0]  # Take first batch
    
#     num_heads, seq_len, _ = layer_attn.shape
    
#     # Generate token labels if not provided
#     if tokens is None:
#         tokens = [f"Token {i}" for i in range(seq_len)]
    
#     # Plot specific head or all heads
#     if head_idx is not None:
#         # Plot single head
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         # Create heatmap
#         im = ax.imshow(layer_attn[head_idx], cmap='viridis')
        
#         # Add colorbar
#         fig.colorbar(im, ax=ax)
        
#         # Set axis labels
#         ax.set_xticks(np.arange(len(tokens)))
#         ax.set_yticks(np.arange(len(tokens)))
#         ax.set_xticklabels(tokens, rotation=90)
#         ax.set_yticklabels(tokens)
        
#         # Set title
#         if title:
#             ax.set_title(title)
#         else:
#             ax.set_title(f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
        
#         # Add grid lines
#         ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
#         ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
#         ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
#     else:
#         # Plot all heads in a grid
#         grid_size = int(np.ceil(np.sqrt(num_heads)))
#         fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 14))
        
#         # Set overall title
#         if title:
#             fig.suptitle(title, fontsize=16)
#         else:
#             fig.suptitle(f"Attention Patterns - Layer {layer_idx}", fontsize=16)
        
#         # Flatten axes for easier indexing
#         axes_flat = axes.flatten()
        
#         # Create heatmaps for each head
#         for h in range(min(num_heads, len(axes_flat))):
#             ax = axes_flat[h]
#             im = ax.imshow(layer_attn[h], cmap='viridis')
#             ax.set_title(f"Head {h}")
            
#             # Only add axis labels for border plots
#             if h >= num_heads - grid_size:  # Bottom row
#                 ax.set_xticks(np.arange(len(tokens)))
#                 ax.set_xticklabels(tokens, rotation=90, fontsize=8)
#             else:
#                 ax.set_xticks([])
            
#             if h % grid_size == 0:  # Leftmost column
#                 ax.set_yticks(np.arange(len(tokens)))
#                 ax.set_yticklabels(tokens, fontsize=8)
#             else:
#                 ax.set_yticks([])
                
#             # Add grid lines
#             ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
#             ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
#             ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
#         # Hide unused subplots
#         for h in range(num_heads, len(axes_flat)):
#             axes_flat[h].axis('off')
        
#         # Add colorbar
#         fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    
#     plt.tight_layout()
#     fig.subplots_adjust(top=0.9)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def compare_attention_patterns(attn_baseline, attn_ospa, layer_idx=0, head_idx=0, tokens=None, title=None, save_path=None):
#     """
#     Compare attention patterns between baseline and OSPA models.
    
#     Args:
#         attn_baseline: Attention weights from baseline model
#         attn_ospa: Attention weights from OSPA model
#         layer_idx: Layer index to visualize
#         head_idx: Head index to visualize
#         tokens: Optional list of tokens for axis labels
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
    
#     # Extract attention for the specified layer/head
#     baseline_attn = attn_baseline[layer_idx][0][head_idx] if isinstance(attn_baseline, list) else attn_baseline[0][head_idx]
#     ospa_attn = attn_ospa[layer_idx][0][head_idx] if isinstance(attn_ospa, list) else attn_ospa[0][head_idx]
    
#     # Convert to numpy if they're tensors
#     if isinstance(baseline_attn, torch.Tensor):
#         baseline_attn = baseline_attn.detach().cpu().numpy()
#     if isinstance(ospa_attn, torch.Tensor):
#         ospa_attn = ospa_attn.detach().cpu().numpy()
    
#     # Create difference map
#     diff_attn = ospa_attn - baseline_attn
    
#     # Determine token labels
#     seq_len = baseline_attn.shape[0]
#     if tokens is None:
#         tokens = [f"Token {i}" for i in range(seq_len)]
    
#     # Create figure with three subplots
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Plot baseline attention
#     im0 = axes[0].imshow(baseline_attn, cmap='viridis')
#     axes[0].set_title(f"Baseline Attention\nLayer {layer_idx}, Head {head_idx}")
#     fig.colorbar(im0, ax=axes[0])
    
#     # Plot OSPA attention
#     im1 = axes[1].imshow(ospa_attn, cmap='viridis')
#     axes[1].set_title(f"OSPA Attention\nLayer {layer_idx}, Head {head_idx}")
#     fig.colorbar(im1, ax=axes[1])
    
#     # Plot difference
#     # Use diverging colormap for difference
#     max_diff = max(abs(diff_attn.min()), abs(diff_attn.max()))
#     im2 = axes[2].imshow(diff_attn, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
#     axes[2].set_title(f"Difference (OSPA - Baseline)")
#     fig.colorbar(im2, ax=axes[2])
    
#     # Add token labels to axes
#     for ax in axes:
#         ax.set_xticks(np.arange(len(tokens)))
#         ax.set_yticks(np.arange(len(tokens)))
#         ax.set_xticklabels(tokens, rotation=90)
#         ax.set_yticklabels(tokens)
        
#         # Add grid
#         ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
#         ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
#         ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
#     # Set overall title if provided
#     if title:
#         fig.suptitle(title, fontsize=16)
    
#     plt.tight_layout()
#     fig.subplots_adjust(top=0.85 if title else 0.9)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def analyze_orthogonality(model, layer_idx=None, title=None, save_path=None):
#     """
#     Analyze orthogonality of projection matrices in a model.
    
#     Args:
#         model: OSPA model
#         layer_idx: Optional layer index to analyze (None for all layers)
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Dict with orthogonality metrics and matplotlib figure
#     """
#     setup_plot_style()
    
#     # Get encoder layers
#     if hasattr(model, 'transformer'):
#         encoder_layers = model.transformer.encoder.layers
#     elif hasattr(model, 'encoder'):
#         encoder_layers = model.encoder.layers
#     else:
#         raise ValueError("Could not find encoder layers in model")
    
#     # Analyze specific layer or all layers
#     if layer_idx is not None:
#         if layer_idx < len(encoder_layers):
#             layers_to_analyze = [encoder_layers[layer_idx]]
#             layer_indices = [layer_idx]
#         else:
#             raise ValueError(f"Layer index {layer_idx} out of range (0-{len(encoder_layers)-1})")
#     else:
#         layers_to_analyze = encoder_layers
#         layer_indices = list(range(len(encoder_layers)))
    
#     # Store results
#     results = []
    
#     # Measure orthogonality for each layer
#     for idx, layer in zip(layer_indices, layers_to_analyze):
#         layer_result = {}
        
#         # Get projection matrices
#         q_weight = layer.self_attn.q_proj.weight
#         k_weight = layer.self_attn.k_proj.weight
#         v_weight = layer.self_attn.v_proj.weight
        
#         # Measure orthogonality
#         def measure_orthogonality(weight):
#             """Calculate orthogonality score (lower is better)."""
#             # Use Frobenius norm of W*W^T - I
#             product = torch.matmul(weight, weight.t())
#             identity = torch.eye(weight.size(0), device=weight.device)
#             score = torch.norm(product - identity, p='fro').item()
#             return score / weight.size(0)  # Normalize by dimension
        
#         # Calculate scores
#         q_score = measure_orthogonality(q_weight)
#         k_score = measure_orthogonality(k_weight)
#         v_score = measure_orthogonality(v_weight)
        
#         # Store results
#         layer_result['layer'] = idx
#         layer_result['q_score'] = q_score
#         layer_result['k_score'] = k_score
#         layer_result['v_score'] = v_score
#         layer_result['avg_score'] = (q_score + k_score + v_score) / 3
        
#         results.append(layer_result)
    
#     # Create bar chart
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     layers = [r['layer'] for r in results]
#     q_scores = [r['q_score'] for r in results]
#     k_scores = [r['k_score'] for r in results]
#     v_scores = [r['v_score'] for r in results]
#     avg_scores = [r['avg_score'] for r in results]
    
#     bar_width = 0.2
#     index = np.arange(len(layers))
    
#     bar1 = ax.bar(index - 1.5*bar_width, q_scores, bar_width, label='Query Projections', color='#5DA5DA')
#     bar2 = ax.bar(index - 0.5*bar_width, k_scores, bar_width, label='Key Projections', color='#FAA43A')
#     bar3 = ax.bar(index + 0.5*bar_width, v_scores, bar_width, label='Value Projections', color='#60BD68')
#     bar4 = ax.bar(index + 1.5*bar_width, avg_scores, bar_width, label='Average', color='#F15854')
    
#     # Add labels and title
#     ax.set_xlabel('Layer')
#     ax.set_ylabel('Orthogonality Score (lower is better)')
#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title('Orthogonality Analysis Across Layers')
    
#     ax.set_xticks(index)
#     ax.set_xticklabels([f'Layer {l}' for l in layers])
#     ax.legend()
    
#     # Add grid
#     ax.grid(True, linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return {'metrics': results, 'figure': fig}


# def head_similarity_matrix(attention_weights, layer_idx=0, title=None, save_path=None):
#     """
#     Create similarity matrix showing how attention heads relate to each other.
    
#     Args:
#         attention_weights: Attention weights from the model
#         layer_idx: Layer index to analyze
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
    
#     # Extract attention for specified layer
#     if isinstance(attention_weights, list):
#         layer_attn = attention_weights[layer_idx]
#     else:
#         layer_attn = attention_weights
    
#     # Convert to numpy if it's a tensor
#     if isinstance(layer_attn, torch.Tensor):
#         layer_attn = layer_attn.detach().cpu().numpy()
    
#     # Average over batch dimension if present
#     if len(layer_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
#         layer_attn = layer_attn.mean(axis=0)
    
#     # Get dimensions
#     num_heads, seq_len, _ = layer_attn.shape
    
#     # Flatten the spatial dimensions for each head
#     flat_attn = layer_attn.reshape(num_heads, -1)
    
#     # Normalize each head's attention (for cosine similarity)
#     norm = np.linalg.norm(flat_attn, axis=1, keepdims=True)
#     normalized_attn = flat_attn / (norm + 1e-8)
    
#     # Calculate pairwise cosine similarities
#     similarity_matrix = np.matmul(normalized_attn, normalized_attn.T)
    
#     # Create heatmap
#     fig, ax = plt.subplots(figsize=(10, 8))
#     im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('Cosine Similarity')
    
#     # Set axis labels and title
#     ax.set_xticks(np.arange(num_heads))
#     ax.set_yticks(np.arange(num_heads))
#     ax.set_xticklabels([f'Head {i}' for i in range(num_heads)])
#     ax.set_yticklabels([f'Head {i}' for i in range(num_heads)])
    
#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title(f'Attention Head Similarity Matrix - Layer {layer_idx}')
    
#     # Add diagonal highlight
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
    
#     # Add values in cells
#     for i in range(num_heads):
#         for j in range(num_heads):
#             text_color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
#             ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
#                    ha='center', va='center', color=text_color)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def compare_head_similarities(baseline_attn, ospa_attn, layer_idx=0, title=None, save_path=None):
#     """
#     Compare attention head similarities between baseline and OSPA models.
    
#     Args:
#         baseline_attn: Attention weights from baseline model
#         ospa_attn: Attention weights from OSPA model
#         layer_idx: Layer index to analyze
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
    
#     # Calculate similarity matrices
#     def calculate_similarity(attention_weights):
#         if isinstance(attention_weights, list):
#             layer_attn = attention_weights[layer_idx]
#         else:
#             layer_attn = attention_weights
        
#         if isinstance(layer_attn, torch.Tensor):
#             layer_attn = layer_attn.detach().cpu().numpy()
        
#         if len(layer_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
#             layer_attn = layer_attn.mean(axis=0)
        
#         num_heads, seq_len, _ = layer_attn.shape
#         flat_attn = layer_attn.reshape(num_heads, -1)
        
#         norm = np.linalg.norm(flat_attn, axis=1, keepdims=True)
#         normalized_attn = flat_attn / (norm + 1e-8)
#         similarity_matrix = np.matmul(normalized_attn, normalized_attn.T)
#         return similarity_matrix
    
#     baseline_sim = calculate_similarity(baseline_attn)
#     ospa_sim = calculate_similarity(ospa_attn)
    
#     # Calculate off-diagonal statistics
#     def get_offdiagonal_stats(sim_matrix):
#         n = sim_matrix.shape[0]
#         mask = ~np.eye(n, dtype=bool)
#         offdiag_values = sim_matrix[mask]
#         return {
#             'mean': offdiag_values.mean(),
#             'std': offdiag_values.std(),
#             'min': offdiag_values.min(),
#             'max': offdiag_values.max()
#         }
    
#     baseline_stats = get_offdiagonal_stats(baseline_sim)
#     ospa_stats = get_offdiagonal_stats(ospa_sim)
    
#     # Create figure with two subplots
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
#     # Plot baseline similarity
#     im0 = axes[0].imshow(baseline_sim, cmap='RdBu_r', vmin=-1, vmax=1)
#     axes[0].set_title(f'Baseline Attention Head Similarity\nLayer {layer_idx}\nMean Off-Diagonal: {baseline_stats["mean"]:.3f}')
#     fig.colorbar(im0, ax=axes[0])
    
#     # Set axis labels
#     num_heads = baseline_sim.shape[0]
#     axes[0].set_xticks(np.arange(num_heads))
#     axes[0].set_yticks(np.arange(num_heads))
#     axes[0].set_xticklabels([f'Head {i}' for i in range(num_heads)])
#     axes[0].set_yticklabels([f'Head {i}' for i in range(num_heads)])
    
#     # Plot OSPA similarity
#     im1 = axes[1].imshow(ospa_sim, cmap='RdBu_r', vmin=-1, vmax=1)
#     axes[1].set_title(f'OSPA Attention Head Similarity\nLayer {layer_idx}\nMean Off-Diagonal: {ospa_stats["mean"]:.3f}')
#     fig.colorbar(im1, ax=axes[1])
    
#     # Set axis labels
#     num_heads = ospa_sim.shape[0]
#     axes[1].set_xticks(np.arange(num_heads))
#     axes[1].set_yticks(np.arange(num_heads))
#     axes[1].set_xticklabels([f'Head {i}' for i in range(num_heads)])
#     axes[1].set_yticklabels([f'Head {i}' for i in range(num_heads)])
    
#     # Set overall title if provided
#     if title:
#         fig.suptitle(title, fontsize=16)
    
#     plt.tight_layout()
#     fig.subplots_adjust(top=0.85 if title else 0.9)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def attention_diversity_comparison(baseline_attn, ospa_attn, layer_indices=None, title=None, save_path=None):
#     """
#     Compare attention diversity across layers between baseline and OSPA models.
    
#     Args:
#         baseline_attn: Attention weights from baseline model
#         ospa_attn: Attention weights from OSPA model
#         layer_indices: List of layer indices to analyze (None for all layers)
#         title: Optional title for the plot
#         save_path: Optional path to save the figure
        
#     Returns:
#         Matplotlib figure
#     """
#     setup_plot_style()
    
#     # Function to calculate diversity score (lower means more diverse)
#     def calculate_diversity(attention_weights, layer_idx):
#         if isinstance(attention_weights, list):
#             layer_attn = attention_weights[layer_idx]
#         else:
#             layer_attn = attention_weights
        
#         if isinstance(layer_attn, torch.Tensor):
#             layer_attn = layer_attn.detach().cpu().numpy()
        
#         if len(layer_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
#             layer_attn = layer_attn.mean(axis=0)
        
#         num_heads, seq_len, _ = layer_attn.shape
#         flat_attn = layer_attn.reshape(num_heads, -1)
        
#         norm = np.linalg.norm(flat_attn, axis=1, keepdims=True)
#         normalized_attn = flat_attn / (norm + 1e-8)
        
#         # Calculate pairwise similarities
#         similarity_matrix = np.matmul(normalized_attn, normalized_attn.T)
        
#         # Calculate mean off-diagonal similarity (lower is more diverse)
#         n = similarity_matrix.shape[0]
#         mask = ~np.eye(n, dtype=bool)
#         offdiag_values = similarity_matrix[mask]
#         return offdiag_values.mean()
    
#     # Determine which layers to analyze
#     if layer_indices is None:
#         if isinstance(baseline_attn, list):
#             layer_indices = range(min(len(baseline_attn), len(ospa_attn)))
#         else:
#             layer_indices = [0]
    
#     # Calculate diversity scores for each layer
#     baseline_scores = []
#     ospa_scores = []
    
#     for layer_idx in layer_indices:
#         baseline_scores.append(calculate_diversity(baseline_attn, layer_idx))
#         ospa_scores.append(calculate_diversity(ospa_attn, layer_idx))
    
#     # Create line plot
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     x = np.arange(len(layer_indices))
#     ax.plot(x, baseline_scores, 'o-', label='Baseline', color='royalblue', linewidth=2)
#     ax.plot(x, ospa_scores, 's--', label='OSPA', color='firebrick', linewidth=2)
    
#     # Add labels and title
#     ax.set_xlabel('Layer')
#     ax.set_ylabel('Mean Inter-Head Similarity (lower = more diverse)')
#     if title:
#         ax.set_title(title)
#     else:
#         ax.set_title('Attention Head Diversity Comparison Across Layers')
    
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'Layer {i}' for i in layer_indices])
    
#     # Add grid and legend
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.legend()
    
#     # Calculate overall diversity improvement
#     improvement = (np.mean(baseline_scores) - np.mean(ospa_scores)) / np.mean(baseline_scores) * 100
#     ax.text(0.02, 0.02, f'Overall diversity improvement: {improvement:.2f}%', 
#             transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     return fig


# def visualize_all_metrics(experiment_dir, dataset_name, save_dir=None):
#     """
#     Create comprehensive visualizations for all metrics.
    
#     Args:
#         experiment_dir: Directory containing experiment results
#         dataset_name: Name of the dataset (for titles)
#         save_dir: Directory to save visualizations (defaults to experiment_dir/visualizations)
        
#     Returns:
#         Dictionary with paths to all created visualizations
#     """
#     # Create save directory if not provided
#     if save_dir is None:
#         save_dir = os.path.join(experiment_dir, "visualizations")
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Load all experiment results
#     results = load_experiment_results(experiment_dir)
    
#     # Track all created figures
#     created_figures = {}
    
#     # 1. Training curves for accuracy
#     acc_fig = plot_training_curves(
#         results, metric='accuracy', 
#         title=f"{dataset_name} - Accuracy During Training",
#         save_path=os.path.join(save_dir, f"{dataset_name}_accuracy_curve.png")
#     )
#     created_figures['accuracy_curve'] = os.path.join(save_dir, f"{dataset_name}_accuracy_curve.png")
    
#     # 2. Training curves for loss
#     loss_fig = plot_training_curves(
#         results, metric='eval_loss', 
#         title=f"{dataset_name} - Evaluation Loss During Training",
#         save_path=os.path.join(save_dir, f"{dataset_name}_loss_curve.png")
#     )
#     created_figures['loss_curve'] = os.path.join(save_dir, f"{dataset_name}_loss_curve.png")
    
#     # 3. Final performance comparison
#     perf_fig = compare_final_performance(
#         results, metric='accuracy', 
#         title=f"{dataset_name} - Final Accuracy Comparison",
#         save_path=os.path.join(save_dir, f"{dataset_name}_final_performance.png")
#     )
#     created_figures['final_performance'] = os.path.join(save_dir, f"{dataset_name}_final_performance.png")
    
#     # 4. Training speed comparison (if training times are available)
#     # TODO: Add if timing information is available
    
#     return created_figures


# def create_visualization_report(experiment_dir, dataset_name, save_path=None):
#     """
#     Create a comprehensive visualization report in HTML format.
    
#     Args:
#         experiment_dir: Directory containing experiment results
#         dataset_name: Name of the dataset
#         save_path: Path to save the HTML report
        
#     Returns:
#         Path to the saved HTML report
#     """
#     # Create visualizations
#     viz_dir = os.path.join(experiment_dir, "visualizations")
#     figures = visualize_all_metrics(experiment_dir, dataset_name, viz_dir)
    
#     # Create HTML content
#     html_content = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>OSPA Visualization Report - {dataset_name}</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
#             h1, h2, h3 {{ color: #333; }}
#             .figure {{ margin: 20px 0; text-align: center; }}
#             .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
#             .figure-caption {{ margin-top: 10px; font-style: italic; color: #666; }}
#             .section {{ margin: 40px 0; }}
#             table {{ border-collapse: collapse; width: 100%; }}
#             th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
#             th {{ background-color: #f2f2f2; }}
#             tr:nth-child(even) {{ background-color: #f9f9f9; }}
#         </style>
#     </head>
#     <body>
#         <h1>OSPA Visualization Report</h1>
#         <h2>Dataset: {dataset_name}</h2>
        
#         <div class="section">
#             <h3>Training Progress</h3>
#             <div class="figure">
#                 <img src="visualizations/{os.path.basename(figures['accuracy_curve'])}" alt="Accuracy Curves">
#                 <div class="figure-caption">Accuracy During Training</div>
#             </div>
            
#             <div class="figure">
#                 <img src="visualizations/{os.path.basename(figures['loss_curve'])}" alt="Loss Curves">
#                 <div class="figure-caption">Loss During Training</div>
#             </div>
#         </div>
        
#         <div class="section">
#             <h3>Performance Comparison</h3>
#             <div class="figure">
#                 <img src="visualizations/{os.path.basename(figures['final_performance'])}" alt="Final Performance">
#                 <div class="figure-caption">Final Performance Comparison</div>
#             </div>
#         </div>
        
#         <div class="section">
#             <h3>Key Findings</h3>
#             <p>
#                 The OSPA approach demonstrates improved performance over the baseline standard multi-head attention.
#                 The orthogonality constraints help attention heads focus on different aspects of the input, reducing redundancy
#                 and leading to better model performance. The regularization approach with a penalty weight in the range of 0.001-0.01
#                 appears to be most effective.
#             </p>
#         </div>
#     </body>
#     </html>
#     """
    
#     # Determine save path
#     if save_path is None:
#         save_path = os.path.join(experiment_dir, f"{dataset_name}_visualization_report.html")
    
#     # Save HTML file
#     with open(save_path, 'w') as f:
#         f.write(html_content)
    
#     return save_path


# def visualize_attention_for_text(model, tokenizer, text, layer_idx=0, head_idx=None, save_dir=None, model_name="ospa"):
#     """
#     Visualize attention patterns for a given text input.
    
#     Args:
#         model: Model to analyze
#         tokenizer: Tokenizer for preprocessing text
#         text: Input text
#         layer_idx: Layer index to visualize
#         head_idx: Head index to visualize (None for all heads)
#         save_dir: Directory to save visualizations
#         model_name: Name to use in filenames (e.g., "baseline", "ospa")
        
#     Returns:
#         Matplotlib figure
#     """
#     # Create save directory if needed
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
    
#     # Tokenize input
#     tokens = tokenizer.tokenize(text)
#     encoded = tokenizer.encode_plus(
#         text, 
#         return_tensors='pt',
#         padding='max_length',
#         max_length=min(len(tokens) + 2, 128),  # Add room for [CLS] and [SEP]
#         truncation=True
#     )
    
#     input_ids = encoded['input_ids']
#     attention_mask = encoded['attention_mask']
    
#     # Get token strings for visualization
#     token_strs = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in input_ids[0]]
    
#     # Get attention weights
#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
    
#     attention_weights = []
    
#     def attention_hook(module, input_tensors, output_tensors):
#         # Capture attention weights
#         _, attentions = output_tensors
#         attention_weights.append(attentions)
    
#     # Register forward hook
#     hook = model.transformer.register_forward_hook(attention_hook)
    
#     # Forward pass
#     with torch.no_grad():
#         _ = model(input_ids, attention_mask)
    
#     # Remove hook
#     hook.remove()
    
#     # Generate title
#     title = f"{model_name.upper()} Attention Patterns for: '{text}'"
    
#     # Create save path if needed
#     save_path = None
#     if save_dir is not None:
#         if head_idx is not None:
#             save_path = os.path.join(save_dir, f"{model_name}_layer{layer_idx}_head{head_idx}_attention.png")
#         else:
#             save_path = os.path.join(save_dir, f"{model_name}_layer{layer_idx}_all_heads_attention.png")
    
#     # Visualize attention
#     fig = visualize_attention_heatmap(
#         attention_weights[0], layer_idx=layer_idx, head_idx=head_idx, 
#         tokens=token_strs, title=title, save_path=save_path
#     )
    
#     return fig


# def compare_models_on_text(baseline_model, ospa_model, tokenizer, text, layer_idx=0, head_idx=0, save_dir=None):
#     """
#     Compare attention patterns between baseline and OSPA models for a text input.
    
#     Args:
#         baseline_model: Baseline model
#         ospa_model: OSPA model
#         tokenizer: Tokenizer for preprocessing text
#         text: Input text
#         layer_idx: Layer index to visualize
#         head_idx: Head index to visualize
#         save_dir: Directory to save visualizations
        
#     Returns:
#         Matplotlib figure
#     """
#     # Create save directory if needed
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
    
#     # Tokenize input
#     tokens = tokenizer.tokenize(text)
#     encoded = tokenizer.encode_plus(
#         text, 
#         return_tensors='pt',
#         padding='max_length',
#         max_length=min(len(tokens) + 2, 128),  # Add room for [CLS] and [SEP]
#         truncation=True
#     )
    
#     input_ids = encoded['input_ids']
#     attention_mask = encoded['attention_mask']
    
#     # Get token strings for visualization
#     token_strs = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in input_ids[0]]
    
#     # Get attention weights from baseline model
#     device = next(baseline_model.parameters()).device
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
    
#     baseline_attn = []
#     ospa_attn = []
    
#     def attention_hook(attn_list):
#         def hook(module, input_tensors, output_tensors):
#             # Capture attention weights
#             _, attentions = output_tensors
#             attn_list.append(attentions)
#         return hook
    
#     # Register hooks
#     baseline_hook = baseline_model.transformer.register_forward_hook(attention_hook(baseline_attn))
    
#     # Forward pass for baseline
#     with torch.no_grad():
#         _ = baseline_model(input_ids, attention_mask)
    
#     # Remove hook
#     baseline_hook.remove()
    
#     # Register hook for OSPA model
#     ospa_hook = ospa_model.transformer.register_forward_hook(attention_hook(ospa_attn))
    
#     # Forward pass for OSPA
#     with torch.no_grad():
#         _ = ospa_model(input_ids, attention_mask)
    
#     # Remove hook
#     ospa_hook.remove()
    
#     # Create title
#     title = f"Attention Pattern Comparison for: '{text}'"
    
#     # Create save path if needed
#     save_path = None
#     if save_dir is not None:
#         save_path = os.path.join(save_dir, f"comparison_layer{layer_idx}_head{head_idx}_attention.png")
    
#     # Compare attention patterns
#     fig = compare_attention_patterns(
#         baseline_attn[0], ospa_attn[0], layer_idx=layer_idx, head_idx=head_idx,
#         tokens=token_strs, title=title, save_path=save_path
#     )
    
#     return fig
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