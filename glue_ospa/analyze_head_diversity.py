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
sns.set(font_scale=1.1) # Slightly smaller default font
sns.set_style("whitegrid")

def get_model_type_from_path(model_path):
    """Infer model type from path/filename."""
    filename = os.path.basename(model_path).lower()
    dirname = os.path.basename(os.path.dirname(model_path)).lower()
    path_str = (dirname + "_" + filename)

    if 'ospa' in path_str:
        return 'ospa'
    elif 'vanilla' in path_str:
        return 'vanilla'
    elif 'linformer' in path_str:
        return 'linformer' # Add other types if needed
    else:
        logger.warning(f"Could not infer model type from path: {model_path}. Assuming 'vanilla'.")
        return 'vanilla' # Default or raise error

def extract_attention_weights(state_dict, args):
    """
    Extract Q, K, V, and Output projection weights for each attention layer.

    Returns:
        list: A list of tuples, where each tuple is (layer_idx, q_weight, k_weight, v_weight, out_weight)
    """
    layer_weights = {} # Use dict to store by layer index first
    keys = list(state_dict.keys())

    # Regex to find layer index and attention components
    # Example patterns: transformer.encoder.layers.0.self_attn.q_proj.weight
    #                 encoder.layers.0.self_attn.k_proj.weight
    # Adapt if your naming convention is different
    pattern = re.compile(r"(?:transformer\.encoder|encoder)\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight")

    for key in keys:
        match = pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            component_type = match.group(2) # 'q_proj', 'k_proj', 'v_proj', 'out_proj'
            weight = state_dict[key]

            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = {}

            # Store the weight tensor based on its type
            layer_weights[layer_idx][component_type + '_weight'] = weight
            # logger.debug(f"Extracted {component_type}_weight for layer {layer_idx}") # Debug log

    # Convert dict to list of tuples, ensuring all components are present
    extracted_weights_list = []
    num_layers = args.nlayers # Use nlayers from args
    logger.info(f"Expecting {num_layers} layers based on args.")

    for layer_idx in range(num_layers):
        if layer_idx in layer_weights:
            layer_data = layer_weights[layer_idx]
            # Check if all required weights were found for this layer
            if all(k in layer_data for k in ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'out_proj_weight']):
                extracted_weights_list.append(
                    (
                        layer_idx,
                        layer_data['q_proj_weight'],
                        layer_data['k_proj_weight'],
                        layer_data['v_proj_weight'],
                        layer_data['out_proj_weight']
                    )
                )
                logger.info(f"Successfully extracted all projection weights for layer {layer_idx}")
            else:
                logger.warning(f"Missing one or more projection weights for layer {layer_idx}. Skipping layer.")
        else:
             logger.warning(f"No attention weights found for layer {layer_idx}.")

    if not extracted_weights_list:
         logger.error("Failed to extract weights for any layer. Check model structure and state dict keys.")

    return extracted_weights_list

def compute_head_vectors(layer_weights_list, nhead, d_model):
    """
    Compute a vector representation for each attention head based on projection columns.
    """
    if not layer_weights_list:
        logger.error("No layer weights provided for head vector computation.")
        return []

    head_dim = d_model // nhead
    if d_model % nhead != 0:
        logger.error(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        return []

    flattened_heads = []
    num_layers = len(layer_weights_list)

    for layer_idx_tuple in layer_weights_list:
        layer_idx, q_weight, k_weight, v_weight, _ = layer_idx_tuple # Ignore out_weight for diversity calc
        # Weights are expected shape [d_model, d_model] for OSPA/Vanilla (if implemented similarly)
        # Or potentially [d_model_out, d_model_in] -> Need to confirm shape logic
        # Assuming square [d_model, d_model] for now based on OSPA implementation

        if q_weight.shape[0] != d_model or q_weight.shape[1] != d_model:
             logger.warning(f"Layer {layer_idx} Q weight shape {q_weight.shape} unexpected. Expected [{d_model}, {d_model}]. Skipping layer for head diversity.")
             continue
        if k_weight.shape[0] != d_model or k_weight.shape[1] != d_model:
             logger.warning(f"Layer {layer_idx} K weight shape {k_weight.shape} unexpected. Expected [{d_model}, {d_model}]. Skipping layer.")
             continue
        if v_weight.shape[0] != d_model or v_weight.shape[1] != d_model:
             logger.warning(f"Layer {layer_idx} V weight shape {v_weight.shape} unexpected. Expected [{d_model}, {d_model}]. Skipping layer.")
             continue


        for head_idx in range(nhead):
            try:
                # Calculate start and end column indices for this head's subspace
                start_col = head_idx * head_dim
                end_col = (head_idx + 1) * head_dim

                # Extract column slices corresponding to the head's subspace
                # Shape of slice: [d_model, head_dim]
                q_head_cols = q_weight[:, start_col:end_col]
                k_head_cols = k_weight[:, start_col:end_col]
                v_head_cols = v_weight[:, start_col:end_col]

                # Flatten the column slices and concatenate
                head_vector = torch.cat([
                    q_head_cols.flatten(),
                    k_head_cols.flatten(),
                    v_head_cols.flatten()
                ]).cpu().numpy() # Move to CPU as numpy array

                flattened_heads.append(head_vector)
            except Exception as e:
                logger.error(f"Error processing head {head_idx} in layer {layer_idx}: {e}")
                continue # Skip this head if error occurs

    logger.info(f"Generated {len(flattened_heads)} head vectors across {num_layers} layers.")
    return flattened_heads

def compute_similarity_and_diversity(head_vectors):
    """Compute cosine similarity matrix and diversity score."""
    if not head_vectors or len(head_vectors) < 2:
        logger.warning("Not enough head vectors to compute similarity.")
        # Return default values indicating failure or invalid input
        return np.zeros((1, 1)), 0.0 # Or None, None

    # Compute pairwise cosine similarity
    try:
        similarity_matrix = cosine_similarity(head_vectors)
        # Ensure diagonal is exactly 1 (it should be, but floating point...)
        np.fill_diagonal(similarity_matrix, 1.0)
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}")
        return np.zeros((len(head_vectors), len(head_vectors))), 0.0

    # Compute diversity score: Average off-diagonal similarity
    # Lower score means higher diversity (less similarity between different heads)
    num_heads_total = similarity_matrix.shape[0]
    if num_heads_total <= 1:
        return similarity_matrix, 0.0 # No off-diagonal elements

    # Sum of all elements minus sum of diagonal, divided by number of off-diagonal elements
    off_diagonal_sum = np.sum(similarity_matrix) - np.sum(np.diag(similarity_matrix))
    num_off_diagonal = num_heads_total * (num_heads_total - 1)
    diversity_score = off_diagonal_sum / num_off_diagonal if num_off_diagonal > 0 else 0.0

    # Clamp score slightly for stability if needed, though raw score is fine
    # diversity_score = max(0.0, min(1.0, diversity_score)) # Optional clamping

    return similarity_matrix, diversity_score

def check_orthogonality(layer_weights_list):
    """
    Check orthogonality of OSPA projection matrices (Q, K, V, O).
    Uses the squared Frobenius norm ||P^T P - I||^2 or ||P P^T - I||^2.
    """
    orthogonality_scores = {}
    all_errors = []

    for layer_idx_tuple in layer_weights_list:
        layer_idx, P_q, P_k, P_v, P_o = layer_idx_tuple # Unpack all four weights

        layer_errors = {}
        for name, P in [('P_Q', P_q), ('P_K', P_k), ('P_V', P_v), ('P_O', P_o)]:
            try:
                rows, cols = P.shape
                identity = None
                prod = None

                # Check column orthogonality (P^T P = I) if rows >= cols (Tall or Square)
                if rows >= cols:
                    prod = P.t() @ P
                    identity = torch.eye(cols, device=P.device, dtype=P.dtype)
                    col_error = torch.norm(prod - identity, p='fro').pow(2).item()
                    layer_errors[f'{name}_col_error'] = col_error
                    # Store col error as the primary error for tall/square
                    layer_errors[f'{name}_error'] = col_error


                # Check row orthogonality (P P^T = I) if cols > rows (Fat)
                # Note: For a true orthogonal matrix (square), both should hold.
                # For projection onto subspace (non-square Stiefel), only one holds.
                # Let's calculate both where applicable for analysis.
                if cols >= rows:
                     prod_row = P @ P.t()
                     identity_row = torch.eye(rows, device=P.device, dtype=P.dtype)
                     row_error = torch.norm(prod_row - identity_row, p='fro').pow(2).item()
                     layer_errors[f'{name}_row_error'] = row_error
                     # If fat matrix, row error is the primary one
                     if cols > rows:
                         layer_errors[f'{name}_error'] = row_error


                # If square, ideally both errors are low. We stored one as primary 'error'.

            except Exception as e:
                logger.error(f"Error computing orthogonality for {name} in layer {layer_idx}: {e}")
                layer_errors[f'{name}_error'] = float('nan') # Mark error for this matrix

        # Calculate average primary error for the layer
        valid_errors = [err for err in [layer_errors.get('P_Q_error'), layer_errors.get('P_K_error'),
                                        layer_errors.get('P_V_error'), layer_errors.get('P_O_error')] if err is not None and not np.isnan(err)]
        avg_error = sum(valid_errors) / len(valid_errors) if valid_errors else float('nan')
        layer_errors['avg_error'] = avg_error
        if not np.isnan(avg_error):
            all_errors.append(avg_error)

        orthogonality_scores[f'layer_{layer_idx}'] = layer_errors

    # Compute overall average error across layers
    if all_errors:
        orthogonality_scores['overall_avg_error'] = sum(all_errors) / len(all_errors)
    else:
        orthogonality_scores['overall_avg_error'] = float('nan')


    return orthogonality_scores

def analyze_model(model_path, output_dir, args):
    """
    Analyze head diversity and orthogonality for a single model.
    """
    logger.info(f"--- Analyzing Model: {model_path} ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load State Dict ---
    try:
        # Use weights_only=True for security if loading untrusted files
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False) # Set False if non-pytorch objects are saved
        logger.info(f"Successfully loaded state dict from {model_path}")
        # Handle nested state dicts common in some saving libraries
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            logger.info("Extracted nested state_dict")
        if isinstance(state_dict, dict) and 'model' in state_dict: # Another common pattern
             state_dict = state_dict['model']
             logger.info("Extracted nested 'model' state_dict")

    except Exception as e:
        logger.error(f"Error loading state dict from {model_path}: {e}")
        return None # Cannot proceed without state dict

    # --- 2. Determine Model Type and Extract Weights ---
    model_type = get_model_type_from_path(model_path)
    logger.info(f"Inferred model type: {model_type}")
    is_ospa = (model_type == 'ospa')

    layer_weights_list = extract_attention_weights(state_dict, args)
    if not layer_weights_list:
        logger.error(f"Failed to extract weights for model: {model_path}")
        return None

    # --- 3. Compute Head Vectors and Diversity ---
    head_vectors = compute_head_vectors(layer_weights_list, args.nhead, args.d_model)
    similarity_matrix, diversity_score = compute_similarity_and_diversity(head_vectors)

    # --- 4. Check Orthogonality (for OSPA models) ---
    orthogonality_scores = None
    if is_ospa:
        logger.info("Checking orthogonality for OSPA model...")
        orthogonality_scores = check_orthogonality(layer_weights_list)
        if orthogonality_scores:
             logger.info(f"Overall Avg Orthogonality Error (Squared Norm): {orthogonality_scores.get('overall_avg_error', 'N/A'):.4e}")

    # --- 5. Compile and Save Results ---
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'diversity_score': diversity_score, # Lower is better (less similar)
        'num_layers_extracted': len(layer_weights_list),
        'num_heads_total': len(head_vectors),
        'args_nhead': args.nhead,
        'args_d_model': args.d_model,
        'orthogonality_scores': orthogonality_scores # Will be None for non-OSPA
    }

    results_json_path = os.path.join(output_dir, "analysis_metrics.json")
    try:
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x) # Handle numpy floats
        logger.info(f"Saved analysis metrics to {results_json_path}")
    except Exception as e:
        logger.error(f"Error saving analysis metrics JSON: {e}")


    # --- 6. Create Visualizations ---
    # Head similarity heatmap
    if similarity_matrix is not None and similarity_matrix.shape[0] > 1:
        try:
            plt.figure(figsize=(10, 8))
            mask = np.eye(similarity_matrix.shape[0], dtype=bool) # Mask diagonal
            sns.heatmap(
                similarity_matrix,
                cmap='viridis_r', # Reversed Viridis: yellow=high sim, purple=low sim
                mask=mask,
                vmin=np.min(similarity_matrix[~mask]) if np.any(~mask) else 0, # Adjust vmin based on off-diagonal
                vmax=1.0, # Max similarity is 1
                square=True,
                xticklabels=max(1, similarity_matrix.shape[1] // 10), # Adjust tick frequency
                yticklabels=max(1, similarity_matrix.shape[0] // 10)
            )
            plt.title(f'Head Similarity (Avg Off-Diag CosSim: {diversity_score:.4f}) - {os.path.basename(model_path)}')
            plt.xlabel('Head Index (Layer * nhead + Head)')
            plt.ylabel('Head Index (Layer * nhead + Head)')
            plt.tight_layout()
            save_path = os.path.join(output_dir, "head_similarity.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"Saved head similarity heatmap to {save_path}")
        except Exception as e:
            logger.error(f"Error creating head similarity visualization: {e}")

    # Orthogonality error plot (for OSPA models)
    if is_ospa and orthogonality_scores and len(orthogonality_scores) > 1:
        try:
            layers = sorted([int(k.split('_')[1]) for k in orthogonality_scores if k.startswith('layer_')])
            if not layers: raise ValueError("No layer data found in orthogonality scores")

            metrics_to_plot = ['P_Q_error', 'P_K_error', 'P_V_error', 'P_O_error', 'avg_error']
            plot_data = {metric: [] for metric in metrics_to_plot}
            plot_layers = []

            for layer_idx in layers:
                 layer_key = f'layer_{layer_idx}'
                 if layer_key in orthogonality_scores:
                     scores = orthogonality_scores[layer_key]
                     has_data = False
                     for metric in metrics_to_plot:
                         # Use primary 'error' key which accounts for shape
                         error_key = metric if metric == 'avg_error' else f"{metric.split('_')[0]}_error"
                         value = scores.get(error_key, float('nan'))
                         plot_data[metric].append(value)
                         if not np.isnan(value): has_data = True
                     if has_data: plot_layers.append(layer_idx) # Only include layers with valid data

            # Filter out metrics with no valid data
            valid_plot_data = {k:v for k,v in plot_data.items() if not all(np.isnan(val) for val in v)}


            if plot_layers and valid_plot_data:
                plt.figure(figsize=(12, 7))
                styles = {'avg_error': '*-', 'P_Q_error': 'o-', 'P_K_error': 's-', 'P_V_error': '^-', 'P_O_error': 'd-'}
                labels = {'avg_error': 'Average Error', 'P_Q_error': 'Query (P_Q)', 'P_K_error': 'Key (P_K)', 'P_V_error': 'Value (P_V)', 'P_O_error': 'Output (P_O)'}

                for metric, values in valid_plot_data.items():
                    plt.plot(plot_layers, values, styles.get(metric, '.-'), label=labels.get(metric, metric), alpha=0.8)

                plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Perfect Orthogonality')
                plt.yscale('log') # Use log scale as errors can be small
                plt.xlabel('Layer Index')
                plt.ylabel('Orthogonality Error (Squared Norm, Log Scale)')
                plt.title(f'OSPA Orthogonality Error per Layer - {os.path.basename(model_path)}')
                plt.xticks(plot_layers) # Ensure ticks are at layer indices
                plt.legend(loc='best')
                plt.grid(True, which="both", ls="--", alpha=0.3) # Grid for both major and minor ticks on log scale
                plt.tight_layout()
                save_path = os.path.join(output_dir, "orthogonality_error.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                logger.info(f"Saved orthogonality error plot to {save_path}")
            else:
                 logger.warning("No valid orthogonality data to plot.")

        except Exception as e:
            logger.error(f"Error creating orthogonality visualization: {e}")

    logger.info(f"--- Analysis complete for: {model_path} ---")
    return results


def compare_models(model_results, output_dir):
    """
    Create comparison visualizations for multiple models based on extracted results.
    """
    if not model_results or len(model_results) < 2:
        logger.warning("Need at least two valid model analysis results to compare.")
        return

    logger.info(f"--- Creating Comparison Visualizations for {len(model_results)} Models ---")
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for comparison plot
    plot_data = []
    for model_path, results in model_results.items():
        if results: # Ensure results are not None
            model_name = os.path.basename(results.get('model_path', 'Unknown')).split('.')[0]
            # Try to make name more readable
            model_name = model_name.replace('_transformer', '').replace('_model', '')
            plot_data.append({
                'name': model_name,
                'diversity': results.get('diversity_score', float('nan')),
                'type': results.get('model_type', 'unknown')
            })
        else:
             logger.warning(f"Skipping comparison for model path due to missing results: {model_path}")

    if len(plot_data) < 2:
        logger.error("Not enough models with valid results for comparison plotting.")
        return

    # Sort data for plotting (e.g., by type then diversity)
    plot_data.sort(key=lambda x: (x['type'] != 'vanilla', x['diversity'])) # Vanilla first, then sorted by diversity

    model_names = [d['name'] for d in plot_data]
    diversity_scores = [d['diversity'] for d in plot_data]
    model_types = [d['type'] for d in plot_data]

    # Create bar chart of diversity scores
    try:
        plt.figure(figsize=(max(8, len(model_names) * 1.5), 6)) # Adjust width based on number of models

        # Define colors for model types
        color_map = {'vanilla': 'skyblue', 'ospa': 'salmon', 'linformer': 'lightgreen', 'unknown': 'grey'}
        colors = [color_map.get(t, 'grey') for t in model_types]

        bars = plt.bar(range(len(model_names)), diversity_scores, color=colors)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right', fontsize=10)

        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=9)

        plt.ylabel('Avg. Off-Diagonal Head Cosine Similarity (Lower is more Diverse)')
        plt.title('Comparison of Attention Head Diversity')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout() # Adjust layout
        save_path = os.path.join(output_dir, "model_diversity_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved diversity comparison plot to {save_path}")
    except Exception as e:
        logger.error(f"Error creating diversity comparison plot: {e}")

    # Save comparison data to JSON
    comparison_data = plot_data # Save the sorted list of dicts
    comparison_json_path = os.path.join(output_dir, "diversity_comparison_data.json")
    try:
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Saved comparison data to {comparison_json_path}")
    except Exception as e:
        logger.error(f"Error saving comparison data JSON: {e}")

    logger.info("--- Model Comparison Complete ---")


# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(
        description='Analyze attention head diversity and orthogonality in Transformer models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # --- Input Arguments ---
    parser.add_argument('--model_path', type=str,
                        help='Path to a single model checkpoint file (.pt) to analyze.')
    parser.add_argument('--model_paths', nargs='+',
                        help='Paths to multiple model checkpoint files (.pt) for comparison.')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results and visualizations.')

    # --- Model Hyperparameters (Required for Interpretation) ---
    # These should match the models being analyzed
    parser.add_argument('--d_model', type=int, required=True,
                        help='Model embedding dimension (required).')
    parser.add_argument('--nhead', type=int, required=True,
                        help='Number of attention heads (required).')
    parser.add_argument('--nlayers', type=int, required=True,
                        help='Number of encoder layers analyzed (required).')
    # dim_feedforward and task are not strictly needed for this analysis script

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.model_paths: # Compare mode
        if len(args.model_paths) < 2:
            parser.error("--model_paths requires at least two model paths for comparison.")
        logger.info(f"Comparing {len(args.model_paths)} models.")

        # Analyze multiple models
        results = {}
        # Base output dir for comparison runs
        base_output_dir = args.output_dir

        for model_path in args.model_paths:
            if not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}. Skipping.")
                continue
            # Create a subdirectory for each model's individual results
            model_name = os.path.basename(model_path).split('.')[0]
            model_output_dir = os.path.join(base_output_dir, model_name)
            analysis_result = analyze_model(model_path, model_output_dir, args)
            # Store result keyed by the original path for reference
            results[model_path] = analysis_result

        # Create comparison visualizations in the main output dir
        compare_models(results, base_output_dir)

    elif args.model_path: # Single model mode
        if not os.path.exists(args.model_path):
             parser.error(f"Model path not found: {args.model_path}")
        logger.info("Analyzing single model.")
        analyze_model(args.model_path, args.output_dir, args)
    else:
        parser.error("Either --model_path (for single analysis) or --model_paths (for comparison) must be specified.")

    logger.info("--- Analysis Script Finished ---")

if __name__ == "__main__":
    # Example Usage:
    # Single: python analyze_diversity.py --model_path path/to/your/model.pt --output_dir results/model_analysi
    # Compare: python analyze_diversity.py --model_paths path/vanilla.pt path/ospa.pt --output_dir results/comparison --d_model 512 --nhead 8 --nlayers 6
    main()