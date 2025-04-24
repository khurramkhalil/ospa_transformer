import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ospa_attention import OSPAMultiHeadAttention
from baseline_models import VanillaMultiHeadAttention


def analyze_orthogonality_over_training(model, train_loader, epochs=5, optimizer=None, 
                                        criterion=None, device='cuda'):
    """
    Track orthogonality metrics during training.
    
    Args:
        model: The transformer model with orthogonal layers
        train_loader: DataLoader for training data
        epochs: Number of epochs to train
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to run on
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Get all orthogonal linear layers
    orth_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'compute_orthogonality_penalty'):
            orth_layers.append((name, module))
    
    # Initialize tracking
    step_count = 0
    max_steps = epochs * len(train_loader)
    steps = []
    orth_metrics = {name: [] for name, _ in orth_layers}
    loss_values = []
    
    # Training loop with tracking
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add orthogonality regularization if applicable
            if hasattr(model, 'get_orthogonality_penalty'):
                orth_penalty = model.get_orthogonality_penalty()
                loss = loss + 0.01 * orth_penalty  # Adjust penalty weight as needed
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record metrics
            steps.append(step_count)
            loss_values.append(loss.item())
            
            # Record orthogonality deviation for each layer
            for name, layer in orth_layers:
                orth_metrics[name].append(layer.compute_orthogonality_penalty().item())
            
            step_count += 1
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Plot orthogonality metrics over training
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(steps, loss_values)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Orthogonality deviation plot
    plt.subplot(2, 1, 2)
    for name, metrics in orth_metrics.items():
        plt.plot(steps, metrics, label=name)
    plt.title('Orthogonality Deviation During Training')
    plt.xlabel('Training Step')
    plt.ylabel('||WW^T - I||_F')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('orthogonality_training.png')
    
    return steps, orth_metrics, loss_values


def analyze_attention_heads(model, data_loader, device='cuda', num_samples=10):
    """
    Analyze similarity and specialization of attention heads.
    
    Args:
        model: The transformer model
        data_loader: DataLoader for input data
        device: Device to run on
        num_samples: Number of samples to analyze
    """
    model.eval()
    
    # Initialize storage for attention weights
    attention_weights = []
    head_outputs = []
    
    # Register hooks to capture attention
    hooks = []
    
    def attn_hook(module, input, output):
        # output is (attn_output, attn_weights)
        attention_weights.append(output[1].detach())
    
    def head_output_hook(module, input, output):
        # Capture individual head outputs before they're combined
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        head_outputs.append(output.detach())
    
    # Find attention modules
    for name, module in model.named_modules():
        if isinstance(module, (OSPAMultiHeadAttention, VanillaMultiHeadAttention)):
            hooks.append(module.register_forward_hook(attn_hook))
    
    # Process samples
    sample_count = 0
    for data, _ in data_loader:
        if sample_count >= num_samples:
            break
            
        data = data.to(device)
        with torch.no_grad():
            _ = model(data)
        
        sample_count += 1
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze attention patterns
    for layer_idx, attn_weights in enumerate(attention_weights):
        # Average over batches
        attn_weights = attn_weights.mean(dim=0)  # [num_heads, seq_len, seq_len]
        
        # Compute similarity between attention heads
        num_heads = attn_weights.shape[0]
        similarity_matrix = torch.zeros((num_heads, num_heads))
        
        for i in range(num_heads):
            for j in range(num_heads):
                # Flatten attention patterns
                head_i = attn_weights[i].view(-1)
                head_j = attn_weights[j].view(-1)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(head_i.unsqueeze(0), head_j.unsqueeze(0))
                similarity_matrix[i, j] = similarity.item()
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Layer {layer_idx}: Attention Head Similarity")
        plt.xlabel("Head Index")
        plt.ylabel("Head Index")
        plt.savefig(f"head_similarity_layer{layer_idx}.png")
        plt.close()
        
        # Compute entropy for each head
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)  # [num_heads, seq_len]
        mean_entropy = entropy.mean(dim=-1)  # [num_heads]
        
        # Plot entropy
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_heads), mean_entropy.cpu().numpy())
        plt.title(f"Layer {layer_idx}: Attention Head Entropy")
        plt.xlabel("Head Index")
        plt.ylabel("Mean Entropy")
        plt.grid(True, axis='y')
        plt.savefig(f"head_entropy_layer{layer_idx}.png")
        plt.close()
    
    return attention_weights


def analyze_representation_quality(model, data_loader, device='cuda'):
    """
    Analyze the quality of representations learned by different models.
    
    Args:
        model: The transformer model
        data_loader: DataLoader for input data
        device: Device to run on
    """
    model.eval()
    
    # Get representations from the last layer
    representations = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            
            # Forward pass to get representations
            if hasattr(model, 'encoder'):
                output = model.encoder(data)
            else:
                output = model(data)
            
            # Get representation of CLS token or average over sequence
            if output.size(0) > 1:  # If we have sequence dimension
                rep = output.mean(dim=0)  # Average over sequence
            else:
                rep = output
            
            representations.append(rep.cpu())
            labels.append(target)
    
    # Concatenate all representations and labels
    representations = torch.cat(representations, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Dimensionality reduction for visualization
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(representations.numpy())
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(representations.numpy())
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    # PCA plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('PCA of Learned Representations')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # t-SNE plot
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE of Learned Representations')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    plt.savefig('representation_visualization.png')
    
    # Calculate clustering metrics
    from sklearn.metrics import silhouette_score
    try:
        silhouette = silhouette_score(representations.numpy(), labels.numpy())
        print(f"Silhouette Score (representation quality): {silhouette:.4f}")
    except:
        print("Could not compute silhouette score (possibly due to number of classes)")
    
    return representations, labels


def analyze_gradient_flow(model, data_loader, device='cuda'):
    """
    Analyze gradient flow during backpropagation.
    
    Args:
        model: The transformer model
        data_loader: DataLoader for input data
        device: Device to run on
    """
    model.train()
    
    # Get a batch of data
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        break
    
    # Forward and backward pass
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Collect gradient norms for each layer
    grad_norms = {}
    grad_stds = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
            grad_stds[name] = param.grad.std().item()
    
    # Plot gradient norms
    plt.figure(figsize=(15, 10))
    
    # Filter out biases and focus on weights
    weight_names = [name for name in grad_norms.keys() if 'weight' in name]
    weight_norms = [grad_norms[name] for name in weight_names]
    
    # Sort by layer depth (assuming layer number is in the name)
    layer_indices = []
    for name in weight_names:
        # Extract layer number if present
        parts = name.split('.')
        layer_idx = next((i for i, part in enumerate(parts) if part.isdigit()), -1)
        if layer_idx != -1:
            layer_indices.append(int(parts[layer_idx]))
        else:
            layer_indices.append(-1)
    
    # Sort by layer index
    sorted_indices = sorted(range(len(layer_indices)), key=lambda i: layer_indices[i])
    sorted_names = [weight_names[i] for i in sorted_indices]
    sorted_norms = [weight_norms[i] for i in sorted_indices]
    
    # Gradient norm plot
    plt.subplot(2, 1, 1)
    plt.bar(range(len(sorted_names)), sorted_norms)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90)
    plt.title('Gradient Norms by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.grid(True, axis='y')
    
    # Compare attention types if both are present
    ospa_names = [name for name in weight_names if 'ospa' in name.lower()]
    vanilla_names = [name for name in weight_names if 'vanilla' in name.lower()]
    
    if ospa_names and vanilla_names:
        ospa_norms = [grad_norms[name] for name in ospa_names]
        vanilla_norms = [grad_norms[name] for name in vanilla_names]
        
        plt.subplot(2, 1, 2)
        plt.boxplot([ospa_norms, vanilla_norms], labels=['OSPA', 'Vanilla'])
        plt.title('Gradient Norm Distribution by Attention Type')
        plt.ylabel('Gradient Norm')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    
    return grad_norms, grad_stds


def estimate_condition_numbers(model, data_loader, device='cuda', n_samples=10):
    """
    Estimate condition numbers of Jacobian matrices for different attention mechanisms.
    
    Args:
        model: The transformer model
        data_loader: DataLoader for input data
        device: Device to run on
        n_samples: Number of samples to use
    """
    model.eval()
    
    # Collect attention modules
    attn_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, (OSPAMultiHeadAttention, VanillaMultiHeadAttention)):
            attn_modules[name] = module
    
    if not attn_modules:
        print("No attention modules found in the model")
        return
    
    # Condition number estimates
    condition_numbers = {name: [] for name in attn_modules}
    
    # Process samples
    sample_count = 0
    for data, _ in data_loader:
        if sample_count >= n_samples:
            break
            
        data = data.to(device)
        
        for name, module in attn_modules.items():
            # Extract weight matrices
            weights = []
            if hasattr(module, 'q_proj'):
                weights.append(module.q_proj.weight)
            if hasattr(module, 'k_proj'):
                weights.append(module.k_proj.weight)
            if hasattr(module, 'v_proj'):
                weights.append(module.v_proj.weight)
            if hasattr(module, 'out_proj'):
                weights.append(module.out_proj.weight)
            
            # Compute condition numbers
            for weight in weights:
                try:
                    # Compute singular values
                    u, s, v = torch.svd(weight)
                    
                    # Condition number is ratio of largest to smallest singular value
                    condition = s[0].item() / s[-1].item()
                    condition_numbers[name].append(condition)
                except:
                    pass
        
        sample_count += 1
    
    # Compute average condition numbers
    avg_conditions = {name: np.mean(conds) for name, conds in condition_numbers.items()}
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(avg_conditions.keys(), avg_conditions.values())
    plt.title('Average Condition Numbers of Weight Matrices')
    plt.xlabel('Module')
    plt.ylabel('Condition Number (lower is better)')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('condition_numbers.png')
    
    return avg_conditions


def analyze_mutual_information(model, data_loader, device='cuda', n_samples=10):
    """
    Analyze mutual information between attention heads.
    
    Args:
        model: The transformer model
        data_loader: DataLoader for input data
        device: Device to run on
        n_samples: Number of samples to use
    """
    model.eval()
    
    # Collect attention outputs
    attention_outputs = []
    
    # Register hooks
    hooks = []
    
    def hook_fn(module, input, output):
        # output[0] is attention output, shape [seq_len, batch_size, embed_dim]
        attention_outputs.append(output[0].detach())
    
    # Add hooks to attention modules
    for name, module in model.named_modules():
        if isinstance(module, (OSPAMultiHeadAttention, VanillaMultiHeadAttention)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Process samples
    sample_count = 0
    for data, _ in data_loader:
        if sample_count >= n_samples:
            break
            
        data = data.to(device)
        with torch.no_grad():
            _ = model(data)
        
        sample_count += 1
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute correlation between attention outputs
    for layer_idx, attn_output in enumerate(attention_outputs):
        # Average over sequence dimension
        attn_output = attn_output.mean(dim=0)  # [batch_size, embed_dim]
        
        # Calculate correlation matrix
        correlation = torch.corrcoef(attn_output.t())
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation.cpu().numpy(), cmap='coolwarm', center=0)
        plt.title(f'Layer {layer_idx}: Correlation Matrix of Attention Features')
        plt.tight_layout()
        plt.savefig(f'attention_correlation_layer{layer_idx}.png')
        plt.close()
    
    return attention_outputs


def compare_models_on_long_sequences(models, seq_lengths=[256, 512, 1024, 2048, 4096], 
                                     d_model=512, batch_size=16, device='cuda'):
    """
    Compare different models on increasingly long sequences.
    
    Args:
        models: Dictionary of models {name: model}
        seq_lengths: List of sequence lengths to test
        d_model: Embedding dimension
        batch_size: Batch size
        device: Device to run on
    """
    results = {
        'model': [],
        'seq_len': [],
        'perplexity': [],
        'accuracy': []
    }
    
    # Generate synthetic data
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        # Create random input and target
        src = torch.randint(0, 10000, (seq_len, batch_size)).to(device)
        tgt = torch.randint(0, 10000, (seq_len * batch_size,)).to(device)
        
        # Embedding
        src_embed = torch.randn(seq_len, batch_size, d_model).to(device)
        
        for name, model in models.items():
            model.eval()
            
            try:
                # Forward pass
                with torch.no_grad():
                    if hasattr(model, 'encoder'):
                        output = model.encoder(src_embed)
                    else:
                        output = model(src_embed)
                
                # Compute loss
                criterion = nn.CrossEntropyLoss()
                
                # Simple projection for loss computation
                proj = nn.Linear(d_model, 10000).to(device)
                logits = proj(output.reshape(-1, d_model))
                
                loss = criterion(logits, tgt)
                perplexity = torch.exp(loss).item()
                
                # Compute accuracy
                pred = logits.argmax(dim=1)
                accuracy = (pred == tgt).float().mean().item()
                
                # Store results
                results['model'].append(name)
                results['seq_len'].append(seq_len)
                results['perplexity'].append(perplexity)
                results['accuracy'].append(accuracy)
                
                print(f"  {name}: Perplexity = {perplexity:.4f}, Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"  Error with {name} at length {seq_len}: {e}")
                # Store failure
                results['model'].append(name)
                results['seq_len'].append(seq_len)
                results['perplexity'].append(float('nan'))
                results['accuracy'].append(float('nan'))
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Perplexity
    plt.subplot(2, 1, 1)
    for model_name in set(results['model']):
        model_results = [(seq_len, perp) for model, seq_len, perp, _ in 
                         zip(results['model'], results['seq_len'], results['perplexity'], results['accuracy']) 
                         if model == model_name and not np.isnan(perp)]
        if model_results:
            seq_lens, perps = zip(*model_results)
            plt.plot(seq_lens, perps, marker='o', label=model_name)
    
    plt.title('Perplexity vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Perplexity (lower is better)')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(2, 1, 2)
    for model_name in set(results['model']):
        model_results = [(seq_len, acc) for model, seq_len, _, acc in 
                         zip(results['model'], results['seq_len'], results['perplexity'], results['accuracy']) 
                         if model == model_name and not np.isnan(acc)]
        if model_results:
            seq_lens, accs = zip(*model_results)
            plt.plot(seq_lens, accs, marker='o', label=model_name)
    
    plt.title('Accuracy vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('long_sequence_comparison.png')
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description='Analysis tools for OSPA models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--analysis', type=str, default='all', 
                        choices=['orthogonality', 'attention', 'representation', 'gradient', 'condition', 'mutual_info', 'long_seq', 'all'],
                        help='Which analysis to run')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run analysis on')
    
    args = parser.parse_args()
    
    # Load model and data
    # This is a placeholder - actual loading would depend on your model and data structure
    model = torch.load(args.model_path)
    model.to(args.device)
    
    # Load data
    # This is a placeholder - actual loading would depend on your dataset
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(
        torch.randn(1000, 512),  # Input features
        torch.randint(0, 10, (1000,))  # Labels
    )
    data_loader = DataLoader(dummy_data, batch_size=32, shuffle=True)
    
    # Run selected analysis
    if args.analysis in ['orthogonality', 'all'] and hasattr(model, 'get_orthogonality_penalty'):
        print("Analyzing orthogonality over training...")
        analyze_orthogonality_over_training(model, data_loader, device=args.device)
    
    if args.analysis in ['attention', 'all']:
        print("Analyzing attention heads...")
        analyze_attention_heads(model, data_loader, device=args.device)
    
    if args.analysis in ['representation', 'all']:
        print("Analyzing representation quality...")
        analyze_representation_quality(model, data_loader, device=args.device)
    
    if args.analysis in ['gradient', 'all']:
        print("Analyzing gradient flow...")
        analyze_gradient_flow(model, data_loader, device=args.device)
    
    if args.analysis in ['condition', 'all']:
        print("Estimating condition numbers...")
        estimate_condition_numbers(model, data_loader, device=args.device)
    
    if args.analysis in ['mutual_info', 'all']:
        print("Analyzing mutual information...")
        analyze_mutual_information(model, data_loader, device=args.device)
    
    if args.analysis in ['long_seq', 'all']:
        print("Comparing models on long sequences...")
        # This requires multiple models, so it's just a placeholder
        models = {'OSPA': model}
        compare_models_on_long_sequences(models, device=args.device)