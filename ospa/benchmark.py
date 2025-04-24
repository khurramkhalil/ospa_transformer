import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from orthogonal_linear import OrthogonalLinear
from ospa_attention import OSPAMultiHeadAttention
from ospa_transformer import OSPATransformer
from baseline_models import VanillaTransformer, LinformerTransformer


def benchmark_linear_layers(sizes=[128, 256, 512, 1024], batch_size=32, seq_len=512, device='cuda'):
    """Benchmark OrthogonalLinear vs nn.Linear."""
    results = {
        'size': [],
        'layer_type': [],
        'forward_time': [],
        'backward_time': [],
        'memory_usage': []
    }
    
    for size in sizes:
        # Create inputs
        x = torch.randn(seq_len, batch_size, size, device=device)
        
        # Standard Linear
        standard = nn.Linear(size, size).to(device)
        
        # OrthogonalLinear with different modes
        orthogonal_init = OrthogonalLinear(size, size, mode='init').to(device)
        orthogonal_regularize = OrthogonalLinear(size, size, mode='regularize').to(device)
        orthogonal_strict = OrthogonalLinear(size, size, mode='strict').to(device)
        
        # Benchmark standard Linear
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm-up
        for _ in range(10):
            _ = standard(x)
            
        # Forward pass timing
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100):
            y = standard(x)
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event) / 100
        
        # Backward pass timing
        loss = y.sum()
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100):
            loss.backward(retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event) / 100
        
        # Memory usage
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        y = standard(x)
        loss = y.sum()
        loss.backward()
        mem_after = torch.cuda.memory_allocated()
        memory_usage = (mem_after - mem_before) / (1024 * 1024)  # MB
        
        # Store results
        results['size'].append(size)
        results['layer_type'].append('nn.Linear')
        results['forward_time'].append(forward_time)
        results['backward_time'].append(backward_time)
        results['memory_usage'].append(memory_usage)
        
        # Benchmark OrthogonalLinear with different modes
        for orth_layer, mode_name in [(orthogonal_init, 'init'), 
                                     (orthogonal_regularize, 'regularize'), 
                                     (orthogonal_strict, 'strict')]:
            # Warm-up
            for _ in range(10):
                _ = orth_layer(x)
                
            # Forward pass timing
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(100):
                y = orth_layer(x)
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event) / 100
            
            # Backward pass timing
            loss = y.sum()
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(100):
                loss.backward(retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event) / 100
            
            # Memory usage
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            y = orth_layer(x)
            loss = y.sum()
            loss.backward()
            mem_after = torch.cuda.memory_allocated()
            memory_usage = (mem_after - mem_before) / (1024 * 1024)  # MB
            
            # Store results
            results['size'].append(size)
            results['layer_type'].append(f'OrthogonalLinear-{mode_name}')
            results['forward_time'].append(forward_time)
            results['backward_time'].append(backward_time)
            results['memory_usage'].append(memory_usage)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Forward time
    for layer_type in df['layer_type'].unique():
        subset = df[df['layer_type'] == layer_type]
        axs[0].plot(subset['size'], subset['forward_time'], marker='o', label=layer_type)
    axs[0].set_title('Forward Pass Time')
    axs[0].set_xlabel('Size')
    axs[0].set_ylabel('Time (ms)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Backward time
    for layer_type in df['layer_type'].unique():
        subset = df[df['layer_type'] == layer_type]
        axs[1].plot(subset['size'], subset['backward_time'], marker='o', label=layer_type)
    axs[1].set_title('Backward Pass Time')
    axs[1].set_xlabel('Size')
    axs[1].set_ylabel('Time (ms)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Memory usage
    for layer_type in df['layer_type'].unique():
        subset = df[df['layer_type'] == layer_type]
        axs[2].plot(subset['size'], subset['memory_usage'], marker='o', label=layer_type)
    axs[2].set_title('Memory Usage')
    axs[2].set_xlabel('Size')
    axs[2].set_ylabel('Memory (MB)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_benchmark.png')
    
    return df


def benchmark_attention(sizes=[128, 256, 512], heads=[4, 8, 16], seq_lens=[128, 256, 512, 1024], 
                        batch_size=32, device='cuda'):
    """Benchmark OSPAMultiHeadAttention vs VanillaMultiHeadAttention vs LinformerSelfAttention."""
    results = {
        'embed_dim': [],
        'num_heads': [],
        'seq_len': [],
        'attn_type': [],
        'forward_time': [],
        'backward_time': [],
        'memory_usage': []
    }
    
    for size in sizes:
        for num_heads in heads:
            if size % num_heads != 0:
                continue  # Skip invalid combinations
                
            for seq_len in seq_lens:
                # Create inputs
                q = torch.randn(seq_len, batch_size, size, device=device)
                k = torch.randn(seq_len, batch_size, size, device=device)
                v = torch.randn(seq_len, batch_size, size, device=device)
                
                # Create attention modules
                from baseline_models import VanillaMultiHeadAttention, LinformerSelfAttention
                vanilla_attn = VanillaMultiHeadAttention(size, num_heads).to(device)
                linformer_attn = LinformerSelfAttention(size, num_heads, k=min(seq_len // 2, 128)).to(device)
                ospa_attn_init = OSPAMultiHeadAttention(size, num_heads, orth_mode='init').to(device)
                ospa_attn_strict = OSPAMultiHeadAttention(size, num_heads, orth_mode='strict').to(device)
                
                # Benchmark each attention type
                for attn_module, attn_name in [
                    (vanilla_attn, 'Vanilla'),
                    (linformer_attn, 'Linformer'),
                    (ospa_attn_init, 'OSPA-init'),
                    (ospa_attn_strict, 'OSPA-strict')
                ]:
                    # Warm-up
                    for _ in range(5):
                        _ = attn_module(q, k, v)
                        
                    # Forward pass timing
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    torch.cuda.synchronize()
                    start_event.record()
                    for _ in range(20):
                        output, _ = attn_module(q, k, v)
                    end_event.record()
                    torch.cuda.synchronize()
                    forward_time = start_event.elapsed_time(end_event) / 20
                    
                    # Backward pass timing
                    loss = output.sum()
                    torch.cuda.synchronize()
                    start_event.record()
                    for _ in range(20):
                        loss.backward(retain_graph=True)
                    end_event.record()
                    torch.cuda.synchronize()
                    backward_time = start_event.elapsed_time(end_event) / 20
                    
                    # Memory usage
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                    output, _ = attn_module(q, k, v)
                    loss = output.sum()
                    loss.backward()
                    mem_after = torch.cuda.memory_allocated()
                    memory_usage = (mem_after - mem_before) / (1024 * 1024)  # MB
                    
                    # Store results
                    results['embed_dim'].append(size)
                    results['num_heads'].append(num_heads)
                    results['seq_len'].append(seq_len)
                    results['attn_type'].append(attn_name)
                    results['forward_time'].append(forward_time)
                    results['backward_time'].append(backward_time)
                    results['memory_usage'].append(memory_usage)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results - focus on sequence length scaling
    plt.figure(figsize=(15, 10))
    
    # Group by embed_dim, num_heads and attn_type
    for size in sizes:
        for num_heads in heads:
            if size % num_heads != 0:
                continue
                
            plt.figure(figsize=(15, 5))
            # Forward time vs seq_len
            plt.subplot(1, 3, 1)
            for attn_type in df['attn_type'].unique():
                subset = df[(df['embed_dim'] == size) & (df['num_heads'] == num_heads) & (df['attn_type'] == attn_type)]
                if not subset.empty:
                    plt.plot(subset['seq_len'], subset['forward_time'], marker='o', label=attn_type)
            plt.title(f'Forward Time (d={size}, h={num_heads})')
            plt.xlabel('Sequence Length')
            plt.ylabel('Time (ms)')
            plt.legend()
            plt.grid(True)
            
            # Backward time vs seq_len
            plt.subplot(1, 3, 2)
            for attn_type in df['attn_type'].unique():
                subset = df[(df['embed_dim'] == size) & (df['num_heads'] == num_heads) & (df['attn_type'] == attn_type)]
                if not subset.empty:
                    plt.plot(subset['seq_len'], subset['backward_time'], marker='o', label=attn_type)
            plt.title(f'Backward Time (d={size}, h={num_heads})')
            plt.xlabel('Sequence Length')
            plt.ylabel('Time (ms)')
            plt.legend()
            plt.grid(True)
            
            # Memory usage vs seq_len
            plt.subplot(1, 3, 3)
            for attn_type in df['attn_type'].unique():
                subset = df[(df['embed_dim'] == size) & (df['num_heads'] == num_heads) & (df['attn_type'] == attn_type)]
                if not subset.empty:
                    plt.plot(subset['seq_len'], subset['memory_usage'], marker='o', label=attn_type)
            plt.title(f'Memory Usage (d={size}, h={num_heads})')
            plt.xlabel('Sequence Length')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'attn_benchmark_d{size}_h{num_heads}.png')
            plt.close()
    
    return df


def benchmark_transformer_models(d_model=512, nhead=8, nlayers=6, seq_lens=[128, 256, 512, 1024, 2048], 
                                batch_size=16, device='cuda'):
    """Benchmark full transformer models on various sequence lengths."""
    results = {
        'model_type': [],
        'seq_len': [],
        'forward_time': [],
        'backward_time': [],
        'memory_usage': [],
        'param_count': []
    }
    
    for seq_len in seq_lens:
        # Create input
        src = torch.randn(seq_len, batch_size, d_model, device=device)
        
        # Create models
        vanilla = VanillaTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=d_model * 4
        ).to(device)
        
        linformer = LinformerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=d_model * 4,
            k=min(seq_len // 2, 256)
        ).to(device)
        
        ospa_init = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=d_model * 4,
            orth_mode='init'
        ).to(device)
        
        ospa_regularize = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=d_model * 4,
            orth_mode='regularize'
        ).to(device)
        
        ospa_strict = OSPATransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dim_feedforward=d_model * 4,
            orth_mode='strict'
        ).to(device)
        
        # Benchmark each model
        for model, model_name in [
            (vanilla, 'Vanilla'),
            (linformer, 'Linformer'),
            (ospa_init, 'OSPA-init'),
            (ospa_regularize, 'OSPA-regularize'),
            (ospa_strict, 'OSPA-strict')
        ]:
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Warm-up
            for _ in range(3):
                _ = model(src)
                
            # Forward pass timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10):
                output = model(src)
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event) / 10
            
            # Backward pass timing
            loss = output.sum()
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10):
                loss.backward(retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event) / 10
            
            # Memory usage
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            output = model(src)
            loss = output.sum()
            loss.backward()
            mem_after = torch.cuda.memory_allocated()
            memory_usage = (mem_after - mem_before) / (1024 * 1024)  # MB
            
            # Store results
            results['model_type'].append(model_name)
            results['seq_len'].append(seq_len)
            results['forward_time'].append(forward_time)
            results['backward_time'].append(backward_time)
            results['memory_usage'].append(memory_usage)
            results['param_count'].append(param_count / 1000000)  # M
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(18, 12))
    
    # Forward time vs seq_len
    plt.subplot(2, 2, 1)
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.plot(subset['seq_len'], subset['forward_time'], marker='o', label=model_type)
    plt.title('Forward Pass Time')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    
    # Backward time vs seq_len
    plt.subplot(2, 2, 2)
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.plot(subset['seq_len'], subset['backward_time'], marker='o', label=model_type)
    plt.title('Backward Pass Time')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    
    # Memory usage vs seq_len
    plt.subplot(2, 2, 3)
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.plot(subset['seq_len'], subset['memory_usage'], marker='o', label=model_type)
    plt.title('Memory Usage')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)
    
    # Parameter count
    plt.subplot(2, 2, 4)
    model_types = df['model_type'].unique()
    param_counts = [df[df['model_type'] == mt]['param_count'].iloc[0] for mt in model_types]
    
    plt.bar(model_types, param_counts)
    plt.title('Parameter Count')
    plt.xlabel('Model Type')
    plt.ylabel('Parameters (M)')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('transformer_benchmark.png')
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarking Orthogonal Subspace Projection Attention')
    parser.add_argument('--benchmark', type=str, default='all', choices=['linear', 'attention', 'transformer', 'all'],
                        help='Which component to benchmark')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run benchmarks on')
    parser.add_argument('--output_dir', type=str, default='benchmarks',
                        help='Directory to save benchmark results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    if args.benchmark in ['linear', 'all']:
        print("Benchmarking linear layers...")
        linear_df = benchmark_linear_layers(device=args.device)
        linear_df.to_csv(os.path.join(args.output_dir, 'linear_benchmark.csv'), index=False)
        
    if args.benchmark in ['attention', 'all']:
        print("Benchmarking attention mechanisms...")
        attn_df = benchmark_attention(device=args.device)
        attn_df.to_csv(os.path.join(args.output_dir, 'attention_benchmark.csv'), index=False)
        
    if args.benchmark in ['transformer', 'all']:
        print("Benchmarking transformer models...")
        transformer_df = benchmark_transformer_models(device=args.device)
        transformer_df.to_csv(os.path.join(args.output_dir, 'transformer_benchmark.csv'), index=False)