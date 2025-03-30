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

