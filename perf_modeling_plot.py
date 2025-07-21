import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
from collections import defaultdict

plt.style.use('default')
plt.rcParams.update({
    #'font.size': 12,
    #'font.family': 'serif',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    #'figure.dpi': 300
})


def parse_dcgm_data(filename):
    """
    Parse DCGM data file and extract GPU metrics over time
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    gpu_data = defaultdict(lambda: defaultdict(list))
    timestamps = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    current_timestamp = 0
    header_line = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for header line with metric names
        if line.startswith('#Entity'):
            header_line = line
            # Extract metric names
            parts = line.split()
            metrics = parts[1:]  # Skip '#Entity'
            continue
        
        # Process GPU data lines
        if line.startswith('GPU'):
            parts = line.split()
            gpu_id = parts[1]  # GPU ID (0, 1, 2, 3, etc.)
            values = parts[2:]  # Metric values
            
            # Store each metric value for this GPU at this timestamp
            for i, metric in enumerate(metrics):
                if i < len(values):
                    try:
                        value = float(values[i])
                        gpu_data[gpu_id][metric].append(value)
                    except ValueError:
                        gpu_data[gpu_id][metric].append(0.0)
    
    return gpu_data, metrics

def generate_colors(num_colors):
    """
    Generate a list of distinct colors for the given number of items
    """
    if num_colors <= 10:
        # Use matplotlib's tab10 colormap for up to 10 colors
        colormap = plt.cm.tab10
        colors = [colormap(i) for i in range(num_colors)]
    elif num_colors <= 20:
        # Use tab20 colormap for up to 20 colors
        colormap = plt.cm.tab20
        colors = [colormap(i) for i in range(num_colors)]
    else:
        # For more colors, use HSV color space to generate distinct colors
        import colorsys
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2       # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
    
    return colors

def apply_sampling_window(data, window_size, aggregation_method='mean'):
    """
    Apply sampling window to data with specified aggregation method
    
    Args:
        data: List of data points
        window_size: Size of the sampling window (number of points to aggregate)
        aggregation_method: Method to aggregate data ('mean', 'max', 'min', 'median')
    
    Returns:
        Tuple of (aggregated_data, new_time_points)
    """
    if window_size <= 1:
        return data, list(range(len(data)))
    
    aggregated_data = []
    new_time_points = []
    
    for i in range(0, len(data), window_size):
        window_data = data[i:i + window_size]
        
        if aggregation_method == 'mean':
            agg_value = np.mean(window_data)
        elif aggregation_method == 'max':
            agg_value = np.max(window_data)
        elif aggregation_method == 'min':
            agg_value = np.min(window_data)
        elif aggregation_method == 'median':
            agg_value = np.median(window_data)
        else:
            agg_value = np.mean(window_data)  # Default to mean
        
        aggregated_data.append(agg_value)
        # Use the middle point of the window as the time point
        new_time_points.append(i + len(window_data) // 2)
    
    return aggregated_data, new_time_points


def create_gpu_metric_plots(gpu_data, metrics, output_dir='gpu_plots', window_size=1, aggregation_method='mean'):
    """
    Create individual plots for each GPU-metric combination
    Generate ALL plots regardless of activity level
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted list of GPU IDs and generate colors dynamically
    gpu_ids = sorted(gpu_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    num_gpus = len(gpu_ids)
    
    # Generate colors for all GPUs
    gpu_colors = generate_colors(num_gpus)
    
    # Create a mapping from GPU ID to color
    color_map = {gpu_id: gpu_colors[i] for i, gpu_id in enumerate(gpu_ids)}
    
    # Define metrics that should have 0-1 range displayed as 0%-100%
    percentage_metrics = {'DRAMA', 'FP16A', 'FP32A', 'FP64A', 'GRACT'}
    
    plot_count = 0
    
    for gpu_id in gpu_ids:
        for metric in metrics:
            if metric in gpu_data[gpu_id] and len(gpu_data[gpu_id][metric]) > 0:
                # Apply sampling window
                original_values = gpu_data[gpu_id][metric]
                values, time_points = apply_sampling_window(original_values, window_size, aggregation_method)
                
                # Create plot for ALL metrics (no filtering)
                plt.figure(figsize=(10, 6))
                color = color_map[gpu_id]
                
                plt.plot(time_points, values, color=color, linewidth=4, marker='o', markersize=6, alpha=0.8)
                
                # Update title to reflect sampling
                '''
                if window_size > 1:
                    title = f'GPU {gpu_id} - {metric} Over Time (Window: {window_size}, {aggregation_method.title()})'
                else:
                    title = f'GPU {gpu_id} - {metric} Over Time'
                '''
                # title = f'GPU {gpu_id} - {metric} Over Time'
                # plt.title(title, fontsize=16, fontweight='bold')
                plt.xlabel('Time Point', fontsize=26)
                
                # Set y-axis label and range based on metric type
                if metric in percentage_metrics:
                    plt.ylabel(f'{metric} (%)', fontsize=26)
                    
                    # Special handling for GRACT - extend range to 110%
                    if metric == 'GRACT':
                        plt.ylim(0, 1.1)  # Set range from 0 to 1.1 (0% to 110%)
                    else:
                        plt.ylim(0, 1)    # Set range from 0 to 1 (0% to 100%)
                    
                    # Format y-axis to show as percentage (0 to 1 becomes 0% to 100%)
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x*100)}%'))
                else:
                    plt.ylabel(f'{metric} Value', fontsize=26)
                
                plt.tick_params(axis='both', which='major', labelsize=26)
                plt.grid(True, alpha=0.3)
                                
                plt.gca().spines['bottom'].set_linewidth(4)
                plt.gca().spines['left'].set_linewidth(4)
                plt.gca().spines['top'].set_linewidth(4)
                plt.gca().spines['right'].set_linewidth(4)
                plt.gca().spines['top'].set_visible(True)
                plt.gca().spines['right'].set_visible(True)

                # Add some styling
                plt.tight_layout()
                
                # Add statistics text box
                mean_val = np.mean(values)
                max_val = np.max(values)
                min_val = np.min(values)
                std_val = np.std(values)
                
                # Format statistics based on metric type
                if metric in percentage_metrics:
                    # Convert to percentage for display
                    stats_text = f'Mean: {mean_val*100:.1f}%, Max: {max_val*100:.1f}%, Min: {min_val*100:.1f}%, Std: {std_val*100:.1f}%'
                else:
                    stats_text = f'Mean: {mean_val:.3f}, Max: {max_val:.3f}, Min: {min_val:.3f}, Std: {std_val:.3f}'
                
                # Add sampling info to stats if window size > 1
                if window_size > 1:
                    original_points = len(original_values)
                    sampled_points = len(values)
                    sampling_info = f'\nOriginal: {original_points} pts, Sampled: {sampled_points} pts'
                    stats_text += sampling_info
                '''
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=16, bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.8))
                '''
                # Save plot with window size in filename if applicable
                if window_size > 1:
                    filename = f'GPU_{gpu_id}_{metric}_window{window_size}_{aggregation_method}.png'
                else:
                    filename = f'GPU_{gpu_id}_{metric}.png'

                # Save plot
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=600, bbox_inches='tight')
                plt.close()
                
                plot_count += 1
                print(f'Created plot {plot_count}: {filename} (Color: GPU {gpu_id})')
            else:
                # Handle case where metric has no data for this GPU
                print(f'Warning: No data found for GPU {gpu_id} - {metric}')
    
    print(f'\nTotal plots created: {plot_count}')
    print(f'Color assignments for {num_gpus} GPUs:')
    for i, gpu_id in enumerate(gpu_ids):
        print(f'  GPU {gpu_id}: Color {i+1}/{num_gpus}')
    
    return plot_count

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse DCGM data and generate GPU metric plots')
    parser.add_argument('filename', help='Path to the DCGM data file')
    parser.add_argument('-w', '--window', type=int, default=1,
                        help='Sampling window size (number of data points to aggregate, default: 1)')
    parser.add_argument('-o', '--output', default='gpu_plots', 
                        help='Output directory for plots (default: gpu_plots)')
    parser.add_argument('-a', '--aggregation', choices=['mean', 'max', 'min', 'median'], 
                        default='mean', help='Aggregation method for sampling window (default: mean)')
    args = parser.parse_args()
    
    # Validate window size
    if args.window < 1:
        print("Error: Window size must be at least 1")
        sys.exit(1)

    try:
        print(f"Parsing DCGM data from: {args.filename}")
        gpu_data, metrics = parse_dcgm_data(args.filename)
        
        print(f"Found {len(gpu_data)} GPUs with metrics: {metrics}")
        print(f"GPUs detected: {sorted(gpu_data.keys())}")
        print(f"Expected total plots: {len(gpu_data)} GPUs × {len(metrics)} metrics = {len(gpu_data) * len(metrics)} plots")
        
        # Show data summary
        for gpu_id in sorted(gpu_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"\nGPU {gpu_id}:")
            for metric in metrics:
                if metric in gpu_data[gpu_id]:
                    count = len(gpu_data[gpu_id][metric])
                    if count > 0:
                        max_val = max(gpu_data[gpu_id][metric])
                        min_val = min(gpu_data[gpu_id][metric])
                        print(f"  {metric}: {count} data points, range: [{min_val:.3f} - {max_val:.3f}]")
                    else:
                        print(f"  {metric}: No data")
                else:
                    print(f"  {metric}: Missing")
        
        print(f"\nCreating ALL GPU-metric plots in '{args.output}' directory...")
        plot_count = create_gpu_metric_plots(gpu_data, metrics, args.output, args.window, args.aggregation)
        
        print(f"\nCompleted! All {plot_count} plots saved to '{args.output}' directory!")
        
        # List all created files
        print(f"\nGenerated files:")
        gpu_ids = sorted(gpu_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        for gpu_id in gpu_ids:
            for metric in metrics:
                if args.window > 1:
                    filename = f'GPU_{gpu_id}_{metric}_window{args.window}_{args.aggregation}.png'
                else:
                    filename = f'GPU_{gpu_id}_{metric}.png'
                print(f"  {filename}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# Run the analysis
if __name__ == "__main__":
    main()