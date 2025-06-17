import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
from collections import defaultdict

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

def create_gpu_metric_plots(gpu_data, metrics, output_dir='gpu_plots'):
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
    
    plot_count = 0
    
    for gpu_id in gpu_ids:
        for metric in metrics:
            if metric in gpu_data[gpu_id] and len(gpu_data[gpu_id][metric]) > 0:
                # Create time axis
                time_points = list(range(len(gpu_data[gpu_id][metric])))
                values = gpu_data[gpu_id][metric]
                
                # Create plot for ALL metrics (no filtering)
                plt.figure(figsize=(12, 6))
                color = color_map[gpu_id]
                
                plt.plot(time_points, values, color=color, linewidth=2, 
                        marker='o', markersize=4, alpha=0.8)
                
                plt.title(f'GPU {gpu_id} - {metric} Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Time Point', fontsize=12)
                plt.ylabel(f'{metric} Value', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Add some styling
                plt.tight_layout()
                
                # Add statistics text box
                mean_val = np.mean(values)
                max_val = np.max(values)
                min_val = np.min(values)
                std_val = np.std(values)
                stats_text = f'Mean: {mean_val:.3f}\nMax: {max_val:.3f}\nMin: {min_val:.3f}\nStd: {std_val:.3f}'
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.8))
                
                # Save plot
                filename = f'GPU_{gpu_id}_{metric}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    parser.add_argument('-o', '--output', default='gpu_plots', 
                        help='Output directory for plots (default: gpu_plots)')
    
    args = parser.parse_args()
    
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
        plot_count = create_gpu_metric_plots(gpu_data, metrics, args.output)
        
        print(f"\nCompleted! All {plot_count} plots saved to '{args.output}' directory!")
        
        # List all created files
        print(f"\nGenerated files:")
        gpu_ids = sorted(gpu_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        for gpu_id in gpu_ids:
            for metric in metrics:
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