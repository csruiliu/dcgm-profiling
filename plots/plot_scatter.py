import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot runtime prediction scatter plot')
parser.add_argument('data_file', type=str, help='Path to the CSV data file')
parser.add_argument('--format', type=str, default='png', 
                    choices=['png', 'pdf', 'svg', 'jpg'],
                    help='Output image format (default: png)')
args = parser.parse_args()

# Read data from CSV file, skipping lines that start with #
data = pd.read_csv(args.data_file, comment='#')

# Extract columns
measured = data['measured'].values
lower = data['smocc_lower'].values
middle = data['smocc_mid'].values
upper = data['smocc_upper'].values
mock = data['mock_smocc'].values  # Added mock_smocc

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 7))

# Plot the four lines with markers
ax.plot(measured, lower, color='teal', linewidth=2, 
        markersize=7, marker='X', label='lower')
ax.plot(measured, middle, color='skyblue', linewidth=2, 
        markersize=7, marker='*', label='middle')
ax.plot(measured, upper, color='orange', linewidth=2, 
        markersize=7, marker='s', label='upper')
ax.plot(measured, mock, color='purple', linewidth=2, 
        markersize=7, marker='o', label='mock')  # Added mock line

# Plot ideal diagonal line
max_val = max(measured.max(), lower.max(), middle.max(), upper.max(), mock.max())
ideal_limit = max_val * 1.2  # 20% beyond max value
ideal_x = [0, ideal_limit]
ideal_y = [0, ideal_limit]
ax.plot(ideal_x, ideal_y, '--', color='magenta', linewidth=2, 
        alpha=0.7, label='ideal')

# Grid
ax.grid(True, linestyle=':', alpha=0.5, color='gray')

# Labels
ax.set_xlabel('Measured runtime (sec)', fontsize=14)
ax.set_ylabel('Predicted runtime (sec)', fontsize=14)

# Axis limits (auto-adjust based on data)
ax.set_xlim(0, ideal_limit)
ax.set_ylim(0, ideal_limit)

# Legend
ax.legend(fontsize=12, loc='lower right')

# Tick parameters
ax.tick_params(labelsize=12)

# Make the plot square
ax.set_aspect('equal')

plt.tight_layout()

# Generate output filename from input data filename
base_name = os.path.splitext(os.path.basename(args.data_file))[0]
output_file = f"{base_name}_scatter.{args.format}"

# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to: {output_file}")

