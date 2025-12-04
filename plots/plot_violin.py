import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read all CSV files
data_files = {
    'BGW-FP64-A100': 'bgw-fp64-a100-ref.csv',
    'BGW-FP64-H100': 'bgw-fp64-h100-ref.csv',
    'LAMMPS-FP32-A100': 'lammps-fp32-a100-ref.csv',
    'LAMMPS-FP32-H100': 'lammps-fp32-h100-ref.csv',
    'MILC-FP32-A100': 'milc-fp32-a100-ref.csv',
    'MILC-FP32-H100': 'milc-fp32-h100-ref.csv',
    'MILC-FP64-A100': 'milc-fp64-a100-ref.csv',
    'MILC-FP64-H100': 'milc-fp64-h100-ref.csv'
}

# Collect relative errors for each SMOCC variant
smocc_variants = ['smocc_lower', 'smocc_mid', 'smocc_upper', 'mock_smocc']
relative_errors = {variant: [] for variant in smocc_variants}

for name, file in data_files.items():
    df = pd.read_csv(file)
    
    for _, row in df.iterrows():
        measured = row['measured']
        
        # Calculate relative error for each variant: (predicted - measured) / measured * 100
        for variant in smocc_variants:
            rel_error = (row[variant] - measured) / measured * 100
            relative_errors[variant].append(rel_error)

# Prepare data for violin plot
data = [relative_errors[variant] for variant in smocc_variants]
labels = ['SMOCC Lower', 'SMOCC Mid', 'SMOCC Upper', 'Mock SMOCC']

# Create violin plot
fig, ax = plt.subplots(figsize=(12, 8))

parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True, widths=0.7)

# Color the violins with different colors
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Customize mean and median lines
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)
parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2)

# Add a horizontal line at y=0 (perfect prediction)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Prediction')

# Customize the plot
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Error (%)', fontsize=16, fontweight='bold')
ax.set_title('Relative Error Distribution Across SMOCC Variants', 
             fontsize=18, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Set y-axis range
ax.set_ylim(-40, 70)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Median'),
    Line2D([0], [0], color='red', linewidth=2, label='Mean'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Perfect Estimation')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Add statistics text
stats_text = []
for i, variant in enumerate(smocc_variants):
    mean_err = np.mean(relative_errors[variant])
    median_err = np.median(relative_errors[variant])
    stats_text.append(f'{labels[i]}: Mean={mean_err:.1f}%, Median={median_err:.1f}%')

# Add text box with statistics
textstr = '\n'.join(stats_text)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('relative_error_violin_plot.png', dpi=300, bbox_inches='tight')
