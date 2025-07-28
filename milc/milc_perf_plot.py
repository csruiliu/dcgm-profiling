import matplotlib.pyplot as plt
import numpy as np


plt.style.use('default')
plt.rcParams.update({
    #'font.size': 12,
    #'font.family': 'serif',
    #'axes.linewidth': 1.2,
    #'axes.spines.top': False,
    #'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'hatch.linewidth': 2,
    'figure.dpi': 600
})

# Fake data (you can replace with your actual measurements)
# Format: [single_node_single_gpu, single_node_multi_gpu]
    
a100_total_measured = [2362.4, 652.8]
a100_total_predicted = [2366.1, 654.3]

h100_total_measured = [1264.1, 356.9]
h100_total_predicted = [1401.6, 416.7]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# X-axis positions for the groups
x = np.arange(2)
width = 0.2  # Width of bars

# Create bars for each measurement type
bars1 = ax.bar(x - 1.5*width, a100_total_measured, width, label='A100 Measurement', 
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x - 0.5*width, a100_total_predicted, width, label='A100 Prediction', 
               color='steelblue', alpha=0.5, hatch='X', edgecolor='black', linewidth=2)

bars3 = ax.bar(x + 0.5*width, h100_total_measured, width, label='H100 Measurement', 
               color='salmon', alpha=0.8, edgecolor='black', linewidth=2)
bars4 = ax.bar(x + 1.5*width, h100_total_predicted, width, label='H100 Prediction', 
               color='salmon', alpha=0.5, hatch='X', edgecolor='black', linewidth=2)

# Customize the plot
#ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Overall Runtime (second)', fontsize=24, fontweight='bold')
# ax.set_title('GPU Performance Comparison: Measured vs Predicted', fontsize=14, fontweight='bold')

ax.set_ylim(0, 2500)
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
ax.set_yticklabels(['0', '500', '1000', '1500', '2000', '2500'])
ax.tick_params(axis='y', labelsize=24)
ytick_labels = ax.get_yticklabels()
for label in ytick_labels:
    label.set_fontweight('bold')

# Set x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(['1 Node x 1 GPU', 
                    '1 Node x 4 GPUs'])
ax.tick_params(axis='x', labelsize=24)
xtick_labels = ax.get_xticklabels()
for label in xtick_labels:
    label.set_fontweight('bold')

# Add legend
ax.legend(loc='upper right', 
          ncol=1, 
          fontsize=24,
          frameon=True,
          fancybox=True,
          framealpha=0.9, 
          shadow=False,
          edgecolor='black',            # Border color
          facecolor='white')

# Add grid for better readability
ax.grid(True, alpha=0.4, axis='y')

 # Enable and style ALL FOUR spines for left subplot
ax.spines['bottom'].set_linewidth(3)  # Bottom axis
ax.spines['left'].set_linewidth(3)    # Left axis
ax.spines['top'].set_linewidth(3)     # Top axis
ax.spines['right'].set_linewidth(3)   # Right axis (will be overridden by twin)

# Add value labels on top of bars (optional)
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points", fontweight='bold',
                   ha='center', va='bottom', fontsize=20)

# Uncomment the lines below if you want value labels on bars
add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)



# Adjust layout and display
plt.tight_layout()

# Save with maximum quality settings
fig.savefig('milc_perf.png', 
            dpi=600,                    # Very high DPI for crisp print
            bbox_inches='tight',
            facecolor='white',          # White background
            edgecolor='none',
            format='png',
            metadata={'Creator': 'Matplotlib'})