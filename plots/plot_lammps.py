import matplotlib.pyplot as plt
import numpy as np

# Set hatch linewidth globally
plt.rcParams['hatch.linewidth'] = 2.0  # Change this value as needed

# Data
categories = ['H100', 'A100-40G', 'A40']
measured = [362, 675, 964]
overlap = [361, 784.3, 752]
sequential = [361, 791.4, 811]

# Calculate error percentages: (value - measured) / measured * 100
overlap_errors = [((ov - meas) / meas * 100) for meas, ov in zip(measured, overlap)]
sequential_errors = [((seq - meas) / meas * 100) for meas, seq in zip(measured, sequential)]

# X positions for the bars
x = np.arange(len(categories))
width = 0.25  # Width of each bar

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
bars1 = ax.bar(x - width, measured, width, label='Measured', 
               color='white', edgecolor='black', linewidth=2)
bars2 = ax.bar(x, overlap, width, label='Overlap', 
               color='white', edgecolor='black', linewidth=2, hatch='//')
bars3 = ax.bar(x + width, sequential, width, label='Sequential', 
               color='white', edgecolor='black', linewidth=1.5, hatch='O')

# Add value labels on bars
for i, (meas, ov, seq, ov_err, seq_err) in enumerate(zip(measured, overlap, sequential, overlap_errors, sequential_errors)):
    # Measured values (no error)
    ax.text(i - width, meas + 20, f'{meas}s', 
            ha='center', va='bottom', fontsize=16)
    # Overlap values (with error on top)
    ax.text(i, ov + 20, f'{ov_err:+.1f}%\n{ov}s', 
            ha='center', va='bottom', fontsize=16)
    # Sequential values (with error on top)
    ax.text(i + width, seq + 20, f'{seq_err:+.1f}%\n{seq}s', 
            ha='center', va='bottom', fontsize=16)

# Customize plot
ax.set_ylabel('Overall Runtime', fontsize=26, labelpad=3)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.legend(loc='upper left', frameon=False, fontsize=22)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylim(bottom=0, top=1100)

# Set ticks to point inward (do this BEFORE hiding spines)
ax.tick_params(axis='both', direction='in', which='both', labelsize=25)

# Set frame (spines) linewidth
frame_linewidth = 3.5  # Change this value to control frame thickness
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(frame_linewidth)

# Hide top and right spines
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the figure as PNG
plt.savefig('plot_lammps.png', dpi=300, bbox_inches='tight')


# plt.show()