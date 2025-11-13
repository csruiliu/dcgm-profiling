import matplotlib.pyplot as plt
import numpy as np

# Set hatch linewidth globally
plt.rcParams['hatch.linewidth'] = 2.0

# Data
categories = ['H100', 'A100-40G', 'A40']

# Use H100 as reference
measured_total = [402, 855, 1968]
measured_quda = [367, 811, 1939]
smocc_upper = [402, 664, 1311]
smocc_mid = [402, 756, 1321]
smocc_lower = [402, 888, 1498]

'''
# Use A100 as reference
measured_total = [402, 855, 1968]
measured_quda = [367, 821, 1939]
smocc_upper = [394, 854, 1545]
smocc_mid = [446, 854, 1548]
smocc_lower = [518, 854, 1556]
'''
# Calculate the non-quda portion
measured_non_quda = [total - quda for total, quda in zip(measured_total, measured_quda)]

# Calculate error percentages
smocc_upper_errors = [((su - meas) / meas * 100) for meas, su in zip(measured_total, smocc_upper)]
smocc_mid_errors = [((sm - meas) / meas * 100) for meas, sm in zip(measured_total, smocc_mid)]
smocc_lower_errors = [((sl - meas) / meas * 100) for meas, sl in zip(measured_total, smocc_lower)]

# X positions for the bars
x = np.arange(len(categories))
width = 0.2  # Width to fit 4 bars

# Create figure with single subplot
fig, ax = plt.subplots(figsize=(14, 8))

# Create stacked bars for Measured (quda at bottom, non-quda on top)
bars1_bottom = ax.bar(x - 1.5*width, measured_quda, width, label='Measured (QUDA)', 
                       color='lightgray', edgecolor='black', linewidth=2)
bars1_top = ax.bar(x - 1.5*width, measured_non_quda, width, bottom=measured_quda, 
                    label='Measured (Other)', color='white', edgecolor='black', hatch='xx', linewidth=2)

# Create other bars
bars2 = ax.bar(x - 0.5*width, smocc_upper, width, label='SMOCC Upper', 
               color='white', edgecolor='black', linewidth=2, hatch='/')
bars3 = ax.bar(x + 0.5*width, smocc_mid, width, label='SMOCC Mid', 
               color='white', edgecolor='black', linewidth=2, hatch='\\')
bars4 = ax.bar(x + 1.5*width, smocc_lower, width, label='SMOCC Lower', 
               color='white', edgecolor='black', linewidth=1.5, hatch='O')

# Find the maximum bar height for adaptive y-axis
all_values = measured_total + smocc_upper + smocc_mid + smocc_lower
max_bar_height = max(all_values)

# Add value labels on bars with 45-degree rotation
annotation_offset = 20  # Space above bars for annotations
for i, (total, quda, non_quda, su, sm, sl, su_err, sm_err, sl_err) in enumerate(zip(
        measured_total, measured_quda, measured_non_quda, smocc_upper, smocc_mid, smocc_lower, 
        smocc_upper_errors, smocc_mid_errors, smocc_lower_errors)):
    
    # Measured stacked bar - show both quda and total values
    ax.text(i - 1.5*width, total + annotation_offset, f'{quda}s+{non_quda}s\n{total}s', 
            ha='center', va='bottom', fontsize=16, rotation=0)
    
    # SMOCC Upper values
    ax.text(i - 0.5*width, su + annotation_offset, f'{su_err:+.1f}%\n{su}s', 
            ha='center', va='bottom', fontsize=16, rotation=0)
    
    # SMOCC Mid values
    ax.text(i + 0.5*width, sm + annotation_offset, f'{sm_err:+.1f}%\n{sm}s', 
            ha='center', va='bottom', fontsize=16, rotation=0)
    
    # SMOCC Lower values
    ax.text(i + 1.5*width, sl + annotation_offset, f'{sl_err:+.1f}%\n{sl}s', 
            ha='center', va='bottom', fontsize=16, rotation=0)

# Set adaptive y-axis limits
# Account for annotation height (roughly 2 lines of text at fontsize 16)
# Estimate: each line of text is about 3-4% of the max value
annotation_height_estimate = max_bar_height * 0.12  # ~12% for 2 lines + spacing
y_max = max_bar_height + annotation_offset + annotation_height_estimate

# Set ylim with some padding at the bottom
ax.set_ylim(0, y_max)

# Set adaptive y-ticks
# Create nice round numbers for ticks
tick_interval = np.ceil(y_max / 6 / 100) * 100  # Round to nearest 100
y_ticks = np.arange(0, y_max + tick_interval, tick_interval)
ax.set_yticks(y_ticks)

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.legend(loc='upper left', frameon=False, fontsize=22)
ax.axhline(y=0, color='black', linewidth=0.8)

# Set ylabel
ax.set_ylabel('Overall Runtime', fontsize=26)

# Set ticks to point inward
ax.tick_params(axis='both', direction='in', which='both', labelsize=25)

# Set frame (spines) linewidth
frame_linewidth = 3.5
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(frame_linewidth)
# Save the figure as PNG
plt.savefig('plot_milc_h100.png', dpi=300, bbox_inches='tight')

# plt.show()