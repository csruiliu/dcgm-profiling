import matplotlib.pyplot as plt
import numpy as np

# Set hatch linewidth globally
plt.rcParams['hatch.linewidth'] = 2.0

# Data
categories = ['H100', 'A100-40G', 'A40']
measured = [362, 655, 964]
smocc_upper = [361, 602, 628]
smocc_mid = [361, 696, 863]
smocc_lower = [361, 828, 1399]

'''
# Data
categories = ['H100', 'A100-40G', 'A40']
measured = [362, 655, 964]
smocc_upper = [288, 655, 714]
smocc_mid = [331, 655, 864]
smocc_lower = [391, 655, 1114]
'''

# Calculate error percentages
smocc_upper_errors = [((su - meas) / meas * 100) for meas, su in zip(measured, smocc_upper)]
smocc_mid_errors = [((sm - meas) / meas * 100) for meas, sm in zip(measured, smocc_mid)]
smocc_lower_errors = [((sl - meas) / meas * 100) for meas, sl in zip(measured, smocc_lower)]

# X positions for the bars
x = np.arange(len(categories))
width = 0.2  # Width to fit 4 bars

# Create figure with single subplot
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
bars1 = ax.bar(x - 1.5*width, measured, width, label='Measured', color='white', edgecolor='black', linewidth=2)
bars2 = ax.bar(x - 0.5*width, smocc_upper, width, label='SMOCC Upper', color='white', edgecolor='black', linewidth=2, hatch='/')
bars3 = ax.bar(x + 0.5*width, smocc_mid, width, label='SMOCC Mid', color='white', edgecolor='black', linewidth=2, hatch='\\')
bars4 = ax.bar(x + 1.5*width, smocc_lower, width, label='SMOCC Lower', color='white', edgecolor='black', linewidth=1.5, hatch='O')

# Add value labels on bars
for i, (meas, su, sm, sl, su_err, sm_err, sl_err) in enumerate(zip(measured, smocc_upper, smocc_mid, smocc_lower, smocc_upper_errors, smocc_mid_errors, smocc_lower_errors)):
    # Measured values
    ax.text(i - 1.5*width, meas + 20, f'{meas}s', ha='center', va='bottom', fontsize=16)
    # SMOCC Upper values
    ax.text(i - 0.5*width, su + 20, f'{su_err:+.1f}%\n{su}s', ha='center', va='bottom', fontsize=16)
    # SMOCC Mid values
    ax.text(i + 0.5*width, sm + 20, f'{sm_err:+.1f}%\n{sm}s', ha='center', va='bottom', fontsize=16)
    # SMOCC Lower values
    ax.text(i + 1.5*width, sl + 20, f'{sl_err:+.1f}%\n{sl}s', ha='center', va='bottom', fontsize=16)

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
plt.savefig('plot_lammps.png', dpi=300, bbox_inches='tight')

# plt.show()