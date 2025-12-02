import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot runtime comparison bar chart')
parser.add_argument('data_file', type=str, help='Path to the CSV data file')
parser.add_argument('--format', type=str, default='png', 
                    choices=['png', 'pdf', 'svg', 'jpg'],
                    help='Output image format (default: png)')
args = parser.parse_args()

# Set hatch linewidth globally
plt.rcParams['hatch.linewidth'] = 2.0

# Read data from CSV file, skipping lines that start with #
data = pd.read_csv(args.data_file, comment='#')

# Extract data
categories = data['category'].values
measured = data['measured'].values
smocc_upper = data['smocc_upper'].values
smocc_mid = data['smocc_mid'].values
smocc_lower = data['smocc_lower'].values
mock_smocc = data['mock_smocc'].values  # Added mock_smocc

# Calculate error percentages
smocc_upper_errors = [((su - meas) / meas * 100) for meas, su in zip(measured, smocc_upper)]
smocc_mid_errors = [((sm - meas) / meas * 100) for meas, sm in zip(measured, smocc_mid)]
smocc_lower_errors = [((sl - meas) / meas * 100) for meas, sl in zip(measured, smocc_lower)]
mock_smocc_errors = [((mock - meas) / meas * 100) for meas, mock in zip(measured, mock_smocc)]  # Added mock errors

# X positions for the bars
x = np.arange(len(categories))
width = 0.18  # Adjusted width to fit 5 bars

# Adaptive figure width based on number of categories
fig_width = max(14, 10 + (len(categories) - 3) * 3)
fig, ax = plt.subplots(figsize=(fig_width, 8))

# Create bars (5 bars now)
bars1 = ax.bar(x - 2*width, measured, width, label='Measured', color='white', edgecolor='black', linewidth=2)
bars2 = ax.bar(x - width, smocc_upper, width, label='SMOCC Upper', color='white', edgecolor='black', linewidth=2, hatch='/')
bars3 = ax.bar(x, smocc_mid, width, label='SMOCC Mid', color='white', edgecolor='black', linewidth=2, hatch='\\')
bars4 = ax.bar(x + width, smocc_lower, width, label='SMOCC Lower', color='white', edgecolor='black', linewidth=1.5, hatch='O')
bars5 = ax.bar(x + 2*width, mock_smocc, width, label='Mock SMOCC', color='white', edgecolor='black', linewidth=2, hatch='x')  # Added mock bars

# Add value labels on bars
for i, (meas, su, sm, sl, mock, su_err, sm_err, sl_err, mock_err) in enumerate(zip(measured, smocc_upper, smocc_mid, smocc_lower, mock_smocc, smocc_upper_errors, smocc_mid_errors, smocc_lower_errors, mock_smocc_errors)):
    # Measured values
    ax.text(i - 2*width, meas + 20, f'{meas}s', ha='center', va='bottom', fontsize=13)
    # SMOCC Upper values
    ax.text(i - width, su + 20, f'{su_err:+.1f}%\n{su}s', ha='center', va='bottom', fontsize=13)
    # SMOCC Mid values
    ax.text(i, sm + 20, f'{sm_err:+.1f}%\n{sm}s', ha='center', va='bottom', fontsize=13)
    # SMOCC Lower values
    ax.text(i + width, sl + 20, f'{sl_err:+.1f}%\n{sl}s', ha='center', va='bottom', fontsize=13)
    # Mock SMOCC values
    ax.text(i + 2*width, mock + 20, f'{mock_err:+.1f}%\n{mock}s', ha='center', va='bottom', fontsize=13)  # Added mock labels

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

# Generate output filename from input data filename
base_name = os.path.splitext(os.path.basename(args.data_file))[0]
output_file = f"{base_name}_bar.{args.format}"

# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to: {output_file}")

# plt.show()