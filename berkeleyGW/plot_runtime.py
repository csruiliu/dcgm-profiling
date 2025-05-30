import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
bar_names = ["No DCGM", "10000ms", "1000ms", "100ms", "10ms", "1ms"]

# Home runtime data
home_runtime_2m_t1 = [830.61, 839.41, 833.33, 839.87, 854.69, 914.25]
home_runtime_2m_t2 = [831.80, 836.08, 832.45, 833.10, 874.87, 898.87]
home_runtime_2m_t3 = [829.72, 837.13, 839.91, 839.80, 822.44, 865.00]

# Home runtime data
home_runtime_10m_t1 = [830.61, 861.32, 861.93, 864.42, 877.29, 910.16]
home_runtime_10m_t2 = [831.80, 840.59, 840.20, 845.41, 851.25, 900.40]
home_runtime_10m_t3 = [829.72, 838.19, 839.75, 847.57, 851.41, 881.78]

# Scratch runtime data
scratch_runtime_10m_t1 = [819.84, 827.71, 825.80, 820.39, 835.10, 859.46]
scratch_runtime_10m_t2 = [819.95, 828.40, 825.80, 820.77, 837.71, 870.32]
scratch_runtime_10m_t3 = [817.97, 825.78, 819.40, 826.49, 833.47, 861.09]

# SHM runtime data
shm_runtime_10m_t1 = [827.23, 828.63, 825.46, 821.44, 800.76, 854.24]
shm_runtime_10m_t2 = [823.95, 816.12, 830.17, 827.32, 839.37, 864.68]
shm_runtime_10m_t3 = [823.23, 816.28, 819.28, 811.08, 830.57, 818.49]

# Calculate average runtime for each type
home_runtime_avg = [(x + y + z) / 3 for x, y, z in zip(home_runtime_10m_t1, home_runtime_10m_t2, home_runtime_10m_t3)]
scratch_runtime_avg = [(x + y + z) / 3 for x, y, z in zip(scratch_runtime_10m_t1, scratch_runtime_10m_t2, scratch_runtime_10m_t3)]
shm_runtime_avg = [(x + y + z) / 3 for x, y, z in zip(shm_runtime_10m_t1, shm_runtime_10m_t2, shm_runtime_10m_t3)]

# Calculate average runtime for 2m home data
home_runtime_2m_avg = [(x + y + z) / 3 for x, y, z in zip(home_runtime_2m_t1, home_runtime_2m_t2, home_runtime_2m_t3)]

# Define color families for each group (each time interval gets its own color family)
color_families = [
    ['#4b7bc8', '#2d5aa8', '#1f4788'],  # Blue family for No DCGM
    ['#f07c7d', '#e74c3c', '#d62728'],  # Red family for 10000ms
    ['#5dc85d', '#3cb43c', '#2ca02c'],  # Green family for 1000ms
    ['#ffb366', '#ff9933', '#ff7f0e'],  # Orange family for 100ms
    ['#be9de9', '#a981d3', '#9467bd'],  # Purple family for 10ms
    ['#c09693', '#a6766f', '#8c564b'],  # Brown family for 1ms
]

# ========== First Plot: Grouped bar plot for 10m data ==========
# Set up the grouped bar plot
x = np.arange(len(bar_names))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Create bars with colors from the same family for each group
# Order: SHM (left), Scratch (middle), Home (right)
shm_bars = []
scratch_bars = []
home_bars = []

for i in range(len(bar_names)):
    # SHM bars (no hatch)
    shm_bar = ax.bar(x[i] - width, shm_runtime_avg[i], width, 
                      color=color_families[i][0], edgecolor='black', linewidth=1)
    shm_bars.append(shm_bar)
    
    # Scratch bars (with diagonal hatch)
    scratch_bar = ax.bar(x[i], scratch_runtime_avg[i], width, 
                         color=color_families[i][1], edgecolor='black', linewidth=1,
                         hatch='//')
    scratch_bars.append(scratch_bar)
    
    # Home bars (with cross hatch)
    home_bar = ax.bar(x[i] + width, home_runtime_avg[i], width, 
                      color=color_families[i][2], edgecolor='black', linewidth=1,
                      hatch='xx')
    home_bars.append(home_bar)

# Add value labels on top of each bar
for i in range(len(bar_names)):
    # SHM bar annotation
    ax.annotate(f'{shm_runtime_avg[i]:.1f}',
                xy=(x[i] - width, shm_runtime_avg[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9)
    # Scratch bar annotation
    ax.annotate(f'{scratch_runtime_avg[i]:.1f}',
                xy=(x[i], scratch_runtime_avg[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9)
    # Home bar annotation
    ax.annotate(f'{home_runtime_avg[i]:.1f}',
                xy=(x[i] + width, home_runtime_avg[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9)

# Create custom legend with hatching patterns
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', label='SHM'),
    plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', hatch='//', label='Scratch'),
    plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', hatch='xx', label='Home')
]
ax.legend(handles=legend_elements, fontsize=14, loc='upper left', ncol = 3)

# Add labels and title
ax.set_xlabel('Sampling Interval', fontsize=18)
ax.set_ylabel('Runtime (Seconds)', fontsize=18)
ax.set_title('Runtime Comparison Across Different Storage Types (10 metrics)', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(bar_names)

# Customize the font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Set y-axis limits to better show the variation
ax.set_ylim(0, 1000)

plt.tight_layout()
plt.savefig('runtime_comparison_ordered_with_hatch.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Second Plot: Single bar plot for 2m home data ==========
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Create single bars for home 2m data
x2 = np.arange(len(bar_names))
width2 = 0.6  # Wider bars since it's a single series

for i in range(len(bar_names)):
    # Home 2m bars
    bar = ax2.bar(x2[i], home_runtime_2m_avg[i], width2, 
                   color=color_families[i][2], edgecolor='black', linewidth=1)
    
    # Add value labels on top of each bar
    ax2.annotate(f'{home_runtime_2m_avg[i]:.1f}',
                 xy=(x2[i], home_runtime_2m_avg[i]),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=12)

# Add labels and title
ax2.set_xlabel('Sampling Interval', fontsize=18)
ax2.set_ylabel('Runtime (Seconds)', fontsize=18)
ax2.set_title('Home Runtime with Different Sampling Intervals (2 metrics)', fontsize=20)
ax2.set_xticks(x2)
ax2.set_xticklabels(bar_names)

# Customize the font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid for better readability
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)

# Set y-axis limits to better show the variation
ax2.set_ylim(0, 1000)

plt.tight_layout()
plt.savefig('home_runtime_2m_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Two figures have been saved:")
print("1. runtime_comparison_ordered_with_hatch.png - Grouped bar plot with 10m data")
print("2. home_runtime_2m_comparison.png - Single bar plot with 2m home data")