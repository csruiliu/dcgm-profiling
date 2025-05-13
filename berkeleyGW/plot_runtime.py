import matplotlib.pyplot as plt

# Data for the plot
bar_names = ["No DCGM", "10000ms", "1000ms", "100ms", "10ms", "1ms"]

# Runtime in seconds
# bar_runtime_t1 = [796, 839.41, 833.33, 839.87, 854.69, 914.25]
# bar_runtime_t2 = [796, 840.08, 859.45, 853.10, 874.87, 848.87]
# bar_runtime_t3 = [796, 837.13, 839.91, 849.80, 822.44, 865.00]  

bar_runtime_t1 = [796, 861.32, 861.93, 864.42, 877.29, 910.16]
bar_runtime_t2 = [796, 840.59, 840.20, 845.41, 851.25, 900.40]
bar_runtime_t3 = [796, 838.19, 839.75, 847.57, 851.41, 881.78]

bar_runtime = [(x + y + z) / 3 for x, y, z in zip(bar_runtime_t1, bar_runtime_t2, bar_runtime_t3)]
print(bar_runtime)

# Modern color palette
colors = plt.get_cmap('tab10').colors

# Create the bar plot with narrower bars
plt.figure(figsize=(9, 5.5))
bars = plt.bar(bar_names, bar_runtime, width=0.4, color=colors[:6])

# Add labels and title
plt.xlabel('Time Interval', fontsize=18)
plt.ylabel('Runtime (Seconds)', fontsize=18)
plt.title('Runtime With Various Sample Rates', fontsize=18)

plt.ylim(0, 1000)

# Customize the font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Position the ticks inside the plot
plt.tick_params(axis='x', direction='in', length=6, width=2)
plt.tick_params(axis='y', direction='in', length=6, width=2)

# Add text annotations on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=14)

plt.savefig('runtimes_eplison.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
