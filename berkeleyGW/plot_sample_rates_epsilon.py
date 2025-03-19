import matplotlib.pyplot as plt

# Data for the plot
bar_names = ["1 Hz", "10 Hz", "100 Hz", "1000 Hz"]
bar_datasize = [78, 769, 6.8 * 10**3, 46 * 10**3]  # Convert kb and mb to bytes
bar_runtime = [335.12, 340.12, 356.12, 379.25]  # Values in seconds
datasize_annotations = ["78KB", "769KB", "6.8MB", "46MB"]
runtime_annotations = ["335.12s", "340.12s", "356.12s", "379.25s"]

# Create the bar plot
plt.figure(figsize=(9, 5.5))
bars = plt.bar(bar_names, bar_datasize, width=0.55, color=['slategrey', 'cornflowerblue', 'teal', 'slateblue'])

# Customize the font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Position the ticks inside the plot
plt.tick_params(axis='x', direction='in', length=6, width=2)
plt.tick_params(axis='y', direction='in', length=6, width=2)

# Add text annotations on top of each bar
for bar, annotation in zip(bars, datasize_annotations):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, annotation, ha='center', va='bottom', fontsize=16)

plt.ylim(10, 10**5)

# Add labels and title
plt.xlabel('Sample Rates', fontsize=18)
plt.ylabel('Size of Sampled Metrics (KBs)', fontsize=18)
plt.title('Various Sample Rates for BerkeleyGW-Epsilon', fontsize=18)
plt.yscale('log')  # Use a logarithmic scale for better visualization

plt.savefig('data_sizes_epsilon.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

# Modern color palette
colors = plt.get_cmap('tab10').colors

# Create the bar plot with narrower bars
plt.figure(figsize=(9, 5.5))
bars = plt.bar(bar_names, bar_runtime, width=0.4, color=colors[:4])

# Add labels and title
plt.xlabel('Sample Rates', fontsize=18)
plt.ylabel('Runtime (Seconds)', fontsize=18)
plt.title('Various Sample Rates for BerkeleyGW-Epsilon', fontsize=18)

plt.ylim(300, 400)

# Customize the font size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Position the ticks inside the plot
plt.tick_params(axis='x', direction='in', length=6, width=2)
plt.tick_params(axis='y', direction='in', length=6, width=2)

# Add text annotations on top of each bar
for bar, annotation in zip(bars, runtime_annotations):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, annotation, ha='center', va='bottom', fontsize=14)

plt.savefig('runtimes_epsilon.png', dpi=300, bbox_inches='tight', pad_inches=0.05)