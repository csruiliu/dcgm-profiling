import matplotlib.pyplot as plt

# Data for the plot
bar_names = ["1000ms", "100ms", "10ms", "1ms"]
bar_datasize = [136, 1.5 * 10**3, 13 * 10**3, 88 * 10**3]  # Convert kb and mb to bytes
bar_runtime = [572.27, 580.55, 593.39, 610.99]  # Values in seconds
datasize_annotations = ["136KB", "1.5MB", "13MB", "88MB"]
runtime_annotations = ["572.27s", "580.55s", "593.39s", "610.99s"]

# Modern color palette
colors = plt.get_cmap('tab10').colors

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

plt.ylim(10, 200 * 10**3)

# Add labels and title
plt.xlabel('Sample Rates', fontsize=18)
plt.ylabel('Size of Sampled Metrics (KBs)', fontsize=18)
plt.title('Various Sample Rates for BerkeleyGW-Sigma', fontsize=18)
plt.yscale('log')  # Use a logarithmic scale for better visualization

plt.savefig('data_sizes_sigma.png', dpi=300, bbox_inches='tight', pad_inches=0.05)