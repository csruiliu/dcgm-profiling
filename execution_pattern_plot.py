import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300
})

def create_dcgm_comparison():
    """Create Figure 2: DCGM Measurement Accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    #fig.suptitle('DCGM Measurement Accuracy Comparison', fontsize=16, fontweight='bold')
    
    time = np.linspace(0, 1, 1000)
    
    # Bursty pattern actual vs DCGM
    # Peak: 400 GB/s for 0.25s, then 0 for 0.75s → Average: 100 GB/s
    mem_actual_bursty = np.where(time <= 0.25, 400, 0)
    mem_dcgm_bursty = np.full_like(time, 100)  # Average over 1 second
    
    compute_actual_bursty = np.where(time <= 0.25, 0.5, 9.33)
    compute_dcgm_bursty = np.full_like(time, 7.0)  # Average over 1 second
    
    # Interleaved pattern (DCGM matches actual) - keeping this at 400 GB/s steady
    mem_actual_interleaved = np.full_like(time, 100)
    mem_dcgm_interleaved = np.full_like(time, 100)
    
   # Color options - choose one of these:
    
    # Option 1: Teal and Orange
    mem_color = '#2E8B8B'  # Teal
    compute_color = '#FF8C00'  # Dark Orange
    
    # Option 2: Purple and Green
    # mem_color = '#8A2BE2'  # Blue Violet
    # compute_color = '#228B22'  # Forest Green
    
    # Option 3: Navy and Coral
    # mem_color = '#191970'  # Midnight Blue
    # compute_color = '#FF7F50'  # Coral
    
    # Option 4: Indigo and Goldenrod
    # mem_color = '#4B0082'  # Indigo
    # compute_color = '#DAA520'  # Goldenrod
    
    # Option 5: Dark Slate Gray and Crimson
    # mem_color = '#2F4F4F'  # Dark Slate Gray
    # compute_color = '#DC143C'  # Crimson
    
    # Bursty comparison
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Memory (left y-axis)
    line1 = ax1.plot(time, mem_actual_bursty, color=mem_color, linestyle='-', 
                     linewidth=4, label='Actual Memory BW', alpha=0.8)
    line2 = ax1.plot(time, mem_dcgm_bursty, color=mem_color, linestyle='--', 
                     linewidth=4, label='Average Memory BW')
    ax1.set_ylabel('Memory Bandwidth (GB/s)', color=mem_color, fontweight='bold', fontsize=20)
    ax1.set_ylim(0, 500)  # Adjusted for new peak of 400 GB/s
    ax1.tick_params(axis='y', labelcolor=mem_color, labelsize=20)
    
    # Set custom x-axis ticks for left subplot
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax1.tick_params(axis='x', labelsize=20)

    # Compute (right y-axis)
    line3 = ax1_twin.plot(time, compute_actual_bursty, color=compute_color, linestyle='-', 
                          linewidth=4, label='Actual Compute', alpha=0.8)
    line4 = ax1_twin.plot(time, compute_dcgm_bursty, color=compute_color, linestyle='--', 
                          linewidth=4, label='Average Compute')
    ax1_twin.set_ylabel('Compute Rate (TFLOP/s)', color=compute_color, fontweight='bold', fontsize=20)
    #ax1_twin.set_ylim(0, 16)
    ax1_twin.set_yticks([0, 4, 8, 12, 16])
    ax1_twin.set_yticklabels(['0', '4', '8', '12', '16'])
    ax1_twin.tick_params(axis='y', labelcolor=compute_color, labelsize=20)
    
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_title('(a) Bursty Pattern\n', fontweight='bold', fontsize=20)
    ax1.grid(True, alpha=0.3)
    
    # Enable and style ALL FOUR spines for left subplot
    ax1.spines['bottom'].set_linewidth(2.5)  # Bottom axis
    ax1.spines['left'].set_linewidth(2.5)    # Left axis
    ax1.spines['top'].set_linewidth(2.5)     # Top axis
    ax1.spines['right'].set_linewidth(2.5)   # Right axis (will be overridden by twin)
    ax1.spines['top'].set_visible(True)      # Make sure top is visible
    ax1.spines['right'].set_visible(True)    # Make sure right is visible
    
    # Style twin axis spines
    ax1_twin.spines['bottom'].set_linewidth(2.5)
    ax1_twin.spines['left'].set_linewidth(2.5)
    ax1_twin.spines['top'].set_linewidth(2.5)
    ax1_twin.spines['right'].set_linewidth(2.5)
    ax1_twin.spines['top'].set_visible(True)
    ax1_twin.spines['left'].set_visible(True)

        # Create combined legend at top center with 2x2 layout
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]

        # Position legend at top center of the figure with 2 columns (2x2 layout)
    ax1.legend(lines, labels, 
              loc='upper center',           # Position at top center
              bbox_to_anchor=(0.7, 1),   # Fine-tune position
              ncol=1,                       # 2 columns for 2x2 layout
              fontsize=15,                  # Adjust font size
              frameon=True,                 # Show frame around legend
              fancybox=True,                # Rounded corners
              shadow=False,                  # Drop shadow
              framealpha=0.9,               # Semi-transparent background
              edgecolor='black',            # Border color
              facecolor='white')            # Background color

    # Interleaved comparison
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Memory (left y-axis) 
    ax2.plot(time, mem_actual_interleaved, color=mem_color, linestyle='-', 
             linewidth=4, label='Actual Memory BW', alpha=0.8)
    ax2.plot(time, mem_dcgm_interleaved, color=mem_color, linestyle='--', 
             linewidth=4, label='Average Memory BW')
    ax2.set_ylabel('Memory Bandwidth (GB/s)', color=mem_color, fontweight='bold', fontsize=20)
    ax2.set_ylim(0, 500)  # Matching the left plot for consistency
    ax2.tick_params(axis='y', labelcolor=mem_color, labelsize=20)
    
    # Set custom x-axis ticks for left subplot
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax2.tick_params(axis='x', labelsize=20)

    # Compute (right y-axis)
    ax2_twin.plot(time, [7.0]*len(time), color=compute_color, linestyle='-', 
                  linewidth=4, label='Actual Compute', alpha=0.8)
    ax2_twin.plot(time, [7.0]*len(time), color=compute_color, linestyle='--', 
                  linewidth=4, label='Average Compute')
    ax2_twin.set_ylabel('Compute Rate (TFLOP/s)', color=compute_color, fontweight='bold', fontsize=20)
    #ax2_twin.set_ylim(0, 16)
    ax2_twin.set_yticks([0, 4, 8, 12, 16])
    ax2_twin.set_yticklabels(['0', '4', '8', '12', '16'])
    ax2_twin.tick_params(axis='y', labelcolor=compute_color, labelsize=20)
    
    ax2.set_xlabel('Time (s)', fontsize=20)
    ax2.set_title('(b) Interleaved Pattern\n', fontweight='bold', fontsize=20)
    ax2.grid(True, alpha=0.3)
    
    # Make axis spines bolder for left subplot
    ax2.spines['bottom'].set_linewidth(2.5)  # x-axis line
    ax2.spines['left'].set_linewidth(2.5)    # y-axis line
    ax2_twin.spines['right'].set_linewidth(2.5)  # right y-axis line

    # Enable and style ALL FOUR spines for left subplot
    ax2.spines['bottom'].set_linewidth(2.5)  # Bottom axis
    ax2.spines['left'].set_linewidth(2.5)    # Left axis
    ax2.spines['top'].set_linewidth(2.5)     # Top axis
    ax2.spines['right'].set_linewidth(2.5)   # Right axis (will be overridden by twin)
    ax2.spines['top'].set_visible(True)      # Make sure top is visible
    ax2.spines['right'].set_visible(True)    # Make sure right is visible
    
    # Style twin axis spines
    ax2_twin.spines['bottom'].set_linewidth(2.5)
    ax2_twin.spines['left'].set_linewidth(2.5)
    ax2_twin.spines['top'].set_linewidth(2.5)
    ax2_twin.spines['right'].set_linewidth(2.5)
    ax2_twin.spines['top'].set_visible(True)
    ax2_twin.spines['left'].set_visible(True)

    # Create combined legend at top center with 2x2 layout
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    # Position legend at top center of the figure with 2 columns (2x2 layout)
    ax2.legend(lines, labels, 
              loc='upper center',           # Position at top center
              bbox_to_anchor=(0.7, 1),   # Fine-tune position
              ncol=1,                       # 2 columns for 2x2 layout
              fontsize=15,                  # Adjust font size
              frameon=True,                 # Show frame around legend
              fancybox=True,                # Rounded corners
              shadow=False,                  # Drop shadow
              framealpha=0.9,               # Semi-transparent background
              edgecolor='black',            # Border color
              facecolor='white')            # Background color
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the legend at the top
    
    return fig
    

def create_dcgm_comparison_fluctuation():
    """Create Figure 2: DCGM Measurement Accuracy - High Quality Version"""
    # Create figure with higher DPI and larger size for better clarity
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # fig.suptitle('DCGM Measurement Accuracy Comparison', fontsize=20, fontweight='bold')
    
    # Option 1: Teal and Orange
    mem_color = '#2E8B8B'  # Teal
    compute_color = '#FF8C00'  # Dark Orange
    
    # Option 2: Purple and Green
    # mem_color = '#8A2BE2'  # Blue Violet
    # compute_color = '#228B22'  # Forest Green
    
    # Option 3: Navy and Coral
    # mem_color = '#191970'  # Midnight Blue
    # compute_color = '#FF7F50'  # Coral
    
    # Option 4: Indigo and Goldenrod
    # mem_color = '#4B0082'  # Indigo
    # compute_color = '#DAA520'  # Goldenrod
    
    # Option 5: Dark Slate Gray and Crimson
    # mem_color = '#2F4F4F'  # Dark Slate Gray
    # compute_color = '#DC143C'  # Crimson
    
    time = np.linspace(0, 1, 10000)  # Higher resolution for fluctuations
    
    # Bursty pattern actual vs DCGM
    mem_actual_bursty = np.where(time <= 0.25, 400, 0)
    mem_dcgm_bursty = np.full_like(time, 100)
    
    compute_actual_bursty = np.where(time <= 0.25, 0.5, 9.33)
    compute_dcgm_bursty = np.full_like(time, 7.0)
    
    # Interleaved pattern with tiny fluctuations to represent individual operations
    # Base values
    base_memory = 100
    base_compute = 7.0
    
    # Create tiny fluctuations to represent 50 billion individual operations
    # Memory fluctuations (8-byte operations)
    np.random.seed(42)  # For reproducible results
    memory_fluctuations = np.random.normal(0, 8, len(time))  # Small variations around baseline
    mem_actual_interleaved = base_memory + memory_fluctuations
    mem_actual_interleaved = np.clip(mem_actual_interleaved, base_memory-20, base_memory+20)  # Limit range
    
    # Compute fluctuations (140 FLOP operations)
    compute_fluctuations = np.random.normal(0, 0.15, len(time))  # Small variations around baseline
    compute_actual_interleaved = base_compute + compute_fluctuations
    compute_actual_interleaved = np.clip(compute_actual_interleaved, base_compute-0.4, base_compute+0.4)  # Limit range
    
    # Add some periodic structure to represent operation batching
    operation_frequency = 50e9  # 50 billion operations per second
    batch_frequency = operation_frequency / 1000  # Batch every 1000 operations
    t_batch = 2 * np.pi * batch_frequency * time
    
    # Add subtle periodic component
    mem_actual_interleaved += 3 * np.sin(t_batch * 0.1) + 2 * np.sin(t_batch * 0.3)
    compute_actual_interleaved += 0.08 * np.sin(t_batch * 0.1) + 0.05 * np.sin(t_batch * 0.3)
    
    # DCGM shows smooth averages (no fluctuations)
    mem_dcgm_interleaved = np.full_like(time, base_memory)
    compute_dcgm_interleaved = np.full_like(time, base_compute)
    
    # Bursty comparison
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Memory (left y-axis)
    line1 = ax1.plot(time, mem_actual_bursty, color=mem_color, linestyle='-', 
                     linewidth=4, label='Actual Memory BW', alpha=0.8)
    line2 = ax1.plot(time, mem_dcgm_bursty, color=mem_color, linestyle='--', 
                     linewidth=4, label='Average Memory BW')
    ax1.set_ylabel('Memory Bandwidth (GB/s)', color=mem_color, fontweight='bold', fontsize=20)
    ax1.set_ylim(0, 500)  # Adjusted for new peak of 400 GB/s
    ax1.tick_params(axis='y', labelcolor=mem_color, labelsize=20)
    
    # Set custom x-axis ticks for left subplot
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax1.tick_params(axis='x', labelsize=20)

    # Compute (right y-axis)
    line3 = ax1_twin.plot(time, compute_actual_bursty, color=compute_color, linestyle='-', 
                          linewidth=4, label='Actual Compute', alpha=0.8)
    line4 = ax1_twin.plot(time, compute_dcgm_bursty, color=compute_color, linestyle='--', 
                          linewidth=4, label='Average Compute')
    ax1_twin.set_ylabel('Compute Rate (TFLOP/s)', color=compute_color, fontweight='bold', fontsize=20)
    #ax1_twin.set_ylim(0, 16)
    ax1_twin.set_yticks([0, 4, 8, 12, 16])
    ax1_twin.set_yticklabels(['0', '4', '8', '12', '16'])
    ax1_twin.tick_params(axis='y', labelcolor=compute_color, labelsize=20)
    
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_title('(a) Bursty Pattern\n', fontweight='bold', fontsize=20)
    ax1.grid(True, alpha=0.3)
    
    # Make axis spines bolder for left subplot
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)

        ax1_twin.spines[axis].set_linewidth(2.0)  # right y-axis line

        # Create combined legend at top center with 2x2 layout
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]

        # Position legend at top center of the figure with 2 columns (2x2 layout)
    ax1.legend(lines, labels, 
              loc='upper center',           # Position at top center
              bbox_to_anchor=(0.7, 1),   # Fine-tune position
              ncol=1,                       # 2 columns for 2x2 layout
              fontsize=15,                  # Adjust font size
              frameon=True,                 # Show frame around legend
              fancybox=True,                # Rounded corners
              shadow=False,                  # Drop shadow
              framealpha=0.9,               # Semi-transparent background
              edgecolor='black',            # Border color
              facecolor='white')            # Background color
    
    # Interleaved comparison with fluctuations
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Memory (left y-axis) - Show fluctuating actual vs smooth DCGM
    ax2.plot(time, mem_actual_interleaved, color=mem_color, linestyle='-', 
             linewidth=2, label='Actual Memory BW', alpha=0.8)  # Thinner line to show fluctuations
    ax2.plot(time, mem_dcgm_interleaved, color=mem_color, linestyle='--', 
             linewidth=4, label='DCGM Memory BW')  # Thick smooth line
    ax2.set_ylabel('Memory Bandwidth (GB/s)', color=mem_color, fontweight='bold', fontsize=16)
    ax2.set_ylim(0, 500)  # Zoomed in to see fluctuations better
    ax2.tick_params(axis='y', labelcolor=mem_color, labelsize=16, width=2, length=8)
    
    # Compute (right y-axis) - Show fluctuating actual vs smooth DCGM
    ax2_twin.plot(time, compute_actual_interleaved, color=compute_color, linestyle='-', 
                  linewidth=2, label='Actual Compute', alpha=0.8)  # Thinner line to show fluctuations
    ax2_twin.plot(time, compute_dcgm_interleaved, color=compute_color, linestyle='--', 
                  linewidth=4, label='DCGM Compute')  # Thick smooth line
    ax2_twin.set_ylabel('Compute Rate (TFLOP/s)', color=compute_color, fontweight='bold', fontsize=16)
    ax2_twin.set_yticks([0, 4, 8, 12, 16])
    ax2_twin.set_yticklabels(['0', '4', '8', '12', '16'])
    ax2_twin.tick_params(axis='y', labelcolor=compute_color, labelsize=20)
    
    ax2.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax2.set_title('(b) Interleaved Pattern\n', fontweight='bold', fontsize=18)
    ax2.grid(True, alpha=0.4, linewidth=1.5)
    
    # Enhanced x-axis ticks
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(['0', '0.5', '1'])
    ax2.tick_params(axis='x', labelsize=16, width=2, length=8)
    
    # Much bolder axis spines
    ax2.spines['bottom'].set_linewidth(3.0)
    ax2.spines['left'].set_linewidth(3.0)
    ax2_twin.spines['right'].set_linewidth(3.0)
    
    # Enhanced legend with larger font and better styling
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    ax2.legend(lines, labels, 
              loc='upper center',           # Position at top center
              bbox_to_anchor=(0.7, 1),   # Fine-tune position
              ncol=1,                       # 2 columns for 2x2 layout
              fontsize=15,                  # Adjust font size
              frameon=True,                 # Show frame around legend
              fancybox=True,                # Rounded corners
              shadow=False,                  # Drop shadow
              framealpha=0.9,               # Semi-transparent background
              edgecolor='black',            # Border color
              facecolor='white')            # Background color
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    return fig

# Generate all figures
if __name__ == "__main__":
    fig = create_dcgm_comparison()
    
    # Save with maximum quality settings
    fig.savefig('dcgm_comparison.png', 
                dpi=600,                    # Very high DPI for crisp print
                bbox_inches='tight',
                facecolor='white',          # White background
                edgecolor='none',
                format='png',
                metadata={'Creator': 'Matplotlib'})
    
    '''
    # Also save as PDF for vector graphics (scalable)
    fig.savefig('dcgm_comparison_vector.pdf', 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='pdf')
    
    # Save as EPS for publications (vector)
    fig.savefig('dcgm_comparison_vector.eps', 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='eps')
    '''
    