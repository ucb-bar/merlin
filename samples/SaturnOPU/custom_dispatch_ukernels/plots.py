import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DATA DEFINITION
# ==========================================

# X-Axis: Matrix Sizes (N x N)
matrix_sizes = np.array([64, 128, 256, 512, 1024, 2048])
x_labels = matrix_sizes.astype(str)

# Baseline (Standard RVV) 
# ONLY defined for first 3 points (64, 128, 256)
baseline_ops_subset = np.array([0.5, 0.51, 0.52]) 
baseline_x_subset = x_labels[:3]

# Ours (RVV + uKernel + OPU) 
# Defined for ALL points
ours_ops = np.array([0.95, 2.25, 4.21, 8.5, 13.0, 19.5])

# ==========================================
# 2. PLOT 1: OPS PER CYCLE (Line Chart)
# ==========================================
plt.figure(figsize=(10, 6))

# Plot "Ours" (Blue Line) - Full Range
plt.plot(x_labels, ours_ops, 
         color='#0033cc', linewidth=3, marker='o', markersize=8, 
         label='Ours (RVV + OPU)')

# Plot "Baseline" (Orange Line) - STOP at 256
plt.plot(baseline_x_subset, baseline_ops_subset, 
         color='#ff8c00', linewidth=3, marker='s', markersize=8, 
         label='Baseline (Standard RVV)')

# Formatting
plt.title('Compute Performance Scaling', fontsize=16, fontweight='bold')
plt.xlabel('Matrix Size (N x N)', fontsize=14)
plt.ylabel('Ops / Cycle', fontsize=14)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle='-', alpha=0.3)
plt.ylim(0, max(ours_ops) * 1.1)

# Save
plt.tight_layout()
plt.savefig('plot_ops_cycle_v2.png', dpi=300)
print("Saved: plot_ops_cycle_v2.png")
plt.close()


# ==========================================
# 3. PLOT 2: SPEEDUP (Bar Chart - First 3 Only)
# ==========================================

# Calculate Speedup for the overlapping points
speedup_factors = ours_ops[:3] / baseline_ops_subset

plt.figure(figsize=(8, 6))

# Create Green Bars
bars = plt.bar(baseline_x_subset, speedup_factors, 
               color='#00b050', edgecolor='black', width=0.6, zorder=3)

# Add Labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{height:.1f}x',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Formatting
plt.title('Speedup vs. Baseline', fontsize=16, fontweight='bold')
plt.xlabel('Matrix Size (N x N)', fontsize=14)
plt.ylabel('Speedup Factor', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
plt.ylim(0, max(speedup_factors) * 1.2) 

# Save
plt.tight_layout()
plt.savefig('plot_speedup_v2.png', dpi=300)
print("Saved: plot_speedup_v2.png")
plt.close()