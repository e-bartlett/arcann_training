import matplotlib.pyplot as plt
import numpy as np

cycle = 0

# Create a figure with 1 row, 3 columns of subplots
fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharex=True)

colors = ['r', 'b', 'k']

for i in range(3):
    # Load data from file
    data = np.genfromtxt(f"{i+1}/lcurve.out", names=True)
    
    for j, name in enumerate(data.dtype.names[1:-1]):  # Skip the first and last columns
        axes[i].plot(data["step"], data[name], label=name, color=colors[j % len(colors)])
    
    axes[i].set_xlabel("Step")
    axes[i].set_ylabel("Loss")
    axes[i].set_xscale("symlog")
    axes[i].set_yscale("log")
    axes[i].grid()
    axes[i].legend()

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig("lcurve_" + str(cycle) + ".png")
