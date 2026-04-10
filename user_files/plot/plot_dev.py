import matplotlib.pyplot as plt
import numpy as np

cycle = '3'

# Create a figure with a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex='all')

colors = ['r', 'b', 'k']
cnt = 0

for i in range(3):
    for j in range(2):
        # Load data from file
        job = 4251126
        if i == 2 and j == 2:
            cnt=-1
        data = np.genfromtxt(f"he/{i+1}/0000{j+1}/model_devi_he_{i+1}_00" + cycle + ".out", skip_header=1)
        cnt += 1
        
        # Extract every other value and append the last one
        step = np.append(data[:, 0][::2], data[-1, 0])
        max_devi_f = np.append(data[:, 4][::2], data[-1, 4])
        min_devi_f = np.append(data[:, 5][::2], data[-1, 5])
        avg_devi_f = np.append(data[:, 6][::2], data[-1, 6])

        # Plot the data
        axes[i][j].plot(step, avg_devi_f, label="avg deviation", color='k')
        axes[i][j].plot(step, min_devi_f, label="min deviation", color='b')
        axes[i][j].plot(step, max_devi_f, label="max deviation", color='r')
        axes[i][j].axhline(y=0.1, color='gray', linestyle='--')
        axes[i][j].grid(True)
        
        axes[i][j].set_ylim(0, 1.5)
        axes[i][j].set_xlim(0, 20000)

        axes[i, j].set_title(f"NNP{i+1}_0000{j+1}")

axes[2,1].set_xlabel("Step", fontsize=12)
axes[1,0].set_ylabel(r"Deviation in Force (eV/\AA)", fontsize=12)

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig("devi_" + cycle + '.png')
