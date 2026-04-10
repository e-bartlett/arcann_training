import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

cycle = '23'
base_path = "he"

# Output directories
plot_dir = "plumed_plots"
os.makedirs(plot_dir, exist_ok=True)

# Copy reference file
shutil.copyfile(f"{base_path}/3/00003/plumed_he.dat", f"{plot_dir}/plumed.dat")

# Load header and data
plumed_file = f"{base_path}/3/00003/plumed_values.txt"
with open(plumed_file) as f:
    header = f.readline().strip().replace('#!', '').strip().split()
    if header[0] == 'FIELDS':
        header = header[1:]

field_indices = {name: i for i, name in enumerate(header)}
plumed_data = np.loadtxt(plumed_file, comments='#')
time = plumed_data[:, field_indices['time']]

# Plot all fields vs time
for name in header:
    if name == 'time':
        continue
    plt.figure(figsize=(8, 5))
    plt.plot(time, plumed_data[:, field_indices[name]])
    plt.xlabel('Time')
    plt.ylabel(name)
    plt.title(f'{name} vs Time')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{name}.png')
    plt.close()

# Create a grid of deviation plots
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        devi_path = f"{base_path}/{i+1}/0000{j+1}/model_devi_he_{i+1}_0{cycle}.out"
        plumed_path = f"{base_path}/{i+1}/0000{j+1}/plumed_values.txt"

        try:
            dev_data = np.genfromtxt(devi_path, skip_header=1)
            plumed_vals = np.loadtxt(plumed_path, comments='#')
        except Exception or UserWarning:
            dev_data = np.zeros((1, 7))
            plumed_vals = np.zeros((1, 8))

        # Subsample and pad with last value
        step = np.append(dev_data[:, 0][::2], dev_data[-1, 0])
        max_devi_f = np.append(dev_data[:, 4][::2], dev_data[-1, 4])
        plumed_steps = np.append(plumed_vals[:, 0] * 1000, plumed_vals[-1, 0] * 1000)
        dsXH_min = np.append(plumed_vals[:, field_indices["dsXH.min"]], plumed_vals[-1, field_indices["dsXH.min"]])
        #dsXH_center = np.append(plumed_vals[:, field_indices["bias1.dsXH.min_cntr"]], plumed_vals[-1, field_indices["bias1.dsXH.min_cntr"]])

        #compute min from trajectories
        import MDAnalysis as mda
        import numpy as np

        L = 12.43  # Box length for PBC

        def compute_distance(coords1, coords2):
            delta = coords2 - coords1
            delta -= L * np.round(delta / L)  # Apply minimum image convention
            return np.linalg.norm(delta, axis=1)  # Vectorized for all H atoms

        pdb_file = "he_1.pdb"
        dcd_file = f"{base_path}/{i+1}/0000{j+1}/he_{i+1}_023.dcd"

        u = mda.Universe(pdb_file, dcd_file)
        hydrogens = u.select_atoms("name H")
        elec = u.select_atoms("name LI")

        traj_min = []
        #with open("min_e_H.txt", "w") as f:
        for ts in u.trajectory:
            h_coords = hydrogens.positions
            e_coord = elec.positions[0]  # Single electron
            dists = compute_distance(h_coords, e_coord)
            min_dist = np.min(dists)
            traj_min.append(min_dist)


        ax_left = ax
        ax_right = ax_left.twinx()

        ax_left.plot(step / 2, max_devi_f, color='b', label="Max Deviation")
        ax_left.axhline(y=0.1, color='gray', linestyle='--')
        ax_left.set_ylim(0, 1.5)
        ax_left.tick_params(axis='y', labelcolor='b')


        ax_right.plot(plumed_steps, dsXH_min, color='r', label="PLUMED min")
        ax_right.plot(step/2, traj_min, color='k', label="Trajectory min")
        #ax_right.plot(plumed_steps, dsXH_center, color='k', label="PLUMED center")
        ax_right.tick_params(axis='y', labelcolor='r')
        ax_right.set_ylim(0, 2.1)

        ax_left.set_title(f"NNP{i+1}_0000{j+1}")
        ax_left.grid(True)


# Shared axis labels
axes[2, 1].set_xlabel("Time (fs)")
axes[1, 0].set_ylabel(r"Max Force Deviation", color='b')


ax = axes[1, 2]
ax2 = ax.twinx()
ax2.set_ylabel(r"PLUMED min ($\AA$)", color='r', labelpad=25)
ax2.tick_params(right=False, labelright=False)



plt.tight_layout()
plt.savefig(f"devi_{cycle}.png")
plt.savefig(f"plumed_plots/devi_{cycle}.png")
plt.show()
