#!/usr/bin/env python3
"""Extract coordinates, energies, and forces from he/XXXXX/* raw CP2K
labeling output directly into extxyz

Also writes the electron coordinates separately: the stride-4
spin-density centroid (cx_A/cy_A/cz_A) for each config, read from
spin_analysis.csv, one row per config.
"""

import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

# Set up directories
working_dir = Path(__file__).resolve().parent
he_dir = working_dir / "he"
out_dir = working_dir / "mace_training_set"
out_dir.mkdir(exist_ok=True)

# Set up constants
Ha_to_eV = np.float64(27.211386245988)
Bohr_to_A = np.float64(0.529177210903)
au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)

n_waters = 64
n_atoms = n_waters * 3
n_configs = 755
box_length = 12.43  # Angstrom, cubic cell

symbols = ["O", "H", "H"] * n_waters
ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"

# Configs that failed labeling and have no valid output to read
with open(he_dir / "failed_indices.txt", "r") as f:
    failed = {int(line) for line in f.read().split()}
f.close()

frames = []
for i in range(n_configs):
    if i in failed:
        continue
    if i % 50 == 0:
        print(i)
    conf = f"{i:05}"

    # read in energy
    with open(he_dir / conf / f"2_labeling_{conf}-Force_Eval.fe", "r") as g:
        e_lines = g.readlines()
        energy = float(e_lines[1].split()[-1]) * Ha_to_eV
    g.close()

    # read in forces
    forces = np.zeros((n_atoms, 3))
    with open(he_dir / conf / f"2_labeling_{conf}-Forces.for", "r") as g:
        f_lines = g.readlines()
        for j in range(n_atoms):
            parts = f_lines[j + 3].split()
            forces[j] = [float(x) * au_to_eV_per_A for x in parts[2:5]]
    g.close()

    # read in coordinates
    coords = np.zeros((n_atoms, 3))
    with open(he_dir / conf / f"he_{conf}.xyz", "r") as f:
        lines = f.readlines()
        del lines[0:2]
        for j in range(n_atoms):
            parts = lines[j].split()
            coords[j] = [float(x) for x in parts[1:4]]
    f.close()

    atoms = Atoms(
        symbols=symbols,
        positions=coords,
        cell=[box_length, box_length, box_length],
        pbc=True,
    )
    atoms.info[ENERGY_KEY] = energy
    atoms.arrays[FORCES_KEY] = forces
    atoms.info["config"] = conf
    frames.append(atoms)
print("coordinates, energies, and forces extracted")

out_path = out_dir / "he_mace_dataset_no_electron.xyz"
write(out_path, frames, format="extxyz")
print(f"wrote {len(frames)} frames to {out_path}")

# electron coordinates: stride-4 spin-density centroid, one row per config
electron_coords = np.zeros((n_configs, 3))
with open(working_dir / "spin_analysis.csv", "r") as f:
    for row in csv.DictReader(f):
        if row["stride"] != "4":
            continue
        i = int(row["config"])
        electron_coords[i] = [float(row["cx_A"]), float(row["cy_A"]), float(row["cz_A"])]
f.close()

electron_path = out_dir / "electron_coords_stride4.xyz"
n_written = 0
with open(electron_path, "w") as f:
    for i in range(n_configs):
        if i in failed:
            continue
        conf = f"{i:05}"
        f.write("1\n")
        f.write(f"config={conf} stride=4\n")
        f.write(f"X {electron_coords[i,0]:.8f} {electron_coords[i,1]:.8f} {electron_coords[i,2]:.8f}\n")
        n_written += 1
f.close()
print(f"wrote {n_written} electron coordinates to {electron_path}")
