from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

# Set up directories
working_dir = Path(__file__).resolve().parent
he_dir = working_dir / "tr3_fixed"
out_dir = working_dir / "mace_training_set"
out_dir.mkdir(exist_ok=True)

# Set up constants
Ha_to_eV = np.float64(27.211386245988)
Bohr_to_A = np.float64(0.529177210903)
au_to_eV_per_A = np.float64(Ha_to_eV / Bohr_to_A)

# System composition
n_silica_sioh = 96            # 32 x (Si, O, H)
n_si = 48
n_o = 144
n_waters = 192
n_atoms = n_silica_sioh + n_si + n_o + n_waters * 3   # 864

cell = [20.251, 14.32, 35.707675]  # <-- your cell lengths in Angstrom (not necessarily cubic)
pbc = True                     

skip_first = 2000
failed = set(range(skip_first))

symbols = (
    ["Si", "O", "H"] * 32
    + ["Si"] * 48
    + ["O"] * 144
    + ["H", "H", "O"] * n_waters
)
assert len(symbols) == n_atoms, f"expected {n_atoms}, got {len(symbols)}"

ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"

# Verify atom order against the first available coordinate file
first_conf = next(i for i in range(0, 10001, 20) if i not in failed)
with open(he_dir / f"interface_{first_conf}.xyz") as f:
    ref_symbols = [l.split()[0] for l in f.readlines()[2:2 + n_atoms]]
assert ref_symbols == symbols, "Atom order in xyz does not match expected symbols"

frames = []
for i in range(0, 10001, 20):
    if i in failed:
        continue
    if i % 100 == 0:
        print(i)
    conf = str(i)

    # read in energy
    with open(he_dir / f"energy.out-1_{conf}.Log", "r") as g:
        e_lines = g.readlines()
    energy = float(e_lines[1].split()[-1]) * Ha_to_eV

    # read in forces
    forces = np.zeros((n_atoms, 3))
    with open(he_dir / f"forces.out-1_{conf}.xyz", "r") as g:
        f_lines = g.readlines()
    for j in range(n_atoms):
        parts = f_lines[j + 3].split()
        forces[j] = [float(x) * au_to_eV_per_A for x in parts[2:5]]

    # read in coordinates
    coords = np.zeros((n_atoms, 3))
    with open(he_dir / f"interface_{conf}.xyz", "r") as f:
        lines = f.readlines()[2:]
    for j in range(n_atoms):
        parts = lines[j].split()
        coords[j] = [float(x) for x in parts[1:4]]

    atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=pbc)
    atoms.info[ENERGY_KEY] = energy
    atoms.arrays[FORCES_KEY] = forces
    atoms.info["config"] = conf
    frames.append(atoms)
print("coordinates, energies, and forces extracted")

out_path = out_dir / "i_mace_dataset3.xyz"
write(out_path, frames, format="extxyz")
print(f"wrote {len(frames)} frames to {out_path}")
