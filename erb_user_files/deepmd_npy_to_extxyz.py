#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/deepmd_npy_to_extxyz.py -- the tested
# copy lives there; this is the copy ArcaNN init places in
# $WORK_DIR/user_files/ for training/prepare.py. Edit the dataset_prep copy,
# then re-copy here.
"""Convert ArcaNN DeePMD training dirs to MACE extxyz for the force model.

Chunk 2 of ``ARCANN_TANDEM_PLAN.md``. ``training/prepare.py``'s MACE branch
calls this on the ``data/<system>_<iter>/`` dirs it discovers plus the seed
``data/init_he*``, writing ``<nnp>/train.xyz`` and ``<nnp>/valid.xyz`` for
``mace_run_train``.

Every frame carries an explicit dummy ``X`` atom at the electron position
(locked decision 1 of the plan: the force model sees the electron, with a
zero force target on it). Two input shapes are accepted:

* **193-particle dir** (the seed ``init_he_v2``): the electron is already
  particle 193 in ``coord.npy`` / ``force.npy``; its force row is zeroed
  here. When ``grain_id.npy`` + ``split_id.npy`` are present they drive the
  split (0/1/2 = train/valid/test, matching ``to_extxyz.py``); test frames
  are written to ``test.xyz`` and left out of the ``mace_run_train`` inputs.
* **192-particle dir** (an ArcaNN ``labeling extract`` output for a new
  iteration): water only. The electron position for each frame comes from
  the matching ``elec_candidates_<iter>_<system>.xyz`` (one single-atom XYZ
  frame per config, produced by ``exploration/extract.py``). No split
  columns, so the split is random by ``--valid-frac``.

Energies go to ``info["REF_energy"]`` and forces to
``arrays["REF_forces"]`` -- ``mace_run_train``'s own default keys, so no
``--energy_key`` / ``--forces_key`` override is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

TYPE_MAP = ["O", "H", "X"]  # DeePMD type index -> element (config.json type_map)
ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"
SPLIT_NAMES = {0: "train", 1: "valid", 2: "test"}


def load_deepmd_dir(deepmd_dir: Path):
    """Return (coord, force, energy, box, type_raw, split_id-or-None)."""
    set_dir = deepmd_dir / "set.000"
    coord = np.load(set_dir / "coord.npy")
    force = np.load(set_dir / "force.npy")
    energy = np.load(set_dir / "energy.npy")
    box = np.load(set_dir / "box.npy")
    type_raw = np.genfromtxt(deepmd_dir / "type.raw", dtype=int).reshape(-1)

    split_path = set_dir / "split_id.npy"
    split_id = np.load(split_path) if split_path.is_file() else None

    n_frames = len(coord)
    n_particles = len(type_raw)
    if coord.shape[1] != n_particles * 3 or force.shape[1] != n_particles * 3:
        sys.exit(
            f"error: {deepmd_dir}: coord/force width {coord.shape[1]}/"
            f"{force.shape[1]} does not match {n_particles} particles from type.raw"
        )
    if len(energy) != n_frames or len(box) != n_frames:
        sys.exit(f"error: {deepmd_dir}: energy/box length mismatch with coord")
    if split_id is not None and len(split_id) != n_frames:
        sys.exit(f"error: {deepmd_dir}: split_id length mismatch with coord")

    return coord, force, energy, box, type_raw, split_id


def electron_positions_from_xyz(elec_xyz: Path, n_frames: int) -> np.ndarray:
    """(n_frames, 3) electron positions from a single-atom-per-frame XYZ."""
    frames = read(elec_xyz, index=":", format="extxyz")
    if len(frames) != n_frames:
        sys.exit(
            f"error: {elec_xyz} has {len(frames)} frames, expected {n_frames} "
            "(one per config in the paired DeePMD dir)"
        )
    positions = np.array([f.positions[0] for f in frames])
    return positions


def build_frame(symbols, positions, forces, energy, cell, config_id):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.info[ENERGY_KEY] = float(energy)
    atoms.arrays[FORCES_KEY] = forces  # plain ndarray: bare "x y z" columns
    atoms.info["config"] = config_id
    # TODO(noise-injection): optional jittered replicas of the X position /
    # water go here -- deferred this loop (plan tactic 1).
    return atoms


def frames_from_dir(deepmd_dir: Path, elec_xyz: Path | None):
    """Yield (split_name, Atoms) for every frame in one DeePMD dir."""
    coord, force, energy, box, type_raw, split_id = load_deepmd_dir(deepmd_dir)
    n_frames = len(coord)
    n_particles = len(type_raw)

    has_dummy = 2 in type_raw
    if has_dummy:
        if n_particles < 2 or type_raw[-1] != 2:
            sys.exit(f"error: {deepmd_dir}: expected the X particle to be last in type.raw")
        symbols = [TYPE_MAP[t] for t in type_raw]
        n_real = n_particles - 1
    else:
        if elec_xyz is None:
            sys.exit(
                f"error: {deepmd_dir} has no X particle and no paired --elec-xyz; "
                "cannot place the electron"
            )
        symbols = [TYPE_MAP[t] for t in type_raw] + ["X"]
        n_real = n_particles
        elec_pos = electron_positions_from_xyz(elec_xyz, n_frames)

    for i in range(n_frames):
        cell = box[i].reshape(3, 3)
        real_xyz = coord[i].reshape(n_particles, 3)
        real_f = force[i].reshape(n_particles, 3)
        if has_dummy:
            positions = real_xyz.copy()
            forces = real_f.copy()
            forces[-1] = 0.0  # zero force target on the electron
        else:
            positions = np.vstack([real_xyz, elec_pos[i]])
            forces = np.vstack([real_f, np.zeros(3)])

        split = int(split_id[i]) if split_id is not None else None
        config_id = f"{deepmd_dir.name}_{i:05d}"
        yield split, build_frame(symbols, positions, forces, energy[i], cell, config_id), n_real


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--deepmd-dir", type=Path, action="append", required=True,
        help="a DeePMD dataset dir (repeatable). 193-particle dirs carry the "
             "electron already; 192-particle dirs each consume one --elec-xyz.",
    )
    parser.add_argument(
        "--elec-xyz", type=Path, action="append", default=[],
        help="electron XYZ for a 192-particle --deepmd-dir (repeatable, "
             "consumed in the order the 192-particle dirs appear).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="output dir; writes train.xyz, valid.xyz (and test.xyz if any "
             "dir has split_id == 2).",
    )
    parser.add_argument(
        "--valid-frac", type=float, default=0.1,
        help="held-out fraction for dirs without split_id (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the --valid-frac split (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    rng = np.random.default_rng(options.seed)

    elec_queue = list(options.elec_xyz)
    buckets = {name: [] for name in SPLIT_NAMES.values()}
    n_real_seen = set()

    for deepmd_dir in options.deepmd_dir:
        if not (deepmd_dir / "set.000" / "coord.npy").is_file():
            sys.exit(f"error: {deepmd_dir} has no set.000/coord.npy")
        type_raw = np.genfromtxt(deepmd_dir / "type.raw", dtype=int).reshape(-1)
        needs_elec = 2 not in type_raw
        elec_xyz = None
        if needs_elec:
            if not elec_queue:
                sys.exit(f"error: {deepmd_dir} needs an --elec-xyz but none left")
            elec_xyz = elec_queue.pop(0)

        pending = []  # frames from this dir with no split_id, for the random split
        for split, atoms, n_real in frames_from_dir(deepmd_dir, elec_xyz):
            n_real_seen.add(n_real)
            if split is None:
                pending.append(atoms)
            else:
                buckets[SPLIT_NAMES[split]].append(atoms)

        if pending:
            idx = rng.permutation(len(pending))
            n_valid = max(1, round(len(pending) * options.valid_frac))
            valid_idx = set(idx[:n_valid].tolist())
            for j, atoms in enumerate(pending):
                buckets["valid" if j in valid_idx else "train"].append(atoms)

    if elec_queue:
        sys.exit(f"error: {len(elec_queue)} unused --elec-xyz file(s)")
    if len(n_real_seen) > 1:
        print(f"warning: mixed real-atom counts across dirs: {sorted(n_real_seen)}")

    options.out.mkdir(parents=True, exist_ok=True)
    for name, frames in buckets.items():
        if name == "test" and not frames:
            continue
        path = options.out / f"{name}.xyz"
        write(path, frames, format="extxyz")
        print(f"wrote {len(frames)} frames to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
