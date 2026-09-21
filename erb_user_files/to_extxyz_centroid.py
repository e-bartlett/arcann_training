#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/to_extxyz_centroid.py -- the tested
# copy lives there; this is the copy ArcaNN init places in
# $WORK_DIR/user_files/, run by training/prepare.py's MACE branch. Edit the
# dataset_prep copy, then re-copy here, dropping the `grains` import (that
# module is not staged into user_files/ -- the seed-length sanity check goes
# with it), same adaptation the vendored deepmd_npy_to_extxyz.py carries.
"""Build / grow the CentroidMACE training set (MACE extxyz, absolute-position target).

**Seed mode** (default): convert ``init_he_v2/`` (DeePMD format), a sibling
of ``../to_extxyz.py``. The label column in those already-reviewed arrays is
the finest-cube-grain (stride 2) spin-density centroid absolute position,
sourced from ``spin_analysis.csv`` by ``build_training_set.py``. This skips
``to_extxyz.py``'s previous-frame-residual step and writes that label
directly (milestone 1 of the centroid head, Phase 3 item 4 --
``CENTROID_HEAD_PLAN.md``). Writes ``{train,valid,test}.xyz``.

**Append mode** (``--extra-deepmd-dir`` + ``--centroid-csv``): the
per-iteration retrain step of the ArcaNN tandem loop (ARCANN_TANDEM_PLAN.md
chunk 5, A9). Each extra dir is a 192-atom ``data/<sys>_<iter>/`` DeePMD
set (water only) from ``labeling extract``; the centroid label for each of
its frames comes from ``centroid_labels.csv`` (written by
``centroid_label_from_cube.py`` from the SPIN_DENSITY cubes), joined on the
``config`` id. Config ids per frame come from a ``--config-ids-file`` (one
id per line, frame order) or, if omitted, default to ``00000..`` with a
warning -- the frame<->config alignment is a **live-work-dir assumption**
(see the Chunk 2 note). New frames are split by ``--valid-frac`` and
**appended** to the existing ``{train,valid}.xyz``; ``config``s already
present are skipped, ``test.xyz`` is untouched.

Uses ``info["REF_centroid"]``, not ``to_extxyz.py``'s ``info["REF_dipole"]``
-- a different quantity (absolute position vs. residual-from-previous-frame)
kept under a different key. Read by our own ``train_centroid.py``, not
``mace_run_train``'s CLI, so no ASE reserved-key workaround is needed.

Examples
--------
    python to_extxyz_centroid.py
    python to_extxyz_centroid.py --deepmd-dir ../init_he_v2 --out mace_training_set_centroid
    python to_extxyz_centroid.py --out mace_training_set_centroid \\
        --extra-deepmd-dir $WORK_DIR/data/he_001 \\
        --config-ids-file $WORK_DIR/data/he_001/config_ids.txt \\
        --centroid-csv centroid_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from deepmd_npy_to_extxyz import TYPE_MAP, load_deepmd_dir  # noqa: E402

N_WATERS = 64
N_ATOMS = N_WATERS * 3  # 192 real atoms, O-H-H per water
SYMBOLS = ["O", "H", "H"] * N_WATERS

SPLIT_NAMES = {0: "train", 1: "valid", 2: "test"}
ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"
CENTROID_KEY = "REF_centroid"


def build_atoms(i, coord, force, energy, cell_lengths, label, grain_id):
    atoms = Atoms(
        symbols=SYMBOLS,
        positions=coord[i, : N_ATOMS * 3].reshape(N_ATOMS, 3),
        cell=cell_lengths[i],
        pbc=True,
    )
    atoms.info[ENERGY_KEY] = float(energy[i])
    atoms.arrays[FORCES_KEY] = force[i, : N_ATOMS * 3].reshape(N_ATOMS, 3)
    # Plain numpy array, not .tolist(): ASE writes a bare "x y z" triplet for
    # arrays but wraps a Python list in its own "_JSON [...]" extension,
    # which a non-ASE extxyz reader may not understand (see to_extxyz.py).
    atoms.info[CENTROID_KEY] = label[i]

    atoms.info["config"] = f"{i:05d}"
    atoms.info["grain_id"] = int(grain_id[i])
    return atoms


def convert(deepmd_dir):
    set_dir = deepmd_dir / "set.000"
    coord = np.load(set_dir / "coord.npy")
    force = np.load(set_dir / "force.npy")
    energy = np.load(set_dir / "energy.npy")
    box = np.load(set_dir / "box.npy")
    grain_id = np.load(set_dir / "grain_id.npy")
    split_id = np.load(set_dir / "split_id.npy")

    n_configs = len(coord)
    label = coord[:, N_ATOMS * 3 :]  # the old dummy-atom column: the absolute centroid label
    cell_lengths = box[:, [0, 4, 8]]  # cubic cell, diagonal entries

    frames = {code: [] for code in SPLIT_NAMES}
    for i in range(n_configs):
        atoms = build_atoms(i, coord, force, energy, cell_lengths, label, grain_id)
        frames[int(split_id[i])].append(atoms)

    return frames


def read_centroid_csv(path):
    """config id -> (cx, cy, cz) Angstrom, from centroid_label_from_cube.py's CSV."""
    labels = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            labels[str(row["config"])] = np.array(
                [float(row["cx_A"]), float(row["cy_A"]), float(row["cz_A"])]
            )
    return labels


def config_ids_for_dir(deepmd_dir, ids_file, n_frames):
    """One config id per frame, in frame order. From ``ids_file`` if given,
    else 00000.. with a warning -- the frame<->config alignment is a
    live-work-dir assumption."""
    if ids_file is not None:
        ids = [ln.strip() for ln in Path(ids_file).read_text().splitlines() if ln.strip()]
        if len(ids) != n_frames:
            sys.exit(f"error: {ids_file} has {len(ids)} ids, {deepmd_dir} has {n_frames} frames")
        return ids
    print(f"warning: no --config-ids-file for {deepmd_dir}; assuming frame i == config "
          f"{'{:05d}'.format(0)}.. (0-indexed). Pass one if labeling dirs differ.")
    return [f"{i:05d}" for i in range(n_frames)]


def build_extra_atoms(coord_row, force_row, energy, cell, centroid, config_id):
    n = len(coord_row) // 3
    atoms = Atoms(
        symbols=["O", "H", "H"] * (n // 3) if n % 3 == 0 else None,
        positions=coord_row.reshape(n, 3), cell=cell, pbc=True,
    )
    atoms.info[ENERGY_KEY] = float(energy)
    atoms.arrays[FORCES_KEY] = force_row.reshape(n, 3)
    atoms.info[CENTROID_KEY] = np.asarray(centroid, dtype=float)
    atoms.info["config"] = config_id
    return atoms


def append_extra(options):
    """Append new labeled configs to an existing {train,valid}.xyz."""
    labels = read_centroid_csv(options.centroid_csv)
    have = set()
    for name in ("train", "valid"):
        path = options.out / f"{name}.xyz"
        if path.is_file():
            have.update(str(a.info.get("config")) for a in read(path, index=":"))

    ids_files = list(options.config_ids_file)
    rng = np.random.default_rng(options.seed)
    # Shared split map (see training/prepare.py::_prepare_mace): keyed
    # "<deepmd_dir.name>_<frameidx:05d>" -> "train"|"valid", so a config
    # lands on the same side for the centroid and the force converters.
    split_map = (
        json.loads(Path(options.split_file).read_text())
        if options.split_file is not None else None
    )
    added = {"train": 0, "valid": 0}
    for deepmd_dir in options.extra_deepmd_dir:
        coord, force, energy, box, type_raw, _ = load_deepmd_dir(deepmd_dir)
        if 2 in type_raw:
            sys.exit(f"error: {deepmd_dir} has an X particle -- pass it to the force "
                     "converter, not here (the centroid is the label, not an atom)")
        n_frames = len(coord)
        ids_file = ids_files.pop(0) if ids_files else None
        config_ids = config_ids_for_dir(deepmd_dir, ids_file, n_frames)

        pending = []
        for i, config_id in enumerate(config_ids):
            if config_id in have:
                continue
            if config_id not in labels:
                sys.exit(f"error: config {config_id} ({deepmd_dir} frame {i}) not in "
                         f"{options.centroid_csv} -- run centroid_label_from_cube.py first")
            pending.append((i, build_extra_atoms(
                coord[i], force[i], energy[i], box[i].reshape(3, 3),
                labels[config_id], config_id,
            )))
            have.add(config_id)

        if pending and split_map is not None:
            for i, atoms in pending:
                key = f"{deepmd_dir.name}_{i:05d}"
                if key not in split_map:
                    sys.exit(f"error: {key} not in --split-file {options.split_file}")
                bucket = "valid" if split_map[key] == "valid" else "train"
                write(options.out / f"{bucket}.xyz", atoms, format="extxyz", append=True)
                added[bucket] += 1
        elif pending:
            idx = rng.permutation(len(pending))
            n_valid = max(1, round(len(pending) * options.valid_frac))
            valid_idx = set(idx[:n_valid].tolist())
            for j, (i, atoms) in enumerate(pending):
                bucket = "valid" if j in valid_idx else "train"
                write(options.out / f"{bucket}.xyz", atoms, format="extxyz", append=True)
                added[bucket] += 1

    if ids_files:
        sys.exit(f"error: {len(ids_files)} unused --config-ids-file")
    for name in ("train", "valid"):
        print(f"appended {added[name]} frame(s) to {options.out / f'{name}.xyz'}")
    return 0


def parse_args(argv=None):
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--deepmd-dir", type=Path, default=here / "init_he_v2",
                        help="seed-mode DeePMD dir to convert (default: %(default)s)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "mace_training_set_centroid",
                        help="output directory (default: %(default)s)")
    parser.add_argument("--extra-deepmd-dir", type=Path, action="append", default=[],
                        help="append mode: a 192-atom data/<sys>_<iter>/ dir (repeatable)")
    parser.add_argument("--config-ids-file", type=Path, action="append", default=[],
                        help="one config id per frame for the matching --extra-deepmd-dir "
                             "(repeatable, same order)")
    parser.add_argument("--centroid-csv", type=Path,
                        help="append mode: centroid_label_from_cube.py output")
    parser.add_argument("--valid-frac", type=float, default=0.1,
                        help="append mode held-out fraction, ignored when --split-file "
                             "is given (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="append mode split RNG seed, ignored when --split-file "
                             "is given (default: %(default)s)")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="append mode: JSON {\"<dir>_<frameidx:05d>\": "
                             "\"train\"|\"valid\"} shared with the force converter so a "
                             "config lands on the same train/valid side for both models")
    options = parser.parse_args(argv)
    if options.extra_deepmd_dir and options.centroid_csv is None:
        parser.error("--extra-deepmd-dir needs --centroid-csv")
    return options


def main(argv=None):
    options = parse_args(argv)

    if options.extra_deepmd_dir:
        return append_extra(options)

    if not (options.deepmd_dir / "set.000" / "coord.npy").is_file():
        sys.exit(f"error: {options.deepmd_dir} has no set.000/coord.npy -- "
                 "run build_training_set.py first")

    frames = convert(options.deepmd_dir)

    options.out.mkdir(parents=True, exist_ok=True)
    for code, name in SPLIT_NAMES.items():
        path = options.out / f"{name}.xyz"
        write(path, frames[code], format="extxyz")
        print(f"wrote {len(frames[code])} frames to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
