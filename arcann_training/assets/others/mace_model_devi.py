#!/usr/bin/env python3
"""Committee model deviation for a MACE LAMMPS exploration (non-he path).

Stock ArcaNN gets ``model_devi_*.out`` from ``pair_style deepmd ... out_file``,
which evaluates the whole committee inside LAMMPS. MACE has no equivalent, so
for ``mlip_engine: "mace"`` without ``hydrated_electron_mode`` the explore job
runs LAMMPS with one model (``pair_style mliap unified``) and then this script
re-evaluates every frame of the deviation dump with all committee models and
writes the file in DeePMD-kit's format, so ``exploration check / deviate /
extract`` work unchanged:

    # step max_devi_v min_devi_v avg_devi_v max_devi_f min_devi_f avg_devi_f

Definitions follow DeePMD-kit (deepmd/infer/model_devi.py):
- force: per atom, the norm over x/y/z of the committee standard deviation;
  max/min/mean over atoms.
- virial: committee standard deviation of each of the 9 virial components
  divided by the atom count; max/min over components, avg = norm / 3.

The dump is the ``dump ... custom ... id type x y z`` that
``exploration prepare`` adds to the LAMMPS input (sorted by id). LAMMPS type
-> element comes from the ``pair_coeff * * <elements>`` line of that input.

Runs in the MACE env (mace-torch, ase, numpy); it does not need ArcaNN.

Usage:
    python mace_model_devi.py --lammps-in X.in --dump X_devi.lammpstrj \
        --out model_devi_X.out --models a.model b.model c.model [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_pair_coeff_elements(lammps_in: Path):
    """Elements from the last 'pair_coeff * * El1 El2 ...' line of a LAMMPS input."""
    elements = None
    for line in lammps_in.read_text().splitlines():
        tokens = line.split("#", 1)[0].split()
        if len(tokens) > 3 and tokens[0] == "pair_coeff" and tokens[1:3] == ["*", "*"]:
            elements = tokens[3:]
    if not elements:
        raise ValueError(
            f"No 'pair_coeff * * <elements>' line in '{lammps_in}': "
            "cannot map LAMMPS atom types to elements."
        )
    return elements


def iter_dump_frames(dump_path: Path):
    """Yield (step, cell 3x3, types, positions) from a LAMMPS text dump, sorted by id.

    Positions are shifted by the box origin (xlo, ylo, zlo) so they sit in an
    ASE cell whose origin is 0.
    """
    with dump_path.open() as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline().split()[0])
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().split()[0])
            box_header = f.readline()  # ITEM: BOX BOUNDS ...
            bounds = np.array([f.readline().split() for _ in range(3)], dtype=float)
            if "xy" in box_header:
                # Triclinic: lines are 'lo_bound hi_bound tilt' (xy, xz, yz)
                xy, xz, yz = bounds[:, 2]
                xlo = bounds[0, 0] - min(0.0, xy, xz, xy + xz)
                xhi = bounds[0, 1] - max(0.0, xy, xz, xy + xz)
                ylo = bounds[1, 0] - min(0.0, yz)
                yhi = bounds[1, 1] - max(0.0, yz)
                zlo, zhi = bounds[2, 0], bounds[2, 1]
                cell = np.array(
                    [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]]
                )
                origin = np.array([xlo, ylo, zlo])
            else:
                cell = np.diag(bounds[:, 1] - bounds[:, 0])
                origin = bounds[:, 0]
            columns = f.readline().split()[2:]  # ITEM: ATOMS id type x y z
            data = np.array([f.readline().split() for _ in range(natoms)], dtype=float)
            col = {name: i for i, name in enumerate(columns)}
            order = np.argsort(data[:, col["id"]])
            data = data[order]
            yield (
                step,
                cell,
                data[:, col["type"]].astype(int),
                data[:, [col["x"], col["y"], col["z"]]] - origin,
            )


def model_devi_row(forces_comm, stress_comm, volume, natoms):
    """One DeePMD-format row (without the step) from committee outputs."""
    f_devi = np.linalg.norm(np.std(forces_comm, axis=0), axis=-1)  # (natoms,)
    if stress_comm is not None:
        virial = -np.asarray(stress_comm).reshape(len(forces_comm), 9) * volume / natoms
        v_devi = np.std(virial, axis=0)  # (9,)
        v_row = [np.max(v_devi), np.min(v_devi), np.linalg.norm(v_devi) / 3.0]
    else:
        v_row = [0.0, 0.0, 0.0]
    return v_row + [np.max(f_devi), np.min(f_devi), np.mean(f_devi)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lammps-in", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", default="float64")
    args = parser.parse_args(argv)

    from ase import Atoms
    from mace.calculators import MACECalculator

    if len(args.models) < 2:
        print("ERROR: need at least 2 committee models.", file=sys.stderr)
        return 1

    elements = read_pair_coeff_elements(args.lammps_in)
    calc = MACECalculator(
        model_paths=args.models, device=args.device, default_dtype=args.default_dtype
    )

    nframes = 0
    with args.out.open("w") as out:
        out.write(
            "#%11s %14s %14s %14s %14s %14s %14s\n"
            % ("step", "max_devi_v", "min_devi_v", "avg_devi_v",
               "max_devi_f", "min_devi_f", "avg_devi_f")
        )
        for step, cell, types, positions in iter_dump_frames(args.dump):
            if types.min() < 1 or types.max() > len(elements):
                raise ValueError(
                    f"Step {step}: atom types {types.min()}..{types.max()} "
                    f"but pair_coeff names {len(elements)} elements."
                )
            atoms = Atoms(
                symbols=[elements[t - 1] for t in types],
                positions=positions,
                cell=cell,
                pbc=True,
            )
            atoms.calc = calc
            atoms.get_forces()
            row = model_devi_row(
                calc.results["forces_comm"],
                calc.results.get("stress_comm"),
                atoms.get_volume(),
                len(atoms),
            )
            out.write("%12d" % step + "".join(" %14.6e" % v for v in row) + "\n")
            out.flush()
            nframes += 1

    print(f"Wrote {nframes} frames to {args.out}")
    return 0 if nframes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
