#!/usr/bin/env python3
# MIRROR of dataset_prep/analyze_dataset.py -- the tested copy lives there;
# this is the copy ArcaNN init places in $WORK_DIR/user_files/ so
# centroid_label_from_cube.py (run by labeling/extract.py) can reuse
# read_cube / select_grains / analyze_density. Edit the dataset_prep copy,
# then re-copy here.
"""Analyze hydrated-electron spin-density cubes.

For every configuration under ``dataset_prep/he/NNNNN`` and every grid grain
(CP2K ``STRIDE``) of the spin-density cube, this computes:

1. **Electron centroid** -- the spin-density-weighted centre of charge, using
   the periodic circular mean.  Each Cartesian axis is mapped onto an angle
   ``theta = 2*pi*x/L``, the complex phases ``exp(i*theta)`` are averaged with
   the density as weight, and the mean angle is mapped back to a coordinate.
   This is wrap-safe: an electron straddling the periodic boundary gives the
   right answer, whereas a naive mean of wrapped voxel coordinates would place
   it in the middle of the box.  The *spin* density is used (not the total
   electron density) so the closed-shell water contribution cancels and only
   the unpaired electron is weighted.

2. **Coordination number** -- how many O-H bonds point at the electron in the
   first solvation shell.  Two standard definitions are reported:

   * ``cn_oh``  : O within ``--ro-cut`` of the centroid *and* the angle between
     the O->H bond vector and the O->electron vector below ``--angle-cut``.
   * ``cn_eh``  : H within ``--rh-cut`` of the centroid (first minimum of the
     e-H radial distribution function).

3. **Closest atom** -- the distance from the centroid to the nearest nucleus
   of any element (``d_min_atom_A``), with its element and atom index, taken
   under the minimum-image convention.  For a cavity-bound electron this is
   the depth of the cavity it sits in.

4. **Radius of gyration** -- the spin-density-weighted RMS displacement from
   the centroid, with displacements taken under the minimum-image convention
   so that periodicity is handled consistently with the centroid.  The full
   gyration tensor is diagonalised to give the principal radii and the
   asphericity.

Geometry is taken from the cube header itself (identical to ``he_NNNNN.xyz``
but guaranteed consistent with the grid); ``--check-geometry`` verifies that
against the ``.xyz`` file.

The cell is derived from the cube header as ``npoints * grid_vector`` and the
code assumes an orthorhombic cell (true here: 12.43 A cubic).

Note on the dataset: each configuration carries duplicate cubes under two
extra file names -- an untagged ``2_labeling_NNNNN-SPIN_DENSITY-1_0.cube``
(same 48^3 grid as stride 4) and a malformed
``...-2_labeling_NNNNN-SPIN_DENSITY-1_0.cube_6-...`` (same 32^3 grid as
stride 6).  Grains are de-duplicated by grid size, preferring the file whose
name carries an explicit stride tag.

Examples
--------
    # everything, all grains, parallel over configurations
    python analyze_dataset.py --out spin_analysis.csv

    # quick look: coarse grains only, first 10 configurations
    python analyze_dataset.py --strides 8 12 --limit 10 --out /tmp/test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BOHR_TO_ANG = 0.529177210903

# 2_labeling_00113-6-SPIN_DENSITY-1_0.cube  ->  config 00113, stride 6
# 2_labeling_00113-SPIN_DENSITY-1_0.cube    ->  config 00113, stride unknown
CUBE_RE = re.compile(
    r"^2_labeling_(?P<config>\d+)-(?:(?P<stride>\d+)-)?SPIN_DENSITY-1_0\.cube$"
)

# Cutoffs are literature values for the hydrated electron and are all
# overridable on the command line.
DEFAULT_RO_CUT = 3.5     # A, first minimum of the e-O RDF
DEFAULT_RH_CUT = 2.6     # A, first minimum of the e-H RDF
DEFAULT_ANGLE_CUT = 45.0  # deg, O-H bond vs. O-electron vector
MAX_OH_BOND = 1.3        # A, sanity check on H -> O assignment

ELEMENTS = {1: "H", 8: "O"}

FIELDS = [
    "config", "stride", "nx", "ny", "nz", "dx_A", "dy_A", "dz_A",
    "cell_a_A", "cell_b_A", "cell_c_A",
    "spin_integral", "neg_frac", "conc_x", "conc_y", "conc_z",
    "cx_A", "cy_A", "cz_A",
    "rg_A", "rg_x_A", "rg_y_A", "rg_z_A",
    "rg_1_A", "rg_2_A", "rg_3_A", "asphericity",
    "cn_oh", "cn_eh", "n_water_shell", "d_min_eO_A", "d_min_eH_A",
    "d_min_atom_A", "closest_atom", "closest_atom_index",
    "cube",
]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

class Cube:
    """A parsed Gaussian cube file (all quantities in atomic units)."""

    __slots__ = ("origin", "npoints", "vectors", "numbers", "coords", "data")

    def __init__(self, origin, npoints, vectors, numbers, coords, data):
        self.origin = origin      # (3,) bohr
        self.npoints = npoints    # (3,) int
        self.vectors = vectors    # (3, 3) bohr, row i = step along axis i
        self.numbers = numbers    # (natoms,) int
        self.coords = coords      # (natoms, 3) bohr
        self.data = data          # (nx, ny, nz)

    @property
    def cell(self):
        """Orthorhombic cell lengths in bohr."""
        return self.npoints * np.diag(self.vectors)

    @property
    def voxel_volume(self):
        return abs(float(np.linalg.det(self.vectors)))


def read_cube(path):
    """Parse a cube file.  Returns a :class:`Cube`, coordinates in bohr."""
    with open(path) as fh:
        fh.readline()  # comment line 1
        fh.readline()  # comment line 2

        fields = fh.readline().split()
        natoms = int(fields[0])
        origin = np.array(fields[1:4], dtype=np.float64)

        npoints = np.empty(3, dtype=np.int64)
        vectors = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            fields = fh.readline().split()
            npoints[i] = int(fields[0])
            vectors[i] = np.array(fields[1:4], dtype=np.float64)

        # A negative atom count marks an orbital cube: the atom block is
        # followed by an extra line listing the orbital indices.
        is_mo = natoms < 0
        natoms = abs(natoms)

        numbers = np.empty(natoms, dtype=np.int64)
        coords = np.empty((natoms, 3), dtype=np.float64)
        for i in range(natoms):
            fields = fh.readline().split()
            numbers[i] = int(fields[0])
            coords[i] = np.array(fields[2:5], dtype=np.float64)

        if is_mo:
            fh.readline()

        ntot = int(np.prod(npoints))
        values = np.asarray(fh.read().split(), dtype=np.float64)

    if values.size < ntot:
        raise ValueError(
            f"{path}: expected {ntot} grid values, found {values.size}"
        )
    # Cube ordering is x outermost, z innermost.
    data = values[:ntot].reshape(tuple(int(n) for n in npoints))

    off_diagonal = vectors - np.diag(np.diag(vectors))
    if np.abs(off_diagonal).max() > 1e-8 * max(np.abs(np.diag(vectors)).max(), 1.0):
        raise ValueError(f"{path}: non-orthorhombic grid is not supported")

    return Cube(origin, npoints, vectors, numbers, coords, data)


def read_xyz(path):
    """Read a single-frame xyz file.  Returns (symbols, coords in angstrom)."""
    with open(path) as fh:
        natoms = int(fh.readline().split()[0])
        fh.readline()  # comment
        symbols = []
        coords = np.empty((natoms, 3), dtype=np.float64)
        for i in range(natoms):
            fields = fh.readline().split()
            symbols.append(fields[0])
            coords[i] = np.array(fields[1:4], dtype=np.float64)
    return symbols, coords


# --------------------------------------------------------------------------
# Periodic helpers
# --------------------------------------------------------------------------

def minimum_image(delta, cell):
    """Wrap displacements into [-L/2, L/2) for an orthorhombic cell."""
    return delta - cell * np.round(delta / cell)


def circular_centroid(weights_1d, coords_1d, length):
    """Circular mean of ``coords_1d`` weighted by ``weights_1d``.

    Each coordinate is mapped to an angle on the periodic axis, the weighted
    mean of the unit phasors is taken, and its argument is mapped back to a
    coordinate in ``[0, length)``.  Also returns the mean resultant length
    ``R`` in [0, 1]: R -> 1 for a sharply localised density, R -> 0 for one
    spread uniformly around the axis (for which no centroid is meaningful).
    """
    total = weights_1d.sum()
    if total <= 0:
        return float("nan"), 0.0
    theta = coords_1d * (2.0 * np.pi / length)
    c = float(np.dot(weights_1d, np.cos(theta)))
    s = float(np.dot(weights_1d, np.sin(theta)))
    mean_angle = np.arctan2(s, c) % (2.0 * np.pi)
    resultant = np.hypot(c, s) / total
    return mean_angle * length / (2.0 * np.pi), float(resultant)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def spin_weights(data, negative):
    """Turn raw spin density into non-negative weights.

    The circular mean and the gyration tensor are only meaningful for
    non-negative weights.  Spin density is positive where the unpaired
    electron lives; small negative lobes come from spin polarisation of the
    solvent and from grid noise.
    """
    if negative == "clip":
        weights = np.clip(data, 0.0, None)
    elif negative == "abs":
        weights = np.abs(data)
    elif negative == "keep":
        weights = data
    else:
        raise ValueError(f"unknown negative handling: {negative}")
    return weights


def analyze_density(cube, negative):
    """Centroid, gyration tensor and QC diagnostics for one cube.

    All returned lengths are in angstrom.
    """
    data = cube.data
    weights = spin_weights(data, negative)

    positive = float(data[data > 0].sum())
    negative_sum = float(-data[data < 0].sum())
    neg_frac = negative_sum / positive if positive > 0 else float("nan")
    spin_integral = float(data.sum()) * cube.voxel_volume

    cell = cube.cell * BOHR_TO_ANG
    steps = np.diag(cube.vectors) * BOHR_TO_ANG
    origin = cube.origin * BOHR_TO_ANG

    # 1-D marginals: summing over the other two axes is exact for the
    # centroid and for the diagonal of the gyration tensor, because a voxel's
    # displacement along one axis depends only on its index along that axis.
    marginals = [
        weights.sum(axis=(1, 2)),
        weights.sum(axis=(0, 2)),
        weights.sum(axis=(0, 1)),
    ]
    total = float(weights.sum())

    centroid = np.empty(3)
    concentration = np.empty(3)
    deltas = []
    for axis in range(3):
        grid = origin[axis] + steps[axis] * np.arange(cube.npoints[axis])
        centroid[axis], concentration[axis] = circular_centroid(
            marginals[axis], grid, cell[axis]
        )
        deltas.append(minimum_image(grid - centroid[axis], cell[axis]))

    # Gyration tensor.  The diagonal follows from the 1-D marginals and the
    # off-diagonal from the three 2-D marginals, because a voxel's
    # displacement along one axis depends only on its index along that axis --
    # far cheaper than forming per-voxel coordinate arrays.
    if total > 0:
        tensor = np.zeros((3, 3))
        for axis in range(3):
            tensor[axis, axis] = float(
                np.dot(marginals[axis], deltas[axis] ** 2)
            ) / total
        # (a, b, axis summed over) -- the reduced array keeps axis order (a, b)
        for a, b, summed in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            plane = weights.sum(axis=summed)
            value = float(deltas[a] @ plane @ deltas[b]) / total
            tensor[a, b] = tensor[b, a] = value

        eigenvalues = np.sort(np.linalg.eigvalsh(tensor))  # ascending, A^2
        trace = float(eigenvalues.sum())
        pair_sum = float(
            eigenvalues[0] * eigenvalues[1]
            + eigenvalues[1] * eigenvalues[2]
            + eigenvalues[2] * eigenvalues[0]
        )
        asphericity = 1.0 - 3.0 * pair_sum / trace**2 if trace > 0 else float("nan")
        rg = float(np.sqrt(trace)) if trace >= 0 else float("nan")
        rg_axes = np.sqrt(np.clip(np.diag(tensor), 0.0, None))
        rg_principal = np.sqrt(np.clip(eigenvalues, 0.0, None))
    else:
        # No positive spin density anywhere: nothing is well defined.
        asphericity = rg = float("nan")
        rg_axes = np.full(3, np.nan)
        rg_principal = np.full(3, np.nan)

    return {
        "centroid": centroid,
        "cell": cell,
        "steps": steps,
        "spin_integral": spin_integral,
        "neg_frac": neg_frac,
        "concentration": concentration,
        "rg": rg,
        "rg_axes": rg_axes,
        "rg_principal": rg_principal,
        "asphericity": asphericity,
    }


def assign_waters(numbers, coords, cell):
    """Assign every H to its nearest O.

    Returns ``(oxygen_index, hydrogen_index, pair_o_for_h, max_bond)`` where
    ``pair_o_for_h[k]`` is the index into ``oxygen_index`` of the O bonded to
    ``hydrogen_index[k]``.
    """
    oxygen_index = np.flatnonzero(numbers == 8)
    hydrogen_index = np.flatnonzero(numbers == 1)
    if oxygen_index.size == 0 or hydrogen_index.size == 0:
        return oxygen_index, hydrogen_index, np.empty(0, dtype=np.int64), 0.0

    delta = coords[hydrogen_index][:, None, :] - coords[oxygen_index][None, :, :]
    delta = minimum_image(delta, cell)
    distance = np.linalg.norm(delta, axis=2)
    nearest = np.argmin(distance, axis=1)
    max_bond = float(distance[np.arange(hydrogen_index.size), nearest].max())
    return oxygen_index, hydrogen_index, nearest, max_bond


def coordination(numbers, coords, cell, electron, ro_cut, rh_cut, angle_cut):
    """Count O-H bonds pointing at the electron in the first solvation shell."""
    oxygen_index, hydrogen_index, pair, max_bond = assign_waters(
        numbers, coords, cell
    )

    o_coords = coords[oxygen_index]
    h_coords = coords[hydrogen_index]

    d_eo = np.linalg.norm(minimum_image(o_coords - electron, cell), axis=1)
    d_eh = np.linalg.norm(minimum_image(h_coords - electron, cell), axis=1)

    # Angle between the O->H bond vector and the O->electron vector.
    oh = minimum_image(h_coords - o_coords[pair], cell)
    oe = minimum_image(electron - o_coords[pair], cell)
    norms = np.linalg.norm(oh, axis=1) * np.linalg.norm(oe, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.einsum("ij,ij->i", oh, oe) / norms
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    in_shell = d_eo[pair] <= ro_cut
    cn_oh = int(np.count_nonzero(in_shell & (angle <= angle_cut)))
    cn_eh = int(np.count_nonzero(d_eh <= rh_cut))

    # Closest atom of any element -- how far the electron sits from the nearest
    # nucleus, i.e. how deep the cavity it occupies is.
    d_all = np.linalg.norm(minimum_image(coords - electron, cell), axis=1)
    nearest_atom = int(np.argmin(d_all)) if d_all.size else -1

    return {
        "cn_oh": cn_oh,
        "cn_eh": cn_eh,
        "n_water_shell": int(np.count_nonzero(d_eo <= ro_cut)),
        "d_min_eO_A": float(d_eo.min()) if d_eo.size else float("nan"),
        "d_min_eH_A": float(d_eh.min()) if d_eh.size else float("nan"),
        "d_min_atom_A": float(d_all[nearest_atom]) if d_all.size else float("nan"),
        "closest_atom": (
            ELEMENTS.get(int(numbers[nearest_atom]), f"Z{numbers[nearest_atom]}")
            if nearest_atom >= 0 else ""
        ),
        "closest_atom_index": nearest_atom,
        "max_oh_bond": max_bond,
    }


# --------------------------------------------------------------------------
# Driving one configuration
# --------------------------------------------------------------------------

def cube_grid_size(path):
    """Number of grid points along x, read from the cube header alone."""
    with open(path) as fh:
        for _ in range(3):
            fh.readline()
        return int(fh.readline().split()[0])


def find_cubes(directory):
    """Spin-density cubes in one configuration directory.

    Returns ``(tagged, untagged)`` where ``tagged`` maps an explicit
    file-name stride to its path and ``untagged`` lists the cubes whose name
    carries no stride.
    """
    tagged, untagged = {}, []
    for path in sorted(directory.iterdir()):
        match = CUBE_RE.match(path.name)
        if match is None:
            continue  # e.g. the malformed duplicate names, and ELECTRON_DENSITY
        stride = match.group("stride")
        if stride is None:
            untagged.append(path)
        else:
            tagged[int(stride)] = path
    return tagged, untagged


def select_grains(directory, strides):
    """Pick one cube per distinct grid grain.

    Grain identity is the grid size; the file-name stride is only a label.
    Duplicated grids are resolved in favour of the explicitly tagged name, and
    an untagged cube is labelled by inferring the ungridded (STRIDE 1) mesh
    from the tagged files.  Returns ``(selected, warnings)`` where
    ``selected`` maps a stride label (or ``None``) to a path.
    """
    tagged, untagged = find_cubes(directory)
    selected, warnings = {}, []

    tagged_grids, full_grid = set(), 0
    for stride, path in tagged.items():
        try:
            nx = cube_grid_size(path)
        except (OSError, ValueError, IndexError) as exc:
            warnings.append(f"cannot read header of {path.name}: {exc}")
            continue
        tagged_grids.add(nx)
        full_grid = max(full_grid, nx * stride)
        if strides is None or stride in strides:
            selected[stride] = path

    for path in untagged:
        try:
            nx = cube_grid_size(path)
        except (OSError, ValueError, IndexError) as exc:
            warnings.append(f"cannot read header of {path.name}: {exc}")
            continue
        if nx in tagged_grids:
            continue  # same grain as a tagged cube
        stride = full_grid // nx if full_grid and nx and full_grid % nx == 0 else None
        if strides is not None and stride not in strides:
            continue
        if stride is None or stride not in selected:
            selected[stride] = path

    return selected, warnings


def analyze_config(directory, options):
    """Analyze every grain of one configuration.  Returns (rows, warnings)."""
    directory = Path(directory)
    config = directory.name
    rows = []

    selected, warnings = select_grains(directory, options.strides)
    warnings = [f"{config}: {w}" for w in warnings]
    if not selected:
        warnings.append(f"{config}: no spin-density cube found")
        return rows, warnings

    reference_geometry = None
    if options.check_geometry:
        xyz_path = directory / f"he_{config}.xyz"
        if xyz_path.is_file():
            _, reference_geometry = read_xyz(xyz_path)
        else:
            warnings.append(f"{config}: {xyz_path.name} not found for geometry check")

    for stride in sorted(selected, key=lambda s: (s is None, s)):
        path = selected[stride]
        try:
            cube = read_cube(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"{config}: failed to read {path.name}: {exc}")
            continue

        density = analyze_density(cube, options.negative)
        cell = density["cell"]
        coords = cube.coords * BOHR_TO_ANG

        if reference_geometry is not None:
            if reference_geometry.shape == coords.shape:
                deviation = np.abs(
                    minimum_image(coords - reference_geometry, cell)
                ).max()
                if deviation > 1e-3:
                    warnings.append(
                        f"{config}/{path.name}: cube geometry differs from xyz "
                        f"by up to {deviation:.4f} A"
                    )
            else:
                warnings.append(
                    f"{config}: xyz has {reference_geometry.shape[0]} atoms, "
                    f"cube has {coords.shape[0]}"
                )

        centroid = density["centroid"]
        if not np.all(np.isfinite(centroid)):
            warnings.append(
                f"{config}/{path.name}: no positive spin density, centroid undefined"
            )
            shell = {
                "cn_oh": "", "cn_eh": "", "n_water_shell": "",
                "d_min_eO_A": "", "d_min_eH_A": "", "d_min_atom_A": "",
                "closest_atom": "", "closest_atom_index": "",
                "max_oh_bond": 0.0,
            }
        else:
            shell = coordination(
                cube.numbers, coords, cell, centroid,
                options.ro_cut, options.rh_cut, options.angle_cut,
            )
            if shell["max_oh_bond"] > MAX_OH_BOND:
                warnings.append(
                    f"{config}: longest H-O assignment is "
                    f"{shell['max_oh_bond']:.2f} A -- check the geometry"
                )

        if np.nanmin(density["concentration"]) < options.min_concentration:
            warnings.append(
                f"{config}/{path.name}: spin density is diffuse "
                f"(min phase concentration "
                f"{np.nanmin(density['concentration']):.2f}); "
                "centroid and Rg are poorly defined"
            )

        rows.append({
            "config": config,
            "stride": "" if stride is None else stride,
            "nx": int(cube.npoints[0]),
            "ny": int(cube.npoints[1]),
            "nz": int(cube.npoints[2]),
            "dx_A": density["steps"][0],
            "dy_A": density["steps"][1],
            "dz_A": density["steps"][2],
            "cell_a_A": cell[0],
            "cell_b_A": cell[1],
            "cell_c_A": cell[2],
            "spin_integral": density["spin_integral"],
            "neg_frac": density["neg_frac"],
            "conc_x": density["concentration"][0],
            "conc_y": density["concentration"][1],
            "conc_z": density["concentration"][2],
            "cx_A": centroid[0],
            "cy_A": centroid[1],
            "cz_A": centroid[2],
            "rg_A": density["rg"],
            "rg_x_A": density["rg_axes"][0],
            "rg_y_A": density["rg_axes"][1],
            "rg_z_A": density["rg_axes"][2],
            "rg_1_A": density["rg_principal"][0],
            "rg_2_A": density["rg_principal"][1],
            "rg_3_A": density["rg_principal"][2],
            "asphericity": density["asphericity"],
            "cn_oh": shell["cn_oh"],
            "cn_eh": shell["cn_eh"],
            "n_water_shell": shell["n_water_shell"],
            "d_min_eO_A": shell["d_min_eO_A"],
            "d_min_eH_A": shell["d_min_eH_A"],
            "d_min_atom_A": shell["d_min_atom_A"],
            "closest_atom": shell["closest_atom"],
            "closest_atom_index": shell["closest_atom_index"],
            "cube": path.name,
        })

    return rows, warnings


def _worker(args):
    directory, options = args
    try:
        return analyze_config(directory, options)
    except Exception as exc:  # keep one bad configuration from killing the run
        return [], [f"{Path(directory).name}: unhandled error: {exc!r}"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def format_row(row):
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = "" if not np.isfinite(value) else f"{value:.6f}"
        else:
            formatted[key] = value
    return formatted


def summarize(rows, stream):
    """Print per-grain means so a run can be eyeballed for convergence."""
    if not rows:
        return
    by_stride = {}
    for row in rows:
        by_stride.setdefault(row["stride"], []).append(row)

    print("\ngrain    n     <Rg>/A   <CN_OH>   <CN_eH>   <N_shell>  "
          "<d_min>/A  <spin int>", file=stream)
    for stride in sorted(by_stride, key=lambda s: (s == "", s)):
        group = by_stride[stride]

        def mean(key):
            values = [
                float(r[key]) for r in group
                if r[key] != "" and np.isfinite(float(r[key]))
            ]
            return float(np.mean(values)) if values else float("nan")

        label = f"stride {stride}" if stride != "" else "untagged"
        print(
            f"{label:<8} {len(group):<5} {mean('rg_A'):7.3f}   "
            f"{mean('cn_oh'):6.2f}    {mean('cn_eh'):6.2f}    "
            f"{mean('n_water_shell'):6.2f}     {mean('d_min_atom_A'):6.3f}   "
            f"{mean('spin_integral'):.4f}",
            file=stream,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent / "he",
        help="directory holding the per-configuration folders (default: ./he)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="CSV output path (default: <root>/../spin_analysis.csv)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="only these configuration ids (e.g. 00113 00114)",
    )
    parser.add_argument(
        "--strides", nargs="+", type=int, default=None,
        help="only these cube strides (e.g. 4 8 12)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many configurations (for quick tests)",
    )
    parser.add_argument(
        "--ro-cut", type=float, default=DEFAULT_RO_CUT,
        help="first-shell e-O cutoff in A (default: %(default)s)",
    )
    parser.add_argument(
        "--rh-cut", type=float, default=DEFAULT_RH_CUT,
        help="e-H cutoff in A for cn_eh (default: %(default)s)",
    )
    parser.add_argument(
        "--angle-cut", type=float, default=DEFAULT_ANGLE_CUT,
        help="max O-H / O-electron angle in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--negative", choices=("clip", "abs", "keep"), default="clip",
        help="how to weight negative spin density (default: %(default)s)",
    )
    parser.add_argument(
        "--min-concentration", type=float, default=0.3,
        help="warn when the circular-mean resultant falls below this "
             "(density too diffuse for a meaningful centroid; default: %(default)s)",
    )
    parser.add_argument(
        "--check-geometry", action="store_true",
        help="verify cube header atoms against he_NNNNN.xyz",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1),
        help="parallel worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress progress and warnings on stderr",
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)

    if not options.root.is_dir():
        sys.exit(f"error: --root {options.root} is not a directory")

    directories = sorted(p for p in options.root.iterdir() if p.is_dir())
    if options.configs:
        wanted = set(options.configs)
        directories = [d for d in directories if d.name in wanted]
        missing = wanted - {d.name for d in directories}
        if missing:
            sys.exit(f"error: no such configuration(s): {', '.join(sorted(missing))}")
    if options.limit is not None:
        directories = directories[: options.limit]
    if not directories:
        sys.exit(f"error: no configuration directories under {options.root}")

    out_path = options.out or options.root.parent / "spin_analysis.csv"
    options.strides = set(options.strides) if options.strides else None

    rows, warnings = [], []
    tasks = [(str(d), options) for d in directories]
    done = 0

    if options.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=options.jobs) as pool:
            futures = [pool.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                config_rows, config_warnings = future.result()
                rows.extend(config_rows)
                warnings.extend(config_warnings)
                done += 1
                if not options.quiet and done % 25 == 0:
                    print(f"  {done}/{len(tasks)} configurations",
                          file=sys.stderr, flush=True)
    else:
        for task in tasks:
            config_rows, config_warnings = _worker(task)
            rows.extend(config_rows)
            warnings.extend(config_warnings)
            done += 1
            if not options.quiet and done % 25 == 0:
                print(f"  {done}/{len(tasks)} configurations",
                      file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (r["config"], r["nx"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))

    if not options.quiet:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"\nwrote {len(rows)} rows for {len(directories)} configurations "
              f"to {out_path}", file=sys.stderr)
        summarize(rows, sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
