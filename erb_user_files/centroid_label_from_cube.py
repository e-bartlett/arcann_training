#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/centroid_label_from_cube.py -- the
# tested copy lives there; this is the copy ArcaNN init places in
# $WORK_DIR/user_files/, run by labeling/extract.py's MACE branch. Edit the
# dataset_prep copy, then re-copy here, dropping the parent-dir sys.path
# insert (analyze_dataset.py is staged alongside this file in user_files/).
"""Extract the spin-density-centroid label for newly labelled configs.

Chunk 5 of ARCANN_TANDEM_PLAN.md, step A8b: run this between ArcaNN's
``labeling extract`` and ``training prepare``. For each labelled config it
reads the stage-2 ``2_labeling_<config>-<stride>-SPIN_DENSITY-1_0.cube``
(emitted by the ``&E_DENSITY_CUBE`` block added to
``erb_user_files/2_he_labeling_XXXXX_*.inp`` -- LSD makes CP2K write a
SPIN_DENSITY cube alongside the ELECTRON_DENSITY one) and appends one row to
a running CSV with the periodic spin-density centroid -- the training target
for the per-iteration ``CentroidMACE`` retrain.

All the cube parsing and the periodic circular-mean centroid are reused
verbatim from ``../analyze_dataset.py`` (``read_cube`` / ``select_grains`` /
``analyze_density`` -> ``circular_centroid``); nothing is reimplemented. The
CSV columns are a subset of ``spin_analysis.csv``'s, same names and order, so
anything that reads that file reads this one.

Examples
--------
    # one or more config dirs, each holding the cube:
    python centroid_label_from_cube.py $WORK_DIR/1-labeling/he/00001 \\
        $WORK_DIR/1-labeling/he/00002 --out centroid_labels.csv

    # or point at the labeling dir and let it find the NNNNN/ subdirs:
    python centroid_label_from_cube.py --labeling-root $WORK_DIR/1-labeling/he
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from analyze_dataset import (  # noqa: E402
    analyze_density,
    read_cube,
    select_grains,
)

# A subset of analyze_dataset.FIELDS, same names/order -> spin_analysis.csv
# readers work unchanged. `config` is the label to_extxyz_centroid.py keys on.
FIELDS = [
    "config", "stride",
    "cell_a_A", "cell_b_A", "cell_c_A",
    "cx_A", "cy_A", "cz_A",
    "spin_integral", "neg_frac", "conc_x", "conc_y", "conc_z",
    "cube",
]


def centroid_row(config_dir: Path, stride: int, negative: str):
    """One CSV row for a config dir, or ``(None, warning)`` if no usable cube."""
    selected, warnings = select_grains(config_dir, {stride})
    if stride not in selected:
        return None, f"{config_dir.name}: no stride-{stride} SPIN_DENSITY cube"
    cube_path = selected[stride]
    cube = read_cube(cube_path)
    density = analyze_density(cube, negative)
    centroid = density["centroid"]
    if not np.all(np.isfinite(centroid)):
        return None, f"{config_dir.name}: no positive spin density, centroid undefined"
    cell = density["cell"]
    conc = density["concentration"]
    row = {
        "config": config_dir.name,
        "stride": stride,
        "cell_a_A": cell[0], "cell_b_A": cell[1], "cell_c_A": cell[2],
        "cx_A": centroid[0], "cy_A": centroid[1], "cz_A": centroid[2],
        "spin_integral": density["spin_integral"],
        "neg_frac": density["neg_frac"],
        "conc_x": conc[0], "conc_y": conc[1], "conc_z": conc[2],
        "cube": cube_path.name,
    }
    return row, None


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config_dirs", nargs="*", type=Path,
                        help="config directories, each holding the SPIN_DENSITY cube")
    parser.add_argument("--labeling-root", type=Path,
                        help="parent dir; its NNNNN/ subdirs are used as config_dirs")
    parser.add_argument("--out", type=Path, default=here / "centroid_labels.csv")
    parser.add_argument("--stride", type=int, default=2,
                        help="cube grid stride to read (default: 2, the finest grain)")
    parser.add_argument("--negative", choices=("clip", "abs", "keep"), default="clip",
                        help="negative spin-density handling (default: clip, "
                             "matching spin_analysis.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute configs already in --out (default: skip them)")
    options = parser.parse_args(argv)

    config_dirs = list(options.config_dirs)
    if options.labeling_root is not None:
        config_dirs += sorted(
            p for p in options.labeling_root.iterdir()
            if p.is_dir() and p.name[:1].isdigit()
        )
    if not config_dirs:
        parser.error("no config dirs given (positional args or --labeling-root)")

    existing = {}
    if options.out.is_file():
        with options.out.open() as fh:
            for row in csv.DictReader(fh):
                existing[row["config"]] = row

    warnings = []
    added = 0
    for config_dir in config_dirs:
        name = config_dir.name
        if name in existing and not options.overwrite:
            continue
        row, warning = centroid_row(config_dir, options.stride, options.negative)
        if warning:
            warnings.append(warning)
            continue
        existing[name] = {k: row[k] for k in FIELDS}
        added += 1

    rows = [existing[k] for k in sorted(existing)]
    with options.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"{added} new row(s); {options.out} now has {len(rows)} config(s)")
    return 1 if warnings and added == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
