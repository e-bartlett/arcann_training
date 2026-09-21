#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/evaluate_centroid.py -- the tested
# copy lives there; this is the copy ArcaNN init places in
# $WORK_DIR/user_files/ so training/check.py can print a held-out centroid
# RMSE. Edit the dataset_prep copy, then re-copy here.
"""Evaluate a trained CentroidMACE checkpoint (Phase 3 item 4, milestone 1).

Chunk 4 of the centroid head (see `CENTROID_HEAD_PLAN.md`). Mirrors
`../eval_dipole_error.py`'s inference pattern (load the model with
`torch.load`, build `AtomicData` via `mace.data.config_from_atoms`/
`AtomicData.from_config` directly rather than `mace_eval_configs`, which
never surfaces a dipole/centroid-shaped output) but reports periodic
(minimum-imaged) distance error against the absolute `REF_centroid` label
instead of a residual, using `loss.minimum_image` -- the same convention
`train_centroid.py`'s own loss is computed with, not a plain Euclidean
difference (wrong near a periodic boundary).

Three checks, per the plan's "Verification / eval plan":

1. Primary: periodic RMSE/MAE/median/max on the held-out test grain.
2. Mandatory baseline: predict the training-set mean absolute centroid
   position (fixed, same prediction for every test config) -- milestone 1
   must clear this by a large margin, not a borderline win.
3. Secondary, non-training sanity check: compare predictions against
   `wannier_electron.csv` (never a training target). Not a pass/fail gate --
   Phase 1 already found centroid/Wannier agree only to ~0.03-0.12 A
   depending on grid resolution, a hard floor regardless of model quality.

Examples
--------
    python evaluate_centroid.py --model mace_electron_centroid_work/he_centroid.model \\
        --train-file mace_training_set_centroid/train.xyz \\
        --test-file mace_training_set_centroid/test.xyz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ase.io
import mace  # noqa: F401  -- import first, see model.py/train_centroid.py
import numpy as np
import torch
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

from loss import minimum_image

CENTROID_KEY = "REF_centroid"


def run_inference(model, atoms_list, device, batch_size):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = [data.config_from_atoms(atoms, head_name="Default") for atoms in atoms_list]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(c, z_table=z_table, cutoff=float(model.r_max), heads=None)
                 for c in configs],
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    centroids = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.to_dict())
        centroids.append(torch_tools.to_numpy(out["centroid"]))
    pred = np.concatenate(centroids, axis=0)
    assert len(pred) == len(atoms_list)
    return pred


def periodic_error(pred, true, cell_lengths):
    delta = pred - true
    delta = delta - cell_lengths * np.round(delta / cell_lengths)
    return np.linalg.norm(delta, axis=1)


def report(label, error):
    print(f"  {label} (n={len(error)}): RMSE={np.sqrt((error ** 2).mean()):.4f} A  "
          f"MAE={error.mean():.4f} A  median={np.median(error):.4f} A  max={error.max():.4f} A")


def cell_lengths_of(atoms_list):
    return np.array([np.diagonal(a.cell.array) for a in atoms_list])


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, default=here / "mace_training_set_centroid" / "train.xyz",
                         help="for the mean-position baseline")
    parser.add_argument("--test-file", type=Path, default=here / "mace_training_set_centroid" / "test.xyz")
    parser.add_argument("--wannier-csv", type=Path, default=here.parent / "wannier_electron.csv")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    options = parser.parse_args(argv)

    torch_tools.set_default_dtype("float64")
    device = torch_tools.init_device(options.device)
    model = torch.load(f=options.model, map_location=device)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_atoms = ase.io.read(options.train_file, index=":")
    test_atoms = ase.io.read(options.test_file, index=":")
    test_true = np.array([a.info[CENTROID_KEY] for a in test_atoms])
    test_cell = cell_lengths_of(test_atoms)

    print(f"model: {options.model}")
    print(f"test set: {options.test_file} (n={len(test_atoms)})")

    # 1. Primary metric.
    pred = run_inference(model, test_atoms, device, options.batch_size)
    report("model", periodic_error(pred, test_true, test_cell))

    # 2. Mandatory baseline: training-set mean absolute centroid position,
    # the same fixed prediction for every test config.
    train_true = np.array([a.info[CENTROID_KEY] for a in train_atoms])
    baseline_pred = np.tile(train_true.mean(axis=0), (len(test_atoms), 1))
    baseline_error = periodic_error(baseline_pred, test_true, test_cell)
    report("baseline (train-mean position)", baseline_error)
    model_rmse = np.sqrt((periodic_error(pred, test_true, test_cell) ** 2).mean())
    baseline_rmse = np.sqrt((baseline_error ** 2).mean())
    print(f"  model/baseline RMSE ratio: {model_rmse / baseline_rmse:.3f} "
          f"(target: < 0.5, i.e. model RMSE under half the baseline's)")

    # 3. Secondary, non-training sanity check against Wannier centers.
    if options.wannier_csv.is_file():
        wannier = {}
        with options.wannier_csv.open() as fh:
            for row in csv.DictReader(fh):
                wannier[row["config"]] = np.array(
                    [float(row["ex_x_A"]), float(row["ex_y_A"]), float(row["ex_z_A"])]
                )
        # extxyz's info parser reads a zero-padded numeric-looking string
        # ("00604") back as a plain int (604), not a string -- confirmed
        # live against mace_training_set_centroid/test.xyz. Re-pad to match
        # wannier_electron.csv's "config" column (zero-padded, e.g. "00000").
        configs = [f"{int(a.info['config']):05d}" for a in test_atoms]
        have = np.array([c in wannier for c in configs])
        if have.any():
            wannier_true = np.array([wannier.get(c, np.zeros(3)) for c in configs])
            wannier_error = periodic_error(pred[have], wannier_true[have], test_cell[have])
            report("model vs. Wannier (not a training target)", wannier_error)
        else:
            print("  no test configs found in wannier_electron.csv -- skipping")
    else:
        print(f"  {options.wannier_csv} not found -- skipping Wannier sanity check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
