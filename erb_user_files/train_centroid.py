#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/train_centroid.py -- the tested copy
# lives there; this is the copy ArcaNN init places in $WORK_DIR/user_files/,
# run by the staged job_mace_centroid_<arch>_<machine>.sh. Edit the
# dataset_prep copy, then re-copy here.
"""Standalone training loop for CentroidMACE (Phase 3 item 4, milestone 1).

Chunk 3 of the centroid head (see `CENTROID_HEAD_PLAN.md`). Deliberately
hand-written, not `mace.tools.train.train()`: that function's `evaluate()`
hardcodes a `torchmetrics.Metric` subclass whose tracked fields
(`delta_es`, `rmse_f`, ...) are energy/force-shaped, and adapting it for a
centroid-only model is more code than this ~150-line loop. Data loading
still reuses `mace.data`/`mace.modules` building blocks and
`mace.tools.scatter`/`torch_geometric.dataloader.DataLoader` (the same
pattern `eval_dipole_error.py` already uses) plus
`mace.modules.compute_avg_num_neighbors` for the one required data
statistic.

Label plumbing: `mace.data.config_from_atoms(atoms, head_name="Default")`
(called with no `key_specification`, same as `eval_dipole_error.py`) reads
*none* of `atoms.info` into `config.properties` -- an empty
`KeySpecification()` has no `info_keys`/`arrays_keys` mappings at all.
`AtomicData.from_config`'s "pass through any extra properties" loop can
never see `REF_centroid`, so there's no risk of its generic per-atom-shape
inference (`(n,) -> (n, 1)`, meant for real per-atom scalars) silently
misinterpreting a 3-vector graph-level label. Instead, `REF_centroid` is
read directly off each `ase.Atoms.info` (like `eval_dipole_error.py`
already does for `REF_dipole`) and attached to the `AtomicData` object by
hand as `true_centroid`, shape `(1, 3)` -- the same shape MACE's own
`dipole` field uses internally. Attached this way, it's a completely
ordinary attribute of the `AtomicData`/PyG `Data` object, so PyTorch
Geometric's default collate batches it (concatenated along dim 0, same as
every other per-graph field) automatically and correctly under `shuffle`,
with no separate index bookkeeping needed to keep labels aligned with
shuffled batches.

Per-epoch train/valid RMSE, MSE, LR, and wall time are also rewritten every
epoch to `{work_dir}/{name}_history.json` (not just printed) -- lets
`plot_centroid_training.py` plot a still-running job's progress and survives
a walltime kill mid-run, same incremental-checkpointing reasoning
`check_mace_gpu.py` already uses for its own report.

Checkpoint as plain `torch.save(model, path)`, matching the convention
`eval_dipole_error.py` already reads via `torch.load`. Pickling this way
records the class by module reference (`model.CentroidMACE`, since this
file is run/imported as the bare top-level module `model`, not a
`dataset_prep.mace_electron.model` package path) -- whatever later loads
the checkpoint (`evaluate_centroid.py`, the chunk-5 diagnostics scripts)
must have `dataset_prep/mace_electron/` on `sys.path` for `torch.load` to
resolve it, same as every other script in this directory already assumes
by living there. Verified this round-trips correctly for the smoke-test
checkpoint below.

Examples
--------
    python train_centroid.py --device cpu --max-num-epochs 5 \\
        --train-file mace_training_set_centroid/train.xyz \\
        --valid-file mace_training_set_centroid/valid.xyz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ase.io
import mace  # noqa: F401  -- import first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
             # before e3nn.o3 loads its constants.pt, or torch>=2.6 rejects it
import torch
from e3nn import o3
from mace import data
from mace.modules import compute_avg_num_neighbors, interaction_classes
from mace.tools import torch_geometric, torch_tools, utils

from loss import periodic_squared_error
from model import CentroidMACE

CENTROID_KEY = "REF_centroid"

# Backbone hyperparameters -- reused exactly from what Phase 3 already
# profiled for this dataset (mace_gpu_report.md, check_mace_gpu.py); every
# one of these is also MACE's own CLI default (mace/tools/arg_parser.py),
# so this is "the values this project already validated" == "MACE's own
# defaults", not a coincidence worth re-deriving.
R_MAX = 6.0
NUM_BESSEL = 8
NUM_POLYNOMIAL_CUTOFF = 5
MAX_ELL = 3
HIDDEN_IRREPS = "128x0e + 128x1o"
NUM_INTERACTIONS = 2
CORRELATION = 3
INTERACTION = "RealAgnosticResidualInteractionBlock"

# Optimizer/scheduler defaults, also matching mace_run_train's own CLI
# defaults (mace/tools/arg_parser.py) -- these are the values MACE's own
# authors picked for this exact backbone, not re-tuned here.
LR = 0.01
WEIGHT_DECAY = 5e-7
LR_FACTOR = 0.8
SCHEDULER_PATIENCE = 50


def build_z_table(*atoms_lists):
    zs = sorted({int(z) for atoms_list in atoms_lists for atoms in atoms_list
                 for z in atoms.get_atomic_numbers()})
    return utils.AtomicNumberTable(zs)


def build_dataset(atoms_list, z_table, r_max):
    dataset = []
    for atoms in atoms_list:
        config = data.config_from_atoms(atoms, head_name="Default")
        atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max, heads=None)
        atomic_data.true_centroid = torch.tensor(
            atoms.info[CENTROID_KEY], dtype=torch.get_default_dtype()
        ).view(1, 3)
        dataset.append(atomic_data)
    return dataset


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_sq_err = 0.0
    n_graphs = 0
    with torch.enable_grad() if train else torch.no_grad():
        for batch in loader:
            batch_dict = batch.to(device).to_dict()
            if train:
                optimizer.zero_grad()
            out = model(batch_dict)
            cell = batch_dict["cell"].view(-1, 3, 3)
            length = torch.diagonal(cell, dim1=1, dim2=2)
            loss, _ = periodic_squared_error(out["centroid"], batch_dict["true_centroid"], length)
            if train:
                loss.backward()
                optimizer.step()
            bs = batch_dict["true_centroid"].shape[0]
            total_sq_err += loss.item() * bs
            n_graphs += bs
    return total_sq_err / n_graphs  # mean squared error, A^2


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", default="he_centroid")
    parser.add_argument("--train-file", type=Path, default=here / "mace_training_set_centroid" / "train.xyz")
    parser.add_argument("--valid-file", type=Path, default=here / "mace_training_set_centroid" / "valid.xyz")
    parser.add_argument("--work-dir", type=Path, default=here / "mace_electron_centroid_work")
    parser.add_argument("--r-max", type=float, default=R_MAX)
    parser.add_argument("--num-bessel", type=int, default=NUM_BESSEL)
    parser.add_argument("--num-polynomial-cutoff", type=int, default=NUM_POLYNOMIAL_CUTOFF)
    parser.add_argument("--max-ell", type=int, default=MAX_ELL)
    parser.add_argument("--hidden-irreps", default=HIDDEN_IRREPS)
    parser.add_argument("--num-interactions", type=int, default=NUM_INTERACTIONS)
    parser.add_argument("--correlation", type=int, default=CORRELATION)
    parser.add_argument("--interaction", default=INTERACTION, choices=list(interaction_classes))
    parser.add_argument("--batch-size", type=int, default=4,
                         help="from mace_gpu_report.md: 4 already uses 94%% of 24GB for the "
                              "heavier EnergyDipolesMACE at this r_max/hidden_irreps/float64 -- "
                              "this model has no energy/force readout, so there's headroom "
                              "above 4 if a real run needs it (default: %(default)s)")
    parser.add_argument("--valid-batch-size", type=int, default=4)
    parser.add_argument("--max-num-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lr-factor", type=float, default=LR_FACTOR)
    parser.add_argument("--scheduler-patience", type=int, default=SCHEDULER_PATIENCE)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                        help="model/training precision. float32 (default) is what the "
                             "ArcaNN tandem loop deploys -- compare_centroid_f32.py "
                             "showed it costs nothing on accuracy or smoothness. Use "
                             "float64 to reproduce the signed-off milestone-1 checkpoint.")
    parser.add_argument("--cueq", action="store_true",
                        help="build CentroidMACE with cuEquivariance fused kernels "
                             "(conv_fusion=False -- see centroid_to_cueq.py for why). "
                             "The deployed model for the ML-IAP exploration driver. "
                             "Needs cuequivariance + cuequivariance_torch importable "
                             "(the mace_electron_cueq env) on a GPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="warm-start: copy learnable weights from a previous "
                             "centroid_<iter>.model (buffers -- avg_num_neighbors, "
                             "z-table -- stay at this run's values). The "
                             "per-iteration knob in ARCANN_TANDEM_PLAN.md.")
    options = parser.parse_args(argv)
    if options.init_from is not None and not options.init_from.is_file():
        parser.error(f"--init-from {options.init_from} not found")
    if options.cueq and options.device != "cuda":
        parser.error("--cueq needs --device cuda (cuEquivariance kernels are GPU-only)")

    if not options.train_file.is_file():
        parser.error(f"{options.train_file} not found -- run to_extxyz_centroid.py first")
    if not options.valid_file.is_file():
        parser.error(f"{options.valid_file} not found -- run to_extxyz_centroid.py first")
    return options


def main(argv=None):
    options = parse_args(argv)
    torch_tools.set_default_dtype(options.dtype)
    torch_tools.set_seeds(options.seed)
    device = torch_tools.init_device(options.device)

    cueq_config = None
    if options.cueq:
        from mace.modules.wrapper_ops import CuEquivarianceConfig

        # conv_fusion=False is mandatory: centroid_to_cueq.py records a
        # conv_fusion=True crash at the batch_size-1 single-graph call the
        # exploration driver (he_mace_md_mliap.py) makes.
        cueq_config = CuEquivarianceConfig(
            enabled=True, layout="ir_mul", group="O3_e3nn",
            optimize_all=True, conv_fusion=False,
        )

    train_atoms = ase.io.read(options.train_file, index=":")
    valid_atoms = ase.io.read(options.valid_file, index=":")
    z_table = build_z_table(train_atoms, valid_atoms)

    train_dataset = build_dataset(train_atoms, z_table, options.r_max)
    valid_dataset = build_dataset(valid_atoms, z_table, options.r_max)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset, batch_size=options.batch_size, shuffle=True, drop_last=False,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_dataset, batch_size=options.valid_batch_size, shuffle=False, drop_last=False,
    )
    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    print(f"z_table={z_table.zs}  avg_num_neighbors={avg_num_neighbors:.3f}  "
          f"train={len(train_dataset)}  valid={len(valid_dataset)}", flush=True)

    model = CentroidMACE(
        r_max=options.r_max,
        num_bessel=options.num_bessel,
        num_polynomial_cutoff=options.num_polynomial_cutoff,
        max_ell=options.max_ell,
        interaction_cls=interaction_classes[options.interaction],
        interaction_cls_first=interaction_classes[options.interaction],
        num_interactions=options.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(options.hidden_irreps),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=options.correlation,
        cueq_config=cueq_config,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}", flush=True)

    if options.cueq:
        n_cuet = sum(
            1 for m in model.modules()
            if "cuequivariance" in (type(m).__module__ or "")
        )
        if n_cuet == 0:
            sys.exit(
                "--cueq set but the model has no cuequivariance modules -- "
                "cuequivariance is not importable in this env (silent e3nn "
                "fallback). Run in the mace_electron_cueq env."
            )
        print(f"cuequivariance modules: {n_cuet}", flush=True)

    if options.init_from is not None:
        prev_state = torch.load(options.init_from, map_location=device).state_dict()
        model_params = dict(model.named_parameters())
        loaded, mismatched = 0, []
        with torch.no_grad():
            for key, tensor in prev_state.items():
                if key not in model_params:
                    continue  # a buffer, or a name the current arch doesn't have
                if model_params[key].shape != tensor.shape:
                    mismatched.append(key)
                    continue
                model_params[key].copy_(tensor)
                loaded += 1
        print(f"warm-started {loaded}/{len(model_params)} param tensors from "
              f"{options.init_from}", flush=True)
        if mismatched:
            print(f"  NOT copied (shape changed -- new elements?): {mismatched}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.lr, weight_decay=options.weight_decay, amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=options.lr_factor, patience=options.scheduler_patience,
    )

    options.work_dir.mkdir(parents=True, exist_ok=True)
    model_path = options.work_dir / f"{options.name}.model"
    history_path = options.work_dir / f"{options.name}_history.json"

    best_valid_mse = float("inf")
    history = []
    for epoch in range(1, options.max_num_epochs + 1):
        t0 = time.time()
        train_mse = run_epoch(model, train_loader, optimizer, device, train=True)
        valid_mse = run_epoch(model, valid_loader, optimizer, device, train=False)
        scheduler.step(valid_mse)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0
        print(f"epoch {epoch:4d}  train_rmse={train_mse ** 0.5:.4f} A  "
              f"valid_rmse={valid_mse ** 0.5:.4f} A  lr={lr:.2e}  {epoch_time:.1f}s", flush=True)
        # Rewritten every epoch (not just at the end) so plot_centroid_training.py
        # can show a run's progress while it's still training, and a walltime
        # kill mid-run still leaves a usable partial history.
        history.append({
            "epoch": epoch, "train_mse": train_mse, "valid_mse": valid_mse,
            "train_rmse": train_mse ** 0.5, "valid_rmse": valid_mse ** 0.5,
            "lr": lr, "epoch_time_s": epoch_time,
        })
        history_path.write_text(json.dumps(history, indent=2))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            torch.save(model, model_path)

    print(f"best valid RMSE: {best_valid_mse ** 0.5:.4f} A -> {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
