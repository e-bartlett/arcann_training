#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/loss.py -- the tested copy lives
# there; this is the copy ArcaNN init places in $WORK_DIR/user_files/ so the
# staged centroid training job can import CentroidMACE. Edit the dataset_prep
# copy, then re-copy here.
"""Periodic loss for the weighted-centroid electron head.

Straight torch port of `analyze_dataset.py::minimum_image` (the same
orthorhombic min-image convention `spin_analysis.csv`'s own accuracy
numbers are computed with), applied to `pred_centroid - true_centroid`
instead of a spin-density displacement.
"""

from __future__ import annotations

import torch


def minimum_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap displacements into [-L/2, L/2) for an orthorhombic cell.

    ``delta``, ``cell``: same shape, broadcastable -- e.g. both [n_graphs, 3]
    for a batch of cubic cells' diagonal lengths.
    """
    return delta - cell * torch.round(delta / cell)


def periodic_squared_error(pred_centroid: torch.Tensor, true_centroid: torch.Tensor,
                            cell: torch.Tensor) -> torch.Tensor:
    """Mean squared minimum-imaged error between predicted and true centroids.

    Returns the training loss (mean of squared per-graph distance, Å^2) and
    the human-readable RMSE (Å, matching how `eval_dipole_error.py` already
    reports errors) as a ``(loss, rmse)`` pair of scalars.
    """
    delta = minimum_image(pred_centroid - true_centroid, cell)
    per_graph_sq = (delta ** 2).sum(dim=-1)  # [n_graphs]
    loss = per_graph_sq.mean()
    rmse = torch.sqrt(loss)
    return loss, rmse
