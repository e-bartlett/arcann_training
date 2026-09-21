#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/readout.py -- the tested copy lives
# there; this is the copy ArcaNN init places in $WORK_DIR/user_files/ so the
# staged centroid training job can import CentroidMACE. Edit the dataset_prep
# copy, then re-copy here.
"""Per-atom readout for the weighted-centroid electron head.

Chunk 2 of Phase 3 item 4 (see `CENTROID_HEAD_PLAN.md`). Same `1x0e + 1x1o`
per-atom output shape and same linear-map structure as MACE's own
`LinearDipoleReadoutBlock` (`mace/modules/blocks.py:160-178`) -- one scalar
(`0e`, the unnormalized softmax logit `w_i` is built from) and one vector
(`1o`, the equivariant position correction `Delta_i`) per atom, per
interaction layer. `model.py::CentroidMACE` does the softmax + weighted
circular-mean aggregation across atoms; this block only produces the
per-atom raw features that feed it.
"""

from __future__ import annotations

import mace  # noqa: F401  -- import first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
             # before e3nn.o3 loads its constants.pt, or torch>=2.6 rejects it
import torch
from e3nn import o3
from mace.modules.wrapper_ops import CuEquivarianceConfig, Linear


class WeightedCentroidReadoutBlock(torch.nn.Module):
    """Linear per-atom map to (logit, delta): irreps_in -> 1x0e + 1x1o."""

    def __init__(self, irreps_in: o3.Irreps, cueq_config: CuEquivarianceConfig = None):
        super().__init__()
        self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = Linear(irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config)

    def forward(self, x):  # [n_nodes, irreps_in] -> [n_nodes, 4]
        return self.linear(x)
