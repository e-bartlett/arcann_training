#!/usr/bin/env python3
# MIRROR of dataset_prep/mace_electron/model.py -- the tested copy lives
# there; this is the copy ArcaNN init places in $WORK_DIR/user_files/ so the
# staged centroid training job can import CentroidMACE. Edit the dataset_prep
# copy, then re-copy here.
"""CentroidMACE: the weighted-centroid electron-position predictor.

Chunk 2 of Phase 3 item 4 (see `CENTROID_HEAD_PLAN.md`). Milestone 1 --
centroid-only, no energy/forces. Close copy of
`mace.modules.models.EnergyDipolesMACE`'s embedding/interaction/product
stack construction and forward pass (verified against the installed
`mace==0.3.16` source, `mace/modules/models.py:1171-1407`), with two
changes: the per-atom readout is `WeightedCentroidReadoutBlock` (raw
`(logit, delta)` instead of `(energy, dipole)`), and the aggregation over
atoms is a softmax-weighted periodic circular mean of `positions + delta`
(``r_e = sum_i w_i (r_i + Delta_i)``, softmax-normalized) instead of an
unconstrained ``scatter_sum`` of per-atom vectors. No energy readout, no
`AtomicEnergiesBlock`, no force/virial machinery -- this milestone isolates
whether the aggregation mechanism itself learns, nothing else.

The aggregation is a batched torch port of `analyze_dataset.py`'s
`circular_centroid` (per-graph phasor scatter-sum + `atan2`, not a scatter
of wrapped Cartesian positions -- averaging wrapped positions directly is
wrong near a periodic boundary). Because `w` sums to exactly 1 by softmax
construction, the `total<=0` NaN guard in the numpy version isn't needed
here.
"""

from __future__ import annotations

from typing import List, Optional, Type

import mace  # noqa: F401  -- import first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
             # before e3nn.o3 loads its constants.pt, or torch>=2.6 rejects it
import torch
from e3nn import o3

from mace.modules import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scatter import scatter_sum

from readout import WeightedCentroidReadoutBlock


def weighted_centroid(atomic_logit, atomic_delta, positions, cell, batch, num_graphs):
    """Softmax-weighted periodic circular mean of ``positions + atomic_delta``.

    Split out of ``CentroidMACE.forward`` so the aggregation math -- the
    highest-risk new piece, per a broadcast bug in ``L[batch]`` silently
    computing garbage while still "looking" equivariant -- can be unit-
    tested directly against a hand-built synthetic batch, without also
    constructing the full interaction/product stack. See
    ``test_weighted_centroid.py``.

    atomic_logit: [n_nodes]; atomic_delta, positions: [n_nodes, 3];
    cell: [n_graphs, 3, 3] (only the diagonal is used -- cubic cells);
    batch: [n_nodes] graph index per atom. Returns ``(centroid, w)``:
    centroid [n_graphs, 3] wrapped to [0, L), w [n_nodes] (sums to 1 per
    graph by softmax construction).
    """
    L = torch.diagonal(cell, dim1=1, dim2=2)  # [n_graphs, 3], cubic cells

    # Segment softmax over atoms in each graph (mace.tools.scatter has no
    # scatter_max, so amax comes from torch's own scatter_reduce_).
    graph_max = torch.zeros(num_graphs, dtype=atomic_logit.dtype, device=atomic_logit.device)
    graph_max.scatter_reduce_(0, batch, atomic_logit, reduce="amax", include_self=False)
    exp_logit = torch.exp(atomic_logit - graph_max[batch])
    denom = scatter_sum(exp_logit, batch, dim=0, dim_size=num_graphs)
    w = exp_logit / denom[batch]  # [n_nodes], sums to 1 per graph by construction

    # Batched analyze_dataset.py::circular_centroid: per-graph phasor
    # scatter-sum + atan2, periodic in L with no explicit wrap needed first
    # (cos/sin are already 2*pi-periodic).
    r_corrected = positions + atomic_delta  # [n_nodes, 3]
    theta = r_corrected * (2.0 * torch.pi / L[batch])
    C = scatter_sum(w.unsqueeze(-1) * torch.cos(theta), batch, dim=0, dim_size=num_graphs)
    S = scatter_sum(w.unsqueeze(-1) * torch.sin(theta), batch, dim=0, dim_size=num_graphs)
    mean_angle = torch.atan2(S, C)
    centroid = (mean_angle % (2.0 * torch.pi)) * L / (2.0 * torch.pi)  # [n_graphs, 3], wrapped [0, L)
    return centroid, w


class CentroidMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        # None (default) reproduces the reviewed milestone-1 e3nn path
        # exactly -- every block below already accepts cueq_config and
        # no-ops on None, so this is additive, not a milestone-1 change.
    ):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        self.cueq_config = cueq_config

        # Embedding -- identical to EnergyDipolesMACE.
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps, cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions, products, readouts -- same per-layer accumulation
        # pattern as EnergyDipolesMACE, WeightedCentroidReadoutBlock in
        # place of LinearDipoleReadoutBlock/NonLinearDipoleReadoutBlock at
        # every layer (no NonLinear variant needed: milestone 1 has no
        # "last layer" nonlinear-gate requirement EnergyDipolesMACE's
        # dipole readout has, since we're not also predicting energy).
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = "Residual" in str(interaction_cls_first)

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(WeightedCentroidReadoutBlock(hidden_irreps, cueq_config=cueq_config))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert len(hidden_irreps) > 1, "need at least l=1 hidden_irreps for the centroid readout"
                hidden_irreps_out = str(hidden_irreps[:2])  # scalars + l=1 vectors for the last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            self.readouts.append(WeightedCentroidReadoutBlock(hidden_irreps_out, cueq_config=cueq_config))

    def forward(self, data):
        num_graphs = data["ptr"].numel() - 1
        batch = data["batch"]

        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"], edge_index=data["edge_index"], shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers,
        )

        logit_contribs = []
        delta_contribs = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_out = readout(node_feats)  # [n_nodes, 4]
            logit_contribs.append(node_out[:, 0])
            delta_contribs.append(node_out[:, 1:])

        atomic_logit = torch.sum(torch.stack(logit_contribs, dim=-1), dim=-1)  # [n_nodes]
        atomic_delta = torch.sum(torch.stack(delta_contribs, dim=-1), dim=-1)  # [n_nodes, 3]

        cell = data["cell"].view(-1, 3, 3)
        centroid, w = weighted_centroid(
            atomic_logit, atomic_delta, data["positions"], cell, batch, num_graphs,
        )
        return {"centroid": centroid, "weights": w, "atomic_delta": atomic_delta}
