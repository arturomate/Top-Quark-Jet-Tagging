"""
src/models/particlenet_lite.py
-------------------------------
ParticleNet-Lite: a compact Dynamic Graph CNN / ParticleNet-inspired model
for jet tagging in **pure PyTorch** (no PyTorch Geometric, no torch_cluster).

Architecture summary
--------------------
1. **Input**
   ``x``    : ``(B, M, F)``  — constituent features (F = 7 from ATLAS preprocessing)
   ``mask`` : ``(B, M)``     — True for valid (non-padded) constituents

2. **Coordinate extraction**
   ``coords = x[:, :, 0:2]``  → (η, φ) used for kNN graph construction.
   Coordinates are kept **fixed** across all layers (same convention as
   ParticleNet: the graph is rebuilt at each layer using original η/φ, not
   the propagated feature representation).

3. **EdgeConv blocks**  ×  len(edge_channels)
   For each point *i* and its *k* nearest neighbours *{j}*:
   ::

       edge_feat_{ij} = MLP( cat( x_i,  x_j − x_i ) )

   Features are aggregated over neighbours with max or mean pooling.
   Output: updated point features of the requested channel width.

4. **Masked global pooling**
   ``mean_pool`` and ``max_pool`` computed over valid particles only,
   then concatenated → ``(B, 2 * last_edge_channels)``.

5. **Classifier MLP**
   Fully-connected head → 1 raw logit per jet.

Why is this better than a flattened constituent MLP?
----------------------------------------------------
- A flattened MLP treats the input as a fixed-length ordered vector,
  so it is sensitive to permutation and to the pT-ordering of constituents.
- EdgeConv operates on a local neighbourhood in (η, φ) and processes
  relative displacements ``x_j − x_i``, making it invariant to global
  translation and naturally exploiting spatial locality in the jet image.
- Masked pooling correctly ignores zero-padded particles.

No PyTorch Geometric dependency
--------------------------------
kNN is implemented with ``torch.cdist``, which is standard PyTorch.
All operations are differentiable w.r.t. features (though not w.r.t.
graph topology, which is acceptable here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# k-NN graph builder
# ──────────────────────────────────────────────────────────────────────────────

def _knn(
    coords:  torch.Tensor,
    mask:    torch.Tensor | None,
    k:       int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a batched kNN graph in coordinate space.

    Parameters
    ----------
    coords : ``(B, M, 2)`` — (η, φ) coordinates of each particle.
    mask   : ``(B, M)`` bool — True for valid particles.  None → all valid.
    k      : Number of nearest neighbours requested.

    Returns
    -------
    idx       : ``(B, M, k)`` int64 — kNN indices.  Invalid slots (fewer
                than *k* real neighbours) are filled with 0 (safe fallback;
                those edges are masked in aggregation).
    valid_nbr : ``(B, M, k)`` bool — True where a real neighbour was found.
    """
    B, M, _ = coords.shape
    k = min(k, M - 1)   # can't have more neighbours than M-1

    dist = torch.cdist(coords.float(), coords.float())   # (B, M, M)

    # Mask out invalid particles as potential neighbours
    if mask is not None:
        invalid_col = (~mask).unsqueeze(1).expand(B, M, M)  # (B, M, M)
        dist = dist.masked_fill(invalid_col, float("inf"))

    # Remove self-loops
    eye = torch.eye(M, device=coords.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))

    # k smallest distances
    knn_dist, idx = dist.topk(k, dim=2, largest=False)    # (B, M, k)
    valid_nbr = knn_dist.isfinite()                        # (B, M, k)

    # Replace inf-index slots with 0 (fallback; masked in aggregation)
    idx = idx.masked_fill(~valid_nbr, 0)

    return idx, valid_nbr


# ──────────────────────────────────────────────────────────────────────────────
# EdgeConv block
# ──────────────────────────────────────────────────────────────────────────────

class EdgeConvBlock(nn.Module):
    """
    One EdgeConv layer.

    For each particle *i* and its *k* nearest neighbours *{j}*::

        edge_feat = cat(x_i, x_j − x_i)          # shape (..., 2*C_in)
        h_ij      = MLP(edge_feat)                 # shape (..., C_out)
        x_i_new   = aggr_{j ∈ kNN(i)} h_ij        # max or mean over k

    Parameters
    ----------
    in_channels  : Feature dimension of input ``x``.
    out_channels : Feature dimension of output.
    k            : Number of nearest neighbours.
    aggr         : ``"max"`` or ``"mean"`` aggregation.
    use_batchnorm: Insert BatchNorm1d after the Linear in the edge MLP.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        k:            int  = 8,
        aggr:         str  = "max",
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.k    = k
        self.aggr = aggr
        self.out_channels = out_channels

        # Shared MLP applied per edge: (2*C_in) → C_out
        layers: list[nn.Module] = [nn.Linear(2 * in_channels, out_channels)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.edge_mlp = nn.Sequential(*layers)

    def forward(
        self,
        x:      torch.Tensor,
        coords: torch.Tensor,
        mask:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : ``(B, M, C_in)``
        coords : ``(B, M, 2)`` — fixed (η, φ) coordinates.
        mask   : ``(B, M)`` bool or None.

        Returns
        -------
        out : ``(B, M, C_out)``  — updated particle features.
              Invalid (padded) particles are zeroed.
        """
        B, M, C = x.shape
        k = min(self.k, M - 1)

        # ── Build kNN graph ───────────────────────────────────────────────────
        idx, valid_nbr = _knn(coords, mask, k)
        # idx: (B, M, k)  valid_nbr: (B, M, k) bool

        # ── Gather neighbour features ─────────────────────────────────────────
        # Reshape idx to (B, M*k), gather along dim=1 of x
        idx_flat = idx.reshape(B, M * k)                     # (B, M*k)
        idx_exp  = idx_flat.unsqueeze(-1).expand(B, M * k, C)  # (B, M*k, C)
        x_j_flat = x.gather(1, idx_exp)                      # (B, M*k, C)
        x_j = x_j_flat.reshape(B, M, k, C)                  # (B, M, k, C)

        x_i = x.unsqueeze(2).expand(B, M, k, C)             # (B, M, k, C)

        # Edge features: [x_i || x_j - x_i]  → (B, M, k, 2C)
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)

        # ── Apply shared MLP ──────────────────────────────────────────────────
        # Flatten to (B*M*k, 2C) for the Linear + BN
        Cout = self.out_channels
        h = self.edge_mlp(edge_feat.reshape(B * M * k, 2 * C))  # (B*M*k, Cout)
        h = h.reshape(B, M, k, Cout)                             # (B, M, k, Cout)

        # ── Aggregate over neighbours ─────────────────────────────────────────
        if self.aggr == "max":
            # Mask invalid neighbour slots to −inf before max
            h = h.masked_fill(~valid_nbr.unsqueeze(-1), float("-inf"))
            out, _ = h.max(dim=2)                       # (B, M, Cout)
            # Replace residual −inf (all neighbours invalid) with 0
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        elif self.aggr == "mean":
            valid_f   = valid_nbr.float().unsqueeze(-1)                  # (B, M, k, 1)
            count     = valid_f.sum(dim=2).clamp(min=1.0)               # (B, M, 1)
            out       = (h * valid_f).sum(dim=2) / count                 # (B, M, Cout)

        else:
            raise ValueError(f"Unknown aggr='{self.aggr}'. Use 'max' or 'mean'.")

        # ── Zero out invalid (padded) query particles ─────────────────────────
        if mask is not None:
            out = out * mask.float().unsqueeze(-1)

        return out    # (B, M, Cout)


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class ParticleNetLite(nn.Module):
    """
    ParticleNet-Lite: compact EdgeConv-based jet classifier.

    Parameters
    ----------
    input_dim             : Feature dimension per constituent (default 7).
    k                     : kNN neighbours per particle.
    edge_channels         : Output channel sizes for each EdgeConv block.
    classifier_hidden_dims: Hidden layer widths of the final MLP.
    dropout               : Dropout probability in the classifier head.
    aggr                  : Neighbour aggregation strategy (``"max"`` / ``"mean"``).
    """

    def __init__(
        self,
        input_dim:              int        = 7,
        k:                      int        = 8,
        edge_channels:          list[int]  = None,
        classifier_hidden_dims: list[int]  = None,
        dropout:                float      = 0.15,
        aggr:                   str        = "max",
    ) -> None:
        super().__init__()

        if edge_channels is None:
            edge_channels = [64, 128, 128]
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [128, 64]

        # Store for introspection
        self.input_dim              = input_dim
        self.k                      = k
        self.edge_channels          = edge_channels
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout                = dropout
        self.aggr                   = aggr

        # ── EdgeConv stack ────────────────────────────────────────────────────
        self.edge_convs = nn.ModuleList()
        in_ch = input_dim
        for out_ch in edge_channels:
            self.edge_convs.append(
                EdgeConvBlock(in_ch, out_ch, k=k, aggr=aggr, use_batchnorm=True)
            )
            in_ch = out_ch

        # ── Classifier head ───────────────────────────────────────────────────
        # Global pooling concatenates masked mean + masked max → 2 * last_ch
        pooled_dim = 2 * edge_channels[-1]

        head: list[nn.Module] = []
        in_dim = pooled_dim
        for h in classifier_hidden_dims:
            head += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        head.append(nn.Linear(in_dim, 1))   # raw logit
        self.classifier = nn.Sequential(*head)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : ``(B, M, F)`` float tensor — constituent features.
        mask : ``(B, M)`` bool tensor — True for valid particles.
               If ``None``, all particles are treated as valid.

        Returns
        -------
        logit : ``(B,)`` float tensor — raw output, no sigmoid.
        """
        # Coordinates for graph construction: first two features are (η, φ)
        coords = x[:, :, :2]    # (B, M, 2) — fixed across all layers

        # EdgeConv layers
        for conv in self.edge_convs:
            x = conv(x, coords, mask)

        # ── Masked global pooling ─────────────────────────────────────────────
        if mask is not None:
            valid = mask.float().unsqueeze(-1)               # (B, M, 1)
            valid_count = valid.sum(dim=1).clamp(min=1.0)   # (B, 1)

            mean_pool = (x * valid).sum(dim=1) / valid_count   # (B, C)

            x_for_max = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            max_pool, _ = x_for_max.max(dim=1)               # (B, C)
            max_pool = torch.nan_to_num(max_pool, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            mean_pool = x.mean(dim=1)           # (B, C)
            max_pool, _ = x.max(dim=1)          # (B, C)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2C)

        logit = self.classifier(pooled).squeeze(-1)          # (B,)
        return logit
