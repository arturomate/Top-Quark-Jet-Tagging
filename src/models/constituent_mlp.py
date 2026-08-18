"""
src/models/constituent_mlp.py
------------------------------
Fully-connected MLP that ingests a flattened constituent-feature vector.

Architecture
------------
The input is a 1-D vector of shape (batch, max_constituents * n_features),
produced by flattening the preprocessed (N, M, 7) constituent tensor.
For the default setting of max_constituents=80 and 7 features, the input
dimension is 560.

Limitations
-----------
This is intentionally a **simple baseline**, not a state-of-the-art model.
Flattening the constituent list discards permutation symmetry (the model
implicitly assumes a fixed ordering of constituents, which here is
pT-descending) and does not exploit any spatial or relational structure.
More expressive architectures — Particle Flow Networks (PFN), ParticleNet,
Particle Transformer — that operate on the constituent set in a
permutation-equivariant or permutation-invariant manner will be implemented
in later notebooks and can be compared fairly against this baseline using the
same evaluation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConstituentMLPClassifier(nn.Module):
    """
    Feedforward MLP operating on a flattened constituent-feature vector.

    Architecture per hidden layer::

        Linear(in, h) → [BatchNorm1d(h)] → ReLU → [Dropout(p)]

    Followed by a final ``Linear(h_last, 1)`` outputting a raw logit.

    Parameters
    ----------
    input_dim     : Flattened input dimension, e.g. 80 * 7 = 560.
    hidden_dims   : List of hidden-layer widths, e.g. ``[512, 256, 128, 64]``.
    dropout       : Dropout probability after each hidden block.  0 → disabled.
    use_batchnorm : Insert ``BatchNorm1d`` after each ``Linear``.
    """

    def __init__(
        self,
        input_dim:     int,
        hidden_dims:   list[int],
        dropout:       float = 0.15,
        use_batchnorm: bool  = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # raw logit

        self.net = nn.Sequential(*layers)

        # Store for introspection / checkpoint metadata
        self.input_dim     = input_dim
        self.hidden_dims   = hidden_dims
        self.dropout       = dropout
        self.use_batchnorm = use_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(batch, input_dim)`` float tensor — flattened constituent features.

        Returns
        -------
        logit : ``(batch,)`` raw scalar output (no sigmoid applied).
        """
        return self.net(x).squeeze(-1)
