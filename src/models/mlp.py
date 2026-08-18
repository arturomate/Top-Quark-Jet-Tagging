"""
src/models/mlp.py
-----------------
High-level feature MLP (original, unchanged) plus the central model factory
``build_model`` that maps YAML ``model.name`` strings to ``nn.Module``
instances.

MODIFIED: ``particlenet_lite`` added to ``build_model`` registry.
          All existing model names (mlp, constituent_mlp, cnn_jetimage) work
          exactly as before.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# High-level feature MLP  (UNCHANGED)
# ------------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Configurable MLP for binary classification on high-level jet substructure
    features.

    Architecture per hidden layer::

        Linear(in, h) -> [BatchNorm1d(h)] -> ReLU -> [Dropout(p)]

    Followed by ``Linear(h_last, 1)`` outputting a raw logit.
    """

    def __init__(
        self,
        input_dim:     int,
        hidden_dims:   list,
        dropout:       float = 0.10,
        use_batchnorm: bool  = True,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim     = input_dim
        self.hidden_dims   = hidden_dims
        self.dropout       = dropout
        self.use_batchnorm = use_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) -> (batch,) raw logits."""
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------------------
# Model factory / registry
# ------------------------------------------------------------------------------

def build_model(model_cfg: dict) -> nn.Module:
    """
    Instantiate a model from the ``model`` section of the YAML config.

    Registered names
    ----------------
    ``"mlp"``              -> MLPClassifier
    ``"constituent_mlp"``  -> ConstituentMLPClassifier
    ``"cnn_jetimage"``     -> CNNJetImageClassifier
    ``"particlenet_lite"`` -> ParticleNetLite

    Parameters
    ----------
    model_cfg : The ``cfg["model"]`` sub-dict.

    Raises
    ------
    ValueError : If ``model_cfg["name"]`` is not registered.
    """
    # Deferred imports prevent circular dependencies at module load time
    from .constituent_mlp  import ConstituentMLPClassifier
    from .cnn              import CNNJetImageClassifier
    from .particlenet_lite import ParticleNetLite

    _REGISTRY = {
        "mlp":             MLPClassifier,
        "constituent_mlp": ConstituentMLPClassifier,
        "cnn_jetimage":    CNNJetImageClassifier,
        "particlenet_lite":ParticleNetLite,
        # Future: "transformer": ParticleTransformer, ...
    }

    name = str(model_cfg.get("name", "")).lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Registered models: {sorted(_REGISTRY.keys())}"
        )

    # -- MLP-family (shared constructor signature) -----------------------------
    if name in ("mlp", "constituent_mlp"):
        return _REGISTRY[name](
            input_dim    = int(model_cfg["input_dim"]),
            hidden_dims  = list(model_cfg["hidden_dims"]),
            dropout      = float(model_cfg.get("dropout", 0.10)),
            use_batchnorm= bool(model_cfg.get("use_batchnorm", True)),
        )

    # -- CNN jet-image ---------------------------------------------------------
    if name == "cnn_jetimage":
        return CNNJetImageClassifier(
            input_channels    = int(model_cfg.get("input_channels",  1)),
            image_size        = int(model_cfg.get("image_size",      40)),
            conv_channels     = list(model_cfg.get("conv_channels",  [16, 32, 64])),
            kernel_size       = int(model_cfg.get("kernel_size",     3)),
            dropout           = float(model_cfg.get("dropout",       0.20)),
            use_batchnorm     = bool(model_cfg.get("use_batchnorm",  True)),
            dense_hidden_dims = list(model_cfg.get("dense_hidden_dims", [128, 64])),
        )

    # -- ParticleNet-Lite ------------------------------------------------------
    if name == "particlenet_lite":
        return ParticleNetLite(
            input_dim              = int(model_cfg.get("input_dim",   7)),
            k                      = int(model_cfg.get("k",           8)),
            edge_channels          = list(model_cfg.get("edge_channels",
                                                         [64, 128, 128])),
            classifier_hidden_dims = list(model_cfg.get("classifier_hidden_dims",
                                                         [128, 64])),
            dropout                = float(model_cfg.get("dropout",  0.15)),
            aggr                   = str(model_cfg.get("aggr",       "max")),
        )

    # Generic fallback for any future model whose constructor accepts **kwargs
    cfg_copy = {k: v for k, v in model_cfg.items() if k != "name"}
    return _REGISTRY[name](**cfg_copy)
