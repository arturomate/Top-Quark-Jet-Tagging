"""
src/models/cnn.py
-----------------
Convolutional neural network for binary top-quark jet tagging from jet images.

Input
-----
Each jet is represented as a 2-D image of shape ``(1, image_size, image_size)``
in the (Δη, Δφ) plane, where pixel intensities encode the constituent pT
distribution.

Architecture
------------
Three ConvBlocks (Conv2d → [BatchNorm2d] → ReLU → MaxPool2d) followed by an
AdaptiveAvgPool2d layer to produce a fixed-size spatial representation
regardless of ``image_size``.  A small fully-connected head then maps to a
single raw logit (no sigmoid).

Why AdaptiveAvgPool2d?
    It decouples the convolutional backbone from the image resolution.  The
    head always receives a 64 × 4 × 4 = 1024-dimensional feature vector
    independent of whether ``image_size`` is 32, 40, or 64, making the
    architecture resolution-agnostic and the code free of hard-coded
    dimension calculations.

Output
------
A raw logit of shape ``(batch,)``.  Sigmoid is applied only during inference,
never inside ``forward``, so that ``F.binary_cross_entropy_with_logits`` can
be used with full numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Building block
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """
    Conv2d → [BatchNorm2d] → ReLU → MaxPool2d(2).

    Parameters
    ----------
    in_channels  : Number of input feature maps.
    out_channels : Number of output feature maps.
    kernel_size  : Convolution kernel size (square).
    use_batchnorm: Whether to insert BatchNorm2d after Conv2d.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int  = 3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2   # "same" padding
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class CNNJetImageClassifier(nn.Module):
    """
    Small CNN for binary classification of jet images.

    Parameters
    ----------
    input_channels    : Number of image channels (1 for single-channel pT maps).
    image_size        : Spatial size of the input image (square assumed).
                        The network is resolution-agnostic via AdaptiveAvgPool2d,
                        but the parameter is stored for introspection.
    conv_channels     : List of output channel counts for the three ConvBlocks.
                        E.g. ``[16, 32, 64]``.
    kernel_size       : Convolution kernel size (same for all blocks).
    dropout           : Dropout probability applied in the dense head.
    use_batchnorm     : Insert BatchNorm2d in conv blocks and BatchNorm1d in
                        the dense head.
    dense_hidden_dims : Widths of the fully-connected hidden layers.
    """

    def __init__(
        self,
        input_channels:    int        = 1,
        image_size:        int        = 40,
        conv_channels:     list[int]  = None,
        kernel_size:       int        = 3,
        dropout:           float      = 0.20,
        use_batchnorm:     bool       = True,
        dense_hidden_dims: list[int]  = None,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [16, 32, 64]
        if dense_hidden_dims is None:
            dense_hidden_dims = [128, 64]

        # Store for introspection / checkpoint metadata
        self.input_channels    = input_channels
        self.image_size        = image_size
        self.conv_channels     = conv_channels
        self.kernel_size       = kernel_size
        self.dropout           = dropout
        self.use_batchnorm     = use_batchnorm
        self.dense_hidden_dims = dense_hidden_dims

        # ── Convolutional backbone ─────────────────────────────────────────
        conv_blocks: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in conv_channels:
            conv_blocks.append(
                _ConvBlock(in_ch, out_ch, kernel_size, use_batchnorm)
            )
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_blocks)

        # AdaptiveAvgPool2d: map any spatial size → fixed 4×4 grid.
        # Flattened dimension = conv_channels[-1] × 4 × 4.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim  = conv_channels[-1] * 4 * 4   # e.g. 64 * 4 * 4 = 1024

        # ── Fully-connected head ───────────────────────────────────────────
        head: list[nn.Module] = []
        in_dim = flat_dim
        for h in dense_hidden_dims:
            head.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                head.append(nn.BatchNorm1d(h))
            head.append(nn.ReLU(inplace=True))
            head.append(nn.Dropout(p=dropout))
            in_dim = h
        head.append(nn.Linear(in_dim, 1))   # raw logit

        self.fc_head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(batch, 1, image_size, image_size)`` float tensor.

        Returns
        -------
        logit : ``(batch,)`` float tensor — raw output, no sigmoid.
        """
        x = self.conv_backbone(x)   # (B, C_last, H', W')
        x = self.pool(x)            # (B, C_last, 4, 4)
        x = x.flatten(start_dim=1) # (B, C_last * 16)
        x = self.fc_head(x)        # (B, 1)
        return x.squeeze(-1)        # (B,)
