"""
src/lightning/classifier.py
----------------------------
Generic PyTorch Lightning wrapper for binary jet classifiers.

Loss
----
Weighted binary cross-entropy with logits, normalised by the sum of weights::

    loss_per_sample = BCEWithLogits(logits, y, reduction="none")
    loss = sum(weight * loss_per_sample) / sum(weight)

This ensures that jets with larger ``training_weight`` contribute
proportionally more to the parameter update.

MODIFIED: ``_shared_step`` now dispatches the forward call with an optional
``mask`` tensor so that point-cloud models (e.g. ParticleNetLite) receive the
constituent validity mask.  The change is a minimal two-line addition; all
other models continue to work exactly as before because their batches do not
contain the ``"mask"`` key.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitBinaryClassifier(pl.LightningModule):
    """
    Lightning wrapper for any binary-classification model that outputs
    a scalar logit per sample.

    Parameters
    ----------
    model      : ``nn.Module`` producing raw logits of shape ``(batch,)``.
                 Must accept ``forward(x)`` at minimum.
                 Point-cloud models additionally accept ``forward(x, mask=...)``.
    opt_cfg    : The ``cfg["optimizer"]`` sub-dict.
    sched_cfg  : The ``cfg["scheduler"]`` sub-dict (may be ``None``).
    threshold  : Decision threshold for accuracy logging (default 0.5).
    """

    def __init__(
        self,
        model:     nn.Module,
        opt_cfg:   dict,
        sched_cfg: dict | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.model     = model
        self._opt_cfg  = opt_cfg
        self._sched_cfg= sched_cfg
        self._threshold= threshold
        self.save_hyperparameters(ignore=["model"])

    # --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``.  No mask support here;
        use ``self.model(x, mask=mask)`` directly when a mask is needed."""
        return self.model(x)

    # --------------------------------------------------------------------------
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x      = batch["x"]         # (B, ...) -- shape depends on input type
        y      = batch["y"]         # (B,) float
        weight = batch["weight"]    # (B,) float

        # Pass mask to the model when the batch contains one (point-cloud models).
        # For all other input types, "mask" is absent and we fall back to the
        # standard single-argument forward call.
        mask = batch.get("mask", None)
        if mask is not None:
            logits = self.model(x, mask=mask)   # (B,)
        else:
            logits = self.model(x)              # (B,)

        # Weighted BCE loss (physics convention: sum(w*L)/sum(w))
        loss_per_sample = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        loss = (weight * loss_per_sample).sum() / weight.sum().clamp(min=1e-8)

        # Batch-level accuracy for monitoring (unweighted; approximate)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self._threshold).float()
            acc   = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True,  batch_size=len(y))
        self.log(f"{stage}_acc",  acc,
                 on_step=False, on_epoch=True, prog_bar=(stage == "val"), batch_size=len(y))

        return loss

    # --------------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # --------------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cfg   = self._opt_cfg
        sched_cfg = self._sched_cfg

        # Build optimiser
        opt_name     = opt_cfg.get("name", "adam").lower()
        lr           = float(opt_cfg.get("lr", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr,
                weight_decay=weight_decay,
                momentum=float(opt_cfg.get("momentum", 0.9)),
            )
        else:
            raise ValueError(
                f"Unknown optimizer: '{opt_name}'. Supported: adam, adamw, sgd."
            )

        if sched_cfg is None:
            return optimizer

        # Build scheduler
        sched_name = sched_cfg.get("name", "").lower()
        monitor    = sched_cfg.get("monitor", "val_loss")

        if sched_name == "reduce_on_plateau":
            # NOTE: verbose=False was removed in PyTorch 2.x; do not pass it.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode    = "min",
                factor  = float(sched_cfg.get("factor", 0.5)),
                patience= int(sched_cfg.get("patience", 3)),
            )
            return {
                "optimizer":    optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   monitor,
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }

        elif sched_name in ("cosine", "cosine_annealing"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(sched_cfg.get("T_max", 30))
            )
            return {
                "optimizer":    optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        else:
            raise ValueError(
                f"Unknown scheduler: '{sched_name}'. "
                "Supported: reduce_on_plateau, cosine."
            )
