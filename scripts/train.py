#!/usr/bin/env python
"""
scripts/train.py
----------------
Training entry point for the ATLAS Top Quark Identification project.

Usage
-----
    python scripts/train.py --config configs/mlp_highlevel.yaml

MODIFIED: After training, saves preprocessing artefacts to
    results/<experiment>/preprocessing/
    - scaler.pkl      if standardize=true  (required for external test eval)
    - preprocessing.json  always (metadata for reproducibility)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.data.datamodule      import TopTaggingDataModule
from src.lightning.classifier import LitBinaryClassifier
from src.models               import build_model
from src.utils.config         import load_config
from src.utils.paths          import ensure_dir, resolve_path
from src.utils.serialization  import save_pickle, save_json, timestamp_str
from src.evaluation.plots     import plot_loss_curves


# ------------------------------------------------------------------------------
# Loss-history callback
# ------------------------------------------------------------------------------

class _LossHistory(pl.Callback):
    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_losses.append(float(v))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a jet classifier using PyTorch Lightning."
    )
    parser.add_argument("--config", required=True,
                        help="Path to the YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # Seed
    seed = int(cfg["experiment"]["seed"])
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Output directories
    exp_name   = cfg["experiment"]["name"]
    root_dir   = resolve_path(cfg["outputs"]["root_dir"])
    exp_dir    = ensure_dir(root_dir / exp_name)
    ckpt_dir   = ensure_dir(exp_dir / "checkpoints")
    log_dir    = ensure_dir(exp_dir / "logs")
    prep_dir   = ensure_dir(exp_dir / "preprocessing")
    fig_dir    = ensure_dir(
        resolve_path(cfg["outputs"]["figure_dir"]) / exp_name
    )

    print(f"\n{'='*62}")
    print(f"  Experiment : {exp_name}")
    print(f"  Output dir : {exp_dir}")
    print(f"  Config     : {Path(args.config).resolve()}")
    print(f"  Seed       : {seed}")
    print(f"  CUDA       : {torch.cuda.is_available()}")
    print(f"{'='*62}\n")

    # DataModule
    dm = TopTaggingDataModule(cfg)
    dm.setup("fit")
    print(f"  Training jets   : {dm.n_train:,}")
    print(f"  Validation jets : {dm.n_val:,}\n")

    # Model
    model    = build_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model           : {cfg['model']['name'].upper()}")
    print(f"  Trainable params: {n_params:,}\n")

    # Lightning module
    lit_model = LitBinaryClassifier(
        model     = model,
        opt_cfg   = cfg["optimizer"],
        sched_cfg = cfg.get("scheduler"),
        threshold = float(cfg["evaluation"].get("threshold", 0.5)),
    )

    # Callbacks
    trainer_cfg   = cfg["trainer"]
    early_stop_cb = EarlyStopping(
        monitor  = "val_loss",
        patience = int(trainer_cfg.get("early_stopping_patience", 7)),
        mode     = "min",
        verbose  = True,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath   = str(ckpt_dir),
        filename  = exp_name + "_{epoch:02d}_{val_loss:.4f}",
        monitor   = "val_loss",
        mode      = "min",
        save_top_k= 1,
        save_last = True,
    )
    loss_history = _LossHistory()

    # Logger
    csv_logger = CSVLogger(save_dir=str(log_dir), name="", version=0)

    # Trainer
    trainer = pl.Trainer(
        max_epochs           = int(trainer_cfg.get("max_epochs", 50)),
        accelerator          = trainer_cfg.get("accelerator", "auto"),
        devices              = int(trainer_cfg.get("devices", 1)),
        deterministic        = bool(trainer_cfg.get("deterministic", True)),
        num_sanity_val_steps = int(trainer_cfg.get("num_sanity_val_steps", 0)),
        callbacks            = [early_stop_cb, checkpoint_cb, loss_history],
        logger               = csv_logger,
        enable_progress_bar  = True,
        log_every_n_steps    = max(1, len(dm.train_dataloader()) // 2),
    )

    # Train
    trainer.fit(lit_model, datamodule=dm)

    best_ckpt   = checkpoint_cb.best_model_path
    best_val    = float(checkpoint_cb.best_model_score or float("nan"))
    epochs_ran  = trainer.current_epoch + 1

    print(f"\n  Training complete.")
    print(f"  Epochs run      : {epochs_ran}")
    print(f"  Best val_loss   : {best_val:.6f}")
    print(f"  Best checkpoint : {best_ckpt}")

    # ------------------------------------------------------------------
    # Save preprocessing artefacts
    # ------------------------------------------------------------------
    #
    # These files are required for correct external test-set evaluation.
    # If standardize=true, the fitted scaler MUST be saved here so that
    # evaluate.py can apply the SAME transform to the test file without
    # refitting on test data (which would be data leakage).
    #
    prep_meta = {
        "input_type":    dm.input_type,
        "standardize":   dm.scaler is not None,
        "n_features":    (
            list(dm.n_features) if hasattr(dm.n_features, "__iter__")
            else int(dm.n_features)
        ),
        "feature_names": dm.feature_names[:20],  # truncate long lists for JSON
        "n_train":       dm.n_train,
        "n_val":         dm.n_val,
        "config_name":   exp_name,
        "timestamp":     timestamp_str(),
    }

    if dm.scaler is not None:
        scaler_path = prep_dir / "scaler.pkl"
        save_pickle(dm.scaler, scaler_path)
        prep_meta["scaler_path"] = str(scaler_path)
        print(f"  Scaler saved    : {scaler_path}")
    else:
        prep_meta["scaler_path"] = None
        print("  Scaler          : not used (standardize=false)")

    save_json(prep_dir / "preprocessing.json", prep_meta)
    print(f"  Preprocessing   : {prep_dir / 'preprocessing.json'}")

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------
    if loss_history.train_losses:
        try:
            plot_loss_curves(
                loss_history.train_losses,
                loss_history.val_losses,
                out_path=fig_dir / "loss_curves.png",
                title=f"{exp_name} — loss curves",
            )
            print(f"  Loss curves     : {fig_dir / 'loss_curves.png'}")
        except Exception as exc:
            print(f"  [WARNING] Could not save loss curves: {exc}")

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------
    manifest = {
        "experiment":      exp_name,
        "seed":            seed,
        "epochs_ran":      epochs_ran,
        "best_val_loss":   best_val,
        "best_checkpoint": str(best_ckpt),
        "n_train":         dm.n_train,
        "n_val":           dm.n_val,
        "feature_names":   dm.feature_names[:20],
        "config":          cfg,
        "timestamp":       timestamp_str(),
    }
    save_json(exp_dir / "run_manifest.json", manifest)

    # Copy config into run directory for reproducibility
    shutil.copy2(args.config, exp_dir / "config_used.yaml")
    (exp_dir / "best_checkpoint.txt").write_text(str(best_ckpt))

    print(f"\n  All outputs saved to : {exp_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
