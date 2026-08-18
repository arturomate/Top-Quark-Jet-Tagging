#!/usr/bin/env python
"""
scripts/evaluate.py
--------------------
Evaluation script for trained jet classifiers.

Supports two modes:

VALIDATION MODE (default)
  python scripts/evaluate.py \\
      --config configs/mlp_highlevel.yaml \\
      --checkpoint <CKPT>

  Evaluates on the validation split of the training file (same as before).
  Outputs: eval_metrics.json, val_predictions.npz, figures in figures/<exp>/

EXTERNAL TEST MODE
  python scripts/evaluate.py \\
      --config configs/mlp_highlevel.yaml \\
      --checkpoint <CKPT> \\
      --data-path data/raw/test_nominal_000.h5 \\
      --split test

  Evaluates on the full external HDF5 file (no train/val split).
  For models with standardize=true, the scaler saved during training is
  loaded from results/<experiment>/preprocessing/scaler.pkl and applied
  to the test features.  A new scaler is NEVER fitted on the test file.
  Outputs: test_metrics.json, test_predictions.npz, figures in figures/<exp>/test/

MODIFIED from previous version:
  - Added --data-path and --split CLI arguments.
  - External test mode uses DataModule.setup_external_eval().
  - Scaler is loaded from preprocessing/scaler.pkl when needed.
  - Output filenames and directories are prefixed with the split name.
  - Validation mode is fully backward-compatible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datamodule      import TopTaggingDataModule
from src.evaluation.metrics   import (
    compute_all_metrics,
    compute_pt_binned_background_rejection,
)
from src.evaluation.plots     import (
    plot_background_rejection_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distributions,
    plot_pt_binned_background_rejection,
)
from src.lightning.classifier import LitBinaryClassifier
from src.models               import build_model
from src.utils.config         import load_config
from src.utils.paths          import ensure_dir, resolve_path
from src.utils.serialization  import load_pickle, save_json, save_pickle, timestamp_str


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

def predict_on_loader(
    lit_module: LitBinaryClassifier,
    loader:     DataLoader,
    device:     str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run inference on a DataLoader and return predictions.

    Handles:
    - Standard models: batch["x"] -> model(x) -> logits
    - Point-cloud models: batch["x"] + batch["mask"] -> model(x, mask=mask)
    """
    lit_module.eval()
    lit_module.to(device)

    all_y, all_score, all_w, all_pt = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x   = batch["x"].to(device)
            y   = batch["y"]
            wgt = batch["weight"]

            if "mask" in batch:
                logits = lit_module.model(
                    x, mask=batch["mask"].to(device)
                ).squeeze(-1).cpu()
            else:
                logits = lit_module.model(x).squeeze(-1).cpu()

            probs = torch.sigmoid(logits)
            all_y.append(y.numpy())
            all_score.append(probs.numpy())
            all_w.append(wgt.numpy())
            if "jet_pt" in batch:
                all_pt.append(batch["jet_pt"].numpy())

    y_true  = np.concatenate(all_y).astype(np.int32)
    y_score = np.concatenate(all_score).astype(np.float32)
    weights = np.concatenate(all_w).astype(np.float32)
    jet_pt  = np.concatenate(all_pt).astype(np.float32) if all_pt else None

    return y_true, y_score, weights, jet_pt


# ------------------------------------------------------------------------------
# Scaler loading
# ------------------------------------------------------------------------------

def _load_scaler_if_required(
    cfg:     dict,
    exp_dir: Path,
    split:   str,
) -> object | None:
    """
    Load the fitted StandardScaler from the training artefacts if needed.

    Rules
    -----
    - If standardize=false: return None (no scaler needed).
    - If standardize=true and split=="val": return None (DataModule handles it
      internally via setup("fit")).
    - If standardize=true and split=="test": load from preprocessing/scaler.pkl.
      Raise FileNotFoundError if the file is missing.
    """
    standardize = bool(cfg["data"].get("standardize", False))
    if not standardize:
        return None
    if split == "val":
        # Validation mode: DataModule fits and applies scaler internally.
        return None

    # External test mode with standardize=true: load the saved scaler.
    scaler_path = exp_dir / "preprocessing" / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] External test evaluation requires the scaler fitted "
            f"during training.\n"
            f"  Expected: {scaler_path}\n"
            f"  This file is saved automatically by train.py.\n"
            f"  Run 'python scripts/train.py --config {cfg.get('__config_path__', 'configs/<config>.yaml')}' "
            #f"  Run 'python scripts/train.py --config {cfg.get(\"__config_path__\", \"configs/<config>.yaml\")}' "
            f"first, or check the experiment output directory.\n"
        )
    scaler = load_pickle(scaler_path)
    print(f"  Scaler loaded from : {scaler_path}")
    return scaler


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained jet classifier."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML experiment config.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--data-path", default=None,
        dest="data_path",
        help=(
            "Path to an external HDF5 file for evaluation. "
            "If omitted, the validation split of the training file is used."
        ),
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test"],
        help=(
            "Which split to evaluate. "
            "'val' (default) uses the validation split of the training file. "
            "'test' uses the full --data-path file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate argument combination
    if args.split == "test" and args.data_path is None:
        sys.exit(
            "[ERROR] --split test requires --data-path to specify the "
            "external test file."
        )
    if args.data_path is not None and args.split == "val":
        print(
            "[WARNING] --data-path is set but --split=val. "
            "Ignoring --data-path and running validation evaluation."
        )
        args.data_path = None

    cfg = load_config(args.config)
    cfg["__config_path__"] = str(Path(args.config).resolve())

    exp_name  = cfg["experiment"]["name"]
    eval_cfg  = cfg.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    sig_effs  = list(eval_cfg.get("signal_efficiencies", [0.5, 0.8]))
    split     = args.split

    # Output directories
    root_dir = resolve_path(cfg["outputs"]["root_dir"])
    exp_dir  = ensure_dir(root_dir / exp_name)

    if split == "test":
        fig_dir       = ensure_dir(
            resolve_path(cfg["outputs"]["figure_dir"]) / exp_name / "test"
        )
        metrics_file  = exp_dir / "test_metrics.json"
        preds_file    = exp_dir / "test_predictions.npz"
        plot_prefix   = "test_"
        note = (
            f"External test file: {args.data_path} "
            f"[{exp_name}] — fixed nominal test evaluation."
        )
    else:
        fig_dir       = ensure_dir(
            resolve_path(cfg["outputs"]["figure_dir"]) / exp_name
        )
        metrics_file  = exp_dir / "eval_metrics.json"
        preds_file    = exp_dir / "val_predictions.npz"
        plot_prefix   = ""
        note = (
            f"Validation split of train_nominal_000.h5 "
            f"[{exp_name}] — NOT final test metrics."
        )

    print(f"\n{'='*62}")
    print(f"  Evaluate   : {exp_name}")
    print(f"  Split      : {split.upper()}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Note       : {note}")
    print(f"{'='*62}\n")

    # ------------------------------------------------------------------
    # DataModule setup
    # ------------------------------------------------------------------
    dm = TopTaggingDataModule(cfg)

    if split == "val":
        # Existing validation flow
        dm.setup("fit")
        loader     = dm.val_dataloader()
        eval_label = "Validation"
    else:
        # External test flow: load scaler if needed, then setup_external_eval
        scaler = _load_scaler_if_required(cfg, exp_dir, split)
        dm.setup_external_eval(
            data_path = resolve_path(args.data_path),
            scaler    = scaler,
        )
        loader     = dm.eval_dataloader()
        eval_label = "Test"

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    model     = build_model(cfg["model"])
    lit_model = LitBinaryClassifier.load_from_checkpoint(
        str(ckpt_path),
        model     = model,
        opt_cfg   = cfg["optimizer"],
        sched_cfg = cfg.get("scheduler"),
        threshold = threshold,
    )
    print(f"  Checkpoint loaded : {ckpt_path}\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true, y_score, sample_weights, jet_pt_mev = predict_on_loader(
        lit_model, loader, device=device
    )
    y_pred     = (y_score >= threshold).astype(np.int32)
    jet_pt_gev = jet_pt_mev / 1000.0 if jet_pt_mev is not None else None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    results  = compute_all_metrics(
        y_true, y_score,
        sample_weight       = sample_weights,
        threshold           = threshold,
        signal_efficiencies = sig_effs,
    )
    roc_data = results.pop("_roc")
    fpr, tpr = roc_data["fpr"], roc_data["tpr"]

    # pT-binned rejection
    pt_result = None
    if (
        eval_cfg.get("compute_pt_binned", True)
        and jet_pt_gev is not None
        and "pt_bins_gev" in eval_cfg
    ):
        pt_result = compute_pt_binned_background_rejection(
            y_true, y_score, jet_pt_gev,
            pt_bins_gev       = list(eval_cfg["pt_bins_gev"]),
            target_signal_eff = sig_effs[0],
            sample_weight     = sample_weights,
        )
        results["pt_binned_rejection"] = pt_result

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"  {eval_label} metrics:")
    print(f"    Accuracy            : {results['accuracy']:.4f}")
    print(f"    Weighted accuracy   : {results['weighted_accuracy']:.4f}")
    print(f"    AUC                 : {results['auc']:.4f}")
    print(f"    Weighted AUC        : {results['weighted_auc']:.4f}")
    for key, wp in results.get("working_points", {}).items():
        rej = wp["background_rejection"]
        rej_str = f"{rej:.1f}" if np.isfinite(rej) else "inf"
        print(
            f"    Bkg rejection"
            f" @ eps_S={wp['target_signal_eff']:.1f}"
            f" (actual {wp['actual_signal_eff']:.3f})"
            f" : {rej_str}"
        )

    # ------------------------------------------------------------------
    # Build flat metrics dict for JSON (comparison-friendly)
    # ------------------------------------------------------------------
    flat = {
        "experiment":      exp_name,
        "config_path":     str(Path(args.config).resolve()),
        "checkpoint_path": str(ckpt_path),
        "evaluated_data_path": str(args.data_path or "validation_split"),
        "split":           split,
        "n_events":        int(len(y_true)),
        "input_type":      cfg["data"].get("input_type", ""),
        "model_name":      cfg["model"].get("name", ""),
        "threshold":       float(threshold),
        "accuracy":        results["accuracy"],
        "weighted_accuracy":results["weighted_accuracy"],
        "auc":             results["auc"],
        "weighted_auc":    results["weighted_auc"],
        "confusion_matrix":results.get("confusion_matrix"),
        "note":            note,
        "timestamp":       timestamp_str(),
    }

    # Flatten working points into top-level keys
    for _key, wp in results.get("working_points", {}).items():
        eps = wp["target_signal_eff"]
        flat[f"background_rejection_at_{eps}"] = wp["background_rejection"]
        flat[f"actual_signal_efficiency_at_{eps}"] = wp["actual_signal_eff"]

    if pt_result is not None:
        flat["pt_binned_rejection"] = pt_result

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    save_json(metrics_file, flat)
    print(f"\n  Metrics saved  : {metrics_file}")

    # ------------------------------------------------------------------
    # Save predictions NPZ
    # ------------------------------------------------------------------
    npz_payload: dict = {
        "y_true":        y_true,
        "y_score":       y_score,
        "sample_weight": sample_weights,
    }
    if jet_pt_mev is not None:
        npz_payload["jet_pt_mev"] = jet_pt_mev
    np.savez_compressed(preds_file, **npz_payload)
    print(f"  Predictions saved : {preds_file}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n  Generating plots ...")

    p1 = plot_score_distributions(
        y_true, y_score, threshold=threshold,
        out_path=fig_dir / f"{plot_prefix}score_distributions.png",
        title=f"{exp_name} — output scores ({split})",
        note=note,
    )
    p2 = plot_roc_curve(
        fpr, tpr, auc=results["auc"],
        out_path=fig_dir / f"{plot_prefix}roc_curve.png",
        title=f"{exp_name} — ROC curve ({split})",
        label=exp_name,
        note=note,
    )
    p3 = plot_background_rejection_curve(
        fpr, tpr, auc=results["auc"],
        working_points=results.get("working_points"),
        out_path=fig_dir / f"{plot_prefix}background_rejection_curve.png",
        title=f"{exp_name} — background rejection ({split})",
        label=exp_name,
        note=note,
    )
    p4 = plot_confusion_matrix(
        y_true, y_pred,
        out_path=fig_dir / f"{plot_prefix}confusion_matrix.png",
        title=f"{exp_name} — confusion matrix ({split}, threshold={threshold})",
        note=note,
    )
    for p in (p1, p2, p3, p4):
        print(f"    {p}")

    if pt_result is not None:
        p5 = plot_pt_binned_background_rejection(
            pt_result,
            out_path=fig_dir / f"{plot_prefix}pt_binned_rejection.png",
            title=(
                f"{exp_name} — pT-binned rejection ({split})"
                f" @ eps_S={sig_effs[0]}"
            ),
            note=note,
        )
        print(f"    {p5}")

    print(f"\n  Figures saved to : {fig_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
