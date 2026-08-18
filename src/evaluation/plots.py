"""
src/evaluation/plots.py
------------------------
Matplotlib plotting functions for binary jet classification evaluation.

All functions follow the same calling convention:
- accept data arrays and an ``out_path`` (Path or str),
- create a figure, save it, and close it cleanly.
- return the output path for logging.

No seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ── Colour conventions ─────────────────────────────────────────────────────────
_BKG_COLOR  = "steelblue"
_SIG_COLOR  = "firebrick"
_BKG_LABEL  = "Background (QCD)"
_SIG_LABEL  = "Signal (Top)"
_ROC_COLOR  = "darkorange"

_RCPARAMS = {
    "figure.dpi":     120,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
}


def _apply_style() -> None:
    plt.rcParams.update(_RCPARAMS)


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list[float],
    val_losses:   list[float],
    out_path:     str | Path,
    title:        str = "Training and validation loss",
) -> Path:
    """
    Plot epoch-level training and validation loss curves.

    Parameters
    ----------
    train_losses : List of training losses (one per epoch).
    val_losses   : List of validation losses (one per epoch).
    out_path     : Destination file path (PNG).
    title        : Figure title.
    """
    _apply_style()
    n = min(len(train_losses), len(val_losses))
    epochs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses[:n], color="steelblue", lw=2, label="Train loss")
    ax.plot(epochs, val_losses[:n],   color="firebrick", lw=2, ls="--", label="Val loss")

    best_ep = int(np.argmin(val_losses[:n])) + 1
    ax.axvline(best_ep, color="gray", ls=":", lw=1.5,
               label=f"Best epoch ({best_ep})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    out_path:  str | Path,
    threshold: float = 0.5,
    title:     str   = "Output score distribution",
    note:      str   = "",
) -> Path:
    """
    Plot signal and background predicted-probability distributions.

    Parameters
    ----------
    y_true    : True binary labels (0 / 1).
    y_score   : Predicted signal probabilities in [0, 1].
    out_path  : Destination file path.
    threshold : Decision threshold drawn as a vertical line.
    title     : Figure title.
    note      : Additional text appended to the title (e.g. dataset note).
    """
    _apply_style()
    bins = np.linspace(0, 1, 51)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, color, label in [
        (0, _BKG_COLOR, _BKG_LABEL),
        (1, _SIG_COLOR, _SIG_LABEL),
    ]:
        mask = y_true == lbl
        ax.hist(y_score[mask], bins=bins, density=True,
                histtype="step", lw=2, color=color, label=label)

    ax.axvline(threshold, color="gray", ls="--", lw=1.5,
               label=f"Threshold = {threshold}")
    ax.set_xlabel("Predicted signal probability")
    ax.set_ylabel("Density")
    full_title = title + (f"\n{note}" if note else "")
    ax.set_title(full_title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    fpr:       np.ndarray,
    tpr:       np.ndarray,
    auc:       float,
    out_path:  str | Path,
    title:     str = "ROC curve",
    label:     str = "Model",
    note:      str = "",
) -> Path:
    """
    Plot the ROC curve (FPR vs TPR).

    Parameters
    ----------
    fpr, tpr : Arrays from ``sklearn.metrics.roc_curve``.
    auc      : AUC value to display in the legend.
    out_path : Destination file path.
    label    : Model name for the legend.
    note     : Additional note appended to the title.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_ROC_COLOR, lw=2,
            label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel(r"Background efficiency $\varepsilon_B$ (FPR)")
    ax.set_ylabel(r"Signal efficiency $\varepsilon_S$ (TPR)")
    full_title = title + (f"\n{note}" if note else "")
    ax.set_title(full_title)
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────────────

def plot_background_rejection_curve(
    fpr:      np.ndarray,
    tpr:      np.ndarray,
    auc:      float,
    out_path: str | Path,
    working_points: dict | None = None,
    title:    str = "Background rejection vs signal efficiency",
    label:    str = "Model",
    note:     str = "",
) -> Path:
    """
    Plot background rejection (1/FPR) vs signal efficiency (TPR).

    Parameters
    ----------
    fpr, tpr        : Arrays from ``sklearn.metrics.roc_curve``.
    auc             : AUC value for the legend.
    working_points  : Optional dict from ``compute_all_metrics``
                      (``cfg["evaluation"]["working_points"]``).
    out_path        : Destination file path.
    """
    _apply_style()
    with np.errstate(divide="ignore", invalid="ignore"):
        rejection = np.where(fpr > 0, 1.0 / fpr, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(tpr, rejection, color=_ROC_COLOR, lw=2,
                label=f"{label} (AUC = {auc:.4f})")

    if working_points:
        ls_cycle = ["--", ":", "-."]
        for i, (key, wp) in enumerate(working_points.items()):
            eps_s = wp.get("actual_signal_eff", wp.get("target_signal_eff"))
            rej   = wp.get("background_rejection")
            if rej is None or not np.isfinite(rej):
                continue
            ls = ls_cycle[i % len(ls_cycle)]
            ax.axvline(eps_s, color="gray", ls=ls, lw=1.2)
            ax.axhline(rej,   color="gray", ls=ls, lw=1.2,
                       label=(
                           rf"$\varepsilon_S$={wp['target_signal_eff']:.1f}: "
                           rf"$1/\varepsilon_B$={rej:.0f}"
                       ))

    ax.set_xlabel(r"Signal efficiency $\varepsilon_S$")
    ax.set_ylabel(r"Background rejection $1/\varepsilon_B$")
    full_title = title + (f"\n{note}" if note else "")
    ax.set_title(full_title)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    out_path:  str | Path,
    title:     str = "Confusion matrix",
    note:      str = "",
) -> Path:
    """
    Plot counts and row-normalised confusion matrices side by side.

    Parameters
    ----------
    y_true   : True binary labels.
    y_pred   : Predicted binary labels (after thresholding).
    out_path : Destination file path.
    """
    _apply_style()
    labels = [_BKG_LABEL, _SIG_LABEL]
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mat, fmt, subtitle in [
        (axes[0], cm,      "d",     "Counts"),
        (axes[1], cm_norm, ".3f",   "Row-normalised"),
    ]:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=mat, display_labels=labels
        )
        disp.plot(ax=ax, colorbar=False, values_format=fmt, cmap="Blues")
        ax.set_title(subtitle)
        ax.tick_params(axis="x", rotation=15)

    full_title = title + (f"\n{note}" if note else "")
    fig.suptitle(full_title, y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────────────

def plot_pt_binned_background_rejection(
    pt_result: dict,
    out_path:  str | Path,
    title:     str = r"$p_T$-binned background rejection",
    note:      str = "",
) -> Path:
    """
    Plot background rejection vs jet pT bin centre.

    Parameters
    ----------
    pt_result : Dict returned by
                ``metrics.compute_pt_binned_background_rejection``.
    out_path  : Destination file path.
    """
    _apply_style()
    centres = np.array(pt_result["bin_centres"])
    rej     = np.array(pt_result["background_rejection"], dtype=float)
    target  = pt_result.get("target_signal_eff", 0.5)

    valid = np.isfinite(rej)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(centres[valid], rej[valid],
                "o-", color=_ROC_COLOR, lw=2, ms=5,
                label=rf"$\varepsilon_S = {target}$")

    ax.set_xlabel(r"Jet $p_T$ [GeV]")
    ax.set_ylabel(r"Background rejection $1/\varepsilon_B$")
    full_title = title + (f"\n{note}" if note else "")
    ax.set_title(full_title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)
