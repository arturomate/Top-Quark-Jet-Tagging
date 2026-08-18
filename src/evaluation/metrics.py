"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for binary jet classification.

All functions accept NumPy arrays and return plain Python scalars or dicts,
so they can be serialised directly to JSON.

Key physics convention
----------------------
- Signal efficiency   ε_S = TPR  (fraction of top jets accepted)
- Background efficiency ε_B = FPR (fraction of QCD jets accepted)
- Background rejection   R_B = 1 / ε_B  (how many QCD jets are rejected)

The primary working point is ε_S = 0.5; ε_S = 0.8 is also computed.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sigmoid helper (avoids importing torch in evaluation code)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for NumPy arrays."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Background rejection
# ──────────────────────────────────────────────────────────────────────────────

def background_rejection_at_signal_efficiency(
    y_true:             np.ndarray,
    y_score:            np.ndarray,
    target_signal_eff:  float,
    sample_weight:      np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute background rejection ``1 / ε_B`` at a fixed signal efficiency.

    Parameters
    ----------
    y_true            : True binary labels (int/float, 0 or 1).
    y_score           : Predicted signal probability scores in [0, 1].
    target_signal_eff : Desired signal efficiency (TPR), e.g. 0.5 or 0.8.
    sample_weight     : Optional per-sample weights passed to ``roc_curve``.

    Returns
    -------
    (rejection, actual_signal_eff) : A tuple where
        - ``rejection`` = 1 / ε_B  (np.inf if no background passes).
        - ``actual_signal_eff`` = the TPR value closest to ``target_signal_eff``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    idx    = int(np.argmin(np.abs(tpr - target_signal_eff)))
    eps_b  = float(fpr[idx])
    eps_s  = float(tpr[idx])

    if eps_b <= 0.0:
        return float("inf"), eps_s
    return 1.0 / eps_b, eps_s


# ──────────────────────────────────────────────────────────────────────────────
# Basic metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_basic_metrics(
    y_true:        np.ndarray,
    y_score:       np.ndarray,
    sample_weight: np.ndarray | None = None,
    threshold:     float = 0.5,
) -> dict:
    """
    Compute accuracy and ROC AUC, both unweighted and weighted.

    Parameters
    ----------
    y_true        : True binary labels.
    y_score       : Predicted signal probability scores.
    sample_weight : Optional per-sample weights.
    threshold     : Decision threshold for binary predictions.

    Returns
    -------
    dict with keys:
        accuracy, weighted_accuracy, auc, weighted_auc
    """
    y_pred = (y_score >= threshold).astype(int)

    acc          = float(accuracy_score(y_true, y_pred))
    weighted_acc = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    auc          = float(roc_auc_score(y_true, y_score))
    w_auc        = float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))

    return {
        "accuracy":          acc,
        "weighted_accuracy": weighted_acc,
        "auc":               auc,
        "weighted_auc":      w_auc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Combined metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true:              np.ndarray,
    y_score:             np.ndarray,
    sample_weight:       np.ndarray | None = None,
    threshold:           float = 0.5,
    signal_efficiencies: list[float] | None = None,
) -> dict:
    """
    Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true               : True binary labels.
    y_score              : Predicted signal probability scores.
    sample_weight        : Optional per-sample weights.
    threshold            : Decision threshold (default 0.5).
    signal_efficiencies  : List of ε_S working points.
                           Defaults to [0.5, 0.8].

    Returns
    -------
    Nested dict containing all metrics and the ROC arrays.
    """
    if signal_efficiencies is None:
        signal_efficiencies = [0.5, 0.8]

    basic = compute_basic_metrics(y_true, y_score, sample_weight, threshold)

    # ROC arrays
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)

    # Background rejection at each working point
    rejection_dict = {}
    for eps_s_target in signal_efficiencies:
        key = f"signal_eff_{eps_s_target:.2f}"
        rej, actual_eps_s = background_rejection_at_signal_efficiency(
            y_true, y_score, eps_s_target, sample_weight
        )
        rejection_dict[key] = {
            "target_signal_eff":  eps_s_target,
            "actual_signal_eff":  actual_eps_s,
            "background_rejection": rej,
        }

    # Confusion matrix
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        **basic,
        "threshold":         threshold,
        "confusion_matrix":  cm,
        "working_points":    rejection_dict,
        # ROC arrays stored separately for plotting
        "_roc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }


# ──────────────────────────────────────────────────────────────────────────────
# pT-binned background rejection
# ──────────────────────────────────────────────────────────────────────────────

def compute_pt_binned_background_rejection(
    y_true:               np.ndarray,
    y_score:              np.ndarray,
    jet_pt_gev:           np.ndarray,
    pt_bins_gev:          list[float],
    target_signal_eff:    float = 0.5,
    sample_weight:        np.ndarray | None = None,
    min_samples_per_class: int = 20,
) -> dict:
    """
    Compute background rejection in bins of jet pT.

    Parameters
    ----------
    y_true                : True binary labels.
    y_score               : Predicted signal probability scores.
    jet_pt_gev            : Jet transverse momentum in **GeV** (same length as y_true).
    pt_bins_gev           : Bin *edges* in GeV, e.g. [350, 550, 750, ...].
    target_signal_eff     : ε_S working point (default 0.5).
    sample_weight         : Optional per-sample weights.
    min_samples_per_class : Minimum signal and background events required per
                            bin.  Bins below this threshold return NaN.

    Returns
    -------
    dict with keys:
        bin_centres, bin_edges, background_rejection, actual_signal_eff
    Each value is a list of length ``n_bins``.
    """
    edges      = np.array(pt_bins_gev, dtype=float)
    n_bins     = len(edges) - 1
    bin_centres= 0.5 * (edges[:-1] + edges[1:])

    rejections = []
    actual_effs= []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (jet_pt_gev >= lo) & (jet_pt_gev < hi)

        n_sig_bin = int((y_true[mask] == 1).sum())
        n_bkg_bin = int((y_true[mask] == 0).sum())

        if n_sig_bin < min_samples_per_class or n_bkg_bin < min_samples_per_class:
            rejections.append(float("nan"))
            actual_effs.append(float("nan"))
            continue

        w_bin = sample_weight[mask] if sample_weight is not None else None
        try:
            rej, actual_eps_s = background_rejection_at_signal_efficiency(
                y_true[mask], y_score[mask], target_signal_eff, w_bin
            )
        except Exception:
            rej, actual_eps_s = float("nan"), float("nan")

        rejections.append(rej if not np.isinf(rej) else float("nan"))
        actual_effs.append(actual_eps_s)

    return {
        "bin_edges":           edges.tolist(),
        "bin_centres":         bin_centres.tolist(),
        "background_rejection":rejections,
        "actual_signal_eff":   actual_effs,
        "target_signal_eff":   target_signal_eff,
    }
