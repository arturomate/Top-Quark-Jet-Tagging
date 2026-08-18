# src/evaluation/__init__.py
from .metrics import (
    compute_basic_metrics,
    background_rejection_at_signal_efficiency,
    compute_all_metrics,
    compute_pt_binned_background_rejection,
)
from .plots import (
    plot_loss_curves,
    plot_score_distributions,
    plot_roc_curve,
    plot_background_rejection_curve,
    plot_confusion_matrix,
    plot_pt_binned_background_rejection,
)

__all__ = [
    "compute_basic_metrics",
    "background_rejection_at_signal_efficiency",
    "compute_all_metrics",
    "compute_pt_binned_background_rejection",
    "plot_loss_curves",
    "plot_score_distributions",
    "plot_roc_curve",
    "plot_background_rejection_curve",
    "plot_confusion_matrix",
    "plot_pt_binned_background_rejection",
]
