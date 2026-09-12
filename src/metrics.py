"""
metrics.py — Evaluation metrics for multi-task binary classification.

Handles:
  - Per-task ROC-AUC with missing-label masking
  - Graceful handling of single-class tasks
  - Mean ROC-AUC across valid tasks
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true: np.ndarray, y_pred: np.ndarray,
                    task_names: list[str] = None):
    """
    Compute per-task and mean ROC-AUC for multi-task binary classification.

    Args:
        y_true: Ground truth labels (N, num_tasks). NaN = missing.
        y_pred: Predicted probabilities (N, num_tasks).
        task_names: Optional list of task names for reporting.

    Returns:
        dict with 'per_task' (dict of task_name -> auc or None),
                    'mean_auc' (float),
                    'valid_tasks' (int),
                    'skipped_tasks' (list of task names skipped and why).
    """
    num_tasks = y_true.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]

    per_task = {}
    skipped = []
    valid_aucs = []

    for i, name in enumerate(task_names):
        # Mask out missing labels (NaN)
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() == 0:
            per_task[name] = None
            skipped.append((name, "no valid labels"))
            continue

        y_t = y_true[:, i][mask]
        y_p = y_pred[:, i][mask]

        unique_classes = np.unique(y_t)
        if len(unique_classes) < 2:
            per_task[name] = None
            skipped.append((name, f"only class {unique_classes[0]} present"))
            continue

        try:
            auc = roc_auc_score(y_t, y_p)
            per_task[name] = round(float(auc), 4)
            valid_aucs.append(auc)
        except Exception as e:
            per_task[name] = None
            skipped.append((name, str(e)))

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    return {
        "per_task": per_task,
        "mean_auc": round(mean_auc, 4),
        "valid_tasks": len(valid_aucs),
        "total_tasks": num_tasks,
        "skipped_tasks": skipped,
    }
