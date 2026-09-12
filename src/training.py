"""
training.py — Training and evaluation loops for multi-task GNN models.

Handles:
  - Training with masked BCEWithLogitsLoss for missing labels
  - Validation evaluation with ROC-AUC
  - Early stopping on validation performance
  - Epoch history tracking
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.data import create_masks
from src.metrics import compute_roc_auc


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss only on valid (non-NaN) labels.

    Args:
        logits: Model output (batch_size, num_tasks), raw logits.
        targets: Ground truth (batch_size, num_tasks), NaN = missing.

    Returns:
        Scalar loss computed only over valid entries.
    """
    mask = create_masks(targets)  # True where label is valid

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Replace NaN with 0 to avoid NaN in loss computation
    # (masked out anyway, but BCEWithLogitsLoss needs finite inputs)
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0

    # Compute element-wise loss
    loss = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )

    # Apply mask: only count valid entries
    loss = (loss * mask.float()).sum() / mask.float().sum()

    return loss


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = masked_bce_loss(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, task_names=None):
    """
    Evaluate model on a dataset split.

    Returns:
        dict with 'loss', 'roc_auc' (per-task and mean), 'predictions', 'labels'
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = masked_bce_loss(logits, batch.y)

        total_loss += loss.item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Convert logits to probabilities for ROC-AUC
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

    auc_results = compute_roc_auc(all_labels, all_probs, task_names)

    return {
        "loss": total_loss / max(num_batches, 1),
        "roc_auc": auc_results,
    }


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs: int = 100, patience: int = 15,
                task_names: list = None, checkpoint_path: str = None):
    """
    Full training loop with early stopping on validation ROC-AUC.

    Args:
        model: The GNN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: torch.device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without improvement).
        task_names: Task names for ROC-AUC reporting.
        checkpoint_path: Path to save best model checkpoint.

    Returns:
        history dict with per-epoch metrics and best results.
    """
    from src.utils import save_checkpoint

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mean_auc": [],
        "val_per_task_auc": [],
    }

    best_val_auc = 0.0
    best_epoch = 0
    best_per_task = {}
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_results = evaluate(model, val_loader, device, task_names)
        val_loss = val_results["loss"]
        val_auc = val_results["roc_auc"]["mean_auc"]
        val_per_task = val_results["roc_auc"]["per_task"]

        # Record history
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_mean_auc"].append(round(val_auc, 4))
        history["val_per_task_auc"].append(val_per_task)

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_per_task = val_per_task
            epochs_without_improvement = 0

            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Best: {best_val_auc:.4f} (ep {best_epoch})"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    history["best_epoch"] = best_epoch
    history["best_val_mean_auc"] = round(best_val_auc, 4)
    history["best_val_per_task_auc"] = best_per_task
    history["epochs_run"] = epoch

    return history
