#!/usr/bin/env python
"""
train.py — Main entry point for ToxGraph experiments.

Usage:
    python train.py --model gcn
    python train.py --model gcn --epochs 100 --hidden_dim 128 --lr 1e-3

Supports GCN, GraphSAGE, GAT and all three ToxGraph variants.
"""

import argparse
import json
import os
import sys
import time

import torch

from src.data import (
    load_tox21,
    inspect_missing_labels,
    create_split,
    load_split,
    get_data_loaders,
    verify_dataset,
    TOX21_TASK_NAMES,
)
from src.models import get_model
from src.training import train_model
from src.utils import set_seed, get_device, count_parameters, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="ToxGraph: Multi-Assay Molecular Toxicity Prediction"
    )
    parser.add_argument("--model", type=str, default="gcn",
                        choices=["gcn", "graphsage", "gat", "toxgraph", "toxgraph_lite", "toxgraph_bottleneck"],
                        help="Model architecture to train")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension size")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory for dataset")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip dataset verification")
    parser.add_argument("--output_root", default="runs/manual",
                        help="Separate reproduction output directory; existing model files are protected")
    return parser.parse_args()


def print_verification_report(report):
    """Pretty-print the dataset verification report."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nTotal molecules:      {report['total_molecules']}")
    print(f"Number of tasks:      {report['num_tasks']}")
    print(f"Task names:           {', '.join(report['task_names'])}")
    print(f"Node features:        {report['num_node_features']}")
    print(f"Edge features:        {report['num_edge_features']}")
    print(f"\nTrain size:           {report['train_size']}")
    print(f"Validation size:      {report['val_size']}")
    print(f"Test size:            {report['test_size']}")
    print(f"\nTotal label entries:  {report['total_label_entries']}")
    print(f"Missing labels:       {report['missing_labels']} "
          f"({report['missing_label_proportion'] * 100:.1f}%)")

    print("\nPer-task label distribution:")
    print(f"  {'Task':<20} {'Valid':>8} {'Pos':>8} {'Neg':>8} {'Pos Rate':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for name, stats in report["per_task_distribution"].items():
        print(f"  {name:<20} {stats['valid']:>8} {stats['positive']:>8} "
              f"{stats['negative']:>8} {stats['pos_rate']:>10.4f}")

    print("\nSanity checks:")
    for check, passed in report["sanity_checks"].items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status:<8} {check}")

    print("=" * 70)


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.output_root, "checkpoints", f"{args.model}_best.pt")
    results_path = os.path.join(args.output_root, "results", f"{args.model}_results.json")
    if any(os.path.exists(p) for p in (checkpoint_path, results_path)):
        raise FileExistsError("Model outputs already exist; choose a new --output_root")
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"ToxGraph — Training {args.model.upper()} model")
    print(f"{'='*70}\n")

    # 1. Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # 2. Device
    device = get_device()

    # 3. Load dataset
    print("\nLoading Tox21 dataset...")
    dataset = load_tox21(root=args.data_root)
    print(f"Loaded {len(dataset)} molecules")

    # 4. Inspect missing label representation
    print("\nInspecting missing label encoding...")
    label_info = inspect_missing_labels(dataset)
    print(f"  Label dtype: {label_info['dtype']}")
    print(f"  Non-NaN unique values: {label_info['non_nan_unique_values']}")
    print(f"  Has NaN: {label_info['has_nan']}")
    print(f"  NaN count: {label_info['nan_count']} / {label_info['total_entries']}")

    # 5. Create or load split
    split_path = os.path.join(args.data_root, "split_indices.json")
    if os.path.exists(split_path):
        print(f"\nLoading existing split from {split_path}")
        train_ds, val_ds, test_ds, split_info = load_split(dataset, args.data_root)
        # Adapt split_info keys
        split_info["train_size"] = len(split_info["train_indices"])
        split_info["val_size"] = len(split_info["val_indices"])
        split_info["test_size"] = len(split_info["test_indices"])
        if split_info["seed"] != args.seed:
            raise ValueError("Requested seed differs from saved split seed")
    else:
        print("\nCreating train/val/test split (80/10/10)...")
        train_ds, val_ds, test_ds, split_info = create_split(
            dataset, seed=args.seed, train_ratio=0.8,
            val_ratio=0.1, test_ratio=0.1, split_dir=args.data_root
        )
    print(f"  Train: {split_info['train_size']}  |  "
          f"Val: {split_info['val_size']}  |  "
          f"Test: {split_info['test_size']}")

    split_info["split_dir"] = args.data_root
    # 6. Verify dataset
    if not args.no_verify:
        report = verify_dataset(dataset, train_ds, val_ds, test_ds, split_info)
        print_verification_report(report)

        # Abort if critical checks fail
        for check, passed in report["sanity_checks"].items():
            if not passed:
                print(f"\n CRITICAL: Sanity check '{check}' failed. Aborting.")
                sys.exit(1)

    # 7. Create data loaders
    train_loader, val_loader, _ = get_data_loaders(
        train_ds, val_ds, batch_size=args.batch_size
    )
    # NOTE: test_loader intentionally not created — test set not used during development

    # 8. Get dataset metadata
    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    num_tasks = sample.y.shape[1]

    # 9. Build model
    print(f"\nBuilding {args.model.upper()} model...")
    model = get_model(
        args.model,
        num_node_features=num_node_features,
        num_tasks=num_tasks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")
    print(f"  Architecture:\n{model}")

    # 10. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 11. Train
    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience})...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        task_names=TOX21_TASK_NAMES[:num_tasks],
        checkpoint_path=checkpoint_path,
    )

    elapsed = time.time() - start_time

    # 12. Report results
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Model:              {args.model.upper()}")
    print(f"  Epochs run:         {history['epochs_run']}")
    print(f"  Best epoch:         {history['best_epoch']}")
    print(f"  Best val mean AUC:  {history['best_val_mean_auc']:.4f}")
    print(f"  Training time:      {elapsed:.1f}s")

    print(f"\n  Per-task validation ROC-AUC (best epoch):")
    for task, auc in history["best_val_per_task_auc"].items():
        if auc is not None:
            print(f"    {task:<20} {auc:.4f}")
        else:
            print(f"    {task:<20} N/A (single class)")

    # 13. Save results
    results = {
        "model": args.model,
        "seed": args.seed,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
        },
        "dataset": {
            "name": "Tox21",
            "source": "MoleculeNet (PyTorch Geometric)",
            "total_molecules": len(dataset),
            "num_tasks": num_tasks,
            "num_node_features": num_node_features,
        },
        "split": split_info,
        "model_parameters": num_params,
        "training": {
            "epochs_run": history["epochs_run"],
            "best_epoch": history["best_epoch"],
            "best_val_mean_auc": history["best_val_mean_auc"],
            "best_val_per_task_auc": history["best_val_per_task_auc"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        },
        "epoch_history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_mean_auc": history["val_mean_auc"],
        },
        "training_time_seconds": round(elapsed, 1),
    }

    save_results(results, results_path)

    print(f"\n  Results saved to:   {results_path}")
    print(f"  Checkpoint saved:   {checkpoint_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
