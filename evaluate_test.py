"""
evaluate_test.py — Final test evaluation, statistical robustness analysis,
gate interpretability, and publication-ready figures.

Strict Protocol:
- Uses the frozen best checkpoints from validation-based selection.
- Historical final evaluation used frozen checkpoints; reruns are reproductions.
- No retraining or tuning is performed.
- Saves machine-readable JSON and CSV results under a new output root.
- Generates figures under that output root, leaving frozen figures intact.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import load_tox21, load_split, get_data_loaders, TOX21_TASK_NAMES
from src.models import get_model
from src.metrics import compute_roc_auc
from src.utils import set_seed, get_device


def evaluate_test_predictions(model, loader, device, task_names):
    """
    Run inference on test set and return raw predictions, probabilities, labels, and AUC.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    auc_results = compute_roc_auc(all_labels, all_probs, task_names)

    return {
        "labels": all_labels,
        "logits": all_logits,
        "probs": all_probs,
        "auc_results": auc_results,
    }


def compute_test_task_statistics(test_dataset, task_names):
    """Compute number of valid, positive, and negative labels per task in the test set."""
    all_labels = torch.cat([d.y for d in test_dataset], dim=0).numpy()
    stats = {}
    for i, name in enumerate(task_names):
        labels_i = all_labels[:, i]
        mask = ~np.isnan(labels_i)
        valid = labels_i[mask]
        pos = int((valid == 1).sum())
        neg = int((valid == 0).sum())
        stats[name] = {
            "valid_count": int(mask.sum()),
            "pos_count": pos,
            "neg_count": neg,
            "pos_rate": round(pos / len(valid), 4) if len(valid) > 0 else 0.0,
            "has_both_classes": bool(pos > 0 and neg > 0)
        }
    return stats, all_labels


def bootstrap_auc_difference(labels, probs_a, probs_b, task_names, n_bootstraps=1000, seed=42):
    """
    Non-parametric bootstrap resampling of test molecules with replacement
    to compute confidence intervals for the difference in mean multi-task ROC-AUC:
    Delta = AUC(Model A) - AUC(Model B)
    """
    rng = np.random.RandomState(seed)
    n_samples = labels.shape[0]
    deltas = []

    for b in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_labels = labels[boot_idx]
        boot_probs_a = probs_a[boot_idx]
        boot_probs_b = probs_b[boot_idx]

        auc_a = compute_roc_auc(boot_labels, boot_probs_a, task_names)
        auc_b = compute_roc_auc(boot_labels, boot_probs_b, task_names)

        # Only consider when both have valid tasks
        if auc_a["valid_tasks"] > 0 and auc_b["valid_tasks"] > 0:
            deltas.append(auc_a["mean_auc"] - auc_b["mean_auc"])

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    ci_median = float(np.median(deltas))
    ci_mean = float(np.mean(deltas))
    p_value = float(np.mean(deltas <= 0))  # one-sided empirical p-value

    return {
        "n_bootstraps": n_bootstraps,
        "mean_delta": round(ci_mean, 4),
        "median_delta": round(ci_median, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value_a_le_b": round(p_value, 4),
        "deltas": deltas.tolist(),
    }


def extract_toxgraph_gates(model, test_loader, device, task_names):
    """
    Extract gate activations g_t = sigmoid(W_t h + b_t) across all test molecules.
    Returns: gate_tensor (N_test, 12, 128)
    """
    model.eval()
    all_gates = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # Forward pass through encoder
            x = model.conv1(batch.x.float(), batch.edge_index)
            x = F.relu(x)
            x = model.conv2(x, batch.edge_index)
            x = F.relu(x)

            # Pooling (no dropout during eval)
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(x, batch.batch)

            # Compute gate for each task
            batch_gates = []
            for t in range(model.num_tasks):
                g_t = torch.sigmoid(model.gate_linears[t](h))  # (batch_size, 128)
                batch_gates.append(g_t.unsqueeze(1))  # (batch_size, 1, 128)

            batch_gates = torch.cat(batch_gates, dim=1)  # (batch_size, 12, 128)
            all_gates.append(batch_gates.cpu())

    all_gates = torch.cat(all_gates, dim=0).numpy()  # (N_test, 12, 128)
    return all_gates


def generate_figures(test_results, task_stats, bootstrap_results, gate_data, output_dir="results/figures", history_dir="results"):
    """Generate all requested publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

    # 1. Model Test ROC-AUC Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    model_names = list(test_results.keys())
    display_names = {
        "gcn": "GCN",
        "gat": "GAT",
        "toxgraph_lite": "ToxGraph-Lite",
        "toxgraph_bottleneck": "ToxGraph-Bottleneck",
        "graphsage": "GraphSAGE",
        "toxgraph": "ToxGraph (Full)"
    }
    x_labels = [display_names.get(m, m) for m in model_names]
    scores = [test_results[m]["auc_results"]["mean_auc"] for m in model_names]
    colors = ['#7f7f7f', '#aec7e8', '#ffbb78', '#ff9896', '#1f77b4', '#2ca02c']

    bars = plt.bar(x_labels, scores, color=colors[:len(model_names)], width=0.55, edgecolor='black', linewidth=1)
    plt.ylabel("Mean Test ROC-AUC", fontweight='bold')
    plt.title("Final Model Evaluation on Tox21 Test Set (783 Molecules)", fontweight='bold', pad=15)
    plt.ylim(0.70, 0.82)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_test_roc_auc_comparison.png"), dpi=300)
    plt.close()

    # 2. GraphSAGE vs ToxGraph Per-Task Test ROC-AUC
    tasks = TOX21_TASK_NAMES
    sage_tasks = [test_results["graphsage"]["auc_results"]["per_task"][t] for t in tasks]
    tox_tasks = [test_results["toxgraph"]["auc_results"]["per_task"][t] for t in tasks]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(tasks))
    width = 0.38
    plt.bar(x - width/2, sage_tasks, width, label="GraphSAGE", color="#1f77b4", edgecolor='black')
    plt.bar(x + width/2, tox_tasks, width, label="ToxGraph (Proposed)", color="#2ca02c", edgecolor='black')
    plt.ylabel("Test ROC-AUC", fontweight='bold')
    plt.title("Per-Task Test ROC-AUC: GraphSAGE vs. ToxGraph", fontweight='bold', pad=15)
    plt.xticks(x, tasks, rotation=45, ha='right')
    plt.ylim(0.60, 0.95)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graphsage_vs_toxgraph_per_task.png"), dpi=300)
    plt.close()

    # 3. Per-Task Delta Chart (ToxGraph - GraphSAGE)
    deltas = np.array(tox_tasks) - np.array(sage_tasks)
    delta_colors = ['#2ca02c' if d > 0 else '#d62728' for d in deltas]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(tasks, deltas, color=delta_colors, edgecolor='black', width=0.55)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.ylabel("Test ROC-AUC Difference (ToxGraph − GraphSAGE)", fontweight='bold')
    plt.title("Per-Task Improvement Delta on Test Set", fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        offset = 4 if h >= 0 else -12
        plt.annotate(f'{h:+.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, offset), textcoords="offset points",
                     ha='center', va=va, fontsize=9, fontweight='bold')
    plt.ylim(min(deltas) - 0.015, max(deltas) + 0.015)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_task_improvement_delta.png"), dpi=300)
    plt.close()

    # 4. Validation Learning Curves for Main Models
    plt.figure(figsize=(10, 5))
    histories = {}
    for m in ["gcn", "gat", "graphsage", "toxgraph"]:
        res_file = os.path.join(history_dir, f"{m}_results.json")
        if os.path.exists(res_file):
            with open(res_file) as f:
                histories[m] = json.load(f)["epoch_history"]["val_mean_auc"]

    palette = {"gcn": "#7f7f7f", "gat": "#aec7e8", "graphsage": "#1f77b4", "toxgraph": "#2ca02c"}
    labels_map = {"gcn": "GCN", "gat": "GAT", "graphsage": "GraphSAGE", "toxgraph": "ToxGraph"}
    for m, vals in histories.items():
        epochs = range(1, len(vals) + 1)
        plt.plot(epochs, vals, label=f"{labels_map[m]} (best {max(vals):.4f})",
                 color=palette.get(m, 'black'), linewidth=2)

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Validation Mean ROC-AUC", fontweight='bold')
    plt.title("Validation ROC-AUC Curves Across Training Epochs", fontweight='bold', pad=15)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_curves_comparison.png"), dpi=300)
    plt.close()

    # 5. Task-vs-Feature Gate Activation Heatmap
    mean_task_gates = gate_data.mean(axis=0)  # (12, 128)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(mean_task_gates, aspect='auto', cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Mean Gate Activation $\sigma(W_t h + b_t)$', fontweight='bold')
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Latent Embedding Feature Dimension (0 to 127)", fontweight='bold')
    ax.set_ylabel("Toxicity Endpoint", fontweight='bold')
    ax.set_title("ToxGraph: Task-Specific Learned Feature Gate Activations", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_gate_heatmap.png"), dpi=300)
    plt.close()

    # 6. Task-to-Task Gate Correlation Heatmap
    gate_corr = np.corrcoef(mean_task_gates)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(gate_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontweight='bold')
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            val = gate_corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
    ax.set_title("ToxGraph: Pairwise Pearson Correlation of Task Gate Patterns", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_gate_similarity_heatmap.png"), dpi=300)
    plt.close()

    print(f"All 6 figures successfully saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Reproduce evaluation without overwriting frozen results")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--history_dir", default="results")
    parser.add_argument("--output_root", default="runs/evaluation")
    args = parser.parse_args()
    if os.path.exists(args.output_root):
        raise FileExistsError("Choose a new --output_root; evaluation outputs are never overwritten")
    for name in ("gcn", "gat", "graphsage", "toxgraph", "toxgraph_lite", "toxgraph_bottleneck"):
        if not os.path.isfile(os.path.join(args.checkpoint_dir, f"{name}_best.pt")):
            raise FileNotFoundError(f"Missing checkpoint for {name}")
    final_dir = os.path.join(args.output_root, "final")
    figure_dir = os.path.join(args.output_root, "figures")
    print("=" * 70)
    print("PHASE 5: FINAL TEST EVALUATION & SCIENTIFIC ANALYSIS")
    print("=" * 70)
    set_seed(42)
    device = get_device()

    # 1. Load Dataset & Test Split
    print("\nLoading dataset and locked test split...")
    dataset = load_tox21(root=args.data_root)
    _, _, test_ds, split_info = load_split(dataset, split_dir=args.data_root)
    print(f"Test split size: {len(test_ds)} molecules (strictly locked)")

    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    num_tasks = sample.y.shape[1]
    task_names = TOX21_TASK_NAMES[:num_tasks]

    # Create test loader
    _, _, test_loader = get_data_loaders(test_ds, test_ds, test_ds, batch_size=64)

    # 2. Test Split Label Distribution & Statistics
    task_stats, test_labels = compute_test_task_statistics(test_ds, task_names)
    print("\nTest Set Label Statistics:")
    print(f"  {'Task':<16} {'Valid':>8} {'Pos':>8} {'Neg':>8} {'Pos Rate':>10} {'2-Class?':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for t in task_names:
        s = task_stats[t]
        print(f"  {t:<16} {s['valid_count']:>8} {s['pos_count']:>8} {s['neg_count']:>8} {s['pos_rate']:>10.4f} {'YES' if s['has_both_classes'] else 'NO':>10}")

    # 3. Final Test Evaluation across all frozen checkpoints
    models_to_evaluate = [
        ("gcn", "checkpoints/gcn_best.pt"),
        ("gat", "checkpoints/gat_best.pt"),
        ("toxgraph_lite", "checkpoints/toxgraph_lite_best.pt"),
        ("graphsage", "checkpoints/graphsage_best.pt"),
        ("toxgraph_bottleneck", "checkpoints/toxgraph_bottleneck_best.pt"),
        ("toxgraph", "checkpoints/toxgraph_best.pt"),
    ]

    models_to_evaluate = [(name, os.path.join(args.checkpoint_dir, os.path.basename(path))) for name, path in models_to_evaluate]
    test_results = {}
    print("\nEvaluating frozen model checkpoints on the test set:")
    for model_name, ckpt_path in models_to_evaluate:
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint {ckpt_path} not found. Skipping {model_name}.")
            continue

        model = get_model(
            model_name,
            num_node_features=num_node_features,
            num_tasks=num_tasks,
            hidden_dim=128,
            dropout=0.3
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        res = evaluate_test_predictions(model, test_loader, device, task_names)
        mean_auc = res["auc_results"]["mean_auc"]
        valid_tasks = res["auc_results"]["valid_tasks"]
        print(f"  {model_name.upper():<20} | Test Mean ROC-AUC: {mean_auc:.4f} | Valid Tasks: {valid_tasks}/{num_tasks} | Selected Epoch: {ckpt['epoch']}")

        test_results[model_name] = {
            "model": model_name,
            "selected_epoch": ckpt["epoch"],
            "val_auc_at_selection": ckpt.get("val_auc", None),
            "auc_results": res["auc_results"],
            "probs": res["probs"],
            "labels": res["labels"],
        }

    # 4. Statistical & Robustness Analysis (Bootstrap CI for ToxGraph vs GraphSAGE)
    print("\nRunning Bootstrap Significance Analysis (1,000 resamples)...")
    probs_tox = test_results["toxgraph"]["probs"]
    probs_sage = test_results["graphsage"]["probs"]
    bootstrap_res = bootstrap_auc_difference(
        test_labels, probs_tox, probs_sage, task_names, n_bootstraps=1000, seed=42
    )

    print(f"  Mean Test Delta (ToxGraph - GraphSAGE): {bootstrap_res['mean_delta']:+.4f}")
    print(f"  Median Test Delta:                      {bootstrap_res['median_delta']:+.4f}")
    print(f"  95% Bootstrap Confidence Interval:     [{bootstrap_res['ci_95'][0]:.4f}, {bootstrap_res['ci_95'][1]:.4f}]")
    print(f"  Empirical p-value (Delta <= 0):         {bootstrap_res['p_value_a_le_b']:.4f}")

    # 5. Gate Interpretability Analysis
    print("\nExtracting learned gate patterns from final ToxGraph checkpoint...")
    tox_model = get_model("toxgraph", num_node_features=num_node_features, num_tasks=num_tasks, hidden_dim=128, dropout=0.3)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "toxgraph_best.pt"), map_location=device, weights_only=True)
    tox_model.load_state_dict(ckpt["model_state_dict"])
    tox_model = tox_model.to(device)
    gate_activations = extract_toxgraph_gates(tox_model, test_loader, device, task_names)
    print(f"  Extracted gate tensor: {gate_activations.shape} (N_molecules, N_tasks, D_hidden)")

    mean_gates = gate_activations.mean(axis=0)  # (12, 128)
    gate_variances_per_dim = mean_gates.var(axis=0)  # across tasks for each latent feature
    top_divergent_dims = np.argsort(gate_variances_per_dim)[::-1][:5].tolist()
    print(f"  Top 5 latent dimensions with highest cross-task gate variance: {top_divergent_dims}")

    # 6. Save Machine-Readable Results in results/final/
    os.makedirs(final_dir, exist_ok=False)

    # Clean test results for JSON (strip raw numpy arrays)
    json_summary = {}
    for m in test_results:
        json_summary[m] = {
            "model": m,
            "selected_epoch": test_results[m]["selected_epoch"],
            "val_auc_at_selection": test_results[m]["val_auc_at_selection"],
            "test_mean_auc": test_results[m]["auc_results"]["mean_auc"],
            "valid_tasks": test_results[m]["auc_results"]["valid_tasks"],
            "per_task_auc": test_results[m]["auc_results"]["per_task"],
        }
    json_summary["task_statistics"] = task_stats
    json_summary["bootstrap_analysis_toxgraph_vs_graphsage"] = {
        "n_bootstraps": bootstrap_res["n_bootstraps"],
        "mean_delta": bootstrap_res["mean_delta"],
        "median_delta": bootstrap_res["median_delta"],
        "ci_95": bootstrap_res["ci_95"],
        "p_value_a_le_b": bootstrap_res["p_value_a_le_b"],
    }
    json_summary["gate_analysis"] = {
        "top_divergent_latent_dimensions": top_divergent_dims,
        "mean_gate_activation_range": [float(mean_gates.min()), float(mean_gates.max())],
    }

    with open(os.path.join(final_dir, "test_evaluation_summary.json"), "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Saved {final_dir}/test_evaluation_summary.json")

    # CSV Summary Table
    csv_rows = []
    for m in test_results:
        csv_rows.append({
            "model": m,
            "selected_epoch": test_results[m]["selected_epoch"],
            "test_mean_auc": test_results[m]["auc_results"]["mean_auc"],
            **{f"test_auc_{t}": test_results[m]["auc_results"]["per_task"][t] for t in task_names}
        })
    df_results = pd.DataFrame(csv_rows)
    df_results.to_csv(os.path.join(final_dir, "test_model_comparison.csv"), index=False)
    print(f"  Saved {final_dir}/test_model_comparison.csv")

    # 7. Generate Figures
    print(f"\nGenerating reproduction figures under {figure_dir}/...")
    generate_figures(test_results, task_stats, bootstrap_res, gate_activations, output_dir=figure_dir, history_dir=args.history_dir)

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE: ALL RESULTS AND FIGURES GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    main()
