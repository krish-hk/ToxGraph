"""Read-only frozen-artifact and inference checks; no training or test evaluation."""
import hashlib
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import numpy as np
    import torch
    from src.data import validate_split_indices
    from src.inference import load_toxgraph_checkpoint, smiles_to_graph_data, predict_toxicity_with_gates, validate_smiles
    from src.metrics import compute_roc_auc
    from src.models import get_model
    from src.training import masked_bce_loss

    manifest = json.loads((ROOT / "submission/frozen_manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    split = json.loads((ROOT / "data/split_indices.json").read_text())
    validate_split_indices(split, 7823)
    assert split["seed"] == 42
    assert [len(split[k]) for k in ("train_indices", "val_indices", "test_indices")] == [6258, 782, 783]
    corrupt = {**split, "test_indices": [split["train_indices"][0], *split["test_indices"][1:]]}
    try:
        validate_split_indices(corrupt, 7823)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping split accepted")
    expected = {"gcn": (.7383, .7605, 19340), "gat": (.7610, .7853, 19852),
                "graphsage": (.7853, .7889, 36876), "toxgraph_lite": (.7691, .7842, 38412),
                "toxgraph_bottleneck": (.7815, .7997, 91692), "toxgraph": (.7890, .7939, 235020)}
    summary = json.loads((ROOT / "results/final/test_evaluation_summary.json").read_text())
    with (ROOT / "results/final/test_model_comparison.csv").open(newline="") as stream:
        csv_rows = {row["model"]: row for row in csv.DictReader(stream)}
    for name, (val, test, params) in expected.items():
        history = json.loads((ROOT / "results" / f"{name}_results.json").read_text())
        checkpoint = torch.load(ROOT / "checkpoints" / f"{name}_best.pt", map_location="cpu", weights_only=True)
        model = get_model(name, num_node_features=9, num_tasks=12)
        model.load_state_dict(checkpoint["model_state_dict"])
        assert sum(p.numel() for p in model.parameters()) == params
        assert checkpoint["val_auc"] == val == history["training"]["best_val_mean_auc"]
        assert checkpoint["epoch"] == summary[name]["selected_epoch"] == history["training"]["best_epoch"]
        assert summary[name]["test_mean_auc"] == test
        assert float(csv_rows[name]["test_mean_auc"]) == test
        assert history["seed"] == 42 and history["hyperparameters"]["max_epochs"] == 60
        assert max(history["epoch_history"]["val_mean_auc"]) == val
        for task, value in summary[name]["per_task_auc"].items():
            assert float(csv_rows[name][f"test_auc_{task}"]) == value
    logits = torch.zeros((1, 2), requires_grad=True)
    loss = masked_bce_loss(logits, torch.tensor([[1., float("nan")]]))
    loss.backward()
    assert logits.grad[0, 1] == 0 and logits.grad[0, 0] != 0
    auc = compute_roc_auc(np.array([[0., 1., np.nan], [1., 1., np.nan]]), np.array([[.1, .5, .5], [.9, .5, .5]]))
    assert auc["mean_auc"] == 1 and auc["valid_tasks"] == 1 and len(auc["skipped_tasks"]) == 2
    for text in ("", " ", "not a molecule"):
        assert not validate_smiles(text)[0]
        try:
            smiles_to_graph_data(text)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid SMILES accepted")
    model, _ = load_toxgraph_checkpoint(str(ROOT / "checkpoints/toxgraph_best.pt"))
    for smiles in ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[Na+]"):
        graph = smiles_to_graph_data(smiles)
        probs, gates = predict_toxicity_with_gates(model, graph)
        with torch.no_grad():
            direct = torch.sigmoid(model(graph.x.float(), graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))).numpy()[0]
        np.testing.assert_array_equal(probs, direct)
        assert probs.shape == (12,) and gates.shape == (12, 128)
        assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    print(f"PASS: {len(manifest)} frozen hashes, splits, six checkpoints/results, masking, ROC-AUC edge cases, and four inference examples.")


if __name__ == "__main__":
    main()
