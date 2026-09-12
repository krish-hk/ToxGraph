"""Seed-42 end-to-end reproduction in a fresh directory, preserving frozen evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ("gcn", "gat", "graphsage", "toxgraph_lite", "toxgraph_bottleneck", "toxgraph")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/reproduction")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or evaluating")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory")
    commands = [[sys.executable, str(ROOT / "train.py"), "--model", name,
                 "--epochs", "60", "--patience", "15", "--lr", "0.001",
                 "--hidden_dim", "128", "--dropout", "0.3", "--batch_size", "64",
                 "--seed", "42", "--data_root", str(ROOT / "data"),
                 "--output_root", str(output)] for name in MODELS]
    evaluation = [sys.executable, str(ROOT / "evaluate_test.py"),
                  "--data_root", str(ROOT / "data"),
                  "--checkpoint_dir", str(output / "checkpoints"),
                  "--history_dir", str(output / "results"),
                  "--output_root", str(output / "evaluation")]
    if args.dry_run:
        for command in commands + [evaluation]:
            print(subprocess.list2cmdline(command))
        return
    env = {**os.environ, "PYTHONHASHSEED": "42", "PYTHONUTF8": "1"}
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    scores = {name: json.loads((output / "results" / f"{name}_results.json").read_text())["training"]["best_val_mean_auc"] for name in MODELS}
    # Record the reproduction's choice BEFORE any reproduction test evaluation.
    (output / "selection_before_test.json").write_text(json.dumps({
        "validation_scores": scores,
        "reproduction_validation_winner": max(scores, key=scores.get),
        "historical_primary_model": "toxgraph",
        "notice": "Reproduction after historical test disclosure; does not replace the frozen benchmark."
    }, indent=2) + "\n")
    subprocess.run(evaluation, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
