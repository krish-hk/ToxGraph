# ToxGraph: Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction

**Track 3 / Graph Neural Networks | Tox21 / MoleculeNet | Graph-level prediction | Mean ROC-AUC**

ToxGraph learns molecule-dependent, endpoint-specific gates over a shared GraphSAGE embedding for 12 assay predictions. The mandatory **2-layer GCN baseline**, **GraphSAGE** and **GAT** comparisons, and a three-variant gating **ablation study** are included.

**Protocol:** seed = 42; fixed random train/validation/test split = **6,258 / 782 / 783**; NaN missing-label masking in loss and ROC-AUC; checkpoint selection uses validation only. Full ToxGraph was the validation-selected primary proposed model before final test evaluation.

## Headline frozen results

| Model | Parameters | Selected epoch | Validation ROC-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Mandatory 2-layer GCN | 19,340 | 58 | 0.7383 | 0.7605 |
| GAT | 19,852 | 57 | 0.7610 | 0.7853 |
| GraphSAGE | 36,876 | 57 | 0.7853 | 0.7889 |
| ToxGraph-Lite | 38,412 | 57 | 0.7691 | 0.7842 |
| ToxGraph-Bottleneck | 91,692 | 57 | 0.7815 | **0.7997** |
| Full ToxGraph (validation-selected primary) | 235,020 | 59 | **0.7890** | **0.7939** |

Full ToxGraph beats the mandatory GCN by **+0.0334** test ROC-AUC and GraphSAGE by **+0.0050**, improving **8/12 endpoints**. The paired molecule bootstrap (1,000 resamples) gives **95% CI [-0.0073, +0.0174]**, empirical **p = 0.198**. This is a positive empirical trend, **not statistically significant**. The reported empirical p is the fraction of bootstrap deltas at or below zero, not a calibrated null-hypothesis test.

**Bottleneck obtained the highest test score, 0.7997.** It remains an ablation; we do not retroactively replace the validation-selected primary model. Neither its superiority nor a regularization explanation is established by this single split.

[Four-page report](REPORT.pdf) · [Report source](REPORT.md) · [Frozen JSON](results/final/test_evaluation_summary.json) · [All per-task results](results/final/test_model_comparison.csv) · [Audit and judge preparation](SUBMISSION.md)

## Hackathon compliance

| Requirement | Explicit evidence |
|---|---|
| Track 3, graph-level MoleculeNet ROC-AUC | Tox21: 7,823 molecules, 12 binary endpoints |
| Mandatory 2-layer GCN; GAT and GraphSAGE comparisons | `src/models.py`; all six models in table above |
| Beat mandatory baseline | Full 0.7939 vs GCN 0.7605 on test |
| Fixed split, seed = 42 | `data/split_indices.json`, `src/data.py`, `src/utils.py` |
| Missing-label masking | `src/training.py`, `src/metrics.py`; single-class tasks skipped |
| Ablation and originality | No gates → static → bottleneck dynamic → full dynamic |
| One-command end-to-end run | `python reproduce.py` after environment installation |
| Report: problem, method, baseline results, ablation, limitations; max 4 pages | `REPORT.md`, `REPORT.pdf` |
| Demo / presentation | `python -m streamlit run app.py`; recording script in `SUBMISSION.md` |
| Recorded demo upload | **Outstanding: no recorded demo is present in this working copy** |
| Citations | References in report; dependencies in requirements files |

The supplied brief does not specify actual Tox21 organizer split indices. This repository's fixed random split is explicit; equivalence to any separately distributed organizer partition cannot be established from the attachment. Team membership, submitting lead, deadline date and upload platform also require external verification.

## Method and originality

Standard multi-task graph classifiers pool a shared molecular embedding and apply task-specific output weights. Those weights already permit different task preferences. ToxGraph adds a **nonlinear, molecule-dependent reweighting** before each endpoint's linear output head:

```text
h = Dropout(MeanPool(ReLU(SAGEConv2(ReLU(SAGEConv1(graph))))))
g_t = sigmoid(W_t h + b_t)
h_t = h * g_t
logit_t = Linear_t(h_t)
```

All encoders have two message-passing layers and hidden width 128. GCN and GraphSAGE use ReLU; GraphSAGE uses mean aggregation. GAT uses four 32-channel first-layer heads, one 128-channel second-layer head, ELU and attention dropout. All use global mean pooling and pooled dropout 0.3. Dropout is disabled at inference.

| Ablation | Mechanism | Interpretation |
|---|---|---|
| GraphSAGE | No gate | Shared encoder with task-specific linear output rows |
| ToxGraph-Lite | `sigmoid(a_t)`, static across molecules | Underperformed GraphSAGE on validation and test |
| ToxGraph-Bottleneck | Shared ReLU projection 128 → 32; per-task 32 → 128 sigmoid gates | Better than static on both splits; below GraphSAGE on validation, highest test score |
| Full ToxGraph | Per-task 128 → 128 sigmoid gate | Highest validation; positive test trend over GraphSAGE |

Static gates can be absorbed into linear head weights at inference; their result does **not** prove diminished representational capacity. The comparison is not parameter-matched, so added capacity and optimization are confounders. The contribution is this implemented task-conditioning design, controlled comparison and analysis on Tox21; global architectural novelty is not claimed.

## Data and scientific scope

PyTorch Geometric `MoleculeNet(root="data", name="Tox21")` downloads and processes data on first use (network access required); later runs reuse `data/tox21/`. Graphs contain nine categorical atom-feature columns and three bond-attribute columns. The frozen models cast atom category indices to floats and use **connectivity but not bond attributes**. There is no learned categorical feature encoder or fitted normalization. The demo calls the same PyG `from_smiles` featurizer.

Of 93,876 assay entries, 16,012 are NaN (17.1%). Masked BCE averages valid entries; missing labels never contribute as negatives. ROC-AUC masks each endpoint independently, skips missing/single-class endpoints, and macro-averages unrounded valid task AUCs before reporting four decimals. All 12 tasks were valid in the frozen test evaluation. Checkpoints are selected on the four-decimal validation mean returned by the existing metric code; ties retain the earlier epoch.

The historical protocol used train data for gradient updates and validation for model/checkpoint selection, followed by final frozen test evaluation. Source inspection confirms no test loader in training. Historical dataset diagnostics do summarize labels across the full dataset; therefore we do not claim that test labels were never inspected. Artifacts are consistent with the stated protocol but cannot independently prove how many historical evaluations occurred. Reproductions after test disclosure are not new untouched-test experiments.

## Reproducibility

Run commands from the repository root. The audit used Windows, Python 3.13.15 and CPU PyTorch 2.14.0+cpu. `requirements-lock.txt` records the **audit environment**, not an independently verified original training environment. Hardware and library changes can change training outcomes; seed locking does not guarantee bitwise reproducibility across devices.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python verify_submission.py
python reproduce.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. If the Windows CPU lock cannot be installed on another platform, install suitable PyTorch and `python -m pip install -r requirements.txt`; this is a different environment and exact score matching is not guaranteed.

**One-command execution:** `python reproduce.py` trains all six configurations (60 epochs maximum, patience 15, Adam lr 0.001, batch 64, hidden 128, dropout 0.3, seed 42), saves validation selection before evaluation, then runs test metrics, bootstrap and six figures. It requires training time; it was dry-run checked during this audit, not retrained. It writes exclusively to a new `runs/reproduction/` directory; choose `--output runs/reproduction-2` for another run. The original split is reused and validated. `python reproduce.py --dry-run` prints the exact commands without computation.

Useful individual commands:

```powershell
# A single reproduction model; change --model for another architecture.
python train.py --model toxgraph --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42 --output_root runs/manual

# Re-evaluate existing frozen checkpoints into NEW outputs (not a new unbiased test).
python evaluate_test.py --output_root runs/evaluation

# Launch the frozen Full ToxGraph demo.
python -m streamlit run app.py

# Regenerate the report from its single Markdown source; does not run models.
python generate_pdf_report.py
```

Training outputs: `<output_root>/checkpoints/*_best.pt` and `<output_root>/results/*_results.json`. Evaluation outputs: `<output_root>/final/` and `<output_root>/figures/`. Existing model/evaluation outputs are refused. Keep the shipped `checkpoints/`, `results/` and `data/split_indices.json` with the submission; `.gitignore` now includes these small frozen evidence files. SHA-256 fingerprints captured before audit edits are in `submission/frozen_manifest.json` and checked by `verify_submission.py`.

## Demo and interpretability

The demo validates SMILES, renders a 2D molecule, returns all 12 **predicted assay activity probabilities**, and shows a molecule-specific 12 × 128 gate heatmap from the frozen Full ToxGraph checkpoint. Probabilities are uncalibrated research outputs, not medical advice or chemical safety determinations. Common examples are demonstration inputs, not evidence of generalization.

The frozen summary gives mean gate activations approximately 0.0013–0.9999 (stored maximum 0.9999746) and highest cross-task variance at dimensions 56, 77, 13, 112 and 115. The preserved correlation figure labels NR-ER versus NR-ER-LBD at approximately r = 0.25 (weak positive similarity), not the 0.65 previously stated in the report. Their similar profiles are consistent with related endpoints using overlapping latent representation subspaces, but this is limited evidence. These are descriptive latent-space patterns, not validated biological mechanisms or atom-level explanations; exact correlation coefficients are not retained in the JSON summary.

## Limitations and next experiments

One random split and one seed do not establish scaffold generalization or training stability. Class imbalance and only 783 test molecules limit precision. Full has 6.37 times GraphSAGE's parameters; Bottleneck uses 60.99% fewer than Full. Capacity-matched heads, new scaffold evaluation, repeated training seeds, edge-aware encoders and calibration would be useful **future experiments on a separately planned evaluation protocol**. None was run or substituted into this benchmark.

See `SUBMISSION.md` for the full scorecard, changelog, remaining submission gaps, final abstract, presentation script and judge Q&A.
