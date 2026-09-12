# ToxGraph: Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction

## Project Overview

**ToxGraph** investigates graph neural networks for multi-task molecular toxicity prediction on the **Tox21** benchmark from MoleculeNet. In multi-assay toxicity prediction, molecular graphs have varying assay coverage with missing ground-truth labels. ToxGraph addresses this via a rigorous multi-task loss-masking framework, a deterministic evaluation protocol, and an endpoint-specific feature gating mechanism.

### Method Description: ToxGraph & Task-Aware Gating

Standard multi-task GNNs share a single global graph representation across all prediction targets, which assumes that all biological assays depend on the same latent molecular sub-structures. However, toxicity endpoints possess fundamentally different mechanisms of action (e.g., nuclear receptor binding vs. cellular stress response pathways).

**ToxGraph** introduces **endpoint-specific feature gating** over a shared GraphSAGE backbone:
1. **Shared Graph Representation**: A 2-layer GraphSAGE encoder aggregates local molecular topology into a shared embedding vector $\mathbf{h} \in \mathbb{R}^{d}$.
2. **Task-Aware Gating**: For each toxicity assay $t \in \{1, \dots, 12\}$, a dedicated gating module dynamically filters the latent space:
   $$\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{h} + \mathbf{b}_t) \in (0, 1)^d$$
   $$\mathbf{h}_t = \mathbf{h} \odot \mathbf{g}_t$$
   where $\odot$ denotes element-wise multiplication, allowing each assay to selectively emphasize or suppress specific molecular features conditioned on the input graph.
3. **Task-Specific Prediction Heads**: Individual linear heads $\text{Linear}_t(\mathbf{h}_t)$ map each gated representation to its respective toxicity outcome logit.

#### Controlled Ablation Models
To evaluate the nature of task-specific conditioning, we compared three distinct gating variants:
- **ToxGraph (Full)**: Full molecule-dependent gating where $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{h} + \mathbf{b}_t)$ ($235,020$ parameters).
- **ToxGraph-Lite**: Static task-specific gating vectors $\mathbf{g}_t = \sigma(\mathbf{a}_t)$ where $\mathbf{a}_t \in \mathbb{R}^d$ ($38,412$ parameters).
- **ToxGraph-Bottleneck**: Parameter-efficient input-conditioned gating through a shared bottleneck projection $\mathbf{z} = \text{ReLU}(\mathbf{W}_{\text{shared}}\mathbf{h} + \mathbf{b}_{\text{shared}})$ with $\mathbf{W}_{\text{shared}} \in \mathbb{R}^{128 \to 32}$ and task gates $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{z} + \mathbf{b}_t)$ with $\mathbf{W}_t \in \mathbb{R}^{32 \to 128}$ ($91,692$ parameters, 60.99% reduction vs. Full ToxGraph).

---

## Dataset & Protocol

- **Dataset**: Tox21 via MoleculeNet (`torch_geometric.datasets.MoleculeNet`)
- **Molecules**: 7,823 compounds
- **Tasks**: 12 nuclear receptor (NR) and stress response (SR) assays
  - `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
- **Features**:
  - Node features: 9 atom properties
  - Edge features: 3 bond attributes
- **Missing Label Handling**:
  - MoleculeNet encodes unmeasured assays as `NaN` (16,012 missing entries, 17.1% sparsity).
  - Unmeasured assays are explicitly masked out in the multi-task binary cross-entropy loss (`BCEWithLogitsLoss`) and per-task ROC-AUC calculation. Missing assays are **never** treated as negative outcomes.

---

## Reproducibility & Split Strategy

- **Random Seed**: `42` applied deterministically across Python `random`, `numpy`, and `torch`.
- **Splitting Strategy**: Deterministic random split (80% train / 10% validation / 10% test).
- **Split Sizes**:
  - Train: 6,258 molecules
  - Validation: 782 molecules
  - Test: 783 molecules (strictly locked — **not** evaluated during model development)
- **Split Preservation**: Split indices are saved to `data/split_indices.json` ensuring identical evaluation subsets across all models.

---

## Installation

### 1. Environment Setup

Python 3.13 is recommended.

```bash
# Clone the repository
git clone https://github.com/krish-hk/ToxGraph.git
cd ToxGraph

# Create and activate a local virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch (CPU or CUDA 12.4+):
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Core requirements
pip install -r requirements.txt
```

---

## Interactive Demonstration Application

An interactive web demonstration is provided using Streamlit to perform real-time toxicity profiling with the frozen **ToxGraph** model checkpoint.

### Launch Command

```bash
streamlit run app.py
```

### Input & Output:
- **Input**: Enter any valid SMILES chemical string (e.g. `Cn1cnc2c1c(=O)n(C)c(=O)n2C` for Caffeine, `CC(=O)Oc1ccccc1C(=O)O` for Aspirin), or select from preset benchmark examples in the sidebar.
- **Processing**: RDKit validates the chemical structure and renders a 2D depiction; the exact 9-dimensional graph featurization is constructed and passed to the frozen `checkpoints/toxgraph_best.pt` model on CPU.
- **Output**:
  1. **Multi-Assay Toxicity Profile**: A sorted horizontal bar chart and complete tabular breakdown of predicted probability of assay activity across all 12 Tox21 assays, distinguishing Nuclear Receptor (NR) from Stress Response (SR) pathways.
  2. **Task-Aware Feature Gating**: A dynamic $12 \times 128$ heatmap visualizing the exact endpoint-specific gate activations $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{h} + \mathbf{b}_t)$ computed for the input molecule.
- **Disclaimer**: Research demonstration only. Predictions are model outputs and are not a substitute for experimental toxicology or safety assessment.

---

## Training Commands

All models use the same unified CLI interface:

```bash
# 1. 2-layer GCN Baseline
python train.py --model gcn --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42

# 2. GraphSAGE Baseline
python train.py --model graphsage --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42

# 3. GAT Baseline
python train.py --model gat --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42

# 4. ToxGraph (Full Model with Dynamic Task-Aware Gating)
python train.py --model toxgraph --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42

# 5. ToxGraph-Lite (Static Vector-Gated Variant)
python train.py --model toxgraph_lite --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42

# 6. ToxGraph-Bottleneck (Shared Bottleneck Input-Conditioned Variant)
python train.py --model toxgraph_bottleneck --epochs 60 --patience 15 --lr 0.001 --hidden_dim 128 --dropout 0.3 --batch_size 64 --seed 42
```

---

## Model Architectures

### 1. 2-Layer GCN
- GCNConv(9 → 128) → ReLU → GCNConv(128 → 128) → ReLU → MeanPool → Dropout(0.3) → Linear(128 → 12)
- Parameters: 19,340

### 2. GraphSAGE
- SAGEConv(9 → 128, mean) → ReLU → SAGEConv(128 → 128, mean) → ReLU → MeanPool → Dropout(0.3) → Linear(128 → 12)
- Parameters: 36,876

### 3. GAT (Graph Attention Network)
- GATConv(9 → 32, heads=4) → ELU → GATConv(128 → 128, heads=1) → ELU → MeanPool → Dropout(0.3) → Linear(128 → 12)
- Parameters: 19,852

### 4. ToxGraph (Proposed Model)
- Shared GraphSAGE backbone (128-d) → For each task $t$: $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{h} + \mathbf{b}_t)$, $\mathbf{h}_t = \mathbf{h} \odot \mathbf{g}_t$, $\text{logit}_t = \text{Linear}_t(\mathbf{h}_t)$
- Parameters: 235,020

### 5. ToxGraph-Lite (Static Vector Gating)
- Shared GraphSAGE backbone (128-d) → For each task $t$: $\mathbf{g}_t = \sigma(\mathbf{a}_t)$, $\mathbf{h}_t = \mathbf{h} \odot \mathbf{g}_t$, $\text{logit}_t = \text{Linear}_t(\mathbf{h}_t)$
- Parameters: 38,412

### 6. ToxGraph-Bottleneck (Bottleneck Input-Conditioned Gating)
- Shared GraphSAGE backbone (128-d) → Shared Bottleneck $\mathbf{z} = \text{ReLU}(\mathbf{W}_{\text{shared}}\mathbf{h} + \mathbf{b}_{\text{shared}})$ (32-d) → For each task $t$: $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{z} + \mathbf{b}_t)$, $\mathbf{h}_t = \mathbf{h} \odot \mathbf{g}_t$, $\text{logit}_t = \text{Linear}_t(\mathbf{h}_t)$
- Parameters: 91,692 (60.99% reduction vs. Full ToxGraph)

---

## Final Test Benchmark Results (Phase 5 — Frozen Single Evaluation)

> **Strict Evaluation Protocol**:
> - Models were completely frozen after validation development.
> - The locked test set ($N = 783$ molecules) was evaluated **exactly once** using best validation-selected checkpoints.
> - Zero post-test hyperparameter tuning, retraining, or model selection was conducted.

### Final Model Comparison (Validation vs. Locked Test)

| Model | Parameters | Selected Epoch | Validation ROC-AUC | Test Mean ROC-AUC | Test vs. GraphSAGE ($\Delta$) |
|---|---|---|---|---|---|
| **GCN** | 19,340 | 58 | 0.7383 | 0.7605 | -0.0284 |
| **GAT** | 19,852 | 57 | 0.7610 | 0.7853 | -0.0036 |
| **ToxGraph-Lite** (Static) | 38,412 | 57 | 0.7691 | 0.7842 | -0.0047 |
| **GraphSAGE** (Backbone) | 36,876 | 57 | 0.7853 | 0.7889 | Baseline (0.0000) |
| **ToxGraph** (Full, Proposed) | 235,020 | 59 | **0.7890** | **0.7939** | **+0.0050** |
| **ToxGraph-Bottleneck** | 91,692 | 57 | 0.7815 | **0.7997** | **+0.0108** |

### Per-Task Test ROC-AUC Breakdown

| Assay Task | Category | GCN | GAT | ToxGraph-Lite | GraphSAGE | ToxGraph (Proposed) | $\Delta$ (ToxGraph - SAGE) | ToxGraph-Bottleneck |
|---|---|---|---|---|---|---|---|---|
| **NR-AR** | Nuclear Receptor | 0.8017 | 0.8062 | 0.7892 | 0.7719 | 0.7682 | -0.0037 | 0.7769 |
| **NR-AR-LBD** | Nuclear Receptor | 0.8391 | 0.8701 | 0.8478 | 0.8404 | 0.8301 | -0.0103 | 0.8558 |
| **NR-AhR** | Nuclear Receptor | 0.8332 | 0.8583 | 0.8684 | 0.8731 | **0.8779** | **+0.0048** | 0.8774 |
| **NR-Aromatase** | Nuclear Receptor | 0.7238 | 0.7503 | 0.7639 | 0.7767 | **0.7775** | **+0.0008** | 0.7784 |
| **NR-ER** | Nuclear Receptor | 0.7248 | 0.7453 | 0.7508 | 0.7607 | **0.7757** | **+0.0150** | 0.7778 |
| **NR-ER-LBD** | Nuclear Receptor | 0.7796 | 0.8581 | 0.8444 | 0.8598 | **0.8722** | **+0.0124** | 0.8909 |
| **NR-PPAR-gamma** | Nuclear Receptor | 0.7109 | 0.6631 | 0.6586 | 0.6390 | **0.6667** | **+0.0277** | 0.6703 |
| **SR-ARE** | Stress Response | 0.6910 | 0.7177 | 0.7177 | 0.7443 | **0.7466** | **+0.0023** | 0.7442 |
| **SR-ATAD5** | Stress Response | 0.7812 | 0.8172 | 0.7972 | 0.7911 | **0.8199** | **+0.0288** | 0.8137 |
| **SR-HSE** | Stress Response | 0.7026 | 0.7354 | 0.7554 | 0.7591 | **0.7600** | **+0.0009** | 0.7627 |
| **SR-MMP** | Stress Response | 0.7613 | 0.8047 | 0.8177 | 0.8496 | 0.8321 | -0.0175 | 0.8423 |
| **SR-p53** | Stress Response | 0.7773 | 0.7973 | 0.7991 | 0.8007 | 0.7999 | -0.0008 | 0.8067 |
| **Mean** | — | 0.7605 | 0.7853 | 0.7842 | 0.7889 | **0.7939** | **+0.0050** | **0.7997** |

- **Win/Loss Ratio**: Full ToxGraph improves over GraphSAGE on **8 out of 12 tasks** (66.7% win rate) on the test split.
- Largest gains were observed on **SR-ATAD5** (+0.0288), **NR-PPAR-gamma** (+0.0277), **NR-ER** (+0.0150), and **NR-ER-LBD** (+0.0124).

---

## Statistical Robustness Analysis (Bootstrap Resampling)

We executed a 1,000-iteration non-parametric bootstrap over test set molecules to evaluate the difference $\Delta = \text{AUC}(\text{ToxGraph}) - \text{AUC}(\text{GraphSAGE})$:
- **Mean Test $\Delta$**: $+0.0050$
- **Median Test $\Delta$**: $+0.0051$
- **95% Bootstrap Confidence Interval**: $[-0.0073, +0.0174]$
- **Empirical $p$-value** ($\Delta \le 0$): $p = 0.1980$

> **Scientific Interpretation**: While ToxGraph demonstrates consistent positive empirical advantage on the majority of endpoints (8/12 assays improved, mean $+0.0050$ ROC-AUC), the 95% bootstrap confidence interval crosses zero ($p = 0.1980$) under the fixed test sample size ($N = 783$) and substantial label sparsity. This is typical for multi-task molecular toxicity benchmarks and indicates positive practical utility while appropriately contextualizing statistical certainty.

---

## Task Gate Interpretability Analysis

Test-time gate tensor analysis $\mathbf{g}_t = \sigma(\mathbf{W}_t \mathbf{h} + \mathbf{b}_t) \in [0, 1]^{128}$ revealed clear endpoint differentiation:
1. **Dynamic Dynamic Range**: Mean gate activations range from $0.0013$ to $0.9999$, demonstrating non-degenerate, highly selective representation masking.
2. **Top Divergent Latent Dimensions**: Dimensions `[56, 77, 13, 112, 115]` exhibit the highest cross-task gate variance, indicating features that are actively filtered in opposing manners across assay classes.
3. **Biological Alignment**: Related receptor pathways (such as NR-ER and NR-ER-LBD) share moderately correlated gating profiles ($r > 0.6$), whereas orthogonal stress-response assays (e.g. SR-ATAD5 vs. NR-AR) deploy distinct subspace masks.
*(Note: Latent dimensions represent learned continuous graph representations and are not claimed to map to specific chemical functional groups without explicit post-hoc atom attribution.)*

---

## Generated Artifacts and Publication Figures

All artifacts are automatically generated by `evaluate_test.py`:

```
results/
├── final/
│   ├── test_evaluation_summary.json         # Machine-readable test metrics, stats, CI, gate analysis
│   └── test_model_comparison.csv            # Tabular per-task and mean test ROC-AUCs
└── figures/
    ├── model_test_roc_auc_comparison.png    # Bar chart of all 6 frozen models on test set
    ├── graphsage_vs_toxgraph_per_task.png   # Paired per-task test comparison (SAGE vs. ToxGraph)
    ├── per_task_improvement_delta.png       # Sorted test ROC-AUC delta chart with zero line
    ├── validation_curves_comparison.png     # Validation ROC-AUC convergence across epochs
    ├── task_gate_heatmap.png                # 12-task by 128-dim mean gate activation heatmap
    └── task_gate_similarity_heatmap.png     # Pairwise Pearson correlation between task gates
```

To re-run test evaluation and regenerate all figures:
```powershell
python evaluate_test.py
```

---

## Project Structure

```
ToxGraph/
├── .gitignore               # Ignores venv/, data/, checkpoints/, results/
├── README.md                # Comprehensive documentation, benchmarks, and analysis
├── requirements.txt         # Verified package dependencies
├── train.py                 # Unified CLI experiment runner
├── evaluate_test.py         # Phase 5 frozen test evaluation, bootstrap, and plotting
├── src/
│   ├── __init__.py
│   ├── data.py              # Dataset downloading, verification, missing-label masking, splits
│   ├── models.py            # GNN architecture registry (GCN, GraphSAGE, GAT, ToxGraph variants)
│   ├── training.py          # Masked BCE loss, training loop, early stopping
│   ├── metrics.py           # Multi-task ROC-AUC with single-class edge-case handling
│   └── utils.py             # Reproducible seeding, parameter counts, checkpoint/result I/O
├── data/                    # Dataset root and deterministic split_indices.json (ignored by Git)
├── checkpoints/             # Model checkpoints (*.pt, ignored by Git)
└── results/                 # JSON execution metrics, test summary, CSV, and figures
```
