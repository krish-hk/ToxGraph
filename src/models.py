"""
models.py — GNN model definitions for Tox21 multi-task toxicity prediction.

Currently implements:
  - GCN: 2-layer Graph Convolutional Network (baseline)

Designed so future models (GraphSAGE, GAT, ToxGraph) can be added and
selected via the same interface.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool


class GCN(torch.nn.Module):
    """
    2-layer Graph Convolutional Network for graph-level multi-task classification.

    Architecture:
        Molecular Graph
        → GCNConv(in, hidden) → ReLU
        → GCNConv(hidden, hidden) → ReLU
        → Global Mean Pooling
        → Dropout
        → Linear(hidden, num_tasks)
        → one logit per Tox21 task
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_tasks)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling (graph-level readout)
        x = global_mean_pool(x, batch)

        # Dropout + classification head
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x  # raw logits (num_graphs, num_tasks)


class GraphSAGE(torch.nn.Module):
    """
    2-layer GraphSAGE network for graph-level multi-task classification.

    Architecture:
        Molecular Graph
        → SAGEConv(in, hidden) → ReLU
        → SAGEConv(hidden, hidden) → ReLU
        → Global Mean Pooling
        → Dropout
        → Linear(hidden, num_tasks)
        → one logit per Tox21 task
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_tasks)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling (graph-level readout)
        x = global_mean_pool(x, batch)

        # Dropout + classification head
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x  # raw logits (num_graphs, num_tasks)


class GAT(torch.nn.Module):
    """
    2-layer Graph Attention Network (GAT) for graph-level multi-task classification.

    Architecture:
        Molecular Graph
        → GATConv(in, hidden_dim // heads, heads=4) → ELU
        → GATConv(hidden_dim, hidden_dim, heads=1, concat=False) → ELU
        → Global Mean Pooling
        → Dropout
        → Linear(hidden, num_tasks)
        → one logit per Tox21 task
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        head_dim = hidden_dim // heads
        self.conv1 = GATConv(num_node_features, head_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_dim, num_tasks)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Global pooling (graph-level readout)
        x = global_mean_pool(x, batch)

        # Dropout + classification head
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x  # raw logits (num_graphs, num_tasks)


class ToxGraph(torch.nn.Module):
    """
    ToxGraph: Task-Aware Graph Neural Network for Multi-Assay Toxicity Prediction.

    Architecture:
        1. Shared Molecular Graph Encoder:
           Molecular Graph
           → SAGEConv(in, hidden) → ReLU
           → SAGEConv(hidden, hidden) → ReLU
           → Global Mean Pooling → shared graph embedding h
           → Dropout

        2. Task-Aware Gating Module & Task-Specific Heads:
           For each assay t in {1, ..., num_tasks}:
           g_t = sigmoid(W_t h + b_t)          [endpoint-specific feature gate]
           h_t = h * g_t                       [task-specific molecular representation]
           logit_t = Linear_t(h_t)             [task prediction head]

        Output: 12 task logits
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 1. Shared GraphSAGE encoder backbone
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 2. Task-aware gating: one learned gating linear layer per toxicity task
        self.gate_linears = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_tasks)
        ])

        # 3. Task-specific prediction heads: one dedicated linear classifier per toxicity task
        self.task_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x, edge_index, batch):
        # Shared GraphSAGE message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling to extract shared representation h
        h = global_mean_pool(x, batch)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Task-aware gating and prediction for each endpoint
        task_logits = []
        for t in range(self.num_tasks):
            # Endpoint-specific feature gate: g_t = sigmoid(W_t h + b_t)
            g_t = torch.sigmoid(self.gate_linears[t](h))

            # Task-specific molecular representation: h_t = h * g_t
            h_t = h * g_t

            # Binary prediction for assay t: logit_t = Linear_t(h_t)
            logit_t = self.task_heads[t](h_t)
            task_logits.append(logit_t)

        # Concatenate into (batch_size, num_tasks)
        return torch.cat(task_logits, dim=-1)


class ToxGraphLite(torch.nn.Module):
    """
    ToxGraph-Lite: Lightweight Task-Aware Graph Neural Network.

    Uses the exact same GraphSAGE backbone as GraphSAGE and ToxGraph, but
    replaces the matrix gate W_t with a learned task-specific gate parameter vector a_t:
        g_t = sigmoid(a_t)              [learned endpoint-specific feature gate vector]
        h_t = h * g_t                   [task-specific molecular representation]
        logit_t = Linear_t(h_t)         [lightweight linear prediction head]

    Parameters: ~38,412 (only +1,536 over GraphSAGE vs +198,144 for full ToxGraph).
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 1. Shared GraphSAGE encoder backbone
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 2. Learned task-specific gate parameter vectors a_t in R^(num_tasks x hidden_dim)
        # Initialized to zeros so sigmoid(a_t) starts at 0.5 (neutral uniform gating)
        self.task_gate_vectors = torch.nn.Parameter(torch.zeros(num_tasks, hidden_dim))

        # 3. Dedicated linear classifier head per toxicity task
        self.task_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x, edge_index, batch):
        # Shared GraphSAGE message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling to extract shared representation h
        h = global_mean_pool(x, batch)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Compute gates for all tasks: g_t = sigmoid(a_t)
        # gates shape: (num_tasks, hidden_dim)
        gates = torch.sigmoid(self.task_gate_vectors)

        # Task-aware gating and prediction
        task_logits = []
        for t in range(self.num_tasks):
            # g_t has shape (hidden_dim,), broadcasts across batch dimension of h (batch_size, hidden_dim)
            h_t = h * gates[t]
            logit_t = self.task_heads[t](h_t)
            task_logits.append(logit_t)

        return torch.cat(task_logits, dim=-1)


class ToxGraphBottleneck(torch.nn.Module):
    """
    ToxGraph-Bottleneck: Parameter-Efficient Input-Conditioned Task-Aware GNN.

    Retains molecule-dependent task-specific gating while dramatically reducing
    parameters by projecting the shared embedding h into a lower-dimensional bottleneck
    space before task-specific gating:
        1. Shared GraphSAGE backbone:
           h = global_mean_pool(GraphSAGE(G))  [h in R^128]
        2. Shared bottleneck transformation:
           z = ReLU(W_shared h + b_shared)     [W_shared: 128 -> 32]
        3. Task-specific gating from bottleneck:
           g_t = sigmoid(W_t z + b_t)          [W_t: 32 -> 128]
        4. Gated task representation & prediction:
           h_t = h * g_t                       [h_t in R^128]
           logit_t = Linear_t(h_t)             [Linear_t: 128 -> 1]

    Parameters: 91,692 (60.99% parameter reduction vs full ToxGraph's 235,020).
    """

    def __init__(self, num_node_features: int, num_tasks: int,
                 hidden_dim: int = 128, bottleneck_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout

        # 1. Shared GraphSAGE encoder backbone
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 2. Shared bottleneck linear layer: 128 -> 32
        self.bottleneck = torch.nn.Linear(hidden_dim, bottleneck_dim)

        # 3. Task-specific gating from bottleneck: 32 -> 128 for each of the 12 tasks
        self.gate_linears = torch.nn.ModuleList([
            torch.nn.Linear(bottleneck_dim, hidden_dim) for _ in range(num_tasks)
        ])

        # 4. Task-specific prediction heads: 128 -> 1 for each of the 12 tasks
        self.task_heads = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x, edge_index, batch):
        # Shared GraphSAGE message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling to extract shared representation h
        h = global_mean_pool(x, batch)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Shared bottleneck transformation: z = ReLU(W_shared h + b_shared)
        z = F.relu(self.bottleneck(h))

        # Molecule-dependent, task-specific gating and prediction
        task_logits = []
        for t in range(self.num_tasks):
            # Input-conditioned gate: g_t = sigmoid(W_t z + b_t)
            g_t = torch.sigmoid(self.gate_linears[t](z))

            # Task-specific molecular representation: h_t = h * g_t
            h_t = h * g_t

            # Task prediction: logit_t = Linear_t(h_t)
            logit_t = self.task_heads[t](h_t)
            task_logits.append(logit_t)

        return torch.cat(task_logits, dim=-1)


# ---- Model registry ----

MODEL_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
    "toxgraph": ToxGraph,
    "toxgraph_lite": ToxGraphLite,
    "toxgraph_bottleneck": ToxGraphBottleneck,
}


def get_model(name: str, **kwargs) -> torch.nn.Module:
    """Instantiate a model by name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
