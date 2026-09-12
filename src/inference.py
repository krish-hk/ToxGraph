"""
inference.py — Reusable inference pipeline for ToxGraph.

Handles:
- SMILES validation with RDKit
- 2D molecular structure rendering
- Molecule graph featurization matching training pipeline exactly
- Model loading and evaluation
- Dynamic gate vector extraction
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils import from_smiles

from src.models import get_model
from src.data import TOX21_TASK_NAMES


# Metadata for Tox21 endpoints
TASK_METADATA = {
    "NR-AR": {"category": "Nuclear Receptor", "full_name": "Androgen Receptor", "type": "NR"},
    "NR-AR-LBD": {"category": "Nuclear Receptor", "full_name": "Androgen Receptor (LBD)", "type": "NR"},
    "NR-AhR": {"category": "Nuclear Receptor", "full_name": "Aryl Hydrocarbon Receptor", "type": "NR"},
    "NR-Aromatase": {"category": "Nuclear Receptor", "full_name": "Aromatase Enzyme", "type": "NR"},
    "NR-ER": {"category": "Nuclear Receptor", "full_name": "Estrogen Receptor", "type": "NR"},
    "NR-ER-LBD": {"category": "Nuclear Receptor", "full_name": "Estrogen Receptor (LBD)", "type": "NR"},
    "NR-PPAR-gamma": {"category": "Nuclear Receptor", "full_name": "PPAR-gamma Receptor", "type": "NR"},
    "SR-ARE": {"category": "Stress Response", "full_name": "Antioxidant Response Element", "type": "SR"},
    "SR-ATAD5": {"category": "Stress Response", "full_name": "Genotoxicity (ATAD5)", "type": "SR"},
    "SR-HSE": {"category": "Stress Response", "full_name": "Heat Shock Element", "type": "SR"},
    "SR-MMP": {"category": "Stress Response", "full_name": "Mitochondrial Membrane Potential", "type": "SR"},
    "SR-p53": {"category": "Stress Response", "full_name": "p53 Pathway Activation", "type": "SR"},
}


def validate_smiles(smiles: str) -> Tuple[bool, Optional[Chem.Mol], Optional[str]]:
    """
    Validate SMILES string using RDKit.
    Returns: (is_valid, mol, error_message)
    """
    if not smiles or not smiles.strip():
        return False, None, "SMILES string is empty."
    
    clean_smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, None, f"RDKit could not parse SMILES '{clean_smiles}'. Please verify chemical syntax."
        if mol.GetNumAtoms() == 0:
            return False, None, "SMILES must contain at least one atom."
        return True, mol, None
    except Exception as e:
        return False, None, f"Error parsing SMILES: {str(e)}"


def render_molecule_image(mol: Chem.Mol, size: Tuple[int, int] = (380, 280)):
    """Render a 2D depiction of the molecule using RDKit."""
    return Draw.MolToImage(mol, size=size)


def smiles_to_graph_data(smiles: str):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Uses from_smiles, exactly matching the 9-dimensional node features
    and connectivity from MoleculeNet Tox21 training pipeline.
    """
    clean_smiles = smiles.strip()
    valid, _, error = validate_smiles(clean_smiles)
    if not valid:
        raise ValueError(error)
    data = from_smiles(clean_smiles)
    if data is None or data.x is None:
        raise ValueError(f"Failed to generate molecular graph for '{clean_smiles}'.")
    return data


def load_toxgraph_checkpoint(checkpoint_path: str = "checkpoints/toxgraph_best.pt",
                             device: str = "cpu"):
    """
    Load the frozen ToxGraph model checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    model = get_model(
        name="toxgraph",
        num_node_features=9,
        num_tasks=12,
        hidden_dim=128,
        dropout=0.3
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_toxicity_with_gates(model, data, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with the frozen ToxGraph model.
    Returns:
        probabilities: (12,) array of predicted probabilities in [0, 1]
        gates: (12, 128) array of learned task-specific gate activations in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # GraphSAGE encoder backbone
        x = model.conv1(x, edge_index)
        x = F.relu(x)
        x = model.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pool
        h = global_mean_pool(x, batch)
        h = F.dropout(h, p=model.dropout, training=False)

        gates = []
        logits = []
        for t in range(model.num_tasks):
            # Dynamic task-specific gating: g_t = sigmoid(W_t h + b_t)
            g_t = torch.sigmoid(model.gate_linears[t](h))
            h_t = h * g_t
            logit_t = model.task_heads[t](h_t)
            gates.append(g_t)
            logits.append(logit_t)

        all_logits = torch.cat(logits, dim=-1)  # shape: (1, 12)
        all_gates = torch.stack(gates, dim=1)   # shape: (1, 12, 128)

        probs = torch.sigmoid(all_logits).squeeze(0).cpu().numpy()
        gates_np = all_gates.squeeze(0).cpu().numpy()

    return probs, gates_np
