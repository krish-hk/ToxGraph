"""
app.py — Interactive Streamlit Demonstration for ToxGraph.

ToxGraph: Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction.
Uses the frozen Full ToxGraph checkpoint to perform multi-task toxicity inference and
visualize dynamic task-specific gating patterns.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.data import TOX21_TASK_NAMES
from src.inference import (
    validate_smiles,
    render_molecule_image,
    smiles_to_graph_data,
    load_toxgraph_checkpoint,
    predict_toxicity_with_gates,
    TASK_METADATA,
)

# Page configuration
st.set_page_config(
    page_title="ToxGraph — Molecular Toxicity Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4B5563;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #92400E;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model(checkpoint_path: str = "checkpoints/toxgraph_best.pt"):
    """Load and cache the frozen ToxGraph model checkpoint."""
    return load_toxgraph_checkpoint(checkpoint_path=checkpoint_path, device="cpu")


def main():
    # Header Section
    st.markdown('<div class="main-title">🧪 ToxGraph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "ToxGraph predicts multiple biological toxicity assay responses directly from molecular graph structure. "
        "By combining a shared 2-layer GraphSAGE backbone with **dynamic task-aware gating modules**, ToxGraph "
        "reweights the shared molecular representation specifically for each biological endpoint."
    )

    st.markdown("---")

    # Preset benchmark examples for quick review
    preset_examples = {
        "Custom SMILES": "",
        "Ethanol (Alcohol)": "CCO",
        "Aspirin (Analgesic)": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine (Stimulant)": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Acetaminophen (Analgesic)": "CC(=O)Nc1ccc(O)cc1",
        "Nicotine (Alkaloid)": "CN1CCC[C@H]1c2cccnc2",
    }

    # Sidebar: Model status & Example Selector
    with st.sidebar:
        st.header("⚙️ Experiment Controls")
        
        # Example Selector
        selected_example = st.selectbox(
            "Select Example Molecule:",
            options=list(preset_examples.keys()),
            index=3, # Default to Caffeine
            help="Choose a pre-defined common molecule or select 'Custom SMILES' to enter your own."
        )

        st.markdown("---")
        st.subheader("Model Status")
        checkpoint_path = "checkpoints/toxgraph_best.pt"
        if os.path.exists(checkpoint_path):
            st.success("✅ Frozen ToxGraph Checkpoint Loaded\n\n`checkpoints/toxgraph_best.pt`")
        else:
            st.error("❌ Checkpoint `checkpoints/toxgraph_best.pt` not found!")

        st.markdown("""
        **Architecture Summary**:
        - **Backbone**: 2-layer GraphSAGE (128-d)
        - **Gating**: Dynamic Linear Gates per task
        - **Parameters**: 235,020
        - **Test ROC-AUC**: **0.7939** (vs. SAGE 0.7889)
        - **Endpoints**: 12 Tox21 assays
        """)

        st.markdown("---")
        st.info("💡 **Note**: Common examples are provided for user convenience only. No biological conclusions should be attached to individual examples.")

    # Determine default SMILES from selection
    initial_smiles = preset_examples[selected_example]
    if not initial_smiles and "current_smiles" in st.session_state:
        initial_smiles = st.session_state["current_smiles"]

    # Input Section
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        smiles_input = st.text_input(
            "Enter SMILES String:",
            value=initial_smiles,
            placeholder="e.g. Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            help="Input a valid chemical SMILES representation."
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("🔬 Analyze Molecule", type="primary", use_container_width=True)

    # Perform analysis if button clicked or if example changed
    should_run = analyze_clicked or (smiles_input and selected_example != "Custom SMILES")

    if should_run:
        st.session_state["current_smiles"] = smiles_input

        # 1. Validation
        is_valid, mol, error_msg = validate_smiles(smiles_input)
        if not is_valid:
            st.error(f"⚠️ Validation Error: {error_msg}")
            return

        # 2. Molecular Properties & Rendering
        col_mol_img, col_mol_info = st.columns([1, 1])
        with col_mol_img:
            st.subheader("Molecular 2D Depiction")
            img = render_molecule_image(mol, size=(380, 260))
            st.image(img, use_container_width=False, caption=f"SMILES: {smiles_input}")

        with col_mol_info:
            st.subheader("Chemical Summary")
            mol_wt = Descriptors.MolWt(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_rings = Descriptors.RingCount(mol)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            c1, c2 = st.columns(2)
            c1.metric("Formula", formula)
            c2.metric("Molecular Weight", f"{mol_wt:.2f} g/mol")
            c3, c4 = st.columns(2)
            c3.metric("Heavy Atoms", num_atoms)
            c4.metric("Total Bonds", num_bonds)

        # 3. Model Inference
        try:
            model, _ = get_cached_model(checkpoint_path)
        except Exception as e:
            st.error(f"Failed to load model checkpoint: {str(e)}")
            return

        try:
            graph_data = smiles_to_graph_data(smiles_input)
        except Exception as e:
            st.error(f"Failed to convert molecule to graph representation: {str(e)}")
            return

        with st.spinner("Executing ToxGraph inference and extracting task gates..."):
            probs, gates = predict_toxicity_with_gates(model, graph_data, device="cpu")

        st.markdown("---")

        # 4. Results Section: Table & Sorted Bar Chart
        st.subheader("📊 Multi-Assay Toxicity Profile")

        # Build results DataFrame
        records = []
        for i, task in enumerate(TOX21_TASK_NAMES):
            meta = TASK_METADATA.get(task, {"category": "Unknown", "full_name": task, "type": "Unknown"})
            records.append({
                "Assay": task,
                "Target Description": meta["full_name"],
                "Category": meta["category"],
                "Predicted Probability of Assay Activity": float(probs[i]),
            })
        
        df_results = pd.DataFrame(records)
        df_sorted = df_results.sort_values(by="Predicted Probability of Assay Activity", ascending=False).reset_index(drop=True)

        tab_chart, tab_table = st.tabs(["📈 Sorted Probability Chart", "📋 Complete Assay Table"])

        with tab_chart:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color by Category: Nuclear Receptor (SteelBlue) vs Stress Response (IndianRed)
            colors = [
                '#3B82F6' if cat == 'Nuclear Receptor' else '#EF4444'
                for cat in df_sorted['Category']
            ]
            
            y_pos = np.arange(len(df_sorted))
            bars = ax.barh(y_pos, df_sorted['Predicted Probability of Assay Activity'], color=colors, height=0.65, edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_sorted['Assay'], fontsize=10, fontweight='bold')
            ax.invert_yaxis()  # Highest probability at top
            ax.set_xlabel("Predicted Probability of Assay Activity", fontweight='bold', fontsize=11)
            ax.set_xlim(0.0, max(1.0, float(df_sorted['Predicted Probability of Assay Activity'].max() * 1.15)))
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Value labels on bars
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}",
                        ha='left', va='center', fontsize=9, fontweight='bold')

            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3B82F6', edgecolor='black', label='Nuclear Receptor (NR)'),
                Patch(facecolor='#EF4444', edgecolor='black', label='Stress Response (SR)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', frameon=True)
            ax.set_title("Predicted Probability of Assay Activity (Sorted)", fontweight='bold', fontsize=12, pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab_table:
            # Format dataframe for display
            display_df = df_sorted.copy()
            display_df["Predicted Probability of Assay Activity"] = display_df["Predicted Probability of Assay Activity"].map("{:.4f}".format)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 5. Novelty Visualization: Task-Aware Feature Gating Heatmap
        st.markdown("---")
        st.subheader("🧠 Task-Aware Feature Gating")
        st.markdown(
            "ToxGraph dynamically reweights the shared molecular representation differently for each toxicity endpoint. "
            "Below is the exact **$12 \\times 128$ task gate activation matrix** $\\mathbf{g}_t = \\sigma(\\mathbf{W}_t \\mathbf{h} + \\mathbf{b}_t)$ "
            "computed dynamically for this submitted molecule."
        )

        fig_gate, ax_gate = plt.subplots(figsize=(14, 5))
        im = ax_gate.imshow(gates, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
        cbar = fig_gate.colorbar(im, ax=ax_gate)
        cbar.set_label(r'Gate Activation $\sigma(W_t h + b_t)$', fontweight='bold')
        
        ax_gate.set_yticks(np.arange(len(TOX21_TASK_NAMES)))
        ax_gate.set_yticklabels(TOX21_TASK_NAMES, fontsize=9, fontweight='bold')
        ax_gate.set_xlabel("Latent GraphSAGE Dimension (0 to 127)", fontweight='bold', fontsize=10)
        ax_gate.set_ylabel("Toxicity Endpoint", fontweight='bold', fontsize=10)
        ax_gate.set_title("Task-Aware Feature Gating Matrix (Molecule-Specific)", fontweight='bold', fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig_gate)
        plt.close()

        st.caption(
            "💡 *Interpretation Note: Learned gate activations modulate continuous graph neural embeddings. "
            "They illustrate dynamic task-specific subspace selection and are not claimed to map directly "
            "to specific chemical functional groups.*"
        )

        # 6. Prominent Scientific & Safety Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Scientific Disclaimer:</strong><br>
            Research demonstration only. Predictions are model outputs and are not a substitute for experimental toxicology or safety assessment.
            Predicted values indicate the model's estimated probability of assay activity under in vitro Tox21 screening protocols.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
