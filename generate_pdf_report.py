"""
generate_pdf_report.py — Compiles the final scientific report into a polished,
publication-ready 4-page PDF using ReportLab Platypus.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "ToxGraph: Task-Aware GNNs for Multi-Assay Molecular Toxicity Prediction")
            self.drawRightString(612 - 54, 755, "Deep Learning Hackathon Final Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 612 - 54, 750)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 34, "Research Demonstration — Tox21 Multi-Task Molecular Benchmark")
        self.drawRightString(612 - 54, 34, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf_report(output_filename="REPORT.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=12,
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )

    bold_body = ParagraphStyle(
        'BoldBody_Custom',
        parent=body_style,
        fontName='Helvetica-Bold',
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        spaceAfter=3,
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.white,
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#92400E"),
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("ToxGraph: Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction", title_style))
    story.append(Paragraph("<b>Track:</b> Graph Neural Networks (GNNs) | <b>Benchmark:</b> Tox21 (MoleculeNet) | <b>Architecture:</b> Task-Gated GraphSAGE", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3A8A"), spaceAfter=8))

    # Section 1: Problem & Motivation
    story.append(Paragraph("1. Problem & Motivation", h1_style))
    story.append(Paragraph(
        "Molecular toxicity prediction is a cornerstone of in silico chemical safety assessment and early-stage drug discovery. "
        "Biochemically, toxicity is inherently <b>multi-endpoint</b>: a single molecule may potently bind estrogen or androgen receptors while "
        "leaving antioxidant or stress-response pathways unaffected. Standard graph neural network (GNN) classifiers routinely collapse "
        "an entire molecule into a single shared graph embedding <b>h</b> &in; R<sup>D</sup> that is passed to independent linear classifiers. "
        "This shared-subspace paradigm imposes a rigid inductive bias, forcing all biological endpoints to evaluate identical latent features.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Research Question:</b> <i>Can task-aware graph representations improve multi-assay molecular toxicity prediction by allowing "
        "each endpoint to dynamically reweight latent molecular features from a shared graph encoder?</i> To address this, we introduce <b>ToxGraph</b>, "
        "a multi-task architecture featuring dynamic, input-conditioned feature gates conditioned on GraphSAGE molecular embeddings.",
        body_style
    ))

    # Section 2: Dataset & Experimental Protocol
    story.append(Paragraph("2. Dataset & Experimental Protocol", h1_style))
    story.append(Paragraph(
        "We evaluate on the <b>Tox21</b> benchmark from MoleculeNet (via PyTorch Geometric):",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Scale & Tasks:</b> 7,823 small molecules across 12 binary toxicity endpoints (7 Nuclear Receptor [NR] and 5 Stress Response [SR] assays).", bullet_style))
    story.append(Paragraph("&bull; <b>Graph Features:</b> 9-dimensional atom features (atomic number, degree, hybridization, formal charge, aromaticity, etc.) and 3-dimensional bond features.", bullet_style))
    story.append(Paragraph("&bull; <b>Missing Labels:</b> 16,012 measurements (17.1%) are missing (encoded as NaN) and strictly masked out during loss and metric calculation.", bullet_style))
    story.append(Paragraph("&bull; <b>Deterministic Split:</b> Fixed 80/10/10 split with <code>seed=42</code>: Train = 6,258, Validation = 782, Test = 783. Locked in <code>data/split_indices.json</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>Metric:</b> Mean ROC-AUC computed strictly over valid (non-masked) tasks with single-class edge case safety.", bullet_style))
    story.append(Paragraph("&bull; <b>Test Set Protocol:</b> The test set was held strictly untouched throughout all development and model selection. It was evaluated <b>exactly once</b> on frozen checkpoints.", bullet_style))

    # Section 3: Methods
    story.append(Paragraph("3. Methods", h1_style))
    story.append(Paragraph(
        "We benchmark standard baselines against the proposed task-aware ToxGraph architecture:",
        body_style
    ))
    story.append(Paragraph("<b>1. 2-Layer GCN:</b> Baseline spectral graph convolution + global mean pooling + linear multi-task classifier (19,340 parameters).", bullet_style))
    story.append(Paragraph("<b>2. GraphSAGE:</b> Spatial inductive message passing with mean neighborhood aggregation (36,876 parameters). Strongest baseline on validation.", bullet_style))
    story.append(Paragraph("<b>3. GAT:</b> 2-layer Graph Attention Network with 4 heads (32-d per head, ELU) computing anisotropic neighbor weights (19,852 parameters).", bullet_style))
    story.append(Paragraph(
        "<b>4. Full ToxGraph (Proposed):</b> A shared 2-layer GraphSAGE encoder extracts graph embedding <b>h</b> &in; R<sup>128</sup>. "
        "For each endpoint <i>t</i> &in; {1, ..., 12}, a dedicated gate module computes a dynamic, molecule-dependent feature mask: "
        "<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>g</b><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>t</i></sub><b>h</b> + <b>b</b><sub><i>t</i></sub>), &nbsp;&nbsp; "
        "<b>h</b><sub><i>t</i></sub> = <b>h</b> &odot; <b>g</b><sub><i>t</i></sub>, &nbsp;&nbsp; "
        "logit<sub><i>t</i></sub> = <b>w</b><sub><i>t</i></sub><sup>T</sup><b>h</b><sub><i>t</i></sub> + <i>c</i><sub><i>t</i></sub><br/>"
        "where <b>g</b><sub><i>t</i></sub> is task-specific, input-conditioned, and trained end-to-end via masked BCE loss (235,020 parameters).",
        body_style
    ))

    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    story.append(Paragraph("4. Ablation Study: Validating the Gating Mechanism", h1_style))
    story.append(Paragraph(
        "To rigorously investigate whether input-dependent gating is the causal factor behind performance changes, we examined four variations:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>GraphSAGE:</b> Backbone without task gating (single shared projection).", bullet_style))
    story.append(Paragraph("&bull; <b>ToxGraph-Lite:</b> Static learned task vectors <b>g</b><sub><i>t</i></sub> = &sigma;(<b>a</b><sub><i>t</i></sub>), constant across all molecules (38,412 params).", bullet_style))
    story.append(Paragraph("&bull; <b>ToxGraph-Bottleneck:</b> Compressed gating via shared bottleneck <b>z</b> &in; R<sup>32</sup>, saving 61% params (91,692 params).", bullet_style))
    story.append(Paragraph("&bull; <b>Full ToxGraph:</b> Direct molecule-conditioned linear gating (235,020 params).", bullet_style))

    # Validation Table
    val_table_data = [
        [Paragraph("Model", table_header), Paragraph("Gating Mechanism", table_header), Paragraph("Params", table_header), Paragraph("Val ROC-AUC", table_header), Paragraph("&Delta; vs. SAGE", table_header)],
        [Paragraph("GCN", table_cell), Paragraph("None (Baseline)", table_cell), Paragraph("19,340", table_cell), Paragraph("0.7383", table_cell), Paragraph("-0.0470", table_cell)],
        [Paragraph("GAT", table_cell), Paragraph("Attention (Shared)", table_cell), Paragraph("19,852", table_cell), Paragraph("0.7610", table_cell), Paragraph("-0.0243", table_cell)],
        [Paragraph("ToxGraph-Lite", table_cell), Paragraph("Static Vectors &sigma;(<b>a</b><sub>t</sub>)", table_cell), Paragraph("38,412", table_cell), Paragraph("0.7691", table_cell), Paragraph("-0.0162", table_cell)],
        [Paragraph("ToxGraph-Bottleneck", table_cell), Paragraph("Bottleneck Input-Conditioned", table_cell), Paragraph("91,692", table_cell), Paragraph("0.7815", table_cell), Paragraph("-0.0038", table_cell)],
        [Paragraph("GraphSAGE", table_cell), Paragraph("None (Backbone)", table_cell), Paragraph("36,876", table_cell), Paragraph("0.7853", table_cell), Paragraph("Baseline (0.0000)", table_cell)],
        [Paragraph("<b>Full ToxGraph</b>", table_cell), Paragraph("<b>Direct Input-Conditioned</b>", table_cell), Paragraph("<b>235,020</b>", table_cell), Paragraph("<b>0.7890</b>", table_cell), Paragraph("<b>+0.0037</b>", table_cell)],
    ]
    t_val = Table(val_table_data, colWidths=[95, 155, 65, 80, 85])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Ablation Finding:</b> Static task gates underperformed GraphSAGE (0.7691 vs 0.7853), proving that static endpoint weighting hurts representation capacity. "
        "In contrast, direct input-conditioned gating achieved the strongest validation score (0.7890), qualifying Full ToxGraph as our final proposed model.",
        body_style
    ))

    # Section 5: Final Test Results
    story.append(Paragraph("5. Final Test Results (Frozen Single Evaluation)", h1_style))
    story.append(Paragraph(
        "All models were frozen at validation-selected checkpoints and evaluated exactly once on the test split (N = 783).",
        body_style
    ))

    test_table_data = [
        [Paragraph("Model", table_header), Paragraph("Params", table_header), Paragraph("Selected Epoch", table_header), Paragraph("Val ROC-AUC", table_header), Paragraph("Test ROC-AUC", table_header), Paragraph("Test &Delta; vs. SAGE", table_header)],
        [Paragraph("GCN", table_cell), Paragraph("19,340", table_cell), Paragraph("58", table_cell), Paragraph("0.7383", table_cell), Paragraph("0.7605", table_cell), Paragraph("-0.0284", table_cell)],
        [Paragraph("GAT", table_cell), Paragraph("19,852", table_cell), Paragraph("57", table_cell), Paragraph("0.7610", table_cell), Paragraph("0.7853", table_cell), Paragraph("-0.0036", table_cell)],
        [Paragraph("ToxGraph-Lite", table_cell), Paragraph("38,412", table_cell), Paragraph("57", table_cell), Paragraph("0.7691", table_cell), Paragraph("0.7842", table_cell), Paragraph("-0.0047", table_cell)],
        [Paragraph("GraphSAGE", table_cell), Paragraph("36,876", table_cell), Paragraph("57", table_cell), Paragraph("0.7853", table_cell), Paragraph("0.7889", table_cell), Paragraph("Baseline (0.0000)", table_cell)],
        [Paragraph("<b>Full ToxGraph (Proposed)</b>", table_cell), Paragraph("<b>235,020</b>", table_cell), Paragraph("<b>59</b>", table_cell), Paragraph("<b>0.7890</b>", table_cell), Paragraph("<b>0.7939</b>", table_cell), Paragraph("<b>+0.0050</b>", table_cell)],
        [Paragraph("ToxGraph-Bottleneck", table_cell), Paragraph("91,692", table_cell), Paragraph("57", table_cell), Paragraph("0.7815", table_cell), Paragraph("0.7997", table_cell), Paragraph("+0.0108", table_cell)],
    ]
    t_test = Table(test_table_data, colWidths=[130, 55, 75, 75, 75, 80])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_test)
    story.append(Spacer(1, 4))

    callout_data = [[
        Paragraph("<b>Transparent Protocol Disclosure:</b> Full ToxGraph was selected based on validation performance prior to test evaluation. "
                  "Although ToxGraph-Bottleneck achieved the highest test ROC-AUC (0.7997), scientific integrity dictates that model selection "
                  "must not be performed post-hoc on the test set. Full ToxGraph remains our designated proposed architecture.", callout_style)
    ]]
    t_callout = Table(callout_data, colWidths=[490])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_callout)

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    story.append(Paragraph("Per-Task Performance & Statistical Robustness", h1_style))
    story.append(Paragraph(
        "Full ToxGraph improved over GraphSAGE on <b>8 of 12 test endpoints</b> (66.7% win rate), with strongest gains observed on: "
        "<b>SR-ATAD5</b> (+0.0288), <b>NR-PPAR-gamma</b> (+0.0277), <b>NR-ER</b> (+0.0150), and <b>NR-ER-LBD</b> (+0.0124).",
        body_style
    ))

    fig_path1 = "results/figures/graphsage_vs_toxgraph_per_task.png"
    if os.path.exists(fig_path1):
        img1 = Image(fig_path1, width=470, height=200)
        story.append(img1)
        story.append(Spacer(1, 4))

    # Section 6: Statistical Analysis
    story.append(Paragraph("6. Statistical Analysis: Bootstrap Resampling", h1_style))
    story.append(Paragraph(
        "To rigorously quantify whether the +0.0050 test ROC-AUC gain reflects authentic model superiority or test-sample variance, "
        "we executed a 1,000-iteration non-parametric bootstrap resampling on the test set:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Mean Test Difference &Delta;:</b> +0.0050 (Median &Delta; = +0.0051)", bullet_style))
    story.append(Paragraph("&bull; <b>95% Bootstrap Confidence Interval:</b> [-0.0073, +0.0174]", bullet_style))
    story.append(Paragraph("&bull; <b>Empirical p-value (H<sub>0</sub>: &Delta; &le; 0):</b> <i>p</i> = 0.1980", bullet_style))
    story.append(Paragraph(
        "<b>Scientific Rigor Statement:</b> <i>The observed improvement is positive across the majority of endpoints but is not statistically significant "
        "at conventional scientific thresholds (p = 0.198 > 0.05).</i> Because the 95% confidence interval spans zero, the result must be characterized "
        "as an encouraging positive empirical trend rather than a proven statistical superiority.",
        body_style
    ))

    # Section 7: Interpretability
    story.append(Paragraph("7. Interpretability: Task-Aware Feature Gating", h1_style))
    story.append(Paragraph(
        "Analysis of test-time gate activations <b>g</b><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>t</i></sub><b>h</b> + <b>b</b><sub><i>t</i></sub>) "
        "revealed non-degenerate representations (activations span 0.0013 to 0.9999) with distinct endpoint selectivity:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Top Divergent Latent Dimensions:</b> Features 56, 77, 13, 112, and 115 exhibited the highest cross-task gate variance, indicating active subspace modulation.", bullet_style))
    story.append(Paragraph("&bull; <b>Biological Subspace Alignment:</b> Related estrogen receptor endpoints (NR-ER and NR-ER-LBD) exhibited highly correlated gate patterns (r = 0.65). "
                           "<i>The similar gate profiles of NR-ER and NR-ER-LBD are consistent with related endpoints using overlapping latent representation subspaces.</i>", bullet_style))
    story.append(Paragraph("<b>Scientific Note:</b> Latent dimensions represent continuous neural embeddings and must <b>not</b> be claimed to correspond directly to chemical functional groups.", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 4 ====================
    story.append(Paragraph("Gating Patterns & Conclusions", h1_style))

    fig_path2 = "results/figures/task_gate_heatmap.png"
    if os.path.exists(fig_path2):
        img2 = Image(fig_path2, width=470, height=175)
        story.append(img2)
        story.append(Spacer(1, 4))

    # Section 8: Conclusion & Limitations
    story.append(Paragraph("8. Conclusion, Limitations & Future Work", h1_style))
    story.append(Paragraph(
        "<b>Conclusion:</b> Task-conditioned molecular representations produced a modest empirical improvement over the shared GraphSAGE backbone "
        "(+0.0050 test ROC-AUC) across 8 of 12 toxicity endpoints. Dynamic input-conditioning is essential, as static gates failed to match baseline performance.",
        body_style
    ))
    story.append(Paragraph("<b>Experimental Limitations:</b>", h2_style))
    story.append(Paragraph("1. <i>Sample Size & Imbalance:</i> Test set has 783 molecules; assays like NR-PPAR-gamma possess only 18 active instances (2.76%), inflating variance.", bullet_style))
    story.append(Paragraph("2. <i>Single Split:</i> Evaluated on one fixed random split (seed=42). Multi-seed scaffold splits are needed to evaluate out-of-distribution generalization.", bullet_style))
    story.append(Paragraph("3. <i>Statistical Significance:</i> The empirical improvement is not statistically significant at p < 0.05 (p = 0.198).", bullet_style))
    story.append(Paragraph("4. <i>Parameter Cost:</i> Full ToxGraph requires 235k parameters. Bottleneck gating (91k params) offers a highly attractive trade-off for resource-constrained settings.", bullet_style))
    story.append(Paragraph("5. <i>Latent Explainability:</i> Gating modulates internal GNN dimensions rather than providing direct atom-level explanations.", bullet_style))

    story.append(Paragraph("<b>Future Work:</b>", h2_style))
    story.append(Paragraph("&bull; Evaluate on expanded benchmarks (ToxCast, ClinTox) using scaffold-constrained cross-validation.", bullet_style))
    story.append(Paragraph("&bull; Combine task gating with integrated gradients or GNNExplainer for atom-level toxicophore attribution.", bullet_style))
    story.append(Paragraph("&bull; Explore mixture-of-experts (MoE) routing for scalable task-aware molecular representation learning.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    story.append(Paragraph(
        "<b>Interactive Demonstration & Artifacts:</b> All source code, frozen checkpoints, and experiment data are fully reproducible. "
        "Run <code>python evaluate_test.py</code> for frozen test benchmarking, or launch the interactive application via <code>streamlit run app.py</code>.",
        body_style
    ))

    # Build the document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF report successfully compiled to {output_filename}")


if __name__ == "__main__":
    build_pdf_report("REPORT.pdf")
