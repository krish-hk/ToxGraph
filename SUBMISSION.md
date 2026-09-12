# ToxGraph submission audit and judge preparation

## Scope and outcome

Audited the authoritative local working copy against the complete supplied hackathon DOCX. The attachment supplies requirements, not permission to alter experimental history. No architecture, feature encoding, forward pass, optimizer, training loss, ROC-AUC calculation, bootstrap calculation or frozen prediction was changed. No retraining, model search or full test reevaluation was performed. Changes are documentation, packaging, validation guards, output routing and demo presentation/input handling.

**CRITICAL scientific invalidation issues found: none established.** This is a bounded code/artifact audit, not proof of historical experimental conduct. The saved indices cover all 7,823 molecules exactly once without overlap, and checkpoint metadata agrees with frozen summaries and histories.

**Submission blockers still external:** the required recorded demo is absent; team membership/submitting lead, deadline date, upload destination and any separately distributed official split require verification. The brief says 5pm but has no date and leaves the upload platform as a placeholder. Do not assert these administrative items are complete.

## Rubric scorecard before and after

This is an evidence/readiness scorecard, not an official score prediction. Scientific performance did not change. The rubric gives weights but no detailed numerical scoring anchors; assigning precise earned points would suggest unsupported certainty.

| Scored criterion | Weight | Before | After | Remaining constraint |
|---|---:|---|---|---|
| Problem quality | 20% | Relevant task; overgeneralized shared-head criticism | Clear multi-endpoint question, scope and abstract | Assay activity is not organism-level toxicity |
| Beats mandatory baseline | 20% | GCN result present, improvement buried | Mandatory 2-layer GCN and +0.0334 test gain explicit at top | Only one fixed random split |
| Depth of solution | 20% | Six architectures and analyses present | Code-verified layers/parameters, equations and ablation ladder | No parameter-matched nonlinear-head control; unused edge attributes |
| Experimental rigor | 15% | Masking and seed present; unsafe overwrite defaults, ignored evidence, no single-command entry | Isolated reproduction wrapper, hash checks, split validation, transparent uncertainty | No fresh full reproduction; original environment provenance incomplete |
| Originality of implementation | 15% | Mechanism explained; static-gate capacity claim overstated | Precise input-conditioned contribution, citations, confounders acknowledged | No comprehensive novelty search or globally novel claim |
| Demo / presentation | 10% | Functional live demo, report without references | Checked default/custom/error flows, four-page report, concise script and Q&A | Required recorded demo still absent |

## Mandatory and administrative checklist

- Track 3 / Graph Neural Networks and Tox21/MoleculeNet graph-level prediction explicitly declared in README/report.
- Required 2-layer GCN, GAT and GraphSAGE implemented and compared using test ROC-AUC.
- Beat baseline: Full 0.7939 versus GCN 0.7605. Primary selection was validation-based.
- Seed = 42, shared fixed split, train-only gradients, validation checkpoint selection; no test loader in training.
- One-command end-to-end entry after installation: `python reproduce.py`; six training jobs then evaluation and figures. Dry-run validated, not retrained.
- Four-page PDF includes problem, method, table with baseline, ablation, limitations and references; every rendered page inspected.
- Code and original small evidence files are now eligible for inclusion in Git. Changes have not been committed, pushed or uploaded by this audit.
- Recorded demo: absent. Record the sequence below and upload it alongside the repository and report.
- Existing course-project team and lead, original-work compliance, submission time/platform and official split equivalence: external verification, not inferred from local code.
- Cite all additional borrowed reference code or assistance actually used by the team; this audit cannot reconstruct unrecorded provenance. No pretrained model is used by the inspected implementation.

## High-impact findings and fixes

1. **Missing one-command workflow:** added `reproduce.py`, preserving the six original 60-epoch configurations, seed and selection-before-test sequence. It uses separate output directories and records reproduction selection before evaluation.
2. **Evidence omitted from repository:** `.gitignore` had excluded checkpoints, all training histories and split indices. Narrow exceptions now include those files (about 6 MB of checkpoints plus histories). `.gitattributes` preserves binary files and frozen JSON/CSV bytes across checkouts. Nothing was published.
3. **Overwrite hazards:** `train.py:main` now refuses existing model output files and defaults to `runs/manual`; `evaluate_test.py:main` refuses an existing output root and writes to `runs/evaluation`. A rerun cannot silently replace the shipped metrics under default commands.
4. **Split validation gaps:** `src/data.py:load_split` formerly accepted unchecked saved indices; `verify_dataset` claimed no overlap and test isolation without testing either. Saved indices now undergo exact coverage/uniqueness/integer checks. Training rejects seed metadata conflicts. The misleading runtime claim of historical test isolation was removed.
5. **Evaluator fresh-run failure:** `evaluate_test.py:generate_figures` now creates its output directory; all six checkpoints are required before evaluation, instead of silently skipping a missing model and later failing on dependent analysis. Configurable history paths ensure reproduction curves use reproduction histories.
6. **Scientific overclaims:** removed statements that static gates prove diminished capacity, input conditioning is essential, bootstrap results establish practical utility, or gate correlations prove biology. Linear task heads already have endpoint-specific weights; static gates can be absorbed into those weights. Parameter count and optimization remain confounders.
7. **Featurization documentation:** report had listed implicit valence and omitted ring membership. Corrected to the implemented PyG fields and explicitly disclosed unused bond attributes and numeric categorical encoding. No featurization code changed.
8. **Protocol language:** code prints whole-dataset label summaries (`train.py:main`, `src/data.py:inspect_missing_labels`, `verify_dataset`). This is not test-gradient training or score tuning, but contradicts a literal claim that test labels were never inspected. Documented openly; diagnostic behavior was not rewritten to erase history.
9. **Statistics wording:** empirical p = 0.198 is a bootstrap lower-tail fraction, not a calibrated null-hypothesis test. Four-decimal metric quantization in bootstrap/selection is preserved and documented. No significance claim is made.
10. **Demo reliability:** `src/inference.py:smiles_to_graph_data` now rejects invalid/empty graphs before PyG can silently create an empty molecule. Restricted checkpoint loading (`weights_only=True`) works with all six originals. The app resolves its checkpoint relative to its own file, differentiates checkpoint presence from successful loading, and displays actual loaded epoch/validation metadata. A stable SMILES widget key and preset callback fix invalid-input recovery that previously reset edited text before analysis.
11. **Report drift and attribution:** old PDF generator duplicated prose and could disagree with REPORT.md. It now renders Markdown as the single source, enforces a maximum of four pages, preserves the result table, and includes baseline/dataset/library citations.
12. **Verified correlation reporting mismatch:** the unchanged `results/figures/task_gate_similarity_heatmap.png` labels NR-ER/NR-ER-LBD at approximately r = 0.25. An older draft had erroneously described this relationship as strongly correlated (~0.65 / r > 0.60). Corrected all prose across README, report and presentation to weak positive similarity supported by the figure (r ≈ 0.25). This affects interpretability wording only, not classifier metrics, predictions or model selection. No numerical correlation was recomputed.
13. **Reproducibility visibility:** added observed environment lock, copy-paste commands, explicit output paths, frozen hash manifest and a read-only verification command. Original training-environment provenance is not invented.

## Optional polish completed

README begins with an AI-screening-friendly declaration and compact compliance matrix. Full and Bottleneck selection roles are explicit in README, report and app. The app labels outputs predicted assay activity probabilities, identifies probabilities as uncalibrated research estimates and retains its latent-gate disclaimer. The report has readable tables, a preserved per-endpoint figure, aligned page numbering and no layout overflow. This file provides reusable abstract, novelty wording, presentation script and defensible Q&A.

## Validation evidence

- All six checkpoints load with `weights_only=True`; parameters, selected epochs and validation AUC match histories and final JSON.
- All 15 frozen artifact SHA-256 hashes match the pre-edit manifest: six checkpoints, six histories, final JSON/CSV and split file.
- Saved split: 6,258 / 782 / 783, seed 42, exhaustive and disjoint. Dataset inspection confirmed 7,823 graphs and 16,012 NaN entries.
- Sampled train/inference graph construction agrees for four dataset rows (0, 1, 100, 1000). This is a sample check, not full dataset identity verification.
- Masked-label gradient is zero; single-class and all-missing metric tasks are skipped. No metric formula changed.
- Ethanol, aspirin, caffeine and a single-atom sodium graph produce finite 12-element probabilities and 12 × 128 gates. Demo helper probabilities exactly equal direct model forward probabilities on these inputs. Invalid/blank SMILES rejected.
- Reproduction wrapper dry run prints six fixed-seed training commands and evaluation. Python syntax/import and CLI checks completed; full retraining and test reevaluation intentionally not run.
- Streamlit AppTest covers default caffeine, invalid input, recovery to custom ethanol and switching to aspirin; 12 assay rows are present on valid predictions, errors are handled visibly.
- PDF is four pages; every rendered page visually reviewed. No frozen figure was regenerated.

## Files changed and concise changelog

| File | Change |
|---|---|
| README.md | Rebuilt submission overview, compliance matrix, exact commands and scientific framing |
| REPORT.md | Four-page scientific narrative, corrected feature/ablation/protocol claims and citations |
| REPORT.pdf | Regenerated four-page report from corrected source |
| generate_pdf_report.py | Single-source Markdown renderer with page-limit guard |
| app.py | Path reliability, stable SMILES/preset state, accurate status/metadata and cautious result presentation |
| src/inference.py | Reject invalid/empty graphs; restricted checkpoint loading |
| src/data.py | Validate saved split; remove unsupported historical-isolation check |
| src/models.py | Correct module documentation only |
| train.py | Separate protected output paths and split/seed conflict guard |
| evaluate_test.py | Separate protected output paths, all-model preflight, configurable histories, mkdir, restricted loading |
| requirements.txt | Add missing ReportLab dependency |
| requirements-lock.txt (new) | Observed audit environment, CPU wheel index |
| .gitignore | Include frozen submission evidence; ignore reproduction/temp outputs |
| .gitattributes (new) | Binary handling and frozen evidence byte preservation |
| reproduce.py (new) | Seed-42 six-model train-to-evaluation orchestration |
| verify_submission.py (new) | Read-only integrity, metadata, metric and inference checks |
| submission/frozen_manifest.json (new) | Pre-edit hashes of 15 original scientific artifacts |
| SUBMISSION.md (new) | This scorecard, changelog and judge preparation pack |

Existing `checkpoints/*_best.pt`, `results/*_results.json` and `data/split_indices.json` are newly visible to Git due to ignore exceptions but **their contents are unchanged**. Final JSON/CSV and all figures remain unchanged. `src/training.py`, `src/metrics.py` and `src/utils.py` remain unchanged.

## Remaining limitations and possible AI-screening penalties

The recorded-demo requirement is not satisfied by a live app or script alone. Organizer split equivalence is unknown from the supplied document. End-to-end execution is implemented but not validated by a fresh training run in this audit. The audit lock records current packages, not verified original-training provenance, and installation was not retried in a clean environment. Reproductions can differ across hardware and library versions. There is one seed/random split, no scaffold benchmark, no capacity-matched head control, no probability calibration and no established statistical improvement over GraphSAGE. Bond attributes are unused; categorical atom codes are cast to floats. These are disclosed limitations, not reasons to change frozen numbers. Report citations cover inspected building blocks but team-level provenance must be accurate. Any uncommitted local evidence must be included in the actual submitted repository/archive.

## Recommended project title

ToxGraph: Task-Aware Graph Neural Networks for Multi-Assay Molecular Toxicity Prediction

## One-line description

A reproducible Track 3 Tox21 benchmark that tests molecule-dependent endpoint gating against GCN, GAT and GraphSAGE, with transparent ablations, uncertainty analysis and a SMILES demo.

## Final abstract

Toxicity screening involves multiple endpoints with incomplete assay coverage. ToxGraph adds molecule-dependent, endpoint-specific gates to a shared two-layer GraphSAGE encoder. We compare the mandatory two-layer GCN baseline, GAT, GraphSAGE, static gates and compressed dynamic gates on 7,823 Tox21 molecules using a fixed seed-42 split and masked multi-task learning. Full ToxGraph was selected on validation ROC-AUC (0.7890) before final test evaluation. Its test mean ROC-AUC is 0.7939 versus GCN 0.7605 and GraphSAGE 0.7889, improving 8/12 endpoints. The GraphSAGE comparison is not statistically significant: paired bootstrap 95% CI [-0.0073, +0.0174], empirical p = 0.198. Bottleneck gating achieved the highest test score (0.7997) but remains an ablation, not a retrospective primary-model selection. Gate analysis illustrates task-dependent latent reweighting without establishing chemical substructure explanations.

## Final novelty statement

ToxGraph augments a shared GraphSAGE molecular embedding with molecule-dependent, endpoint-specific sigmoid gates before separate linear assay heads. Its implementation contribution is the explicit task-conditioning design and a no-gate/static/compressed-dynamic/direct-dynamic comparison on Tox21. Unlike fixed linear endpoint weights, the gates change with the input molecule. This is not a claim that gating is globally unprecedented; unequal parameter counts and optimization effects limit causal attribution of the observed gains.

## Final results narrative

Full ToxGraph was selected before test evaluation because it achieved the highest validation mean ROC-AUC, 0.7890. On the frozen test split it achieved 0.7939, exceeding the mandatory GCN baseline (0.7605) by +0.0334 and GraphSAGE (0.7889) by +0.0050, with gains on 8 of 12 endpoints. The paired bootstrap interval [-0.0073, +0.0174] crosses zero and empirical p = 0.198; the improvement is a positive empirical trend, not statistically significant. Static gating underperformed GraphSAGE on both splits. Both dynamic variants exceeded static gating, but Bottleneck was below GraphSAGE on validation. Bottleneck later achieved the highest observed test mean, 0.7997, with 60.99% fewer parameters than Full. We report this result prominently without retrospectively selecting it as the original primary model.

## Two-to-three-minute presentation script

[0:00-0:25, problem] Toxicity is not one binary property. Tox21 measures twelve different assay endpoints, and many molecules have missing measurements. Our Track 3 project asks whether the same molecular representation should be reweighted differently for each assay. We use 7,823 molecular graphs and mask missing labels rather than treating them as negatives.

[0:25-0:55, method] We start with a two-layer GraphSAGE encoder and mean-pool it into a 128-dimensional molecular embedding. For every endpoint, a sigmoid gate depends on that molecule's embedding. We multiply the embedding by the gate and apply the endpoint's linear head. Standard heads already have different weights for each endpoint; our addition lets the reweighting change with the molecule.

[0:55-1:25, experiments] We include the mandatory two-layer GCN, plus GAT and GraphSAGE. Our ablation ladder goes from no gate to static gates, compressed dynamic gates, and full dynamic gates. Every model uses the same saved seed-42 split. Full ToxGraph was selected using validation only, with ROC-AUC 0.7890. Its test score is 0.7939 versus GCN 0.7605 and GraphSAGE 0.7889. Eight of twelve endpoints improve.

[1:25-1:50, honesty] The gain over GraphSAGE is only 0.0050. Its bootstrap confidence interval crosses zero, so we do not call it statistically significant. Bottleneck actually scores highest on test, at 0.7997. We show that prominently, but keep Full as the model selected before seeing test results. This motivates another planned study, not rewriting the selection history.

[1:50-2:20, demo] Here is caffeine as a SMILES string. The demo validates it, draws the molecule and returns all twelve predicted assay activity probabilities. These are uncalibrated research outputs, not a safety verdict. This heatmap shows each endpoint's gate over the 128 latent dimensions. Changing the molecule changes its task gates; the dimensions are not identified chemical substructures.

[2:20-2:45, close] The main result is a complete, reproducible comparison with a modest positive trend and a useful compressed-gating result. The report and source disclose one random split, parameter-count confounding and unused bond attributes. Next we would use a separately planned scaffold evaluation and capacity-matched controls. The repository provides one-command reproduction and preserves the original frozen evidence.

## Recorded-demo shot list

Record the actual running app, not a simulated interface. Show the model metadata and uncertainty sidebar; enter caffeine; show the molecule, chart and complete 12-row table; scroll to the gate heatmap; switch to ethanol; demonstrate one invalid SMILES error; finish on the frozen results table in README. Use the timed script above. Confirm that the saved video plays and include it in the required upload. A live app link alone does not satisfy the attached recorded-demo requirement.

## Judge questions and defensible answers

**What is actually novel?** The implemented addition is molecule-dependent endpoint gating over a shared molecular embedding, evaluated with static and bottleneck controls. Gating itself is established; our contribution is the particular implementation and experimental investigation, not a global first claim.

**Why not separate models for each endpoint?** Joint learning shares statistical strength and computation when labels are sparse. Separate endpoint models are a sensible untested comparison; these experiments do not prove multi-task learning is always superior.

**Why GraphSAGE?** It was the strongest baseline on validation (0.7853 versus GAT 0.7610 and GCN 0.7383), and reusing it gives a direct backbone comparison. That choice does not use test ranking.

**Why only +0.0050?** GraphSAGE is already a strong baseline, and sparse, imbalanced endpoint labels make small differences noisy. We cannot identify the cause from this one run. The mandatory GCN gain is larger (+0.0334), but the stronger baseline is the relevant challenge for the gating claim.

**Is it statistically significant?** No. The 95% molecule-bootstrap interval is [-0.0073, +0.0174]. The stored empirical p = 0.198 is the bootstrap nonpositive-delta fraction, not a calibrated null test. It also does not measure training-seed variability.

**Why does Bottleneck have the best test score?** It may have benefited from compression, optimization or sampling variation. We did not establish which explanation is correct. It achieves 0.7997 with 91,692 parameters versus Full's 235,020.

**Why not declare Bottleneck the winner?** We do label it the highest observed test scorer. Calling it the original selected primary model would use hindsight: Full won validation and was designated before test evaluation. A later model choice needs fresh evaluation data.

**Does the heatmap map to substructures?** No. It shows latent gate values. Atom attribution, perturbation tests and biological validation would be needed for chemical explanations. Gate magnitude alone also ignores embedding magnitude and head weights.

**How are missing labels handled?** NaN entries contribute neither BCE loss nor endpoint ROC-AUC. ROC-AUC skips single-class/all-missing tasks, then averages valid task AUCs. All twelve tasks are valid on the frozen test split.

**Was test used for tuning?** The documented protocol says no, and inspected code uses only train batches for gradients and validation for checkpoints. Whole-dataset label summaries are printed by historical diagnostics. We disclose that distinction and cannot prove historical invocation counts from artifacts alone.

**How reproducible is the split?** Exact saved indices are shipped and hashed. They cover all 7,823 rows once, with 6,258/782/783 partitions and seed 42. Dataset ordering and versions matter; indices alone are not a cross-version molecule identity guarantee.

**Why random rather than scaffold splitting?** The frozen benchmark used a deterministic random split. It does not test unseen-scaffold generalization, and the attachment does not supply a Tox21 scaffold split. We disclose this and would plan a scaffold study without replacing these results.

**Is Full overparameterized?** It has 6.37 times GraphSAGE's parameters, so capacity is a plausible confounder. We cannot diagnose overfitting solely from parameter count. Bottleneck and a future parameter-matched MLP-head comparison address different parts of this question.

**Do static gates add expressive power?** With linear heads they can be absorbed into the head weights. Their inferior run does not prove reduced capacity. Initialization and optimization can still differ from the ungated model.

**Are bond features used?** No. The dataset contains three bond attributes but all frozen models use only connectivity and nine numeric atom-feature codes. An edge-aware encoder would be a new experiment, not a silent correction to these scores.

**What would you do next?** Predefine scaffold evaluation and capacity-matched controls, repeat seeds, assess calibration and edge-aware features, and validate gate explanations through perturbations. Treat the current test as already disclosed for those future models.

## Exact commands

Run from the repository root; Python 3.13 was used for this audit.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python verify_submission.py
python reproduce.py
python -m streamlit run app.py
```

On Linux/macOS use `source .venv/bin/activate`; platform-compatible installation may require the unpinned `requirements.txt` instead of the Windows CPU audit lock. To inspect all reproduction commands without training: `python reproduce.py --dry-run`. To reevaluate the shipped checkpoints into separate results: `python evaluate_test.py --output_root runs/evaluation`. To regenerate the report without model execution: `python generate_pdf_report.py`.

## Scientific preservation confirmation

**Prediction-changing code modified: no for valid molecular inputs.** Invalid/empty input now fails explicitly; saved-split validation rejects corruption. These are boundary checks, not changes to accepted molecule featurization or logits. Model computation, preprocessing, optimization and metrics are unchanged. No prediction-changing post-test experimental improvement was implemented. New reproduction orchestration is labeled post-disclosure and stores its outputs separately. Frozen benchmark results, checkpoints, original split bytes and historical test protocol were not silently altered.
