# Polymer RGCN: Multi-Property Regression + XAI

Predict five polymer properties under extreme missingness using a relational GCN with a 3-edge representation (single/double/triple bonds; aromatic bonds excluded from relation types). Developed for the **NeurIPS - Open Polymer Prediction 2025** Kaggle competition, featuring cross-validation, hyperparameter optimization, explainability for FFV via fragment masking, and external prediction utilities.

## Features

- **Dataset**: NeurIPS 2025 competition dataset (8,835 polymers)
- **Targets**: `Tg`, `FFV`, `Tc`, `Density`, `Rg` (ground truth from molecular dynamics simulations)
- **Representation**: 3-edge (0=SINGLE, 1=DOUBLE, 2=TRIPLE); aromatic bonds tagged `-1` (excluded from message passing) and captured via node features
- **Loss + Selection**: Masked multitask regression with weighted MAE (wMAE) aligned to competition evaluation metric; selection on `val_wmae`
- **XAI**: Murcko fragment masking for FFV with confidence-based gating; optional rule mining from fragment attributions
- **Text Input**: BigSMILES attempted via round-trip; falls back to canonical SMILES when needed
- **Validation Framework**: High confidence predictions validated against experimental data with 78% Category A accuracy

## Contents

### Core Pipeline

- **Building Graphs**: `build_data.py`, `prep_data.py`
- **Model + Training**: `model.py`, `data_module.py`
- **Experiments**: `hyper.py` (Optuna), `stat_val.py` (CV), `final_eval.py` (ensembles)

### Prediction and Analysis
- **Inference + XAI**: `predict.py`
- **Interactive Notebooks**:
  - `FFV_XAI_visual.ipynb` - : Single-molecule FFV prediction with visual XAI 


### Utilities
- **Core**: `logger.py`, `memory_tracker.py`, `config.py`
- **BigSMILES Support**: `bigsmiles_utils/`
- **Data Cleaning**: `polymer_data_cleaning.py`
- **Visualization**: `xai_viz.py`

## Installation

### 1. Environment Setup

Create and activate a Python 3.10+ environment:

```bash
# Using conda (recommended)
conda create -n polymer_rgcn python=3.10
conda activate polymer_rgcn

# Or using venv
python -m venv polymer_env
source polymer_env/bin/activate  # On Windows: polymer_env\Scripts\activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

**Important Notes:**
- PyTorch and PyG (torch-geometric) wheels must match your CUDA/Python version. If pip resolution fails:
  1. Install PyTorch first: https://pytorch.org/get-started/locally/
  2. Then install torch-geometric: https://pytorch-geometric.readthedocs.io
- RDKit:
  - Via pip: uses `rdkit-pypi` (included in requirements.txt)
  - Via conda: prefer `conda install -c rdkit rdkit` and remove `rdkit-pypi` from requirements.txt

## Dataset

### Overview

The default dataset (NeurIPS Open Polymer Prediction 2025) contains **8,835 polymer structures** with severe property imbalance:

| Property | Coverage | Molecules | Notes |
|----------|----------|-----------|-------|
| FFV | 89.3% | 7,892 | Primary target |
| Tg | 5.8% | 511 | Sparsest property |
| Tc | 8.3% | 737 | Sparse |
| Density | 6.9% | 613 | Sparse |
| Rg | 7.0% | 614 | Sparse |

**Complete cases**: 0/8,835 (0.0%) - No molecule has all five properties measured

### Data Splits

Stratified splitting based on property availability patterns:
- **Training**: 6,183 molecules (70.0%)
- **Validation**: 1,326 molecules (15.0%)
- **Test**: 1,326 molecules (15.0%)

Property distributions are preserved across splits to ensure representative validation.

### Default Paths

Defined in `config.py`:
- `data_dir`: `/workspace/data/polymer_data` (override as needed)
- `origin_data_path`: `<data_dir>/polymer_properties.csv`
- `output_dir`: `<data_dir>/graphs_3edge`

### Input CSV Format

Minimum required columns:
- `id`: Unique identifier
- `SMILES`: Repeating unit/polymer representation (wildcards `*` allowed)
- Targets: Any subset of `Tg`, `FFV`, `Tc`, `Density`, `Rg` (NaN allowed for missing values)

### Output Files

After running `build_data.py`:
- **Graphs**: `polymer_properties_regression_primary_graphs.pt`
- **Metadata**: `polymer_properties_regression_primary_meta.csv`
- **BigSMILES Coverage**: `repr_used.csv`
- **Cleaning Log**: `polymer_dataset_cleaning_log.txt`

## Quickstart

### 1. Build Graphs (3-Edge v2)

**Minimal** (uses config.py defaults):
```bash
python build_data.py
```

**Flexible** (custom paths):
```bash
python prep_data.py \
  --dataset_path /path/to/polymer_properties.csv \
  --output_dir   /path/to/output/graphs_3edge \
  --substructure_types primary
```

### 2. Hyperparameter Optimization

Minimize validation wMAE using Optuna:
```bash
python hyper.py
```

Creates an Optuna SQLite database in the output directory and trains trials with pruning.

### 3. Statistical Validation (Cross-Validation)

```bash
python stat_val.py
```

Produces per-fold checkpoints and aggregated metrics under:
- `<output_dir>/<task>_<type>_cv_results/cv*/checkpoints/`
- `<output_dir>/<task>_<type>_cv_results/cv*_summary.json`

### 4. Final Evaluation + Ensembling

```bash
python final_eval.py
```

Selects best checkpoints, evaluates on test set, and builds:
- Global best-5 ensemble (ordinal Category-A first; wMAE tie-breaks)
- Per-property ensembles
- Output: `<output_dir>/<task>_<type>_final_eval/best_models.json`

### 5. Predict on External Candidates (+ XAI)

**Basic best-5 ensemble**:
```bash
python predict.py --input_csv /path/to/polymer_candidates.csv --ensemble best5
```

**With XAI JSON output + rule mining**:
```bash
python predict.py \
  --input_csv /path/to/polymer_candidates.csv \
  --emit_xai_json 1 \
  --mine_rules 1
```

**Full ensemble (all CV models)**:
```bash
python predict.py --input_csv /path/to/polymer_candidates.csv --ensemble full
```

**BigSMILES coverage dry-run** (pre-check without predictions):
```bash
python predict.py \
  --input_csv /path/to/polymer_candidates.csv \
  --dry_run 100 \
  --predict_repr auto \
  --bigsmiles_required 0
```

**Outputs**:
- Results CSV next to input (or use `--output_csv` to specify)
- XAI JSONs in `xai/` directory if `--emit_xai_json 1`
- Fragment rules: `ffv_rules_from_screen.csv` and optional PNG summary if `--mine_rules 1`


## Interactive XAI Notebook

### FFV_XAI_visual.ipynb (Recommended)

**Features**:
- 2-cell notebook for single-molecule FFV prediction + visual XAI
- Loads best-5 ensemble from `best_models.json`
- Displays checkpoint paths used for transparency
- Visual fragment highlighting (lemon green = positive, red = negative)
- Table of fragment SMILES with attribution scores

**Requirements**:
- Core libraries from `requirements.txt`
- Jupyter: `pip install notebook` (or use JupyterLab)
- **Model checkpoints**: Download pre-trained models from Zenodo: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
  - Extract to `output/` directory
  - Required files:
    - `output/polymer_properties_regression_final_eval/best_models.json`
    - `output/polymer_properties_regression_cv_results/cv*/checkpoints/*.ckpt`

**Usage**:
1. Download and extract model checkpoints from Zenodo (see above)
2. Start Jupyter: `jupyter notebook`
3. Open `FFV_XAI_visual.ipynb`
4. Run Cell 1 to load models (shows checkpoint paths)
5. Paste your SMILES in Cell 2 and run to see FFV + XAI visualization

**Output**:
- Predicted FFV value
- Molecule image with fragment highlights (no atom numbering)
- Table of `substructure_smiles` with attribution scores (sorted by magnitude)



## XAI Analysis Tools

### Extract Fragment Rules from Predictions

After running predictions with XAI:

```bash
python analyze_ffv_xai_rules.py \
  --pred_csv output/external_with_preds_3edge.csv \
  --train_csv output/polymer_properties.csv \
  --min_support 5 \
  --confidence high \
  --level_filter A
```

**Parameters**:
- `--min_support`: Minimum molecules per fragment to report (default: 5)
- `--confidence`: Filter by model confidence: `high`, `medium`, `low`, `any` (default: `high`)
- `--level_filter`: Filter by ordinal error level: `A` (≤5%), `B` (≤10%), `C` (≤20%), `D`, `none` (default: `none`)

**Outputs** (in `output/FFV_XAI/` or `output/FFV_XAI_cat_A/`):
- `ffv_fragment_rules_summary.csv` - All fragments with statistics
- `ffv_fragment_rules_top_increasing.csv` - FFV-enhancing fragments
- `ffv_fragment_rules_top_decreasing.csv` - FFV-reducing fragments
- `ffv_fragment_instances.csv` - Instance-level data
- `ffv_fragment_positive_instances.csv` - Positive attribution instances
- `ffv_fragment_negative_instances.csv` - Negative attribution instances
- `ffv_fragment_rules_summary.json` - Compact JSON with top fragments and examples

### Validate Confidence as Accuracy Proxy

Verify that high confidence predictions correlate with prediction accuracy:

```bash
python analyze_confidence_accuracy_correlation.py
```

**Output**: `output/confidence_accuracy_correlation_analysis.csv`

**Results** (on validation set with 7,892 polymers having experimental FFV):
- **Precision**: 78.04% of high confidence predictions achieve Category A accuracy (≤5% error)
- **Odds Ratio**: 1.99× - high confidence predictions are twice as likely to be accurate
- **Error Reduction**: 36% lower mean absolute error for high confidence predictions
- **Statistical Significance**: p < 0.001

This validates high confidence as a reliable filter for screening polymers without experimental data.

## Configuration Highlights (`config.py`)

### Task Configuration
- **Task Type**: `classification = False` (multitask regression)
- **Properties**: `property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']`

### Graph Representation
- **Edge Types**: `num_edge_types = 3` (SINGLE/DOUBLE/TRIPLE bonds)
- **Aromatic Handling**: Aromatic edges marked `-1` and excluded during message passing; aromaticity captured via node features

### Loss Weighting Strategy

Adaptive inverse-proportional weighting to balance gradient contributions across properties despite extreme data imbalance:

```python
competition_weights = {
    'Tg': 1.000,      # Reference (sparsest property at 5.7% training coverage)
    'FFV': 0.073,     # Primary target (89.3% training coverage)
    'Tc': 0.693,      # 8.3% training coverage
    'Density': 0.834, # 7.0% training coverage
    'Rg': 0.832,      # 7.0% training coverage
}
```

**Formula**: `w_p = r_ref / r_p` where `r_ref` is the availability rate for Tg (sparsest property)

**Effective Impact** (weight × availability): ~0.057-0.065 for all properties, achieving near-perfect balance

**Note**: These weights were derived from training set property distributions and achieve balanced gradient contributions across all five properties. See manuscript for detailed methodology.

### Text Representation
- **BigSMILES Support**: `USE_BIGSMILES = True`
- **Prediction Mode**: `PREDICT_REPR = 'auto'` (try BigSMILES, fallback to SMILES)
- **Required**: `BIGSMILES_REQUIRED = False` (keep molecules that fail BigSMILES conversion)

### XAI Configuration
- **Default Method**: Murcko scaffold decomposition
- **Top-K Fragments**: `XAI_TOPK = 8`
- **JSON Output**: Controlled by `EMIT_XAI_JSON` flag
- **Confidence Gating**: XAI only computed for high confidence predictions (customizable)

### Calibration
- **FFV Calibrator**: Available but disabled by default (`CALIBRATE_FFV = False`)

## Method Summary

### Multi-Task Learning Under Extreme Imbalance

- **Masked Loss**: Only molecules with experimental labels contribute to loss for each property
- **Weighted MAE**: Inverse-proportional weighting prevents abundant properties from dominating optimization
- **wMAE Metric**: Aligns with competition-style scoring for model selection and hyperparameter optimization

### 3-Edge v2 Relation Design

- **Clearer Attribution**: Aromatic edges excluded from relation types for simpler message passing
- **Node Features**: Aromaticity captured at atom level where chemically relevant
- **Separation of Concerns**: Structural connectivity (edges) vs. electronic properties (nodes)

### Model Selection and Ensembling

- **HPO Target**: `val_wmae` minimization
- **Selection Strategy**: Ordinal Category-A (tight prediction bins) as primary metric, wMAE for tie-breaking
- **Ensemble Types**:
  - Best-5 global ensemble (recommended)
  - Full CV ensemble (all folds)
  - Per-property ensembles highlighting target-specific strengths

### FFV Explainability (XAI)

- **Method**: Murcko fragment masking with attribution scoring
- **Confidence Gating**: XAI only for high-confidence predictions to ensure reliability
- **Validation**: 78% of high confidence predictions achieve ≤5% error
- **Output Formats**:
  - Visual highlights (lemon green/red color-coding)
  - Fragment SMILES with attribution scores
  - JSON for programmatic access
  - Aggregated rules for chemist-readable SAR

### Confidence-Based Reliability Framework

- **High Confidence**: Ensemble std ≤ 75th percentile (≤0.009 for FFV)
- **Validated Accuracy**: 78% precision for Category A (≤5% error)
- **Screening Utility**: Reliable filter for polymers without experimental data

See `METHODS.md` and `PREDICT_RUN_BRIEF.md` for full methodological details.

## Troubleshooting

### Installation Issues

**PyTorch/PyG Installation**:
1. Install PyTorch first matching your CUDA version: https://pytorch.org/get-started/locally/
2. Then install torch-geometric: https://pytorch-geometric.readthedocs.io

**RDKit**:
- Via pip: `rdkit-pypi` package (included in requirements.txt)
- Via conda: `conda install -c rdkit rdkit` (remove `rdkit-pypi` from requirements.txt)

**Parquet Support**:
- Ensure `pyarrow` is installed for Parquet input files

### Runtime Issues

**Windows Paths**:
- Update `Configuration.data_dir` in `config.py` or
- Pass `--dataset_path`/`--output_dir` to scripts with Windows-style paths

**CUDA Issues**:
- Code automatically falls back to CPU
- Force CPU mode: `--device cpu` in `predict.py`

**Memory Issues**:
- Reduce batch size in config
- Use CPU mode for prediction
- Process large datasets in chunks (predict.py supports resume via checkpointing)

**Missing Checkpoints**:
- Ensure `stat_val.py` and `final_eval.py` have completed successfully
- Check `output/polymer_properties_regression_final_eval/best_models.json` exists
- Verify checkpoint paths in `output/polymer_properties_regression_cv_results/cv*/checkpoints/`

## Citation

If this code or methodology supports your work, please cite:

```bibtex
@software{polymer_rgcn_3edge,
  title = {Polymer RGCN: Multi-Property Prediction with Explainable AI},
  author = {Onawole, Abdulmujeeb T; },
  year = {2025},
  note = {3-edge relational GCN with Murcko-based FFV explainability}
}
```

And acknowledge:
- 3-edge  graph representation with aromatic exclusion
- Adaptive loss weighting for extreme data imbalance
- Confidence-validated fragment attribution framework
- NeurIPS 2025 Open Polymer Prediction dataset

## License

[Add your chosen license, e.g., MIT, Apache 2.0]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description of changes

## Contact

atonawole@gmail.com

## Acknowledgments

- NeurIPS 2025 Open Polymer Prediction Competition
- University of Notre Dame for dataset organization
- PyTorch Geometric team
- RDKit community
