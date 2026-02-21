# 🧠 Automated Multiclass Diagnosis of Neuropsychiatric Disorders from Resting-State EEG using GNNs


[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/github/license/Adxrsh-17/EEG-Detection-Alzheimer-s-FTD-MCI-MDD-GNN)](./LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Adxrsh-17/EEG-Detection-Alzheimer-s-FTD-MCI-MDD-GNN/actions)

---

## 📊 Project Workflow

```mermaid
flowchart TD
    A[Phase 0: Data Ingestion & Unification ✅] --> B[Phase 1: Harmonization & Preprocessing ✅]
    B --> C[Phase 2: Feature Engineering & Graph Construction ⏳]
    C --> C2[Phase 2b: COmBat Batch-Effect Harmonization ⏳]
    C2 --> D[Phase 3: GNN Modeling & Training ⏳]
    D --> E[Phase 4: Evaluation & Explainability ⏳]
```

---

## 📌 Project Overview

This project implements a **production-grade pipeline** for the automated diagnosis of four neuropsychiatric disorders:  

- **Alzheimer’s Disease (AD)**  
- **Frontotemporal Dementia (FTD)**  
- **Mild Cognitive Impairment (MCI)**  
- **Major Depressive Disorder (MDD)**  

using **resting-state EEG data** and **Graph Neural Networks (GNNs)**.

The core idea is to transform complex, multi-channel EEG time-series data into a **graph-based representation of brain connectivity**. This allows us to leverage GNNs to learn intricate patterns of neural synchronization that may be indicative of specific disorders.

The pipeline is **modular, robust, and reproducible**, unifying three distinct public datasets into a single, analysis-ready format suitable for advanced machine learning.

---

## 📌 Project Workflow & Status

The project is structured into several phases. ✅ indicates **completed**, ⏳ indicates **in progress/future work**.

---

### ✅ Phase 0: Data Ingestion & Unification  
**Concept**: Collect three disparate datasets and create a unified list of EEG recordings from the collected datasets.  

**Implementation**:  
- `code/create_metadata.py` → Generates `master_metadata.csv` as the single source of truth.  

---

### ✅ Phase 1: Harmonization & Preprocessing  
**Concept**: Standardize all EEG recordings and apply automated preprocessing for consistency.  

**Implementation**:  
- `code/harmonize_data.py` → Converts all recordings to **200Hz, 19 channels, Eyes-Closed state** in `.fif` format.  
- `code/validate_data.py` → Performs quality control to ensure integrity of signal and presence of required channels.  
- `code/preprocess_data.py` → Applies filtering, ICA-based artifact removal, and segmentation into **2-second clean epochs**.  

---

### ⏳ Phase 2: Feature Engineering & Graph Construction  
**Concept**: Transform clean EEG epochs into a **graph format** suitable for GNNs, then correct for inter-dataset recording differences.

**Planned Implementation**:  
- **Nodes**: 19 EEG channels.  
- **Node Features**: Extracted via **Empirical Wavelet Transform (EWT)** across EEG bands (Delta, Theta, Alpha, Beta, Gamma). Features include **spectral power** and **Shannon entropy**.  
- **Edges**: Functional connectivity via **Weighted Phase Lag Index (wPLI)**, yielding a **19×19 weighted adjacency matrix**.
- **Batch-Effect Removal**: **COmBat** harmonization (see below) is applied to the extracted node features after graph construction.

---

#### 🔬 COmBat Harmonization — Removing Dataset Batch Effects

**Why it is needed**  
This project combines EEG recordings from three independent sources (ds004504, CAUEEG, figshare_mdd). Even after standardising sampling rate and channel layout, systematic non-biological differences remain — differences in amplifier hardware, electrode impedance tolerances, recording environment, and study protocol. These *batch effects* can cause a classifier to learn "which dataset did this come from?" instead of "which disorder does this patient have?", inflating apparent performance during development while degrading real-world generalisation.

**What COmBat does**  
COmBat (Johnson *et al.*, 2007) was originally developed for microarray gene-expression data and has since been validated for neuroimaging (Fortin *et al.*, 2017, 2018). It models each feature with a linear mixed model:

```
y_ijv = α_v + X_ij β_v + γ_iv + δ_iv ε_ijv
```

where `y_ijv` is the observed value of feature *v* for sample *j* in batch *i*; `α_v` is the overall mean; `X_ij β_v` captures the effect of biological covariates (here: diagnosis); `γ_iv` is the additive batch shift; `δ_iv` is the multiplicative batch scale; and `ε_ijv` is the residual error. Critically, COmBat uses **empirical Bayes shrinkage** to pool information across features when estimating these parameters, which makes it robust even when individual batches contain only a handful of subjects.

The biological covariate (`diagnosis`) is included in the model so that variance attributable to the disorder is *protected* and not removed along with the batch effects.

**Where it is applied in this pipeline**  
COmBat is run **after** `create_graphs.py` has extracted EWT node features and stored them in PyTorch Geometric `.pt` graph files. The script:

1. Loads the 10-dimensional node-feature vector (log-power + Shannon entropy for each of 5 bands) for every channel of every epoch-graph.  
2. Stacks these into a `(n_features × n_graphs)` matrix — the format expected by neuroCombat.  
3. Runs COmBat with `original_dataset_source` as the **batch** variable and `diagnosis` as a **protected biological covariate**.  
4. Reshapes the corrected features and writes them back into the `.pt` files **in-place**.

**References**  
- Johnson, W.E., Li, C., Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127. https://doi.org/10.1093/biostatistics/kxj037  
- Fortin, J-P. *et al.* (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047  
- neuroCombat Python package: https://github.com/Jfortin1/neuroCombat_py

**Implementation**: `code/combat_harmonization.py`

---

### ⏳ Phase 3: GNN Modeling, Training & Augmentation  
**Concept**: Train a **Graph Neural Network** to classify EEG graphs into AD, FTD, MCI, MDD, or CN.  

**Planned Implementation**:  
- **Data Splitting**: Subject-wise stratified split (train/val/test).  
- **Class Imbalance**: Use **SMOTE** + **weighted cross-entropy loss**.  
- **GNN Model**: **Graph Attention Network (GAT)** using *PyTorch Geometric*.  
- **Training**: k-fold cross-validation for hyperparameter tuning and robust evaluation.  

---

### ⏳ Phase 4: Evaluation & Explainability (XAI)  
**Concept**: Evaluate performance and interpret model decisions.  

**Planned Implementation**:  
- **Metrics**: Macro F1-Score, class-wise precision/recall, confusion matrix.  
- **Explainability**: **GNNExplainer** to identify influential brain regions and connections, highlighting **potential biomarkers**.  

---

## ⚙️ Setup & Installation

**Prerequisite**: Python **3.9+**

```bash
# Clone Repository
git clone <your-repository-url>
cd Neuropsychiatric_EEG_GNN_Final

# Create Virtual Environment
python -m venv venv

# Activate Environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```
---

## 📂 Data Acquisition & Placement

This pipeline requires **three public datasets** which must be **downloaded manually**.  

⚠️ **Important**: After downloading and unzipping, place them into the correct subdirectories within the `data/raw/` folder.  
The scripts rely on the following exact structure:

```csharp
Neuropsychiatric_EEG_GNN_Final/
└── data/
    └── raw/
        ├── ds004504/
        │   ├── ... (files from OpenNeuro)
        │   └── .gitkeep
        │
        ├── caueeg/
        │   ├── ... (files from CAUEEG GitHub)
        │   └── .gitkeep
        │
        └── figshare_mdd/
            ├── ... (files from Figshare)
            └── .gitkeep
```

### ***Download Links and Instructions:***

- **Dataset 1: Alzheimer's & FTD (ds004504)**

Download from: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504)

Action: Place its contents into data/raw/ds004504/.

- **Dataset 2: Alzheimer's & MCI (CAUEEG)**

Download from: [CAUEEG GitHub](https://github.com/ipis-mjkim/caueeg-dataset)

Action: Place its contents into data/raw/caueeg/.

- **Dataset 3: Major Depressive Disorder (Figshare)**

Download from: [Figshare EEG Data](https://figshare.com/articles/dataset/EEG_Data_New/4244171)

Action: Place its contents into data/raw/figshare_mdd/.

---

## ▶️ Execution Pipeline (Completed Stages)

Run the following scripts sequentially from the project root (after activating venv).

Step A: Gather & Unify Raw Data (Phase 0)

```python
# Unify Metadata
python code/create_metadata.py
# Output: data/master_metadata.csv
```

Step B: Harmonize & Preprocess Data (Phase 1)

```python
# Harmonize Dataset
python code/harmonize_data.py
# Outputs:
#   - Cleaned files → data/processed/harmonized/
#   - Index file    → data/harmonized_metadata.csv

# Validate Harmonized Data
python code/validate_data.py

# Preprocess Harmonized Data
python code/preprocess_data.py
```

Step C: Feature Engineering, Graph Construction & COmBat Harmonization (Phase 2)

```python
# Build graph dataset (node features via EWT + edges via wPLI)
python code/create_graphs.py
# Outputs:
#   - Graph files    → data/processed/graphs/
#   - Index file     → data/graph_metadata.csv

# Apply COmBat to remove dataset-specific batch effects from node features
python code/combat_harmonization.py
# Outputs:
#   - Updated graph files (features corrected in-place)
#   - Index file          → data/combat_graph_metadata.csv
```
