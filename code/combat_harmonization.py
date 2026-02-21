"""
Phase 2, Step 2: COmBat Feature Harmonization

Background
----------
When combining EEG recordings from multiple independent datasets (e.g., ds004504,
CAUEEG, figshare_mdd), systematic non-biological differences arise from variations
in acquisition hardware, recording protocols, and site-specific processing. These
differences — collectively called **batch effects** — can confound downstream machine
learning by causing a classifier to learn *which dataset a sample came from* rather
than *which disorder it represents*.

**COmBat** (Johnson et al., 2007; originally developed for genomics) addresses this
problem using an empirical Bayes framework:

  1. A linear model separates biological variance (e.g., diagnosis) from batch
     (dataset-source) variance.
  2. Batch-specific additive (location) and multiplicative (scale) shift parameters
     are estimated from the data using empirical Bayes shrinkage, which is
     particularly robust when batch sizes are small.
  3. The estimated batch effects are removed, yielding harmonized features that
     preserve inter-subject biological variability.

In this project, COmBat is applied to the **node feature matrix** extracted in
`create_graphs.py`. Each node (EEG channel) carries 10 features: log-power and
Shannon entropy for each of the five EEG frequency bands (Delta, Theta, Alpha,
Beta, Gamma).  These 10 × 19 = 190 features per epoch-graph are stacked across
all subjects, COmBat is run once (with `original_dataset_source` as the batch
variable and `diagnosis` as a biological covariate to protect), and the corrected
features are written back into the saved `.pt` graph files.

Reference
---------
Johnson, W.E., Li, C., Rabinovic, A. (2007). Adjusting batch effects in
microarray expression data using empirical Bayes methods. Biostatistics, 8(1),
118-127. https://doi.org/10.1093/biostatistics/kxj037

NeuroComBat Python package (Fortin et al., 2017):
https://github.com/Jfortin1/neuroCombat_py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuroCombat import neuroCombat
from tqdm import tqdm

# --- Configuration ---
GRAPH_METADATA_PATH = Path("../data/graph_metadata.csv")
PREPROCESSED_METADATA_PATH = Path("../data/preprocessed_metadata.csv")
OUTPUT_METADATA_PATH = Path("../data/combat_graph_metadata.csv")
LABEL_MAP_PATH = Path("../data/label_mapping.json")

# Number of node features per channel (must match create_graphs.py)
N_NODE_FEATURES = 10  # 5 bands × (log_power + entropy)
N_CHANNELS = 19


def load_node_features(graph_records: list[dict]) -> np.ndarray:
    """
    Load node feature matrices from saved graph `.pt` files.

    Returns
    -------
    features : ndarray of shape (n_graphs, N_CHANNELS * N_NODE_FEATURES)
        Flattened node feature matrix for all graphs, suitable for COmBat
        which expects features in rows and samples in columns.
    """
    all_features = []
    for record in graph_records:
        graph = torch.load(record["graph_file_path"], weights_only=True)
        # x has shape (N_CHANNELS, N_NODE_FEATURES) — flatten to a 1-D vector
        all_features.append(graph.x.numpy().flatten())
    return np.array(all_features)  # shape: (n_graphs, N_CHANNELS * N_NODE_FEATURES)


def save_harmonized_features(graph_records: list[dict], harmonized: np.ndarray) -> None:
    """
    Write corrected node features back into the graph `.pt` files in-place.

    Parameters
    ----------
    graph_records : list of metadata dicts (each must have 'graph_file_path').
    harmonized    : ndarray of shape (n_graphs, N_CHANNELS * N_NODE_FEATURES)
                    — the COmBat-corrected feature matrix.
    """
    for i, record in enumerate(tqdm(graph_records, desc="Updating graphs")):
        graph = torch.load(record["graph_file_path"], weights_only=True)
        corrected_x = harmonized[i].reshape(N_CHANNELS, N_NODE_FEATURES)
        graph.x = torch.tensor(corrected_x, dtype=torch.float)
        torch.save(graph, record["graph_file_path"])


def main() -> None:
    print("--- Starting Phase 2, Step 2: COmBat Feature Harmonization ---")

    # --- Load metadata ---
    if not GRAPH_METADATA_PATH.exists():
        print(f"[FATAL] Graph metadata not found at {GRAPH_METADATA_PATH}. "
              "Run create_graphs.py first.")
        sys.exit(1)
    if not PREPROCESSED_METADATA_PATH.exists():
        print(f"[FATAL] Preprocessed metadata not found at {PREPROCESSED_METADATA_PATH}. "
              "Run preprocess_data.py first.")
        sys.exit(1)

    graph_df = pd.read_csv(GRAPH_METADATA_PATH)
    pre_df = pd.read_csv(PREPROCESSED_METADATA_PATH)[
        ["subject_id", "original_dataset_source"]
    ]

    # Attach dataset-source (batch label) to each graph record
    merged = graph_df.merge(pre_df, on="subject_id", how="left")
    if merged["original_dataset_source"].isna().any():
        print("[WARNING] Some graph records could not be matched to a dataset source. "
              "These records will be excluded from harmonization.")
        merged = merged.dropna(subset=["original_dataset_source"])

    graph_records = merged.to_dict("records")
    n_graphs = len(graph_records)
    print(f"  Found {n_graphs} graph files across "
          f"{merged['original_dataset_source'].nunique()} dataset(s).")

    # Check that there are at least two distinct batches; otherwise COmBat is a no-op
    unique_batches = merged["original_dataset_source"].unique()
    if len(unique_batches) < 2:
        print("[INFO] Only one dataset source detected — COmBat requires ≥2 batches. "
              "Skipping harmonization and copying metadata unchanged.")
        merged.to_csv(OUTPUT_METADATA_PATH, index=False)
        print(f"✅ Metadata saved (unchanged) to: {OUTPUT_METADATA_PATH}")
        return

    # --- Build COmBat inputs ---
    print("  Loading node features from graph files…")
    feature_matrix = load_node_features(graph_records)
    # COmBat expects shape (n_features, n_samples)
    data_matrix = feature_matrix.T  # → (190, n_graphs)

    # Batch vector (integer-encoded dataset source)
    batch_labels, batch_index = np.unique(
        merged["original_dataset_source"].values, return_inverse=True
    )
    batch = batch_index + 1  # neuroCombat expects 1-indexed batches

    # Biological covariate: diagnosis (protect from removal by COmBat)
    with open(LABEL_MAP_PATH) as f:
        label_map: dict = json.load(f)
    diagnosis_covariate = merged["diagnosis"].map(label_map).values

    covars = pd.DataFrame({
        "batch": batch,
        "diagnosis": diagnosis_covariate,
    })

    print("  Running COmBat — this may take a moment…")
    combat_result = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col="batch",
        categorical_cols=["diagnosis"],
    )

    harmonized_data = combat_result["data"].T  # → (n_graphs, 190)

    # --- Write harmonized features back to disk ---
    print("  Saving harmonized features to graph files…")
    save_harmonized_features(graph_records, harmonized_data)

    # --- Save updated metadata ---
    merged.to_csv(OUTPUT_METADATA_PATH, index=False)

    print("\n--- COmBat Harmonization Complete ---")
    print(f"  Harmonized {n_graphs} graphs across batches: {list(batch_labels)}")
    print(f"✅ Updated metadata saved to: {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
