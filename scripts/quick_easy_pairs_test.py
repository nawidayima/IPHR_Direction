#!/usr/bin/env python3
"""Quick test: Do easy-pair contradictions show any probe signal?

Run locally (no GPU needed):
    python scripts/quick_easy_pairs_test.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Paths
RUN_DIR = Path("experiments/run_20251228_204835_expand_dataset")
ACTIVATIONS_PATH = RUN_DIR / "activations/residual_stream_activations.pt"
GEO_CSV_PATH = RUN_DIR / "trajectories/geography.csv"

print("=" * 60)
print("QUICK TEST: Easy Pairs Only")
print("=" * 60)
print()

# Load data
print("Loading data...")
data = torch.load(ACTIVATIONS_PATH)
geo_df = pd.read_csv(GEO_CSV_PATH)
metadata = data['metadata']
labels = data['labels'].numpy()

# Get easy pair IDs
easy_pairs = set(geo_df[geo_df['difficulty'] == 'easy']['pair_id'].values)
print(f"Easy pairs: {len(easy_pairs)}")

# Filter to geography, Question B, EASY only
easy_b_indices = []
for i, m in enumerate(metadata):
    if (m['domain'] == 'geography' and
        m['question_type'] == 'B' and
        m['pair_id'] in easy_pairs):
        easy_b_indices.append(i)

easy_b_indices = np.array(easy_b_indices)
filtered_labels = labels[easy_b_indices]

n_total = len(filtered_labels)
n_contra = filtered_labels.sum()
n_honest = n_total - n_contra

print(f"\nFiltered dataset:")
print(f"  Total samples: {n_total}")
print(f"  Contradictions: {n_contra}")
print(f"  Honest: {n_honest}")
print(f"  Balance: {n_contra/n_total*100:.1f}% contradiction")

if n_total < 10:
    print("\nWARNING: Too few samples for reliable cross-validation!")
    print("Results will be very noisy.")

# Cross-validation setup
n_folds = min(5, min(n_contra, n_honest))  # Can't have more folds than minority class
if n_folds < 2:
    print("\nERROR: Not enough samples in each class for cross-validation")
    exit(1)

print(f"\nUsing {n_folds}-fold cross-validation")

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Test each layer
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for layer in data['layers']:
    acts = data['activations'][layer].numpy()
    X = acts[easy_b_indices]
    y = filtered_labels

    # DiM cross-validation
    dim_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Compute DiM direction
        mean_contra = X_train[y_train == 1].mean(axis=0)
        mean_honest = X_train[y_train == 0].mean(axis=0)
        direction = mean_contra - mean_honest
        direction = direction / np.linalg.norm(direction)

        # Score and evaluate
        scores = X_test @ direction
        auc = roc_auc_score(y_test, scores)
        dim_aucs.append(auc)

    dim_aucs = np.array(dim_aucs)

    # LR cross-validation
    lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    lr_aucs = cross_val_score(lr, X, y, cv=skf, scoring='roc_auc')

    # Interpret
    dim_mean = dim_aucs.mean()
    lr_mean = lr_aucs.mean()
    best = max(dim_mean, lr_mean)

    if best >= 0.7:
        status = "*** SIGNAL DETECTED ***"
    elif best >= 0.6:
        status = "* weak signal *"
    else:
        status = "no signal"

    print(f"\nLayer {layer}:")
    print(f"  DiM: {dim_mean:.3f} ± {dim_aucs.std():.3f}  (folds: {dim_aucs.round(2)})")
    print(f"  LR:  {lr_mean:.3f} ± {lr_aucs.std():.3f}  (folds: {lr_aucs.round(2)})")
    print(f"  --> {status}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
If AUC > 0.6: Easy pairs show some signal - contradictions there
              might be more "rationalization-like". Consider:
              - Expanding easy pairs dataset
              - Focusing probe on easy pairs only

If AUC ~ 0.5: Even easy contradictions show no signal.
              The model's behavior is uniformly "confusion-like"
              regardless of difficulty. PIVOT TO ICRL.
""")