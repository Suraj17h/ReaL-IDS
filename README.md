# Network Intrusion Detection System — Upgraded Pipeline
## Design Decision Reference

---

## Overview

This IDS classifies network flow records into 7 classes:
`Normal | DDoS | DoS | BruteForce | Botnet | Infiltration | LowSeverityAttack`

Using a two-model ensemble: **XGBoost + Deep Neural Network**.

---

## Files

| File | Purpose |
|---|---|
| `upgraded_ids_notebook.ipynb` | Main pipeline — run this end to end |
| `upgraded_dataProc.py` | Data cleaning & label consolidation |
| `XGB_Final_Model.json` | Saved XGBoost model |
| `dnn_ids_model.keras` | Saved DNN model |
| `scaler.pkl` | StandardScaler (fit on train only) |
| `label_encoder.pkl` | LabelEncoder for class names |
| `feature_columns.json` | Selected feature names (post-selection) |
| `decision_config.json` | Infiltration threshold (tuned on val set) |
| `plots/` | All generated analysis plots |

---

## Examiner Feedback — Responses

### 1. "Accuracy is too high — possible overfitting"

**Response:** Overfitting is now explicitly **measured and visualised**:

- **Learning curves** for both XGBoost (train vs val log-loss by boosting round) and DNN (train vs val loss/accuracy by epoch) are plotted. Overfitting manifests as diverging curves; convergence indicates generalisation.
- **Train/test accuracy gap** is computed. A gap < 3% is reported as acceptable.
- **Regularisation** is applied systematically:
  - XGBoost: `reg_alpha=0.1` (L1), `reg_lambda=1.0` (L2), `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=20`
  - DNN: `Dropout(0.3)`, `kernel_regularizer=L2(1e-4)`, `BatchNormalization`, `ReduceLROnPlateau`, `EarlyStopping`
- **3-fold cross-validation** is performed during model selection to estimate out-of-sample performance.

---

### 2. "Sampling technique must be logical and thorough"

**Original approach:** Randomly undersample Normal class to 2.5M rows. This is arbitrary — the 2.5M figure has no theoretical basis, and it discards potentially informative boundary samples.

**Upgraded approach: SMOTETomek**

- **SMOTE** (Synthetic Minority Over-sampling Technique): Generates synthetic samples for minority classes by interpolating between k nearest neighbours in feature space. Avoids exact duplication while improving minority class representation.
- **Tomek Links**: Identifies pairs of samples from different classes that are each other's nearest neighbours (i.e. they are borderline samples). The majority-class member of each pair is removed, cleaning the decision boundary.
- **Applied only on training data** — the test set is split off first, preventing any synthetic samples from appearing in evaluation.
- **Before/after visualisation** included to demonstrate the effect.

---

### 3. "Feature selection techniques must be used"

Three-stage pipeline applied in sequence:

| Stage | Method | Threshold | Why |
|---|---|---|---|
| 1 | Variance Threshold | < 0.01 | Zero-variance features carry no signal |
| 2 | Correlation Pruning | \|r\| > 0.95 | Highly correlated features are redundant (multicollinearity) |
| 3 | XGBoost SelectFromModel | Mean importance | Model-aware, accounts for feature interactions |

Correlation heatmap and feature importance bar chart are plotted to document the selection rationale.

---

### 4. "Models must have a reason to be used"

A **baseline comparison** is run with 3-fold cross-validation before finalising model choice:

| Model | Hypothesis |
|---|---|
| Logistic Regression | Is data linearly separable? |
| Random Forest | Do non-linear trees help over linear? |
| XGBoost | Does boosting outperform bagging? |
| DNN (ensemble) | Can neural representations add value over trees? |

Results are plotted as a horizontal bar chart with error bars. The chosen models are justified by empirical CV scores, not assumed.

**XGBoost** is chosen because:
- Gradient boosting sequentially corrects residual errors, producing superior accuracy on tabular/structured data.
- Built-in regularisation (L1/L2).
- Handles class imbalance and sparse features natively.

**DNN** is added as an ensemble component because:
- Learns non-linear feature combinations through hidden layers that trees cannot represent with axis-aligned splits.
- Complementary inductive bias → ensemble reduces variance.

---

### 5. "More parallels — decisions justified throughout"

Every section of the notebook includes a written justification cell explaining *why* the step is being done, not just *what* is being done. Additional justified decisions include:

- **Why 3 hidden layers (128→64→32)**: Progressive dimensionality reduction mimics an encoder; avoids bottleneck on the first layer.
- **Why BatchNormalization**: Stabilises gradient flow; reduces sensitivity to weight initialisation.
- **Why Infiltration gets a custom threshold**: Missing an infiltration event has higher operational cost than a false positive. Threshold is swept on the **validation set** (not test set) to avoid evaluation leakage.
- **Why StandardScaler**: Network flow features span wildly different scales; normalisation ensures equal gradient contribution from all features in the DNN.
- **Why 60/40 ensemble weighting**: XGBoost is the stronger tabular model; DNN adds complementary signal.
- **Why train/val/test split (not just train/test)**: Validation set is used for early stopping and threshold tuning. Using the test set for these would make the test evaluation optimistic (lookahead bias).

---

## Pipeline Order

```
Raw CSV
  → dataProc.py          (clean + consolidate labels)
  → notebook Section 4   (3-stage feature selection)
  → notebook Section 5   (train/val/test split → SMOTETomek on train only)
  → notebook Section 6   (StandardScaler fit on train, transform all)
  → notebook Section 7   (model selection CV)
  → notebook Section 8   (XGBoost + learning curves)
  → notebook Section 9   (DNN + learning curves)
  → notebook Section 10  (ensemble + threshold tuning on val)
  → notebook Section 11  (final evaluation on test — untouched)
```
