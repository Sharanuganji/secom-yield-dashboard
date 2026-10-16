# SECOM SPC & Yield Analysis Dashboard

**SPC control charts, process capability, and ML-based yield prediction on the UCI SECOM semiconductor manufacturing dataset.**

## Problem

Semiconductor fabs generate thousands of sensor readings per wafer. Process engineers need to:
- Detect when a process drifts out of statistical control (excursions)
- Quantify process capability (Cp/Cpk) against spec limits
- Predict which wafers will fail before they ship

This project builds an end-to-end pipeline that does all three, deployed as an interactive dashboard.

## Dataset

[UCI SECOM](https://archive.ics.uci.edu/dataset/273/secom+semiconductor+manufacturing+data) — 1,567 wafers, 590 sensor features, 6.64% fail rate (104 fails / 1,463 passes), collected over 12 months (2008).

## Approach

| Notebook | What it does |
|---|---|
| `01_eda_cleaning` | Load, impute NaNs, drop 116 constant features, visualize missingness |
| `02_spc_control_charts` | I-MR charts, CUSUM, Western Electric rules on top-variance sensors |
| `03_process_capability` | Cp/Cpk calculation with assumed spec limits (mean ± 3σ) |
| `04_yield_prediction` | Random Forest + Logistic Regression with SMOTE; PR-AUC evaluation |
| `05_excursion_detection` | Rolling z-score anomaly detection with tunable window/threshold |

**Key design decisions:**
- **I-MR charts** (not X-bar/R) because each row is a single wafer, not a subgroup
- **PR-AUC** instead of accuracy — with 93/7 class imbalance, a trivial all-pass model scores 93% accuracy but is useless
- **SMOTE + class_weight="balanced"** to handle the 15:1 class imbalance
- **Specs assumed as mean ± 3σ** since SECOM has no real product spec sheet (documented as a limitation)

## Results

| Metric | Value |

| CUSUM signals (X163) | 151 |
| Western Electric violations | 216 (mostly Rule 4: sustained drift) |
| RF PR-AUC | **0.236** (vs. 0.066 baseline → 3.5× improvement) |
| LR PR-AUC | 0.177 |
| Top features by importance | X163, X162, X298, X25, X297 |

The low absolute PR-AUC is expected — published papers on SECOM report 0.2–0.4. The signal is real but weak: most of the 590 sensors are uncorrelated with the rare failure mode.

## Dashboard

Interactive Streamlit app: select a sensor, view SPC charts, capability indices, and excursion flags.

**[Live Demo](https://Sharanuganji.streamlit.app/secom-yield-dashboard)** ← replace with your deployed URL

## Project Structure
   
