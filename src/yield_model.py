"""Yield prediction model for semiconductor pass/fail classification."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def train_yield_model(X, y, model_type="rf", use_smote=True, test_size=0.2):
    """
    Train a yield prediction model.
    Returns: model, metrics dict
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # Train model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "lr":
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_bal, y_train_bal)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics (use PR-AUC, NOT accuracy)
    ap_score = average_precision_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "pr_auc": ap_score,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Pass", "Fail"]
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fail_rate_test": y_test.mean(),
    }

    return model, scaler, metrics


def plot_pr_curve(y_true, y_proba, ap_score, title="PR Curve", save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.axhline(y_true.mean(), color="gray", linestyle="--",
               label=f"Baseline (fail rate={y_true.mean():.3f})")
    ax.fill_between(recall, precision, alpha=0.1, color="blue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} (PR-AUC = {ap_score:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[idx], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features by Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()   