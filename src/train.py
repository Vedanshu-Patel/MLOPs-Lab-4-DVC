"""
train.py
--------
Uses PCA for dimensionality reduction and Isolation Forest for anomaly detection
on the processed credit card dataset. Flags unusual spending behavior.

Run: python src/train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

PROCESSED_PATH = "data/processed/CC_PROCESSED.csv"
MODEL_PATH = "models/isolation_forest.joblib"
SCALER_PATH = "models/scaler.joblib"
PCA_PATH = "models/pca.joblib"
METRICS_PATH = "reports/metrics.json"
FIGURES_DIR = "reports/figures"

CONTAMINATION = 0.05  # Assume 5% of customers are anomalies


def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load data
    print(f"Loading: {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Shape: {df.shape}")

    # ── Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ── PCA — reduce to 10 components for model, 2 for visualization
    print("\nApplying PCA...")
    pca_full = PCA(n_components=10, random_state=42)
    X_pca = pca_full.fit_transform(X_scaled)

    explained = pca_full.explained_variance_ratio_.cumsum()
    print(f"  Variance explained by 10 components: {explained[-1]*100:.1f}%")

    # PCA for visualization (2 components)
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    # ── PCA explained variance plot
    print("Generating PCA explained variance plot...")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), pca_full.explained_variance_ratio_.cumsum(), "bo-", linewidth=2, markersize=8)
    plt.axhline(y=0.80, color="r", linestyle="--", label="80% threshold")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA — Cumulative Explained Variance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/pca_variance.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR}/pca_variance.png")

    # ── Train Isolation Forest
    print(f"\nTraining Isolation Forest (contamination={CONTAMINATION})...")
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    labels = model.fit_predict(X_pca)  # -1 = anomaly, 1 = normal

    # Convert to binary: 1 = anomaly, 0 = normal
    anomaly_flags = (labels == -1).astype(int)
    n_anomalies = anomaly_flags.sum()
    n_normal = len(anomaly_flags) - n_anomalies
    anomaly_pct = (n_anomalies / len(anomaly_flags)) * 100

    print(f"\n── Model Metrics ──")
    print(f"  Total customers:  {len(df)}")
    print(f"  Normal:           {n_normal} ({100-anomaly_pct:.1f}%)")
    print(f"  Anomalies:        {n_anomalies} ({anomaly_pct:.1f}%)")

    # Anomaly scores
    scores = model.decision_function(X_pca)
    avg_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    print(f"  Anomaly score — mean: {avg_score:.4f}, min: {min_score:.4f}, max: {max_score:.4f}")

    # ── Save metrics
    metrics = {
        "model": "IsolationForest + PCA",
        "n_samples": len(df),
        "n_components_pca": 10,
        "variance_explained": round(float(explained[-1]), 4),
        "contamination": CONTAMINATION,
        "n_anomalies": int(n_anomalies),
        "n_normal": int(n_normal),
        "anomaly_percentage": round(anomaly_pct, 2),
        "anomaly_score_mean": round(avg_score, 4),
        "anomaly_score_min": round(min_score, 4),
        "anomaly_score_max": round(max_score, 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {METRICS_PATH}")

    # ── PCA 2D scatter plot with anomalies highlighted
    print("Generating anomaly scatter plot...")
    plt.figure(figsize=(9, 6))
    normal_mask = anomaly_flags == 0
    anomaly_mask = anomaly_flags == 1

    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                c="#3498db", alpha=0.4, s=10, label=f"Normal ({n_normal})", edgecolors="none")
    plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1],
                c="#e74c3c", alpha=0.8, s=25, label=f"Anomaly ({n_anomalies})", edgecolors="none")

    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("PCA + Isolation Forest — Anomaly Detection")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/anomaly_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR}/anomaly_scatter.png")

    # ── Anomaly score distribution
    print("Generating anomaly score distribution...")
    plt.figure(figsize=(9, 4))
    plt.hist(scores[normal_mask], bins=50, alpha=0.6, color="#3498db", label="Normal")
    plt.hist(scores[anomaly_mask], bins=20, alpha=0.8, color="#e74c3c", label="Anomaly")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="Decision boundary")
    plt.xlabel("Anomaly Score (lower = more anomalous)")
    plt.ylabel("Count")
    plt.title("Isolation Forest — Anomaly Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/anomaly_scores.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR}/anomaly_scores.png")

    # ── Save models
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca_full, PCA_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"PCA saved: {PCA_PATH}")

    return metrics


if __name__ == "__main__":
    metrics = train()
    print("\n✓ Training complete!")