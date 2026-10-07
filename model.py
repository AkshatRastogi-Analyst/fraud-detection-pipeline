"""
Step 2 - Machine learning: read the clean table, train 2 models, evaluate properly.

Usage:  python model.py
Output: outputs/metrics.csv, outputs/confusion_matrix.png, outputs/pr_curve.png,
        outputs/feature_importance.png, outputs/predictions.csv  (use predictions.csv in Power BI)
"""
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             precision_recall_curve, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_URL = os.environ.get("DB_URL")
SQLITE_PATH = "outputs/fraud.db"
RANDOM_STATE = 42
os.makedirs("outputs", exist_ok=True)

# ---------- LOAD ----------
if DB_URL:
    from sqlalchemy import create_engine
    df = pd.read_sql("SELECT * FROM transactions_clean", create_engine(DB_URL))
else:
    with sqlite3.connect(SQLITE_PATH) as conn:
        df = pd.read_sql("SELECT * FROM transactions_clean", conn)
print(f"Loaded {len(df):,} rows | fraud rate = {df['Class'].mean():.4%}")

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour_of_day"]
X, y = df[FEATURES], df["Class"]

# Stratified split keeps the same tiny fraud % in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------- MODELS ----------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
}

rows, probs = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    rows.append({
        "model": name,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, p),
        "pr_auc": average_precision_score(y_test, p),   # best single metric for imbalanced data
    })
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("outputs/metrics.csv", index=False)
print("\n", metrics.to_string(index=False))

best_name = metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]
best = models[best_name]
print(f"\nBest model by PR-AUC: {best_name}")

# ---------- CHARTS ----------
# 1) Confusion matrix of best model
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, (probs[best_name] >= 0.5).astype(int), display_labels=["Legit", "Fraud"], ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix - {best_name}")
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png", dpi=150); plt.close(fig)

# 2) Precision-recall curves
fig, ax = plt.subplots(figsize=(6, 4))
for name, p in probs.items():
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(y_test, p):.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve"); ax.legend()
fig.tight_layout(); fig.savefig("outputs/pr_curve.png", dpi=150); plt.close(fig)

# 3) Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(6, 4))
imp.plot.barh(ax=ax); ax.set_title("Top features - Random Forest")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150); plt.close(fig)

# ---------- EXPORT FOR POWER BI ----------
out = X_test.copy()
out["actual_class"] = y_test.values
out["fraud_probability"] = probs[best_name]
out["predicted_class"] = (out["fraud_probability"] >= 0.5).astype(int)
out["risk_band"] = pd.cut(out["fraud_probability"], [-0.01, 0.2, 0.5, 0.8, 1.0],
                          labels=["Low", "Medium", "High", "Critical"]).astype(str)
out.to_csv("outputs/predictions.csv", index=False)
print("\nSaved metrics, charts and predictions.csv in outputs/")
