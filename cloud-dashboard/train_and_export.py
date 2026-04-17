"""
Train the Stacking Ensemble model and export it as model.joblib
for the Flask cloud dashboard to load at startup.

Replicates the exact pipeline from train_model_99_honest.py
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# ── 1. Locate Dataset ──────────────────────────────────────────
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "99", "multi_sensor_data.csv"),
    os.path.join(os.path.dirname(__file__), "..", "backend-server", "multi_sensor_data.csv"),
]

filename = None
for p in DATA_PATHS:
    if os.path.exists(p):
        filename = p
        break

if filename is None:
    print("Error: multi_sensor_data.csv not found in any expected location.")
    sys.exit(1)

print(f"Loading dataset from: {os.path.abspath(filename)}")
df = pd.read_csv(filename)
df['original_index'] = df.index
sensors = ["mq2_ratio", "mq135_ratio", "mq7_ratio"]

# ── 2. Feature Engineering ──────────────────────────────────────
print("--- Step 1: Feature Engineering ---")
for s in sensors:
    df[f"log_{s}"] = np.log10(df[s] + 1e-6)
    df[f"{s}_mean"] = df[s].rolling(window=5, center=True).mean()
    df[f"{s}_std"] = df[s].rolling(window=5, center=True).std()

df["mq2_mq7_inter"] = df["mq2_ratio"] * df["mq7_ratio"]
df["mq2_mq135_inter"] = df["mq2_ratio"] * df["mq135_ratio"]

# Stability Filter
for s in sensors:
    df[f"{s}_roc"] = df[s].pct_change().abs()
df = df[df[[f"{s}_roc" for s in sensors]].max(axis=1) < 0.05]
df.dropna(inplace=True)

FEATURE_COLS = [
    "log_mq2_ratio", "log_mq135_ratio", "log_mq7_ratio",
    "mq2_ratio_mean", "mq135_ratio_mean", "mq7_ratio_mean",
    "mq2_ratio_std", "mq135_ratio_std", "mq7_ratio_std",
    "mq2_mq7_inter", "mq2_mq135_inter"
]

# ── 3. Honest Split ────────────────────────────────────────────
print("--- Step 2: Honest Data Splitting ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['original_index']))

df_train_raw = df.iloc[train_idx]
df_test = df.iloc[test_idx]

# ── 4. Augmentation (Train Only) ──────────────────────────────
print(f"--- Step 3: Augmenting Train Data (Original: {len(df_train_raw)}) ---")
df_augmented = [df_train_raw]
for _ in range(20):
    df_noisy = df_train_raw.copy()
    for s in sensors:
        level = np.random.uniform(0.005, 0.015)
        noise = np.random.normal(0, level, size=len(df_noisy))
        df_noisy[s] = df_noisy[s] * (1 + noise)
    df_augmented.append(df_noisy)

df_train = pd.concat(df_augmented).reset_index(drop=True)
print(f"--- Augmentation Complete: {len(df_train)} training rows ---")

X_train = df_train[FEATURE_COLS]
y_train = df_train["label"]
X_test = df_test[FEATURE_COLS]
y_test = df_test["label"]

# ── 5. SMOTE ──────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ── 6. Stacking Ensemble ─────────────────────────────────────
print("\n--- Step 4: Training Stacking Ensemble ---")
xgb = XGBClassifier(
    n_estimators=400, max_depth=12, learning_rate=0.03,
    subsample=0.8, random_state=42, eval_metric='mlogloss'
)
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    class_weight='balanced', random_state=42
)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), max_iter=1000,
    alpha=0.0001, random_state=42
)

model = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)],
    final_estimator=LogisticRegression(),
    passthrough=True
)

model.fit(X_train_res, y_train_res)

# ── 7. Evaluate ──────────────────────────────────────────────
acc = model.score(X_test, y_test)
print(f"\n[RESULT] HONEST STACKED ACCURACY: {acc*100:.2f}%")
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ── 8. Export ────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), "model.joblib")
joblib.dump({
    'model': model,
    'feature_cols': FEATURE_COLS,
    'accuracy': acc,
    'label_map': {
        0: 'Clean Air',
        1: 'Low Gas',
        2: 'Medium Gas',
        3: 'High Gas',
        4: 'Critical'
    }
}, output_path)

print(f"\n[OK] Model exported to: {os.path.abspath(output_path)}")
print(f"   Features: {len(FEATURE_COLS)}")
print(f"   Accuracy: {acc*100:.2f}%")
