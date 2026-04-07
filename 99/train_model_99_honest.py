import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# 1. Load Dataset
filename = "multi_sensor_data.csv"
if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit()

df = pd.read_csv(filename)
# Add original index to track samples during augmentation
df['original_index'] = df.index
sensors = ["mq2_ratio", "mq135_ratio", "mq7_ratio"]

# 2. Feature Engineering (BEFORE Augmentation/Split to avoid leakage)
print("--- Step 1: Feature Engineering ---")
for s in sensors:
    df[f"log_{s}"] = np.log10(df[s] + 1e-6)
    # Use center-aligned rolling to keep features temporal but local
    df[f"{s}_mean"] = df[s].rolling(window=5, center=True).mean()
    df[f"{s}_std"] = df[s].rolling(window=5, center=True).std()

df["mq2_mq7_inter"] = df["mq2_ratio"] * df["mq7_ratio"]
df["mq2_mq135_inter"] = df["mq2_ratio"] * df["mq135_ratio"]

# Stability Filter
for s in sensors:
    df[f"{s}_roc"] = df[s].pct_change().abs()
df = df[df[[f"{s}_roc" for s in sensors]].max(axis=1) < 0.05]

df.dropna(inplace=True)

feature_cols = [
    "log_mq2_ratio", "log_mq135_ratio", "log_mq7_ratio",
    "mq2_ratio_mean", "mq135_ratio_mean", "mq7_ratio_mean",
    "mq2_ratio_std", "mq135_ratio_std", "mq7_ratio_std",
    "mq2_mq7_inter", "mq2_mq135_inter"
]

# 3. Honest Split (Based on groups of original samples)
# This ensures that noisy versions of the SAME data point don't leak into test set
print("--- Step 2: Honest Data Splitting ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['original_index']))

df_train_raw = df.iloc[train_idx]
df_test = df.iloc[test_idx]

# 4. Augmentation (ONLY on Train Set)
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

X_train = df_train[feature_cols]
y_train = df_train["label"]

X_test = df_test[feature_cols]
y_test = df_test["label"]

# 5. Class Balancing (SMOTE)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. SUPER-AI STACKING ENSEMBLE
print("\n--- Step 4: Training Honest Super-AI Stacking Ensemble ---")
xgb = XGBClassifier(n_estimators=400, max_depth=12, learning_rate=0.03, subsample=0.8, random_state=42, eval_metric='mlogloss')
rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, alpha=0.0001, random_state=42)

super_model = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)],
    final_estimator=LogisticRegression(),
    passthrough=True
)

super_model.fit(X_train_res, y_train_res)

# 7. Ultimate Results
acc = super_model.score(X_test, y_test)
print(f"\n🏆 HONEST STACKED ACCURACY: {acc*100:.4f}%")

y_pred = super_model.predict(X_test)
print("\n--- Final Scientific Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(super_model, X_test, y_test, cmap='viridis', normalize='true')
plt.title(f"Honest Meta-Stacked AI: {acc*100:.2f}%")
plt.savefig("honest_super_ai_confusion_matrix.png")

# Educational distillation
from sklearn.tree import DecisionTreeClassifier, export_text
dt_distilled = DecisionTreeClassifier(max_depth=12)
y_pseudo = super_model.predict(X_train_res)
dt_distilled.fit(X_train_res, y_pseudo)

with open("esp32_99_honest_rules.txt", "w") as f:
    f.write(export_text(dt_distilled, feature_names=feature_cols))
print("\nSuccess! Final Honest results saved to honest_super_ai_confusion_matrix.png and esp32_99_honest_rules.txt")
