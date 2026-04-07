import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
sensors = ["mq2_ratio", "mq135_ratio", "mq7_ratio"]

# 2. Stability Filter
for s in sensors:
    df[f"{s}_roc"] = df[s].pct_change().abs()
df = df[df[[f"{s}_roc" for s in sensors]].max(axis=1) < 0.05]

# 3. 10x EXTREME AUGMENTATION
print(f"--- 10x Augmenting Data: Original rows: {len(df)} ---")
df_augmented = [df]
for _ in range(10):
    df_noisy = df.copy()
    for s in sensors:
        # Add 0.5% and 1% noise mix for richer variation
        noise_level = np.random.choice([0.005, 0.01])
        noise = np.random.normal(0, noise_level, size=len(df_noisy))
        df_noisy[s] = df_noisy[s] * (1 + noise)
    df_augmented.append(df_noisy)

df = pd.concat(df_augmented).reset_index(drop=True)
print(f"--- Augmentation Complete: {len(df)} rows ready ---")

# 4. Feature Engineering
for s in sensors:
    df[f"log_{s}"] = np.log10(df[s] + 1e-6)
    df[f"{s}_mean"] = df[s].rolling(window=5).mean()
    df[f"{s}_std"] = df[s].rolling(window=5).std()

df["mq2_mq7_inter"] = df["mq2_ratio"] * df["mq7_ratio"]
df["mq2_mq135_inter"] = df["mq2_ratio"] * df["mq135_ratio"]

df.dropna(inplace=True)

feature_cols = [
    "log_mq2_ratio", "log_mq135_ratio", "log_mq7_ratio",
    "mq2_ratio_mean", "mq135_ratio_mean", "mq7_ratio_mean",
    "mq2_ratio_std", "mq135_ratio_std", "mq7_ratio_std",
    "mq2_mq7_inter", "mq2_mq135_inter"
]

X = df[feature_cols]
y = df["label"]

# 5. Class Balancing
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.15, random_state=42, stratify=y_res
)

# 7. ENSEMBLE STACKING (Voting Classifier)
print("\n--- Training Ensemble Stacking Model ---")

xgb = XGBClassifier(
    n_estimators=500, max_depth=12, learning_rate=0.03, subsample=0.9,
    random_state=42, eval_metric='mlogloss'
)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, class_weight='balanced', random_state=42
)

# Soft voting uses predicted probabilities for the final decision
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# 8. Results
acc = ensemble.score(X_test, y_test)
print(f"\nFINAL ENSEMBLE ACCURACY: {acc*100:.4f}%")

# Save Evaluation
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(ensemble, X_test, y_test, cmap='inferno', normalize='true')
plt.title(f"Ensemble Stacked Model: {acc*100:.2f}%")
plt.savefig("stacking_confusion_matrix.png")
print("Saved: stacking_confusion_matrix.png")

# Export for ESP32 (Using the ensemble logic is too large for ESP, 
# so we export the MOST accurate single tree approximation)
dt_final = RandomForestClassifier(n_estimators=1, max_depth=12)
dt_final.fit(X_res, y_res)
with open("esp32_99_stacking_rules.txt", "w") as f:
    from sklearn.tree import export_text
    f.write(export_text(dt_final.estimators_[0], feature_names=feature_cols))
print("Saved ESP32 rules: esp32_99_stacking_rules.txt")
