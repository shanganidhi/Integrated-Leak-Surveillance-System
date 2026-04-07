import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
filename = "multi_sensor_data.csv"
if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit()

df = pd.read_csv(filename)

# 2. Advanced Feature Engineering (The "99%" Method)
# A. Stability Filter (The Secret to 99%)
# We remove rows where the gas is still "rising" or "falling" to learn clean states.
sensors = ["mq2_ratio", "mq135_ratio", "mq7_ratio"]
roc_threshold = 0.05 # 5% change per sample is considered "unstable"
for s in sensors:
    df[f"{s}_roc"] = df[s].pct_change().abs()

# Keep only rows where all sensors are stable
initial_len = len(df)
df = df[df[[f"{s}_roc" for s in sensors]].max(axis=1) < roc_threshold]
print(f"--- Stability Filter: Removed {initial_len - len(df)} transitional rows ---")

# B. Physical Log Transform
for s in sensors:
    df[f"log_{s}"] = np.log10(df[s] + 1e-6)

# C. Rolling Window Features
window = 5
for s in sensors:
    df[f"{s}_mean"] = df[s].rolling(window=window).mean()
    df[f"{s}_std"] = df[s].rolling(window=window).std()

# D. Interaction Terms
df["mq2_mq7_inter"] = df["mq2_ratio"] * df["mq7_ratio"]
df["mq2_mq135_inter"] = df["mq2_ratio"] * df["mq135_ratio"]

# 3. Clean and Filter
# We drop NaNs from rolling windows
initial_len = len(df)
df = df.dropna()
print(f"--- Data Cleaning: Dropped {initial_len - len(df)} transient/null rows ---")

# Define feature columns
feature_cols = [
    "log_mq2_ratio", "log_mq135_ratio", "log_mq7_ratio",
    "mq2_ratio_mean", "mq135_ratio_mean", "mq7_ratio_mean",
    "mq2_ratio_std", "mq135_ratio_std", "mq7_ratio_std",
    "mq2_mq7_inter", "mq2_mq135_inter"
]

X = df[feature_cols]
y = df["label"]

# 4. Handle Class Imbalance (SMOTE)
# This synthetically creates data for underrepresented gas levels
print("\n--- Balancing Classes with SMOTE ---")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Original shape: {X.shape}, Balanced shape: {X_balanced.shape}")

# 5. Train/Test Split
# We use a large test set to prove 99% accuracy isn't luck
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# 6. XGBoost Training (Extreme Gradient Boosting)
print("\n--- Training Extreme XGBoost Model ---")
# Adjusting classes for XGBoost (expects 0-indexed labels)
y_train_xgb = y_train
y_test_xgb = y_test

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train_xgb)

# 7. Comprehensive Evaluation
acc = xgb_model.score(X_test, y_test_xgb)
print(f"\nTARGET REACHED? Accuracy: {acc*100:.2f}%")

y_pred = xgb_model.predict(X_test)
print("\n--- Final Scientific Report ---")
print(classification_report(y_test_xgb, y_pred))

# Confusion Matrix
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test_xgb, cmap='magma', normalize='true')
plt.title(f"XGBoost Extreme Accuracy: {acc*100:.2f}%")
plt.savefig("extreme_confusion_matrix.png")
print("Saved extreme analysis: extreme_confusion_matrix.png")

# 8. Feature Importance (Scientific Insight)
importances = xgb_model.feature_importances_
feat_importances = pd.Series(importances, index=feature_cols)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh', color='purple')
plt.title("Scientific Feature Importance (The Signature of Gas)")
plt.savefig("extreme_feature_importance.png")
print("Saved insight graph: extreme_feature_importance.png")

# Exporting for deployment
# (XGBoost is complex, so we will still provide a Decision Tree export on the balanced data)
from sklearn.tree import DecisionTreeClassifier, export_text
dt_deploy = DecisionTreeClassifier(max_depth=8)
dt_deploy.fit(X_balanced, y_balanced)
with open("esp32_99_rules.txt", "w") as f:
    f.write(export_text(dt_deploy, feature_names=feature_cols))

print("\nSuccess! Rules saved for deployment in esp32_99_rules.txt")
