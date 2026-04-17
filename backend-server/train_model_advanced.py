import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, export_text
import os

# 1. Load Dataset
filename = "multi_sensor_data.csv"
if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit()

df = pd.read_csv(filename)

# 2. Advanced Feature Engineering (Sliding Window)
# We calculate moving averages and standard dev to smooth noise and detect trends
window_size = 5
base_features = ["mq2_ratio", "mq135_ratio", "mq7_ratio", "mq2_delta", "mq135_delta", "mq7_delta"]

# Create rolling features
for feat in base_features:
    df[f"{feat}_mean"] = df[feat].rolling(window=window_size).mean()
    df[f"{feat}_std"] = df[feat].rolling(window=window_size).std()

# Drop the first few rows that have NaNs due to the window
advanced_features = [f"{feat}_mean" for feat in base_features] + [f"{feat}_std" for feat in base_features]
df_clean = df.dropna(subset=advanced_features + ["label"])

X = df_clean[advanced_features]
y = df_clean["label"]

# 3. Standardization (Scale features to mean=0, std=1)
# This is critical for making various sensors comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training: Advanced Random Forest
print("\n--- Training Advanced Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=12, 
    class_weight="balanced", # Handles imbalance
    random_state=42
)
rf_model.fit(X_train, y_train)

# 6. Evaluation
acc = rf_model.score(X_test, y_test)
print(f"--- Accuracy Achieved: {acc*100:.2f}% ---")

y_pred = rf_model.predict(X_test)
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Plot Normalized Confusion Matrix (IEEE standard)
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Greens', normalize='true')
plt.title(f"Advanced Model Accuracy: {acc*100:.1f}%")
plt.savefig("advanced_confusion_matrix.png")
print("Saved analysis: advanced_confusion_matrix.png")

# 7. Extract simplified logic for ESP32
# Since scaled/windowed features are complex, we'll fit a simpler DT on the same data
print("\n--- Exporting Simplified Edge Rules ---")
dt_light = DecisionTreeClassifier(max_depth=5, class_weight="balanced")
dt_light.fit(X_train, y_train)

tree_rules = export_text(dt_light, feature_names=advanced_features)
with open("esp32_advanced_rules.txt", "w") as f:
    f.write(tree_rules)
    # Also save scaling parameters
    f.write("\n\n--- Standard Scaler Values (Mean / Scale) ---\n")
    for i, name in enumerate(advanced_features):
        f.write(f"{name}: mean={scaler.mean_[i]:.4f}, scale={scaler.scale_[i]:.4f}\n")

print("Success! Rules and Scaler constants saved to esp32_advanced_rules.txt")
# plt.show() # Commented for non-blocking
