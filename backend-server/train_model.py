import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, export_text
import os

# 1. Load Dataset
filename = "multi_sensor_data.csv"
if not os.path.exists(filename):
    print(f"Error: {filename} not found. Please collect some labeled gas data first!")
    exit()

df = pd.read_csv(filename)

# 2. Feature Selection (Research-grade selection)
features = [
    "mq2_ratio", "mq135_ratio", "mq7_ratio",
    "mq2_delta", "mq135_delta", "mq7_delta"
]

# Clean data: drop rows with NaNs in features (especially the first row Deltas)
df_clean = df.dropna(subset=features + ["label"])

X = df_clean[features]
y = df_clean["label"]

# 3. Data Distribution Check
print("\n--- Class Distribution ---")
print(y.value_counts(normalize=True) * 100)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training: Random Forest
print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluation
y_pred = rf_model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues')
plt.title("Random Forest: Gas Classification Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Saved evaluation graph: confusion_matrix.png")

# 7. Feature Importance (Critical for IEEE Paper)
importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=features)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(6).plot(kind='barh')
plt.title("Sensor Feature Importance")
plt.xlabel("Importance Score")
plt.savefig("feature_importance.png")
print("Saved research graph: feature_importance.png")

# 8. ESP32 Deployment: Decision Tree Approximation
# Random Forest is too large for ESP32 RAM if not using specialized libraries.
# We train a simpler Decision Tree on the SAME data to get "human readable" rules.
print("\n--- Generating Lightweight Rules for ESP32 ---")
dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train, y_train)

tree_rules = export_text(dt_model, feature_names=features)
print("\n[ESP32 IF-THEN Rules]:")
print(tree_rules)

# Save the rules to a file for easy copying
with open("esp32_rules.txt", "w") as f:
    f.write(tree_rules)
    
print("\nSuccess! Rules saved to esp32_rules.txt")
# plt.show()
