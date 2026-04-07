import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from micromlgen import port
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

# 2. Replicate High-Accuracy Feature Engineering
# (We must match exactly what the ESP32 will do)
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

# 3. Handle Imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 4. Train "Edge-Optimized" Model
# 15 trees is a sweet spot for ESP32 memory vs accuracy
print("Training Edge-Optimized Model...")
clf = RandomForestClassifier(n_estimators=15, max_depth=10, random_state=42)
clf.fit(X_res, y_res)

# 5. Export to C++
print("Exporting to C++ (TinyML)...")
c_code = port(clf, classname="GasClassifier", features=feature_cols)

output_path = "esp32_tinyml_gas/Model.h"
with open(output_path, "w") as f:
    f.write(c_code)

print(f"✅ Success! TinyML Model saved to: {output_path}")
print(f"Feature count: {len(feature_cols)}")
