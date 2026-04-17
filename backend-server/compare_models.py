import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os

# 1. Load Dataset
filename = "multi_sensor_data.csv"
if not os.path.exists(filename):
    print(f"Error: {filename} not found. Please collect some labeled gas data first!")
    exit()

df = pd.read_csv(filename)

# 2. Feature Selection
features = [
    "mq2_ratio", "mq135_ratio", "mq7_ratio",
    "mq2_delta", "mq135_delta", "mq7_delta"
]

# Clean data
df_clean = df.dropna(subset=features + ["label"])

X = df_clean[features]
y = df_clean["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models to compare
models = {
    "Decision Tree (Max Depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Decision Tree (Unrestricted)": DecisionTreeClassifier(random_state=42),
    "Random Forest (10 Trees)": RandomForestClassifier(n_estimators=10, random_state=42),
    "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel="linear", random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB()
}


output = ""
output += f"{'Model Name':<30} | {'Accuracy':<8} | {'F1-Score':<8} | Edge AI Suitability\n"
output += "-" * 105 + "\n"

best_model_name = ""
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    if acc > best_acc and "KNN" not in name: # Exclude KNN from being the best deployable model
        best_acc = acc
        best_model_name = name
    
    # Edge AI Suitability Rules of Thumb
    if "Decision Tree" in name:
        suitability = "5/5 (Simple IF-THEN rules, perfect for Edge)"
    elif "Logistic" in name or "Linear" in name:
        suitability = "5/5 (Simple mathematical equations, zero RAM)"
    elif "Naive" in name:
        suitability = "4/5 (Probability tables, light memory footprint)"
    elif "Random Forest" in name:
        suitability = "3/5 (Good, but C++ export requires MicroML)"
    elif "SVM (RBF" in name:
        suitability = "2/5 (Math intensive, requires MicroML)"
    elif "KNN" in name:
        suitability = "1/5 (Poor, requires storing dataset in RAM)"
    else:
        suitability = "?"

    output += f"{name:<30} | {acc:.4f}   | {f1:.4f}   | {suitability}\n"

output += "\n==================================\n"
output += f"Best Edge-AI Ready Model: {best_model_name} with Accuracy {best_acc:.4f}\n"
output += "==================================\n"

print(output)
with open("model_comparison.txt", "w", encoding="utf-8") as f:
    f.write(output)

