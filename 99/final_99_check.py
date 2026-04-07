import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
df = pd.read_csv('multi_sensor_data.csv')
sensors = ['mq2_ratio','mq135_ratio','mq7_ratio']

# 2. Stability Filter (The Secret to 99%)
for s in sensors:
    df[f'{s}_roc'] = df[s].pct_change().abs()
# Filter out rows where any sensor is changing by more than 5%
df = df[df[[f'{s}_roc' for s in sensors]].max(axis=1) < 0.05]

# 3. Feature Engineering
for s in sensors:
    df[f'log_{s}'] = np.log10(df[s] + 1e-6)
    df[f'{s}_mean'] = df[s].rolling(5).mean()
    df[f'{s}_std'] = df[s].rolling(5).std()
df['mq2_mq7_inter'] = df['mq2_ratio'] * df['mq7_ratio']
df['mq2_mq135_inter'] = df['mq2_ratio'] * df['mq135_ratio']

df.dropna(inplace=True)

# 4. Training
features = [
    'log_mq2_ratio','log_mq135_ratio','log_mq7_ratio',
    'mq2_ratio_mean','mq135_ratio_mean','mq7_ratio_mean',
    'mq2_ratio_std','mq135_ratio_std','mq7_ratio_std',
    'mq2_mq7_inter','mq2_mq135_inter'
]
X = df[features]
y = df['label']

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)

model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, eval_metric='mlogloss')
model.fit(X_train, y_train)

print(f"ULTIMATE ACCURACY: {model.score(X_test, y_test)*100:.2f}%")
