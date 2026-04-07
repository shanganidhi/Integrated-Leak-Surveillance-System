import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('multi_sensor_data.csv')
sensors=['mq2_ratio','mq135_ratio','mq7_ratio']
for s in sensors:
    df[f'log_{s}']=np.log10(df[s]+1e-6)
    df[f'{s}_m']=df[s].rolling(5).mean()
    df[f'{s}_s']=df[s].rolling(5).std()
df['mq2_7']=df['mq2_ratio']*df['mq7_ratio']
df.dropna(inplace=True)
features = ['log_mq2_ratio','log_mq135_ratio','log_mq7_ratio','mq2_ratio_m','mq135_ratio_m','mq7_ratio_m','mq2_ratio_s','mq135_ratio_s','mq7_ratio_s','mq2_7']
X=df[features]
y=df['label']
sm=SMOTE(random_state=42)
X_b,y_b=sm.fit_resample(X,y)
X_tr,X_te,y_tr,y_te=train_test_split(X_b,y_b,test_size=0.3,random_state=42,stratify=y_b)
m=XGBClassifier(n_estimators=200,max_depth=6,eval_metric='mlogloss')
m.fit(X_tr,y_tr)
print(f"Accuracy: {m.score(X_te,y_te)*100:.2f}%")
