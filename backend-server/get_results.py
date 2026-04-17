import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("multi_sensor_data.csv")
df = df[df['label'] > 0]

# Mapping labels to approx PPM for fit guidance
ppm_map = {1: 10, 2: 50, 3: 100, 4: 300}
df['ppm_target'] = df['label'].map(ppm_map)

def get_fit(ratio_col, sensor_name):
    ratios = df[ratio_col].values
    ppms = df['ppm_target'].values
    valid = (np.isfinite(ratios)) & (ratios > 0)
    if len(ratios[valid]) < 2:
        return 0.0, 0.0
    m, b = np.polyfit(np.log10(ppms[valid]), np.log10(ratios[valid]), 1)
    return m, b

m2, b2 = get_fit('mq2_ratio', 'MQ2')
m135, b135 = get_fit('mq135_ratio', 'MQ135')
m7, b7 = get_fit('mq7_ratio', 'MQ7')

print("\n--- FINAL CALIBRATION CONSTANTS ---")
print(f"MQ-2:   M={m2:.6f}, B={b2:.6f}")
print(f"MQ-135: M={m135:.6f}, B={b135:.6f}")
print(f"MQ-7:   M={m7:.6f}, B={b7:.6f}")
