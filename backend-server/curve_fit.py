import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Target file containing the multi-sensor dataset
filename = "multi_sensor_data.csv"

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit()

df = pd.read_csv(filename)

# Filter for rows where label > 0 (gas exposure)
# Supervised training requires labeled events beyond the baseline.
df_exposed = df[df["label"] > 0].copy()

if len(df_exposed) == 0:
    print("===================================================================")
    print("No gas exposure data (label > 0) found in the dataset yet.")
    print("Please record some events setting labels 1-4 on your ESP32 first.")
    print("===================================================================")
    exit()

# Map labels to estimated PPM values for supervised log-regression mapping.
# These values are empirical approximations to guide the curve fitting.
ppm_map = {
    1: 10,
    2: 50,
    3: 100,
    4: 300
}

df_exposed["ppm_target"] = df_exposed["label"].map(ppm_map)

def fit_sensor_curve(sensor_name, ratio_col):
    print(f"\n--- Fitting log-regression for {sensor_name} ---")
    
    # Extract ratios and target PPMs
    ratios = df_exposed[ratio_col].values
    ppms = df_exposed["ppm_target"].values
    
    # Clean data to avoid log errors (drop NaNs, non-positives)
    valid = (np.isfinite(ratios)) & (ratios > 0) & (np.isfinite(ppms)) & (ppms > 0)
    log_ratios = np.log10(ratios[valid])
    log_ppms = np.log10(ppms[valid])
    
    if len(log_ratios) < 2:
        print(f"Not enough valid data points to fit a curve for {sensor_name}.")
        return None, None
    
    # Linear fit: log10(Ratio) = m * log10(PPM) + b
    # This follows the power law relationship Ratio = scale * PPM^m
    m, b = np.polyfit(log_ppms, log_ratios, 1)
    
    print(f"Computed Slope (m): {m:.4f}")
    print(f"Computed Intercept (b): {b:.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(log_ppms, log_ratios, label="Experimental Data", alpha=0.6, edgecolors='k')
    plt.plot(log_ppms, m * log_ppms + b, color='red', linewidth=2, label=f"Log-Fit: y={m:.2f}x+{b:.2f}")
    plt.xlabel("log10(PPM target)")
    plt.ylabel(f"log10({ratio_col} observed)")
    plt.title(f"{sensor_name} Empirical Sensitivity Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = f"{sensor_name.lower().replace('-', '_')}_curve.png"
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    
    return m, b

# Compute curves for the sensor array
results = {}
results["MQ-2"] = fit_sensor_curve("MQ-2", "mq2_ratio")
results["MQ-135"] = fit_sensor_curve("MQ-135", "mq135_ratio")
results["MQ-7"] = fit_sensor_curve("MQ-7", "mq7_ratio")

print("\n================== FIRMWARE UPDATE RECOMMENDATIONS ==================")
print("Copy these constants into your Arduino code for localized PPM calculation:")
for sensor, res in results.items():
    if res and res[0] is not None:
        m, b = res
        prefix = sensor.replace('-', '')
        print(f"\n// {sensor} Constants")
        print(f"#define {prefix}_M_SLOPE {m:.4f}")
        print(f"#define {prefix}_B_INTERCEPT {b:.4f}")
print("======================================================================")

# plt.show()
