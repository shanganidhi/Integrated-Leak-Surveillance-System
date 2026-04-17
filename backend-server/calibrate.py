import pandas as pd
import numpy as np

# Path to the dataset collected over the 7 hour baseline air run
filename = "multi_sensor_data.csv"

try:
    df = pd.read_csv(filename, encoding='utf-8')
    # Filter for baseline run where label is 0 (clean air)
    df_clean = df[df['label'] == 0]
    
    if len(df_clean) == 0:
        print("No clean air data (label 0) found in the dataset.")
        exit()

    def compute_R0(rs_values, ratio, sensor_name):
        # Drop NaNs just in case
        rs_values = rs_values.dropna()
        if len(rs_values) == 0:
            return None

        # Sort the Rs values
        rs_sorted = np.sort(rs_values)
        
        # Take the top 10% of Rs values (highest resistance = cleanest air)
        top_10_percent = rs_sorted[int(0.9 * len(rs_sorted)):]
        
        # Calculate the average of these top 10% values
        avg_rs = np.mean(top_10_percent)
        
        # Calculate R0 based on the clean air ratio from the datasheet
        r0 = avg_rs / ratio
        
        print(f"[{sensor_name}] Top 10% Avg Rs: {avg_rs:.2f} Ohms -> R0: {r0:.2f} Ohms")
        return r0

    print("================ CALIBRATION RESULTS ================")
    # Clean air ratios taken from respective sensor datasheets
    # MQ-2 clean air ratio ≈ 9.8
    R0_mq2 = compute_R0(df_clean["mq2_rs"], 9.8, "MQ-2")
    
    # MQ-135 clean air ratio ≈ 3.6
    R0_mq135 = compute_R0(df_clean["mq135_rs"], 3.6, "MQ-135")
    
    # MQ-7 clean air ratio ≈ 27.0
    R0_mq7 = compute_R0(df_clean["mq7_rs"], 27.0, "MQ-7")
    
    print("=====================================================")
    print("Replace these R0 values in your ESP32 code after your 7-hour run!")

except FileNotFoundError:
    print(f"Error: {filename} not found.")
except pd.errors.EmptyDataError:
    print(f"Error: {filename} is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
