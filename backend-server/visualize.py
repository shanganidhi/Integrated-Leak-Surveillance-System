import pandas as pd
import matplotlib.pyplot as plt

filename = "multi_sensor_data.csv"

try:
    df = pd.read_csv(filename)
    
    # Drop rows with NaN in rs columns to plot cleanly
    df = df.dropna(subset=['mq2_rs', 'mq135_rs', 'mq7_rs'])
    
    # Create subplots for the three sensors
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(df["mq2_rs"], color='blue')
    axes[0].set_title("MQ-2 Resistance (Rs) over Time")
    axes[0].set_ylabel("Resistance (Ω)")
    axes[0].grid(True)
    
    axes[1].plot(df["mq135_rs"], color='green')
    axes[1].set_title("MQ-135 Resistance (Rs) over Time")
    axes[1].set_ylabel("Resistance (Ω)")
    axes[1].grid(True)
    
    axes[2].plot(df["mq7_rs"], color='red')
    axes[2].set_title("MQ-7 Resistance (Rs) over Time")
    axes[2].set_ylabel("Resistance (Ω)")
    axes[2].set_xlabel("Sample Index")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: {filename} not found.")
except Exception as e:
    print(f"An error occurred while trying to plot: {e}")
