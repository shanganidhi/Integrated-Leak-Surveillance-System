# Future Improvements & ML Deployment Plan

This document outlines the roadmap for elevating the custom gas detection system into a research-grade IoT / ML pipeline. It details the process from finalizing calibration to deploying Edge AI on the ESP32.

## Phase 1: Baseline Calibration (Current)
- **Goal:** Establish a robust 7-hour clean air dataset.
- **Process:**
  - Run the ESP32 array measuring MQ-2, MQ-135, and MQ-7 continuously for 7 hours in a clean environment.
  - Log data to the local server.
  - Run `calibrate.py` to extract the top 10% highest Resistance ($R_s$) values (indicative of the cleanest air readings).
  - Compute the precise baseline resistance ($R_0$) using datasheet specific clean air ratios.
- **Outcome:** Hardcode empirically validated $R_0$ values back into the Arduino sketch.

## Phase 2: Controlled Gas Exposure & Data Collection
- **Goal:** Build the labeled dataset required for supervised machine learning.
- **Process:**
  - Introduce controlled gas emissions separately to map specific sensor responses:
    - **LPG/Smoke:** Target MQ-2
    - **Ammonia/Perfume/Alcohol:** Target MQ-135
    - **Carbon Monoxide Sources:** Target MQ-7
  - Use the Serial Monitor to set manual labels (0 = Clean Air, 1 = Scenario A, 2 = Scenario B, etc.).
  - Continuously log the derived electrical parameters ($R_s$, Ratio, PPM estimates) rather than just raw ADC values.

## Phase 3: Machine Learning Model Development
- **Goal:** Train a model capable of multi-sensor data fusion to classify exposure levels and differentiate gas types.
- **Models to Evaluate:**
  - **Decision Tree Classifier:** Lightweight and highly suitable for translation into C++ IF/ELSE rules for Edge AI deployment.
  - **Random Forest:** Better overall accuracy, suitable for a cloud or edge-server deployment.
  - **Isolation Forest / One-Class SVM:** Suitable for pure anomaly detection (distinguishing non-clean air from clean air without knowing the specific gas).
- **Process:**
  - Split dataset into training and testing sets.
  - Feature engineering: Utilize the $R_s$/$R_0$ ratios from all three sensors concurrently.
  - Generate a Confusion Matrix to evaluate accuracy, precision, and recall.

## Phase 4: Edge AI & IoT Upgrades
- **Goal:** Move intelligence directly to the ESP32 to reduce reliance on the centralized server.
- **Process:**
  - Export trained Decision Tree to a C header file (using `m2cgen` or similar tools).
  - Integrate the exported model into the ESP32 sketch so it actively classifies data locally.
  - **Networking Upgrade:** Migrate from HTTP POST to MQTT for lower latency and better power efficiency.
  - **Cloud Output:** Connect the MQTT broker to Firebase or a modern React/Flutter dashboard for real-time visualization.
  - **Paper/Project Deliverable:** Present the Confusion Matrix, the response/recovery times, and the accuracy of the local ESP32 predictions compared to lab setups.
