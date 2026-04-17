# 🧪 SnO2-Gas-Analytics

An Adaptive Drift-Compensated Edge-Based Gas Monitoring and Anomaly Detection System built using ESP32 and MQ-2 (SnO₂) semiconductor gas sensor.

📌 Project Overview

SnO2-Gas-Analytics is an intelligent IoT-based gas monitoring system designed to:
- Perform real-time gas detection
- Compensate long-term sensor drift
- Extract electrical and transient features
- Run edge-based anomaly detection
- Log structured ML-ready datasets
- Provide real-time monitoring through a mobile app

Unlike traditional MQ-2 projects that only read raw ADC values, this system performs resistance modeling, adaptive baseline correction, and real-time anomaly detection.

🚀 Key Features
- **Electrical Modeling:** ADC to Voltage conversion, Sensor resistance (Rs) calculation, Baseline resistance (R0) calibration, Rs/R0 ratio computation, Log-scale PPM estimation.
- **Novel Contributions:** Adaptive Baseline Drift Correction, Edge-Based Z-Score Anomaly Detection, Transient Feature Extraction (dRs/dt, response time), Exposure Index Calculation, ML-ready structured dataset logging.

🧠 System Architecture
ESP32 (Edge) -> Feature Extraction -> WiFi -> Backend Server -> CSV Logging + ML Processing -> Mobile App Visualization

📊 Derived Features Logged
| Feature | Description |
| :--- | :--- |
| ADC | Raw analog value |
| Vrl | Load voltage |
| Rs | Sensor resistance |
| R0 | Baseline resistance |
| Rs/R0 | Normalized response |
| PPM | Log-scale estimated concentration |

🔬 Adaptive Baseline Algorithm
To compensate long-term drift:
R0_new = MovingAverage(Lowest 5% Rs values over 24h window)
This allows automatic recalibration during clean-air conditions.

🛠 Hardware Used
- ESP32 Dev Module
- MQ-2 Gas Sensor (SnO₂)
- 5V heater supply
- WiFi connectivity

📈 Machine Learning Pipeline
1. Feature normalization
2. Outlier detection
3. Isolation Forest / Z-score anomaly detection
4. Model export for edge deployment

---
License: MIT
Repository: [SnO2-Gas-Analytics](https://github.com/aashishniranjanb/SnO2-Gas-Analytics)
