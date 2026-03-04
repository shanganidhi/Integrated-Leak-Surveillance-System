# 🛡️ Integrated Leak Surveillance System

Integrated Leak Surveillance System is an adaptive, edge-based multi-sensor leak and air-quality monitoring platform built using ESP32 and multiple MQ series gas sensors (MQ-2, MQ-135, MQ-7).

📌 Project Overview

This project provides firmware, backend, and mobile components for:
- Multi-sensor real-time gas detection (MQ-2, MQ-135, MQ-7)
- Adaptive baseline / drift compensation
- Electrical and transient feature extraction
- Edge-based anomaly detection and structured dataset logging
- Real-time visualization via a mobile app

Unlike traditional MQ-2 projects that only read raw ADC values, this system performs resistance modeling, adaptive baseline correction, and real-time anomaly detection.

🚀 Key Features
- **Electrical Modeling:** ADC to Voltage conversion, Sensor resistance (Rs) calculation, Baseline resistance (R0) calibration, Rs/R0 ratio computation, Log-scale PPM estimation for each sensor (MQ-2, MQ-135, MQ-7).
- **Novel Contributions:** Adaptive Baseline Drift Correction, Edge-Based Z-Score Anomaly Detection, Transient Feature Extraction (dRs/dt, response time), Exposure Index Calculation, ML-ready structured dataset logging.

🧠 System Architecture
ESP32 (Edge) -> Multi-sensor Feature Extraction -> WiFi -> Backend Server -> CSV Logging + ML Processing -> Mobile App Visualization

📊 Derived Features Logged
| Feature | Description |
| :--- | :--- |
| ADC | Raw analog value for each sensor |
| Vrl | Load voltage for each sensor |
| Rs | Sensor resistance for each sensor |
| R0 | Baseline resistance (per-sensor calibration) |
| Rs/R0 | Normalized response |
| PPM | Log-scale estimated concentration (per-sensor, model-dependent) |

🔬 Adaptive Baseline Algorithm
To compensate long-term drift:
R0_new = MovingAverage(Lowest 5% Rs values over 24h window)
This allows automatic recalibration during clean-air conditions.

🛠 Hardware Used
- ESP32 Dev Module
- MQ-2 Gas Sensor (SnO₂)
- MQ-135 Gas Sensor (air-quality, NH3, benzene, alcohols)
- MQ-7 Gas Sensor (carbon monoxide)
- 5V heater supply
- WiFi connectivity

📈 Machine Learning Pipeline
1. Feature normalization
2. Outlier detection
3. Isolation Forest / Z-score anomaly detection
4. Model export for edge deployment

---
License: MIT
