# Research Novelty

SnO2-Gas-Analytics introduces several high-impact innovations to basic gas sensing:

## 1. Adaptive Baseline Drift Correction
Instead of a static `R0`, the system implements a moving average of the local minima (lowest 5% of Rs values). This compensates for sensor aging and environmental drift without manual recalibration.

## 2. Edge-Based Anomaly Detection
The system calculates a running Z-score on-device. If `| (Rs - mean) / std | > threshold`, an anomaly is flagged immediately at the edge, reducing latency and reliance on the cloud.

## 3. Transient Response Feature Extraction
By monitoring `dRs/dt` (the rate of change), the system can potentially classify the type of gas based on its unique adsorption/desorption fingerprint.
