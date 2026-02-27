# MQ-2 Calibration Guide

Proper calibration is essential for accurate gas concentration estimation.

## 1. Preheating (Burn-in)
New SnO2 sensors require a significant preheating period.
- **Duration:** Minimum 24-48 hours.
- **Purpose:** Stabilizes the sensor resistance (Rs) in clean air.

## 2. R0 Calculation (Clean Air)
The baseline resistance `R0` is the resistance of the sensor in clean air.
1. Run the system in clean outdoor air for 30 minutes.
2. Note the average `Rs` value.
3. Calculate `R0 = Rs / 9.8` (The datasheet ratio for clean air is approx 9.8).
4. Update `float R0` in `hardware-esp32/esp32_mq2_main.ino`.

## 3. RL Identification
Check the load resistor on your MQ-2 module.
- 102 Code = 1,000 ohms (1k)
- 202 Code = 2,000 ohms (2k)
- Most "Better" setups use a 20k resistor for higher sensitivity at lower concentrations.
