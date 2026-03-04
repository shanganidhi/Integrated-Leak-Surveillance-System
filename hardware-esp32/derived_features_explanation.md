# Derived Electrical Features

Instead of raw ADC values, we extract physically meaningful features.

## 1. Sensor Resistance (Rs)
Measured using the voltage divider formula:
`Rs = ((VCC - Vrl) * RL) / Vrl`
This tracks the actual chemical change on the SnO2 surface.

## 2. Rs/R0 Ratio
Normalization against a baseline (`R0`) makes the system robust against sensor variance and environmental shifts.

## 3. Log-Scale PPM
Gas concentration follows a non-linear relationship:
`log(PPM) = (log(Rs/R0) - b) / m`
Using Propane sensitivity curves: `m = -0.47`, `b = 0.08`.
