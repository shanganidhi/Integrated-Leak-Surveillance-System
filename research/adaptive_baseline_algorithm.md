# Adaptive Baseline Algorithm

The primary challenge in MOS (Metal Oxide Semiconductor) sensors is baseline drift.

## Algorithm Overview
1. **Windowing:** Store `Rs` values in a 24-hour circular buffer.
2. **Filtering:** Identify the lowest 5th percentile of resistance values (representing cleanest air).
3. **Updating:** `R0` is updated as the moving average of these filtered values.

`R0_new = (R0_old * (N-1) + MinimaAvg) / N`

This ensures that the `Rs/R0` ratio remains consistent even as the sensor surface degrades over time.
