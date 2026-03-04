# Controlled Calibration Protocol - SnO2-Gas-Analytics

Follow these steps to collect the data needed for your custom logarithmic PPM curve.

## 🧪 Experiment Setup
1. **Environment:** Calm room, no fans, constant temperature if possible.
2. **Tools:** A standard unlit but filled butane lighter.
3. **Serial Monitor:** Open the Arduino Serial Monitor at **115200 baud**.

## 📝 Labeling Labels
Use the Serial Monitor to send the following numbers before starting each test:
- **0:** Baseline / Clean Air
- **1:** Level 1 (Shortest burst)
- **2:** Level 2 (Short burst)
- **3:** Level 3 (Medium burst)
- **4:** Level 4 (Strong burst)

## 🏗️ Step-by-Step Procedure

### Phase 1: Baseline (Label 0)
1. Type `0` in Serial Monitor and press Enter.
2. Let it run for 5 minutes in clean air.
3. Ensure `Ratio` stays around 0.9 - 1.1 with `R0 = 11660`.

### Phase 2: Exposure Tests (Labels 1 to 4)
Repeat the following for each level (1, 2, 3, 4):

1. **Set Label:** Type the level number (e.g., `1`) in Serial and press Enter.
2. **Wait:** Ensure the sensor is stable before the burst.
3. **Burst:** Release gas from the lighter towards the sensor for the specified time:
   - **Level 1:** ~0.5 second
   - **Level 2:** ~1.5 seconds
   - **Level 3:** ~3 seconds
   - **Level 4:** ~5 seconds
4. **Recovery:** DO NOT change the label yet. Let the sensor recover in clean air until the `Ratio` returns close to your baseline.
5. **Reset Label:** Type `0` in Serial to return to baseline tracking.
6. **Repeat:** Repeat this 3 times for each level to get better averages.

## 📊 Data Submission
After finishing all levels, stop the server and send me the new `mq2_data.csv`. I will then calculate your custom `m` (slope) and `b` (intercept) for the final PPM equation!
