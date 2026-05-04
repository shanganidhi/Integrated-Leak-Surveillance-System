#include <WiFi.h>
#include <HTTPClient.h>
#include "Model.h" // The generated TinyML model

// ================= CONFIG =================
const char* ssid = "Redmi Note 10S";
const char* password = "123456789";

// ================= CLOUD DASHBOARD =================
// Set this to the IP of the machine running app.py
// Find it via: python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
const char* CLOUD_SERVER = "http://10.17.175.188:5000/log";
bool cloudEnabled = true;

// ================= PINS =================
#define MQ2_PIN   35
#define MQ135_PIN 34
#define MQ7_PIN   32

// ================= CONSTANTS =================
#define RL 20000.0
#define ADC_MAX 4095.0
#define VCC 3.3

// Over-night calibrated baseline R0 values
float R0_MQ2   = 25770.0;
float R0_MQ135 = 10582.0;
float R0_MQ7   = 27584.0;

// ================= TINYML BUFFER =================
#define WINDOW_SIZE 5
float history[WINDOW_SIZE][3]; // Store ratio2, ratio135, ratio7
int bufferIdx = 0;
bool bufferFull = false;

Eloquent::ML::Port::GasClassifier model;

// ================= SETUP =================
void setup() {
    Serial.begin(115200);
    analogReadResolution(12);

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected! TinyML + Cloud Ready.");
    Serial.print("Cloud Dashboard: ");
    Serial.println(CLOUD_SERVER);
}

// ================= HELPERS =================
float getMean(int sensorIdx) {
    float sum = 0;
    int count = bufferFull ? WINDOW_SIZE : bufferIdx;
    for (int i = 0; i < count; i++) sum += history[i][sensorIdx];
    return sum / count;
}

float getStd(int sensorIdx, float mean) {
    float sumSq = 0;
    int count = bufferFull ? WINDOW_SIZE : bufferIdx;
    if (count < 2) return 0;
    for (int i = 0; i < count; i++) sumSq += pow(history[i][sensorIdx] - mean, 2);
    return sqrt(sumSq / (count - 1));
}

// ================= CLOUD LOGGING =================
void sendToCloud(float r2, float r135, float r7,
                 float ppm2, float ppm135, float ppm7,
                 int prediction) {
    if (!cloudEnabled || WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    http.begin(CLOUD_SERVER);
    http.addHeader("Content-Type", "application/json");

    // Build JSON payload
    String json = "{";
    json += "\"mq2_ratio\":" + String(r2, 4) + ",";
    json += "\"mq135_ratio\":" + String(r135, 4) + ",";
    json += "\"mq7_ratio\":" + String(r7, 4) + ",";
    json += "\"mq2_ppm\":" + String(ppm2, 2) + ",";
    json += "\"mq135_ppm\":" + String(ppm135, 2) + ",";
    json += "\"mq7_ppm\":" + String(ppm7, 2) + ",";
    json += "\"prediction\":" + String(prediction);
    json += "}";

    int httpCode = http.POST(json);
    if (httpCode == 200) {
        String response = http.getString();
        Serial.print("[CLOUD] OK -> ");
        Serial.println(response);
    } else {
        Serial.print("[CLOUD] Error: ");
        Serial.println(httpCode);
    }
    http.end();
}

// ================= MAIN LOOP =================
void loop() {
    // 1. Read Raw Sensors
    float v2 = (analogRead(MQ2_PIN) / ADC_MAX) * VCC;
    float v135 = (analogRead(MQ135_PIN) / ADC_MAX) * VCC;
    float v7 = (analogRead(MQ7_PIN) / ADC_MAX) * VCC;

    // 2. Calculate Ratios
    float r2 = ((VCC - v2) * RL / v2) / R0_MQ2;
    float r135 = ((VCC - v135) * RL / v135) / R0_MQ135;
    float r7 = ((VCC - v7) * RL / v7) / R0_MQ7;

    // 3. Approximate PPM (for display only)
    float ppm2 = pow(10, 2.0 - r2 / 3.5);
    float ppm135 = pow(10, 1.5 - r135 / 2.0);
    float ppm7 = pow(10, 2.5 - r7 / 5.0);

    // 4. Update Circular Buffer
    history[bufferIdx][0] = r2;
    history[bufferIdx][1] = r135;
    history[bufferIdx][2] = r7;
    bufferIdx++;
    if (bufferIdx >= WINDOW_SIZE) {
        bufferIdx = 0;
        bufferFull = true;
    }

    // 5. Feature Engineering (Match Python!)
    float m2 = getMean(0), m135 = getMean(1), m7 = getMean(2);
    float s2 = getStd(0, m2), s135 = getStd(1, m135), s7 = getStd(2, m7);

    float features[11] = {
        (float)log10(r2 + 1e-6), (float)log10(r135 + 1e-6), (float)log10(r7 + 1e-6),
        m2, m135, m7,
        s2, s135, s7,
        r2 * r7, r2 * r135
    };

    // 6. Run TinyML Inference (Edge)
    int prediction = model.predict(features);

    // 7. Display Result
    Serial.println("---------------------------------");
    Serial.print("AI DECISION: ");
    switch(prediction) {
        case 0: Serial.println("CLEAN AIR"); break;
        case 1: Serial.println("LOW GAS DETECTED"); break;
        case 2: Serial.println("MEDIUM GAS DETECTED"); break;
        case 3: Serial.println("HIGH GAS - WARNING!"); break;
        case 4: Serial.println("CRITICAL - EVACUATE!"); break;
    }
    Serial.print("Data: R2:"); Serial.print(r2);
    Serial.print(" | R135:"); Serial.print(r135);
    Serial.print(" | R7:"); Serial.println(r7);

    // 8. Send to Cloud Dashboard
    sendToCloud(r2, r135, r7, ppm2, ppm135, ppm7, prediction);

    delay(2000);
}
