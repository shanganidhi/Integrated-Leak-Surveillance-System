/*
 * ESP32 Multi-Sensor Gas Detection Logger
 * Sensors: MQ-2, MQ-135, MQ-7 (Analog)
 * Features: WiFi, NTP, Electrical Parameter Logging, HTTP Logging to Python Server
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "time.h"
#include <math.h>

// --- CONFIGURATION ---
const char* ssid = "Airtel_Barathy";
const char* password = "6384@bank";
const char* serverUrl = "http://192.168.1.5:5000/log";

// ADC / VREF
#define ADC_MAX 4095.0
#define VCC 3.3

// Sensor pins (adjust as per wiring)
#define MQ2_PIN 34
#define MQ135_PIN 35
#define MQ7_PIN 32

// Per-sensor load resistor (ohms)
const float RL_MQ2 = 20000.0;
const float RL_MQ135 = 20000.0;
const float RL_MQ7 = 20000.0;

// Default R0 values (should be calibrated in clean-air conditions per-sensor)
float R0_MQ2 = 10000.0;
float R0_MQ135 = 10000.0;
float R0_MQ7 = 10000.0;

// Per-sensor log-linear model params (placeholders — calibrate for accuracy)
// PPM model: log10(ppm) = (log10(Rs/R0) - b) / m
const float MQ2_M = -0.47;
const float MQ2_B = 0.08;

const float MQ135_M = -0.38; // placeholder
const float MQ135_B = 0.10;  // placeholder

const float MQ7_M = -0.35; // placeholder
const float MQ7_B = 0.12;  // placeholder

// NTP Settings
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 19800;  // India (IST)
const int   daylightOffset_sec = 0;

unsigned long previousMillis = 0;
const long interval = 5000; // Log every 5 seconds

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
}

float readVoltage(int pin) {
  int adc = analogRead(pin);
  return (adc / ADC_MAX) * VCC;
}

float computeRs(float Vrl, float RL) {
  if (Vrl < 0.001) Vrl = 0.001;
  return ((VCC - Vrl) * RL) / Vrl;
}

float computePPM(float Rs, float R0, float m, float b) {
  float ratio = Rs / R0;
  if (ratio <= 0) ratio = 1e-6;
  float logppm = (log10(ratio) - b) / m;
  return pow(10, logppm);
}

String sensorJson(const char* name, int adc, float voltage, float Rs, float R0, float ratio, float ppm) {
  String s = "{";
  s += "\"name\":\"" + String(name) + "\",";
  s += "\"adc\":" + String(adc) + ",";
  s += "\"voltage\":" + String(voltage) + ",";
  s += "\"Rs\":" + String(Rs) + ",";
  s += "\"R0\":" + String(R0) + ",";
  s += "\"ratio\":" + String(ratio) + ",";
  s += "\"ppm\":" + String(ppm);
  s += "}";
  return s;
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
      Serial.println("Failed to obtain time");
      return;
    }

    char dateStr[20], timeStr[20];
    strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &timeinfo);
    strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeinfo);

    // Read each sensor
    int adc_mq2 = analogRead(MQ2_PIN);
    float v_mq2 = (adc_mq2 / ADC_MAX) * VCC;
    float Rs_mq2 = computeRs(v_mq2, RL_MQ2);
    float ratio_mq2 = Rs_mq2 / R0_MQ2;
    float ppm_mq2 = computePPM(Rs_mq2, R0_MQ2, MQ2_M, MQ2_B);

    int adc_mq135 = analogRead(MQ135_PIN);
    float v_mq135 = (adc_mq135 / ADC_MAX) * VCC;
    float Rs_mq135 = computeRs(v_mq135, RL_MQ135);
    float ratio_mq135 = Rs_mq135 / R0_MQ135;
    float ppm_mq135 = computePPM(Rs_mq135, R0_MQ135, MQ135_M, MQ135_B);

    int adc_mq7 = analogRead(MQ7_PIN);
    float v_mq7 = (adc_mq7 / ADC_MAX) * VCC;
    float Rs_mq7 = computeRs(v_mq7, RL_MQ7);
    float ratio_mq7 = Rs_mq7 / R0_MQ7;
    float ppm_mq7 = computePPM(Rs_mq7, R0_MQ7, MQ7_M, MQ7_B);

    // Build JSON payload
    String jsonData = "{";
    jsonData += "\"date\":\"" + String(dateStr) + "\",";
    jsonData += "\"time\":\"" + String(timeStr) + "\",";
    jsonData += "\"sensors\":[";
    jsonData += sensorJson("mq2", adc_mq2, v_mq2, Rs_mq2, R0_MQ2, ratio_mq2, ppm_mq2);
    jsonData += "," + sensorJson("mq135", adc_mq135, v_mq135, Rs_mq135, R0_MQ135, ratio_mq135, ppm_mq135);
    jsonData += "," + sensorJson("mq7", adc_mq7, v_mq7, Rs_mq7, R0_MQ7, ratio_mq7, ppm_mq7);
    jsonData += "],";
    jsonData += "\"label\":0"; // optional label
    jsonData += "}";

    // Send to Server
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");

      int httpResponseCode = http.POST(jsonData);
      if (httpResponseCode > 0) {
        Serial.printf("Posted data, response: %d\n", httpResponseCode);
      } else {
        Serial.printf("HTTP POST failed: %s\n", http.errorToString(httpResponseCode).c_str());
      }
      http.end();
    }

    // Console output
    Serial.printf("[%s %s] MQ-2: %.2f ppm, MQ-135: %.2f ppm, MQ-7: %.2f ppm\n", dateStr, timeStr, ppm_mq2, ppm_mq135, ppm_mq7);
  }
}
