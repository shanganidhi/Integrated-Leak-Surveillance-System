/*
 * ESP32 Gas Detection Logger - Calibration Phase
 * Sensor: MQ-2 (Analog)
 * Features: WiFi, NTP, Electrical Parameter Logging, Manual Labeling for Calibration
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "time.h"
#include <math.h>

// --- CONFIGURATION ---
const char* ssid = "Airtel_Barathy";
const char* password = "6384@bank";
const char* serverUrl = "http://192.168.1.5:5000/log";

// Pins
#define MQ2_PIN 34 

// MQ-2 Parameters
#define RL 20000.0   // 20k ohm
#define ADC_MAX 4095.0
#define VCC 3.3
float R0 = 11660.0; // Calibrated from 24h burn-in data

// Log scale model for Propane (Default constants, will be refined after calibration)
#define M_SLOPE -0.47
#define B_INTERCEPT 0.08

// NTP Settings
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 19800;  // India (IST)
const int   daylightOffset_sec = 0;

unsigned long previousMillis = 0;
const long interval = 2000; // Faster sampling for calibration (2 seconds)

int exposureLevel = 0; // 0=Baseline, 1-4=Gas Levels (set via Serial)

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
  
  Serial.println("--- Calibration Mode Active ---");
  Serial.println("Enter Label (0-4) in Serial Monitor:");
  Serial.println("0: Baseline, 1: Level 1, 2: Level 2, 3: Level 3, 4: Level 4");
}

void loop() {
  // Check for Serial input to update exposureLevel
  if (Serial.available() > 0) {
    int incoming = Serial.parseInt();
    if (incoming >= 0 && incoming <= 4) {
      exposureLevel = incoming;
      Serial.print(">>> Label updated to: ");
      Serial.println(exposureLevel);
    }
    // Clear buffer
    while(Serial.available() > 0) Serial.read();
  }

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // 1. Read MQ-2
    int adcValue = analogRead(MQ2_PIN);
    float Vrl = (adcValue / ADC_MAX) * VCC;
    if (Vrl < 0.01) Vrl = 0.01; 
    float Rs = ((VCC - Vrl) * RL) / Vrl;

    // 2. Derive Parameters
    float ratio = Rs / R0;
    float logppm = (log10(ratio) - B_INTERCEPT) / M_SLOPE;
    float ppm = pow(10, logppm);

    // 3. Get Time
    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
      Serial.println("Failed to obtain time");
      return;
    }

    char dateStr[20], timeStr[20];
    strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", &timeinfo);
    strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &timeinfo);

    // 4. Send to Server
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");

      String jsonData = "{";
      jsonData += "\"date\":\"" + String(dateStr) + "\",";
      jsonData += "\"time\":\"" + String(timeStr) + "\",";
      jsonData += "\"adc\":" + String(adcValue) + ",";
      jsonData += "\"voltage\":" + String(Vrl) + ",";
      jsonData += "\"Rs\":" + String(Rs) + ",";
      jsonData += "\"R0\":" + String(R0) + ",";
      jsonData += "\"ratio\":" + String(ratio) + ",";
      jsonData += "\"ppm\":" + String(ppm) + ",";
      jsonData += "\"label\":" + String(exposureLevel);
      jsonData += "}";

      int httpResponseCode = http.POST(jsonData);
      http.end();
      
      Serial.print("[" + String(dateStr) + " " + String(timeStr) + "] ");
      Serial.print("Gas: "); Serial.print(ppm); 
      Serial.print(" PPM | Ratio: "); Serial.print(ratio);
      Serial.print(" | Label: "); Serial.println(exposureLevel);
    }
  }
}
