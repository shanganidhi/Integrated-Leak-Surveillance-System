/*
 * ESP32 Gas Detection Logger
 * Sensor: MQ-2 (Analog)
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

// Pins
#define MQ2_PIN 34 

// MQ-2 Parameters
#define RL 20000.0   // 20k ohm (standard for modules, better for ML)
#define ADC_MAX 4095.0
#define VCC 3.3
float R0 = 10000.0; // Recalibrate in clean air after 24h: R0 = Rs / 9.8

// Log scale model for Propane
#define M_SLOPE -0.47
#define B_INTERCEPT 0.08

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

void loop() {
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
      jsonData += "\"ppm\":" + String(ppm);
      jsonData += "}";

      int httpResponseCode = http.POST(jsonData);
      http.end();
      
      Serial.print("[" + String(dateStr) + " " + String(timeStr) + "] ");
      Serial.print("Gas: "); Serial.print(ppm); Serial.println(" PPM");
    }
  }
}
