# System Overview

SnO2-Gas-Analytics is a comprehensive IoT ecosystem for professional gas monitoring.

## 1. Hardware Layer (Edge)
ESP32 reads analog data, performs electrical modeling, and transmits via HTTP.

## 2. Communication Layer
WPA2 Secured WiFi sends JSON payloads to the backend.

## 3. Storage & Analytics Layer (Fog/Cloud)
Python Flask server provides logging and serves as a data gateway for ML training.

## 4. UI Layer
Cross-platform app provides real-time visualization and alert management.
