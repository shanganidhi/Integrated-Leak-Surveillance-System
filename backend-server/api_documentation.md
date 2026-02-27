# Backend API Documentation

The backend server is a Flask-based application designed to receive sensor data from the ESP32 and store it in a CSV format.

## Endpoint: `POST /log`

Receives JSON data from the sensor.

### Request Body (JSON)
```json
{
  "date": "2026-02-28",
  "time": "00:13:00",
  "adc": 450,
  "voltage": 0.36,
  "Rs": 16333.33,
  "R0": 10000.0,
  "ratio": 1.6333,
  "ppm": 12.5
}
```

### Responses
- **200 OK:** Data successfully logged.
- **400 Bad Request:** Missing JSON or invalid format.
- **500 Internal Server Error:** CSV write error.

## Storage
Data is appended to `mq2_data.csv` in the server directory.
