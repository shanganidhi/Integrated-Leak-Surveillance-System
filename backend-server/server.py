from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

file_name = "multi_sensor_data.csv"

# Create file with header if not exists
def initialize_csv():
    header = [
        "date", "time", 
        "mq2_adc", "mq2_rs", "mq2_ratio", "mq2_ppm", "mq2_delta",
        "mq135_adc", "mq135_rs", "mq135_ratio", "mq135_ppm", "mq135_delta",
        "mq7_adc", "mq7_rs", "mq7_ratio", "mq7_ppm", "mq7_delta",
        "label"
    ]
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        print(f"Initialized {file_name} with headers.")

@app.route('/log', methods=['POST'])
def log_data():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400

        # Extract values
        row = [
            data.get('date'),
            data.get('time'),
            data.get('mq2_adc'),
            data.get('mq2_rs'),
            data.get('mq2_ratio'),
            data.get('mq2_ppm'),
            data.get('mq2_delta'),
            data.get('mq135_adc'),
            data.get('mq135_rs'),
            data.get('mq135_ratio'),
            data.get('mq135_ppm'),
            data.get('mq135_delta'),
            data.get('mq7_adc'),
            data.get('mq7_rs'),
            data.get('mq7_ratio'),
            data.get('mq7_ppm'),
            data.get('mq7_delta'),
            data.get('label', 0)
        ]

        # Log to console for real-time feedback
        print(f"[{data.get('date')} {data.get('time')}] MQ2: {data.get('mq2_ppm', 0):.2f} | MQ135: {data.get('mq135_ppm', 0):.2f} | MQ7: {data.get('mq7_ppm', 0):.2f} | Label: {data.get('label', 0)}")

        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error logging data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    initialize_csv()
    # Listen on all interfaces so ESP32 can connect
    print("Starting multi-sensor server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
