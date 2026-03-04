from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

# CSV file to store per-sensor rows
file_name = "multi_sensor_data.csv"

# Create file with header if not exists
def initialize_csv():
    header = ["Date", "Time", "Sensor", "ADC", "Voltage", "Rs", "R0", "Ratio", "PPM", "Label"]
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

        date = data.get('date')
        time = data.get('time')
        label = data.get('label', 0)

        # If payload contains a sensors array, log one CSV row per sensor
        sensors = data.get('sensors')
        if sensors and isinstance(sensors, list):
            with open(file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                for s in sensors:
                    name = s.get('name')
                    adc = s.get('adc')
                    voltage = s.get('voltage')
                    Rs = s.get('Rs')
                    R0 = s.get('R0')
                    ratio = s.get('ratio')
                    ppm = s.get('ppm')
                    print(f"[{date} {time}] {name} Gas: {ppm} PPM | Label: {label}")
                    writer.writerow([date, time, name, adc, voltage, Rs, R0, ratio, ppm, label])

            return jsonify({"status": "success"}), 200

        # Backwards-compatible single-sensor payload
        adc = data.get('adc')
        voltage = data.get('voltage')
        Rs = data.get('Rs')
        R0 = data.get('R0')
        ratio = data.get('ratio')
        ppm = data.get('ppm')
        sensor_name = data.get('sensor', 'mq')

        print(f"[{date} {time}] {sensor_name} Gas: {ppm} PPM | Label: {label}")
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([date, time, sensor_name, adc, voltage, Rs, R0, ratio, ppm, label])

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error logging data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    initialize_csv()
    # Listen on all interfaces so ESP32 can connect
    print("Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
