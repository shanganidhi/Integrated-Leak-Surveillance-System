from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

file_name = "mq2_data.csv"

# Create file with header if not exists
def initialize_csv():
    header = ["Date", "Time", "ADC", "Voltage", "Rs", "R0", "Ratio", "PPM"]
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
        adc = data.get('adc')
        voltage = data.get('voltage')
        Rs = data.get('Rs')
        R0 = data.get('R0')
        ratio = data.get('ratio')
        ppm = data.get('ppm')

        # Log to console for real-time feedback
        print(f"[{date} {time}] Gas: {ppm:.2f} PPM | Ratio: {ratio:.4f}")

        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([date, time, adc, voltage, Rs, R0, ratio, ppm])

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error logging data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    initialize_csv()
    # Listen on all interfaces so ESP32 can connect
    print("Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
