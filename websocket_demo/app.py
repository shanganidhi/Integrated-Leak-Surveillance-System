"""
SnO2 Gas Analytics — Cloud Dashboard Server
============================================
Flask server that:
  1. Ingests sensor data from ESP32 via POST /log
  2. Runs ML inference server-side
  3. Serves a premium real-time dashboard
  4. Provides REST APIs for the frontend
"""

import os
import csv
import json
import time
import math
import random
import threading
from datetime import datetime
from collections import deque

import numpy as np
import joblib
import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from scipy.stats import linregress

# Suppress annoying sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DATA_FILE = os.path.join(os.path.dirname(__file__), "multi_sensor_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# In-memory circular buffer for fast API access
MAX_HISTORY = 200
readings_buffer = deque(maxlen=MAX_HISTORY)
alert_log = deque(maxlen=50)

# ML Models
ml_model = None
anomaly_model = None
feature_cols = None
label_map = {0: 'Clean Air', 1: 'Low Gas', 2: 'Medium Gas', 3: 'High Gas', 4: 'Critical'}
model_accuracy = 0.0

# Intelligence Tracking
recent_cgi = deque(maxlen=10) # Keep last 10 for slope/trend

# Rolling window for feature engineering (matches ESP32)
WINDOW_SIZE = 5
ratio_window = deque(maxlen=WINDOW_SIZE)

# Demo mode state
demo_mode = False
demo_thread = None

# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════
def load_model():
    """Load the trained ML model from joblib."""
    global ml_model, anomaly_model, feature_cols, label_map, model_accuracy
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        ml_model = data['model']
        anomaly_model = data.get('anomaly_model')
        feature_cols = data['feature_cols']
        label_map = data.get('label_map', label_map)
        model_accuracy = data.get('accuracy', 0.0)
        print(f"[OK] ML Model loaded (accuracy: {model_accuracy*100:.2f}%)")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")
        print("   Run: python train_and_export.py first")
        print("   Dashboard will work in demo-only mode.")

def initialize_csv():
    """Create the CSV file with headers if it doesn't exist."""
    header = [
        "timestamp", "date", "time",
        "mq2_ratio", "mq135_ratio", "mq7_ratio",
        "mq2_ppm", "mq135_ppm", "mq7_ppm",
        "prediction", "label_name"
    ]
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(header)

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (matches training pipeline)
# ═══════════════════════════════════════════════════════════════
def compute_features(mq2_ratio, mq135_ratio, mq7_ratio):
    """
    Compute the 11 features expected by the model.
    Uses the rolling window buffer for mean/std.
    """
    ratio_window.append((mq2_ratio, mq135_ratio, mq7_ratio))
    
    ratios = list(ratio_window)
    n = len(ratios)
    
    r2_vals = [r[0] for r in ratios]
    r135_vals = [r[1] for r in ratios]
    r7_vals = [r[2] for r in ratios]
    
    # Log features
    log_r2 = math.log10(mq2_ratio + 1e-6)
    log_r135 = math.log10(mq135_ratio + 1e-6)
    log_r7 = math.log10(mq7_ratio + 1e-6)
    
    # Rolling mean
    mean_r2 = sum(r2_vals) / n
    mean_r135 = sum(r135_vals) / n
    mean_r7 = sum(r7_vals) / n
    
    # Rolling std
    def std(vals, mean):
        if len(vals) < 2:
            return 0.0
        return math.sqrt(sum((v - mean)**2 for v in vals) / (n - 1))
    
    std_r2 = std(r2_vals, mean_r2)
    std_r135 = std(r135_vals, mean_r135)
    std_r7 = std(r7_vals, mean_r7)
    
    # Interaction
    inter_r2_r7 = mq2_ratio * mq7_ratio
    inter_r2_r135 = mq2_ratio * mq135_ratio
    
    return [
        log_r2, log_r135, log_r7,
        mean_r2, mean_r135, mean_r7,
        std_r2, std_r135, std_r7,
        inter_r2_r7, inter_r2_r135
    ]

def run_inference(features):
    """Run the ML model on computed features."""
    if ml_model is None:
        return 0, {0: 1.0}, False
    
    X = np.array([features])
    prediction = int(ml_model.predict(X)[0])
    
    # Anomaly Detection
    is_anomaly = False
    if anomaly_model is not None:
        is_anomaly = bool(anomaly_model.predict(X)[0] == -1)
    
    # Get probabilities if available
    try:
        proba = ml_model.predict_proba(X)[0]
        classes = ml_model.classes_
        prob_dict = {int(c): float(p) for c, p in zip(classes, proba)}
    except Exception:
        prob_dict = {prediction: 1.0}
    
    return prediction, prob_dict, is_anomaly

def calculate_intelligence(r2, r135, r7, prediction):
    """Compute Risk, CGI, Trend, and Events"""
    # 1. Risk Score (0-100) — INVERTED: lower ratios = more gas = higher risk
    #    Typical clean-air ratios: r2~7.6, r135~3.2, r7~12.5
    #    Typical critical ratios:  r2~4.8, r135~2.3, r7~7.5
    risk_r2 = max(0, min(100, (8.0 - r2) / 3.5 * 100))
    risk_r135 = max(0, min(100, (3.5 - r135) / 1.5 * 100))
    risk_r7 = max(0, min(100, (13.0 - r7) / 6.0 * 100))
    risk_score = risk_r2 * 0.4 + risk_r135 * 0.3 + risk_r7 * 0.3
    risk_score = max(0, min(100, risk_score))
    
    # 2. Combined Gas Index (CGI) — lower = more dangerous
    cgi = (r2 + r135 + r7) / 3.0
    recent_cgi.append(cgi)
    
    # 3. Gas Trend Prediction (linear regression over recent CGI)
    trend_slope = 0.0
    if len(recent_cgi) >= 5:
        y = list(recent_cgi)
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        trend_slope = float(slope)
    
    # 4. Event Detection Engine
    event = "Nominal"
    if trend_slope < -0.3:
        event = "Sudden Leak Detected"
    elif trend_slope < -0.08 and prediction >= 2:
        event = "Slow Build-up"
    elif prediction == 4:
        event = "Critical Exposure"
    elif prediction >= 3:
        event = "Continuous Exposure"
        
    return {
        "risk_score": round(risk_score, 1),
        "cgi": round(cgi, 2),
        "trend_prediction": round(cgi + (trend_slope * 5), 2),
        "event_status": event,
        "slope": round(trend_slope, 4)
    }

# ═══════════════════════════════════════════════════════════════
# DEMO MODE — Generates realistic fake sensor data
# ═══════════════════════════════════════════════════════════════
def demo_generator():
    """Background thread that generates simulated sensor data."""
    global demo_mode
    
    # Base values for each scenario
    scenarios = [
        {"name": "Clean Air", "r2": 7.6, "r135": 3.2, "r7": 12.5, "duration": 30},
        {"name": "Low Gas",   "r2": 7.0, "r135": 3.1, "r7": 12.0, "duration": 20},
        {"name": "Medium Gas","r2": 6.2, "r135": 2.9, "r7": 11.0, "duration": 15},
        {"name": "High Gas",  "r2": 5.5, "r135": 2.6, "r7": 9.5,  "duration": 12},
        {"name": "Critical",  "r2": 4.8, "r135": 2.3, "r7": 7.5,  "duration": 8},
    ]
    
    scenario_idx = 0
    step = 0
    
    while demo_mode:
        scenario = scenarios[scenario_idx]
        
        # Add realistic noise
        noise_factor = 0.03
        r2 = scenario["r2"] + random.gauss(0, scenario["r2"] * noise_factor)
        r135 = scenario["r135"] + random.gauss(0, scenario["r135"] * noise_factor)
        r7 = scenario["r7"] + random.gauss(0, scenario["r7"] * noise_factor)
        
        # Compute features, predict, and intelligence
        features = compute_features(r2, r135, r7)
        prediction, probabilities, is_anomaly = run_inference(features)
        intel = calculate_intelligence(r2, r135, r7, prediction)
        
        now = datetime.now()
        reading = {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "mq2_ratio": round(r2, 4),
            "mq135_ratio": round(r135, 4),
            "mq7_ratio": round(r7, 4),
            "mq2_ppm": round(10 ** (2.0 - r2 / 3.5), 2),
            "mq135_ppm": round(10 ** (1.5 - r135 / 2.0), 2),
            "mq7_ppm": round(10 ** (2.5 - r7 / 5.0), 2),
            "prediction": prediction,
            "label_name": label_map.get(prediction, "Unknown"),
            "probabilities": probabilities,
            "is_anomaly": is_anomaly,
            "intelligence": intel,
            "features": features,
            "demo": True
        }
        
        readings_buffer.append(reading)
        
        # WEBSOCKET EMIT
        socketio.emit('new_data', reading)
        
        # Log alerts for significant events
        if prediction >= 3:
            alert_log.append({
                "timestamp": now.isoformat(),
                "level": "CRITICAL" if prediction == 4 else "WARNING",
                "message": f"{label_map[prediction]} detected — MQ-2: {r2:.2f}, MQ-7: {r7:.2f}",
                "prediction": prediction
            })
        
        step += 1
        if step >= scenario["duration"]:
            step = 0
            scenario_idx = (scenario_idx + 1) % len(scenarios)
        
        time.sleep(2)

# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/log', methods=['POST'])
def log_data():
    """Ingest sensor data from ESP32."""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON data"}), 400
        
        mq2_ratio = float(data.get('mq2_ratio', 0))
        mq135_ratio = float(data.get('mq135_ratio', 0))
        mq7_ratio = float(data.get('mq7_ratio', 0))
        
        # Compute features, predict, and intelligence
        features = compute_features(mq2_ratio, mq135_ratio, mq7_ratio)
        prediction, probabilities, is_anomaly = run_inference(features)
        intel = calculate_intelligence(mq2_ratio, mq135_ratio, mq7_ratio, prediction)
        
        now = datetime.now()
        reading = {
            "timestamp": now.isoformat(),
            "date": data.get('date', now.strftime("%Y-%m-%d")),
            "time": data.get('time', now.strftime("%H:%M:%S")),
            "mq2_ratio": mq2_ratio,
            "mq135_ratio": mq135_ratio,
            "mq7_ratio": mq7_ratio,
            "mq2_ppm": float(data.get('mq2_ppm', 0)),
            "mq135_ppm": float(data.get('mq135_ppm', 0)),
            "mq7_ppm": float(data.get('mq7_ppm', 0)),
            "prediction": prediction,
            "label_name": label_map.get(prediction, "Unknown"),
            "probabilities": probabilities,
            "is_anomaly": is_anomaly,
            "intelligence": intel,
            "features": features,
            "esp32_prediction": int(data.get('prediction', -1)),
            "demo": False
        }
        
        readings_buffer.append(reading)
        
        # WEBSOCKET EMIT
        socketio.emit('new_data', reading)
        
        # Log to CSV
        with open(DATA_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                reading["timestamp"], reading["date"], reading["time"],
                mq2_ratio, mq135_ratio, mq7_ratio,
                reading["mq2_ppm"], reading["mq135_ppm"], reading["mq7_ppm"],
                prediction, reading["label_name"]
            ])
        
        # Log alerts
        if prediction >= 3:
            alert_log.append({
                "timestamp": now.isoformat(),
                "level": "CRITICAL" if prediction == 4 else "WARNING",
                "message": f"{label_map[prediction]} - MQ-2: {mq2_ratio:.2f}",
                "prediction": prediction
            })
        
        print(f"[{reading['time']}] R2:{mq2_ratio:.2f} R135:{mq135_ratio:.2f} R7:{mq7_ratio:.2f} -> {label_map[prediction]}")
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "label": label_map[prediction],
            "probabilities": probabilities
        }), 200
        
    except Exception as e:
        import traceback
        print(f"!!! SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/latest')
def api_latest():
    """Get the most recent reading."""
    if not readings_buffer:
        return jsonify({"status": "no_data"}), 200
    
    latest = readings_buffer[-1]
    return jsonify({
        "status": "ok",
        "data": latest,
        "model_accuracy": model_accuracy,
        "total_readings": len(readings_buffer),
        "model_loaded": ml_model is not None
    })

@app.route('/api/history')
def api_history():
    """Get historical readings for charting."""
    n = min(int(request.args.get('n', 100)), MAX_HISTORY)
    data = list(readings_buffer)[-n:]
    return jsonify({
        "status": "ok",
        "data": data,
        "count": len(data)
    })

@app.route('/api/stats')
def api_stats():
    """Get aggregate statistics."""
    if not readings_buffer:
        return jsonify({"status": "no_data"})
    
    data = list(readings_buffer)
    
    predictions = [d['prediction'] for d in data]
    mq2_vals = [d['mq2_ratio'] for d in data]
    mq135_vals = [d['mq135_ratio'] for d in data]
    mq7_vals = [d['mq7_ratio'] for d in data]
    
    def stats(vals):
        return {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "avg": round(sum(vals)/len(vals), 4),
            "current": round(vals[-1], 4)
        }
    
    # Count predictions
    pred_counts = {}
    for p in predictions:
        pred_counts[str(p)] = pred_counts.get(str(p), 0) + 1
    
    return jsonify({
        "status": "ok",
        "total_readings": len(data),
        "mq2": stats(mq2_vals),
        "mq135": stats(mq135_vals),
        "mq7": stats(mq7_vals),
        "prediction_counts": pred_counts,
        "alerts": list(alert_log)[-10:],
        "model_accuracy": model_accuracy,
        "model_loaded": ml_model is not None
    })

@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts."""
    return jsonify({
        "status": "ok",
        "alerts": list(alert_log)
    })

@app.route('/api/demo/start', methods=['POST'])
def demo_start():
    """Start demo mode with simulated data."""
    global demo_mode, demo_thread
    if demo_mode:
        return jsonify({"status": "already_running"})
    
    demo_mode = True
    demo_thread = threading.Thread(target=demo_generator, daemon=True)
    demo_thread.start()
    return jsonify({"status": "demo_started"})

@app.route('/api/demo/stop', methods=['POST'])
def demo_stop():
    """Stop demo mode."""
    global demo_mode
    demo_mode = False
    return jsonify({"status": "demo_stopped"})

@app.route('/api/demo/status')
def demo_status():
    """Check if demo mode is active."""
    return jsonify({"demo_active": demo_mode})

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SnO2 Gas Analytics — Cloud Dashboard")
    print("="*60)
    
    load_model()
    initialize_csv()
    
    print(f"\n[WEB] Dashboard: http://localhost:5000")
    print(f"[API] ESP32 Endpoint: POST http://<THIS_IP>:5000/log")
    print(f"[DEMO] Demo API: POST http://localhost:5000/api/demo/start")
    print("="*60 + "\n")
    
    app.config['SECRET_KEY'] = 'sno2_secret!'
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
