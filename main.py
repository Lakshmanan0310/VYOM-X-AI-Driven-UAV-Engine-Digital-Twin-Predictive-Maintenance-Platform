"""
ENGINE-TWIN: Universal Main Launcher Script (SIH26054)
Launches the real-time Digital Twin backend server and opens the operator dashboard.
Supports both FastAPI/Uvicorn (if installed) and Built-In Standard Library Server (zero-install fallback).
"""
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[INFO] Opening ENGINE-TWIN Operator Dashboard in browser: {url}")
    webbrowser.open(url)

def main():
    print("=" * 70)
    print("  ENGINE-TWIN: AI-Enabled Real-Time Digital Twin System (SIH26054)")
    print("  Aero Piston Engine Health Monitoring for MALE UAVs (DRDO)")
    print("=" * 70)
    print("  - Telemetry Engine: 20 Hz Real-Time Integration")
    print("  - Physics Core: Mean Value Engine Model (MVEM) Calibrated to VRDE 2.2L")
    print("  - State Fusion: 12-State Extended Kalman Filter (EKF)")
    print("  - AI Prognostics: Autoencoder Anomaly Trigger + TCN Classifier + RUL")
    print("  - Explainability: Fast Local SHAP Feature Attribution Engine")
    print("  - Operator Dashboard: http://127.0.0.1:8000")
    print("=" * 70)

    # Launch browser in separate background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Try FastAPI/Uvicorn first; fallback to Built-in Standalone Server
    try:
        import uvicorn
        from engine_service import app
        print("[INFO] Starting FastAPI High-Performance WebSocket Engine Server...")
        uvicorn.run("engine_service:app", host="127.0.0.1", port=8000, log_level="warning")
    except ImportError:
        from standalone_server import run_standalone_server
        print("[INFO] FastAPI/Uvicorn not found. Starting Zero-Dependency Built-In Standalone Server...")
        run_standalone_server(port=8000)

if __name__ == "__main__":
    main()
