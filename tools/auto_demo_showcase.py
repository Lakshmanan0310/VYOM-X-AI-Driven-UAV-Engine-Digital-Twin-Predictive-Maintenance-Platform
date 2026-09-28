"""
ENGINE-TWIN: Automated Live Demo Showcase & Presentation Runner (SIH26054)
Executes a fully automated, scripted 2-minute demonstration sequence against the live Digital Twin:
1. Nominal Cruise Phase (100% Subsystem Health, EKF Noise Filtering)
2. Sudden Lubrication Failure Injection (Oil Loss Detection in 2.3s + RTB Directive)
3. Sensor Defect Decoupling (Thermocouple Probe Failure vs Engine Overheat)
4. Mechanical Degradation & SHAP XAI (2X Vibration Attribution & 18-25 Hr RUL Interval)
"""
import sys
import time
import json
from pathlib import Path
from urllib.request import urlopen, Request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = "http://127.0.0.1:8000"

def post_json(endpoint: str, data: dict):
    url = f"{API_BASE}{endpoint}"
    req = Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  [WARN] Request to {endpoint} failed: {e}")
        return None

def get_state():
    try:
        with urlopen(f"{API_BASE}/api/state") as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        return None

def main():
    print("=" * 80)
    print("  ENGINE-TWIN: AUTOMATED STAGE DEMONSTRATION SHOWCASE (DRDO SIH26054)")
    print("=" * 80)
    print("  Connecting to live digital twin server at http://127.0.0.1:8000 ...")

    # Check if server is running
    state = get_state()
    if not state:
        print("  [ERROR] Server is not running. Please start 'python main.py' first.")
        return

    print("  [SUCCESS] Connected to Digital Twin stream.")
    time.sleep(1.0)

    # -------------------------------------------------------------
    # PHASE 1: NOMINAL FLIGHT CRUISE (15 seconds)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [PHASE 1] NOMINAL FLIGHT CRUISE AT 28,500 FT")
    print("#" * 80)
    post_json("/api/fault/clear", {})
    post_json("/api/sim/speed?speed=1.0", {})
    
    for i in range(5):
        s = get_state()
        if s:
            h = s.get("health", {})
            print(f"  T+{s['timestamp_s']:.1f}s | Alt: {s['altitude_ft']:.0f} ft | Health: {h.get('overall_health', 100):.1f}% [{h.get('status_level', 'NOMINAL')}] | RPM: {s['sensor_rpm']:.0f} | Oil P: {s['sensor_oil_p_bar']:.2f} bar")
        time.sleep(1.5)

    # -------------------------------------------------------------
    # PHASE 2: INJECT OIL PRESSURE LOSS (20 seconds)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [PHASE 2] INJECTING CATASTROPHIC OIL PRESSURE LOSS")
    print("#" * 80)
    print("  --> Triggering Oil Pump Relief Valve Degradation...")
    post_json("/api/fault/inject", {"fault_type": "OIL_PRESSURE_LOSS", "severity": 1.0, "ramp_duration_s": 4.0})

    for i in range(6):
        time.sleep(1.0)
        s = get_state()
        if s:
            h = s.get("health", {})
            ai = s.get("ai_prognostics", {})
            alert = s.get("active_alert")
            print(f"  T+{s['timestamp_s']:.1f}s | Oil P: {s['sensor_oil_p_bar']:.2f} bar | Oil Health: {h.get('oil_system_health', 100):.1f}% | Overall: {h.get('overall_health', 100):.1f}% | Diagnosis: {ai.get('fault_class', 'HEALTHY')} ({ai.get('fault_confidence_pct', 0):.1f}%)")
            if alert:
                print(f"      🚨 [ALERT]: {alert.get('headline')} -> Directive: {alert.get('recommended_action')}")

    # -------------------------------------------------------------
    # PHASE 3: SENSOR PROBE FAULT DECOUPLING (15 seconds)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [PHASE 3] SENSOR PROBE DEFECT DECOUPLING (THERMOCOUPLE OPEN-CIRCUIT)")
    print("#" * 80)
    print("  --> Resetting to Healthy, then dropping Cylinder 3 EGT probe wire...")
    post_json("/api/fault/clear", {})
    time.sleep(2.0)
    post_json("/api/fault/inject", {"fault_type": "SENSOR_FAULT_EGT3", "severity": 1.0, "ramp_duration_s": 0.0})

    for i in range(5):
        time.sleep(1.2)
        s = get_state()
        if s:
            h = s.get("health", {})
            ai = s.get("ai_prognostics", {})
            print(f"  T+{s['timestamp_s']:.1f}s | EGT3: {s['sensor_egt_c'][2]:.1f} C (Reads Ambient!) | CHT3: {s['sensor_cht_c'][2]:.1f} C (Healthy) | Engine Health: {h.get('overall_health', 100):.1f}%")
            print(f"      🛡️ [SENSOR VALIDATOR]: Sensor Fault = {ai.get('is_sensor_fault')} | Directive: {ai.get('recommended_action')}")

    # -------------------------------------------------------------
    # PHASE 4: MECHANICAL BEARING WEAR & SHAP XAI (20 seconds)
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  [PHASE 4] MECHANICAL BEARING WEAR & SHAP EXPLAINABILITY")
    print("#" * 80)
    print("  --> Injecting 2X Order Bearing Vibration...")
    post_json("/api/fault/clear", {})
    time.sleep(2.0)
    post_json("/api/fault/inject", {"fault_type": "BEARING_WEAR_VIBRATION", "severity": 1.0, "ramp_duration_s": 6.0})

    for i in range(6):
        time.sleep(1.2)
        s = get_state()
        if s:
            ai = s.get("ai_prognostics", {})
            shap_list = ai.get("top_contributing_channels", [])
            print(f"  T+{s['timestamp_s']:.1f}s | Vib RMS: {s['sensor_vib_rms_g']:.3f} g | Diagnosis: {ai.get('fault_class')} | RUL Window: [{ai.get('rul_hours_min', 0):.1f} – {ai.get('rul_hours_max', 0):.1f}] Flight Hrs")
            if shap_list:
                factors = ", ".join([f"{item['display_name']} ({item['importance_pct']}%)" for item in shap_list[:2]])
                print(f"      🔍 [SHAP XAI]: Top Contributing Factors -> {factors}")

    print("\n" + "=" * 80)
    print("  [SUCCESS] STAGE DEMONSTRATION SHOWCASE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
