"""
ENGINE-TWIN: Zero-Dependency Standalone HTTP & SSE Server (SIH26054)
Uses Python standard library (http.server, urllib, json, threading) to serve the
interactive dashboard and stream 20 Hz digital twin telemetry without requiring FastAPI/Uvicorn.
"""
import os
import sys
import json
import time
import math
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.flight_profile import FlightProfile, FlightState
from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.sensors import SensorModel, SensorReadings
from simulation.fault_injector import FaultInjector, FaultType, FaultConfig
from digital_twin.health_index import HealthIndexEngine
from digital_twin.ekf_estimator import ExtendedKalmanFilter
from digital_twin.twin_state import DigitalTwinState, SubsystemHealth, AIHealthState, StateLevel
from models.sensor_validator import SensorValidator
from models.anomaly_autoencoder import AnomalyDetector
from models.fault_classifier import FaultDiagnosisEngine
from models.rul_estimator import RULEstimator
from xai.shap_explainer import FastSHAPExplainer
from xai.alert_generator import AlertGenerator

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

class StandaloneEngineRuntime:
    def __init__(self):
        self.flight_gen = FlightProfile()
        self.mvem_physical = MeanValueEngineModel()
        self.mvem_baseline = MeanValueEngineModel()
        self.sensors = SensorModel(seed=42)
        self.injector = FaultInjector(self.mvem_physical, self.sensors)
        self.ekf = ExtendedKalmanFilter()
        self.health_engine = HealthIndexEngine()
        self.sensor_validator = SensorValidator()
        self.rul_engine = RULEstimator()

        # AI Models
        models_dir = PROJECT_ROOT / "models" / "saved_models"
        ae_path = models_dir / "anomaly_autoencoder.pt"
        clf_path = models_dir / "fault_classifier.pt"

        self.anomaly_detector = AnomalyDetector(str(ae_path) if ae_path.exists() else None)
        self.fault_classifier = FaultDiagnosisEngine(str(clf_path) if clf_path.exists() else None)
        self.shap_explainer = FastSHAPExplainer(self.fault_classifier.model if self.fault_classifier.is_trained else None)

        self.sim_time_s = 1000.0
        self.elapsed_sortie_s = 0.0
        self.time_scale = 1.0
        self.is_running = True
        self.total_mission_duration_s = 3600.0

        # Flight Sandbox Override States
        self.manual_override = False
        self.override_throttle = 72.0
        self.override_altitude = 28500.0
        self.override_oat = -42.5

        self.latest_state: Optional[DigitalTwinState] = None
        self.latest_alert: Optional[Dict] = None
        self.active_fault_name = "HEALTHY"
        self.lock = threading.Lock()

        # Warm up thermal filters to flight operating equilibrium
        self._warmup_thermal_equilibrium()

    def _warmup_thermal_equilibrium(self):
        flight = self.flight_gen.get_standard_mission_state(self.sim_time_s, self.total_mission_duration_s)
        for _ in range(250):
            p = self.mvem_physical.step(flight, dt_s=0.1)
            self.sensors.sample(p, flight, dt_s=0.1)
            self.mvem_baseline.step(flight, dt_s=0.1)

    def reset(self):
        with self.lock:
            self.sim_time_s = 1000.0
            self.elapsed_sortie_s = 0.0
            self.manual_override = False
            self.mvem_physical.reset(idle=False)
            self.mvem_baseline.reset(idle=False)
            self.sensors = SensorModel(seed=int(time.time()))
            self.injector = FaultInjector(self.mvem_physical, self.sensors)
            self.ekf = ExtendedKalmanFilter()
            self.rul_engine.reset()
            self.latest_alert = None
            self.active_fault_name = "HEALTHY"
            self._warmup_thermal_equilibrium()

    def step(self, dt_s: float = 0.05) -> DigitalTwinState:
        with self.lock:
            self.sim_time_s += dt_s * self.time_scale
            self.elapsed_sortie_s += dt_s * self.time_scale
            t = self.sim_time_s

            flight = self.flight_gen.get_standard_mission_state(t, self.total_mission_duration_s)
            
            # Apply Manual Flight Sandbox Override if enabled
            if self.manual_override:
                flight.throttle_pct = max(0.0, min(100.0, self.override_throttle))
                flight.altitude_ft = max(0.0, min(35000.0, self.override_altitude))
                flight.ambient_temp_c = max(-60.0, min(50.0, self.override_oat))
                alt_m = flight.altitude_ft * 0.3048
                t_c, p_bar, rho = FlightProfile.get_isa_atmosphere(alt_m)
                flight.ambient_pressure_bar = p_bar
                flight.air_density_kgpm3 = rho

            self.injector.update(t)

            true_physical = self.mvem_physical.step(flight, dt_s=dt_s)
            sensor_meas = self.sensors.sample(true_physical, flight, dt_s=dt_s)
            mvem_expected = self.mvem_baseline.step(flight, dt_s=dt_s)

            self.ekf.predict(mvem_expected, dt_s=dt_s)
            self.ekf.update(sensor_meas)
            ekf_est = self.ekf.get_estimated_state()

            sensor_dict = {
                "rpm": sensor_meas.rpm,
                "manifold_pressure": sensor_meas.manifold_pressure_bar,
                "oil_pressure": sensor_meas.oil_pressure_bar,
                "oil_temp": sensor_meas.oil_temp_c,
                "fuel_flow": sensor_meas.fuel_flow_lph,
                "egt_c": sensor_meas.egt_c,
                "cht_c": sensor_meas.cht_c,
                "vibration_rms": sensor_meas.vibration_rms_g,
                "bus_voltage": sensor_meas.bus_voltage_v,
                "ambient_temp_c": sensor_meas.ambient_temp_c
            }
            mvem_dict = {
                "rpm": mvem_expected.rpm,
                "manifold_pressure": mvem_expected.manifold_pressure_bar,
                "oil_pressure": mvem_expected.oil_pressure_bar,
                "oil_temp": mvem_expected.oil_temp_c,
                "fuel_flow": mvem_expected.fuel_flow_lph,
                "egt_c": mvem_expected.egt_c,
                "cht_c": mvem_expected.cht_c,
                "vibration_rms": mvem_expected.vibration_rms_g,
                "bus_voltage": mvem_expected.bus_voltage_v
            }
            residuals = self.health_engine.compute_residuals(sensor_dict, mvem_dict)
            sensor_check = self.sensor_validator.validate(sensor_dict, mvem_dict, residuals)

            if sensor_check.is_sensor_fault:
                # Decouple faulty sensor probe from mechanical health evaluation
                mech_residuals = dict(residuals)
                if sensor_check.faulty_channel:
                    mech_residuals[sensor_check.faulty_channel] = 0.0
                subsystem_health = self.health_engine.evaluate_health(mech_residuals)
                subsystem_health.status_level = StateLevel.ADVISORY
            else:
                subsystem_health = self.health_engine.evaluate_health(residuals)

            is_anom, raw_mse, anom_thresh, anom_index = self.anomaly_detector.detect(residuals)

            if sensor_check.is_sensor_fault:
                fault_class = "SENSOR_FAULT_EGT3" if "egt" in (sensor_check.faulty_channel or "") else "SENSOR_FAULT_GENERAL"
                conf = sensor_check.confidence_pct
                sev = 0.25
            elif is_anom or subsystem_health.overall_health < 85.0:
                fault_class, conf, sev, _ = self.fault_classifier.diagnose(residuals)
            else:
                fault_class = "HEALTHY"
                conf = 99.8
                sev = 0.0

            rul_pred = self.rul_engine.update(t, subsystem_health.overall_health)
            shap_exps = self.shap_explainer.explain(residuals, fault_class)

            alert = AlertGenerator.generate(
                timestamp_s=t,
                fault_class=fault_class,
                confidence_pct=conf,
                severity=sev,
                is_sensor_fault=sensor_check.is_sensor_fault,
                sensor_check_explanation=sensor_check.explanation,
                shap_explanations=shap_exps,
                rul_prediction=rul_pred,
                overall_health=subsystem_health.overall_health
            )
            self.latest_alert = alert.to_dict() if alert else None

            ai_health_state = AIHealthState(
                anomaly_detected=is_anom,
                anomaly_score=anom_index,
                anomaly_raw_mse=raw_mse,
                anomaly_threshold=anom_thresh,
                fault_class=fault_class,
                fault_confidence_pct=conf,
                fault_severity=sev,
                is_sensor_fault=sensor_check.is_sensor_fault,
                rul_hours_mean=rul_pred.rul_hours_mean,
                rul_hours_min=rul_pred.rul_hours_min,
                rul_hours_max=rul_pred.rul_hours_max,
                top_contributing_channels=shap_exps,
                recommended_action=alert.recommended_action if alert else "Continue nominal mission profile."
            )

            state = DigitalTwinState(
                timestamp_s=round(t, 2),
                altitude_ft=round(flight.altitude_ft, 1),
                ambient_temp_c=round(flight.ambient_temp_c, 2),
                throttle_pct=round(flight.throttle_pct, 1),
                airspeed_mps=round(flight.airspeed_mps, 1),
                mission_phase=flight.phase,
                sensor_rpm=round(sensor_meas.rpm, 1),
                sensor_map_bar=round(sensor_meas.manifold_pressure_bar, 3),
                sensor_oil_p_bar=round(sensor_meas.oil_pressure_bar, 2),
                sensor_oil_t_c=round(sensor_meas.oil_temp_c, 1),
                sensor_coolant_t_c=round(sensor_meas.coolant_temp_c, 1),
                sensor_fuel_flow_lph=round(sensor_meas.fuel_flow_lph, 2),
                sensor_egt_c=[round(x, 1) for x in sensor_meas.egt_c],
                sensor_cht_c=[round(x, 1) for x in sensor_meas.cht_c],
                sensor_vib_rms_g=round(sensor_meas.vibration_rms_g, 3),
                sensor_bus_v=round(sensor_meas.bus_voltage_v, 2),
                mvem_expected_power_hp=round(mvem_expected.power_hp, 1),
                mvem_expected_rpm=round(mvem_expected.rpm, 1),
                mvem_expected_map_bar=round(mvem_expected.manifold_pressure_bar, 3),
                mvem_expected_oil_p_bar=round(mvem_expected.oil_pressure_bar, 2),
                mvem_expected_oil_t_c=round(mvem_expected.oil_temp_c, 1),
                mvem_expected_egt_c=[round(x, 1) for x in mvem_expected.egt_c],
                mvem_expected_cht_c=[round(x, 1) for x in mvem_expected.cht_c],
                mvem_expected_fuel_lph=round(mvem_expected.fuel_flow_lph, 2),
                mvem_expected_vib_rms_g=round(mvem_expected.vibration_rms_g, 3),
                estimated_rpm=ekf_est["estimated_rpm"],
                estimated_map_bar=ekf_est["estimated_map_bar"],
                estimated_oil_p_bar=ekf_est["estimated_oil_p_bar"],
                estimated_oil_t_c=ekf_est["estimated_oil_t_c"],
                estimated_egt_c=ekf_est["estimated_egt_c"],
                estimated_cht_c=ekf_est["estimated_cht_c"],
                residuals=residuals,
                health=subsystem_health,
                ai_prognostics=ai_health_state,
                active_fault=self.active_fault_name,
                provenance="SIMULATED"
            )
            self.latest_state = state
            return state

twin_runtime = StandaloneEngineRuntime()

# Background Simulation Thread (20 Hz)
def simulation_loop():
    while twin_runtime.is_running:
        twin_runtime.step(dt_s=0.05)
        time.sleep(0.05)

class DigitalTwinHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, format, *args):
        # Suppress routine 20 Hz GET polling logs to keep console clean
        if "/api/state" in format or "/api/stream" in format or "/static" in format:
            return
        super().log_message(format, *args)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(DASHBOARD_DIR / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/static/app.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.end_headers()
            with open(DASHBOARD_DIR / "app.js", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/static/engine_view3d.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.end_headers()
            with open(DASHBOARD_DIR / "engine_view3d.js", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if twin_runtime.latest_state:
                payload = twin_runtime.latest_state.to_dict()
                payload["active_alert"] = twin_runtime.latest_alert
                payload["active_fault"] = twin_runtime.active_fault_name
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            else:
                self.wfile.write(b'{"status":"initializing"}')
            return

        elif parsed.path == "/api/stream":
            # Server-Sent Events (SSE) stream at 20 Hz
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    if twin_runtime.latest_state:
                        payload = twin_runtime.latest_state.to_dict()
                        payload["active_alert"] = twin_runtime.latest_alert
                        payload["active_fault"] = twin_runtime.active_fault_name
                        msg = f"data: {json.dumps(payload)}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    time.sleep(0.05)
            except Exception:
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        if parsed.path == "/api/fault/inject":
            try:
                data = json.loads(body.decode("utf-8"))
                f_type_str = data.get("fault_type", "HEALTHY")
                f_enum = FaultType(f_type_str)
                sev = float(data.get("severity", 1.0))
                ramp = float(data.get("ramp_duration_s", 1.0))

                cfg = FaultConfig(
                    fault_type=f_enum,
                    start_time_s=twin_runtime.sim_time_s,
                    ramp_duration_s=ramp,
                    severity=sev
                )
                twin_runtime.injector.inject_fault(cfg)
                twin_runtime.active_fault_name = f_type_str
                resp = {"status": "SUCCESS", "injected_fault": f_type_str, "severity": sev}
            except Exception as e:
                resp = {"status": "ERROR", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/flight/override":
            try:
                data = json.loads(body.decode("utf-8"))
                enabled = bool(data.get("enabled", True))
                twin_runtime.manual_override = enabled
                if "throttle_pct" in data:
                    twin_runtime.override_throttle = float(data["throttle_pct"])
                if "altitude_ft" in data:
                    twin_runtime.override_altitude = float(data["altitude_ft"])
                if "ambient_temp_c" in data:
                    twin_runtime.override_oat = float(data["ambient_temp_c"])
                resp = {
                    "status": "SUCCESS",
                    "manual_override": twin_runtime.manual_override,
                    "throttle_pct": twin_runtime.override_throttle,
                    "altitude_ft": twin_runtime.override_altitude,
                    "ambient_temp_c": twin_runtime.override_oat
                }
            except Exception as e:
                resp = {"status": "ERROR", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/fault/clear":
            twin_runtime.injector.clear_faults()
            twin_runtime.active_fault_name = "HEALTHY"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status":"SUCCESS","message":"Engine reset to healthy baseline."}')
            return

        elif parsed.path == "/api/sim/reset":
            twin_runtime.reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status":"SUCCESS","message":"Simulation reset."}')
            return

        elif "/api/sim/speed" in parsed.path:
            qs = parse_qs(parsed.query)
            spd = float(qs.get("speed", [1.0])[0])
            twin_runtime.time_scale = max(0.2, min(10.0, spd))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "SUCCESS", "time_scale": twin_runtime.time_scale}).encode("utf-8"))
            return

def run_standalone_server(port: int = 8000):
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("127.0.0.1", port), DigitalTwinHTTPHandler)
    print(f"ENGINE-TWIN Standalone Server running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        twin_runtime.is_running = False
        server.server_close()

if __name__ == "__main__":
    run_standalone_server()
