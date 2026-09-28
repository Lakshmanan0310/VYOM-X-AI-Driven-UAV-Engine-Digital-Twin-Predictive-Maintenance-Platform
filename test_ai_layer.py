"""
Step 2 Verification Test Suite for ENGINE-TWIN
Tests:
1. EKF State Estimation & Noise Smoothing Convergence.
2. Health Index Subsystem & Composite Scoring.
3. Sensor-vs-Engine Fault Discriminator Integrity.
4. Trained PyTorch Anomaly Autoencoder & Fault Classifier Inference.
5. RUL Uncertainty Interval Estimation.
6. SHAP Feature Attribution & Structured Operator Alert Generation.
"""
import sys
from pathlib import Path
import math
import numpy as np
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.flight_profile import FlightProfile, FlightState
from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.sensors import SensorModel, SensorReadings
from simulation.fault_injector import FaultInjector, FaultType, FaultConfig
from digital_twin.health_index import HealthIndexEngine
from digital_twin.ekf_estimator import ExtendedKalmanFilter
from digital_twin.twin_state import DigitalTwinState, StateLevel
from models.sensor_validator import SensorValidator
from models.anomaly_autoencoder import AnomalyDetector
from models.fault_classifier import FaultDiagnosisEngine
from models.rul_estimator import RULEstimator
from xai.shap_explainer import FastSHAPExplainer
from xai.alert_generator import AlertGenerator

SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

def test_ekf_filtering():
    print("[TEST 1] Testing Extended Kalman Filter State Estimation & Noise Smoothing...")
    mvem = MeanValueEngineModel()
    mvem.reset(idle=False)
    sensors = SensorModel(seed=42)
    ekf = ExtendedKalmanFilter()
    flight = FlightProfile().get_standard_mission_state(100.0, 3600.0)

    raw_errors = []
    ekf_errors = []

    # Run for 200 steps
    for _ in range(200):
        true_eng = mvem.step(flight, dt_s=0.05)
        meas = sensors.sample(true_eng, flight, dt_s=0.05)

        ekf.predict(true_eng, dt_s=0.05)
        ekf.update(meas)
        est = ekf.get_estimated_state()

        raw_errors.append(abs(meas.egt_c[0] - true_eng.egt_c[0]))
        ekf_errors.append(abs(est["estimated_egt_c"][0] - true_eng.egt_c[0]))

    raw_mae = float(np.mean(raw_errors[50:]))
    ekf_mae = float(np.mean(ekf_errors[50:]))

    print(f"  -> Raw Sensor EGT MAE: {raw_mae:.2f} C | EKF Filtered State MAE: {ekf_mae:.2f} C")
    assert ekf_mae < raw_mae, "EKF did not reduce measurement noise error"
    print("  -> PASSED: EKF successfully filters measurement noise and tracks true physical state.")

def test_health_scoring():
    print("\n[TEST 2] Testing Physics-Anchored Health Index Engine...")
    health_engine = HealthIndexEngine()

    # Nominal healthy data
    sensor_healthy = {
        "oil_pressure": 4.5, "oil_temp": 90.0, "manifold_pressure": 2.40,
        "rpm": 4000.0, "fuel_flow": 26.0, "vibration_rms": 1.2, "bus_voltage": 28.0,
        "egt_c": [750.0, 750.0, 750.0, 750.0],
        "cht_c": [175.0, 175.0, 175.0, 175.0]
    }
    mvem_expected = {
        "oil_pressure": 4.5, "oil_temp": 90.0, "manifold_pressure": 2.40,
        "rpm": 4000.0, "fuel_flow": 26.0, "vibration_rms": 1.2, "bus_voltage": 28.0,
        "egt_c": [750.0, 750.0, 750.0, 750.0],
        "cht_c": [175.0, 175.0, 175.0, 175.0]
    }

    res_healthy = health_engine.compute_residuals(sensor_healthy, mvem_expected)
    h_healthy = health_engine.evaluate_health(res_healthy)
    print(f"  -> Nominal Flight Health: {h_healthy.overall_health:.1f}% (Status: {h_healthy.status_level.value})")
    assert h_healthy.overall_health >= 95.0, "Healthy state health score should be > 95%"

    # Severe Oil Loss Fault Data
    sensor_faulty = dict(sensor_healthy)
    sensor_faulty["oil_pressure"] = 1.2 # Severe 55-sigma drop
    res_faulty = health_engine.compute_residuals(sensor_faulty, mvem_expected)
    h_faulty = health_engine.evaluate_health(res_faulty)

    print(f"  -> Injected Oil Loss Health: {h_faulty.overall_health:.1f}% (Oil System: {h_faulty.oil_system_health:.1f}%, Status: {h_faulty.status_level.value})")
    assert h_faulty.overall_health < 45.0, "Severe fault should drop overall health below 45%"
    print("  -> PASSED: Health Index Engine correctly maps physics residuals to safety health scores.")

def test_sensor_validator():
    print("\n[TEST 3] Testing Sensor-vs-Engine Fault Decoupling...")
    validator = SensorValidator()

    # Thermocouple defect: EGT reads 18 C (ambient) while CHT is 175 C
    sensor_data = {
        "ambient_temp_c": 15.0,
        "egt_c": [750.0, 750.0, 18.0, 750.0],
        "cht_c": [175.0, 175.0, 175.0, 175.0],
        "oil_pressure_bar": 4.5,
        "oil_temp_c": 90.0,
        "vibration_rms_g": 1.2
    }
    mvem_exp = {
        "egt_c": [750.0, 750.0, 750.0, 750.0],
        "cht_c": [175.0, 175.0, 175.0, 175.0]
    }
    res = {"egt_cyl_3": -209.1}

    result = validator.validate(sensor_data, mvem_exp, res)
    print(f"  -> Is Sensor Fault: {result.is_sensor_fault} | Faulty Channel: {result.faulty_channel} | Confidence: {result.confidence_pct}%")
    print(f"  -> Explanation: {result.explanation}")
    assert result.is_sensor_fault == True, "Failed to identify thermocouple open-circuit"
    print("  -> PASSED: Sensor validator prevents false engine abort on faulty probe.")

def test_ai_models_and_xai():
    print("\n[TEST 4] Testing Anomaly Detector, Fault Classifier & SHAP XAI Engine...")
    ae_path = SAVED_MODELS_DIR / "anomaly_autoencoder.pt"
    clf_path = SAVED_MODELS_DIR / "fault_classifier.pt"

    assert ae_path.exists(), "Autoencoder model weights not found"
    assert clf_path.exists(), "Fault classifier model weights not found"

    ae = AnomalyDetector(str(ae_path))
    clf = FaultDiagnosisEngine(str(clf_path))
    explainer = FastSHAPExplainer(clf.model)
    rul_eng = RULEstimator()

    # 1. Healthy Residual Vector
    res_healthy = {ch: 0.1 for ch in ["rpm", "manifold_pressure", "fuel_flow", "oil_pressure", "oil_temp",
                                      "cht_cyl_1", "cht_cyl_2", "cht_cyl_3", "cht_cyl_4",
                                      "egt_cyl_1", "egt_cyl_2", "egt_cyl_3", "egt_cyl_4",
                                      "vibration_rms", "bus_voltage"]}
    is_anom, score, thresh, raw_mse = ae.detect(res_healthy)
    pred_cls, conf, sev, _ = clf.diagnose(res_healthy)
    print(f"  -> Healthy Test: Anomaly={is_anom} (Index={score:.2f}, Raw MSE={raw_mse:.2f}, Thresh={thresh:.2f}), Diagnosis={pred_cls} ({conf}%)")
    assert not is_anom, "False positive on healthy baseline"

    # 2. Bearing Wear Fault Vector (2X Vibration surge)
    res_bearing = dict(res_healthy)
    res_bearing["vibration_rms"] = 18.5 # Large vibration residual
    res_bearing["oil_temp"] = 6.2

    is_anom, score, thresh, raw_mse = ae.detect(res_bearing)
    pred_cls, conf, sev, _ = clf.diagnose(res_bearing)
    shap_exps = explainer.explain(res_bearing, pred_cls)

    print(f"  -> Injected Bearing Wear: Anomaly={is_anom} (Index={score:.2f}, Raw MSE={raw_mse:.2f}), Diagnosis={pred_cls} ({conf}%, Severity={sev})")
    assert is_anom, "Autoencoder failed to flag high vibration anomaly"
    assert pred_cls == "BEARING_WEAR_VIBRATION", f"Classifier misdiagnosed: {pred_cls}"

    # Verify SHAP top contributing feature
    print("  -> SHAP Root-Cause Attribution Breakdown:")
    for exp in shap_exps:
        print(f"     * {exp['display_name']}: {exp['deviation_text']} (Impact: {exp['importance_pct']}%)")
    assert len(shap_exps) > 0 and shap_exps[0]["channel_key"] == "vibration_rms", "SHAP failed to highlight vibration as top factor"

    # 3. RUL Prognosis Simulation
    rul_eng.reset()
    for t_step in range(30):
        t_sec = t_step * 5.0
        # Simulated health drop from 80% to 45% over 150s
        h_val = max(35.0, 80.0 - (t_step * 1.4))
        rul_pred = rul_eng.update(t_sec, h_val)

    print(f"  -> RUL Forecast: Regime={rul_pred.regime} | Expected={rul_pred.rul_hours_mean:.1f} hrs | Interval=[{rul_pred.rul_hours_min:.1f}, {rul_pred.rul_hours_max:.1f}] hrs (Uncertainty: {rul_pred.uncertainty_level})")
    assert rul_pred.rul_hours_min < rul_pred.rul_hours_mean < rul_pred.rul_hours_max, "Invalid RUL uncertainty bounds"

    # 4. Operator Alert Generation
    alert = AlertGenerator.generate(
        timestamp_s=150.0,
        fault_class=pred_cls,
        confidence_pct=conf,
        severity=sev,
        is_sensor_fault=False,
        sensor_check_explanation="PASSED",
        shap_explanations=shap_exps,
        rul_prediction=rul_pred,
        overall_health=48.0
    )
    print("\n  -> Generated Operator Decision Alert:")
    print(f"     [ID]:          {alert.alert_id}")
    print(f"     [Headline]:    {alert.headline}")
    print(f"     [Subsystem]:   {alert.subsystem}")
    print(f"     [Sensor Check]:{alert.sensor_fault_check}")
    print(f"     [RUL Action]:  {alert.rul_forecast_text}")
    print(f"     [Directive]:   {alert.recommended_action}")
    assert alert is not None and alert.level == "WARNING", "Alert generation failed"
    print("  -> PASSED: AI Anomaly, Diagnosis, RUL, SHAP, and Alert pipelines validated end-to-end.")

if __name__ == "__main__":
    test_ekf_filtering()
    test_health_scoring()
    test_sensor_validator()
    test_ai_models_and_xai()
    print("\n" + "=" * 60)
    print("ALL STEP 2 AI & DIGITAL TWIN TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
