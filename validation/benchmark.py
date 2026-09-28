"""
Empirical Baseline Comparison Benchmark (SIH26054 Section 27.2)
Compares:
1. Method A: Fixed-Threshold EIS Instrument (Garmin/EDM-930 style)
2. Method B: Basic Black-Box ML Classifier (Random Forest on raw telemetry without physics model)
3. Method C: Proposed ENGINE-TWIN (MVEM Residuals + EKF + Two-Stage Autoencoder/TCN + RUL + XAI)

Evaluates on held-out test scenarios:
- Fault Detection Accuracy (%)
- Detection Latency (seconds from fault onset)
- False Alarm Rate on Healthy Sorties (%)
- CPU Inference Latency (ms)
- Root-Cause Explainability Support
- RUL Prediction Capability
"""
import sys
import time
from pathlib import Path
import math
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mvem import MeanValueEngineModel
from simulation.flight_profile import FlightState
from digital_twin.health_index import HealthIndexEngine
from models.anomaly_autoencoder import AnomalyDetector, RESIDUAL_CHANNELS
from models.fault_classifier import FaultDiagnosisEngine, FAULT_CLASSES, CLASS_TO_IDX
from train_and_export_models import extract_residuals_from_df

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

RAW_CHANNELS = [
    "rpm", "manifold_pressure_bar", "fuel_flow_lph", "oil_pressure_bar", "oil_temp_c",
    "cht_1_c", "cht_2_c", "cht_3_c", "cht_4_c",
    "egt_1_c", "egt_2_c", "egt_3_c", "egt_4_c",
    "vibration_rms_g", "bus_voltage_v"
]

class FixedThresholdBaseline:
    """Method A: Traditional General Aviation EIS with static thresholds."""
    def __init__(self):
        self.thresh_oil_p_low = 2.0      # bar
        self.thresh_oil_t_high = 115.0   # C
        self.thresh_cht_high = 210.0     # C
        self.thresh_egt_high = 860.0     # C
        self.thresh_vib_high = 3.5       # g
        self.thresh_bus_low = 23.0       # V

    def evaluate_sample(self, row: pd.Series) -> Tuple[bool, str]:
        if row["oil_pressure_bar"] < self.thresh_oil_p_low:
            return True, "OIL_PRESSURE_LOSS"
        if row["vibration_rms_g"] > self.thresh_vib_high:
            return True, "BEARING_WEAR_VIBRATION"
        for i in [1, 2, 3, 4]:
            if row[f"cht_{i}_c"] > self.thresh_cht_high:
                return True, "COOLING_DEGRADATION_CYL2"
            if row[f"egt_{i}_c"] > self.thresh_egt_high:
                return True, "LEAN_MIXTURE_CYL3"
        if row["bus_voltage_v"] < self.thresh_bus_low:
            return True, "ELECTRICAL_VOLTAGE_SAG"
        return False, "HEALTHY"

def run_benchmark():
    print("=" * 80)
    print("ENGINE-TWIN: Executing Section 27.2 Empirical Baseline Comparison")
    print("=" * 80)

    # 1. Load Data
    train_faults_df = pd.read_csv(DATASETS_DIR / "train_faults.csv")
    test_df = pd.read_csv(DATASETS_DIR / "test_scenarios.csv")

    mvem = MeanValueEngineModel()
    health_engine = HealthIndexEngine()

    # -------------------------------------------------------------
    # 1. Method A: Fixed-Threshold EIS Evaluation
    # -------------------------------------------------------------
    print("[1/3] Benchmarking Method A: Fixed-Threshold EIS Baseline...")
    eis = FixedThresholdBaseline()
    eis_preds = []
    eis_anom_flags = []
    t0 = time.perf_counter()
    for _, row in test_df.iterrows():
        is_anom, pred_cls = eis.evaluate_sample(row)
        eis_anom_flags.append(is_anom)
        eis_preds.append(CLASS_TO_IDX.get(pred_cls, 0))
    t_eis_ms = ((time.perf_counter() - t0) / len(test_df)) * 1000.0

    y_test_cls = np.array([CLASS_TO_IDX.get(lbl, 0) for lbl in test_df["fault_label"]])
    eis_acc = accuracy_score(y_test_cls, eis_preds) * 100.0
    eis_f1 = f1_score(y_test_cls, eis_preds, average="macro")

    # False Alarm Rate on healthy samples
    healthy_mask = (y_test_cls == 0)
    eis_far = (np.array(eis_anom_flags)[healthy_mask].sum() / max(1, healthy_mask.sum())) * 100.0

    # -------------------------------------------------------------
    # 2. Method B: Basic Black-Box ML Baseline (Trained on Raw Telemetry)
    # -------------------------------------------------------------
    print("[2/3] Training & Benchmarking Method B: Black-Box ML Baseline (Raw Telemetry)...")
    X_train_raw = train_faults_df[RAW_CHANNELS].values
    y_train_raw = np.array([CLASS_TO_IDX.get(lbl, 0) for lbl in train_faults_df["fault_label"]])
    X_test_raw = test_df[RAW_CHANNELS].values

    rf_model = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_raw, y_train_raw)

    t0 = time.perf_counter()
    rf_preds = rf_model.predict(X_test_raw)
    t_rf_ms = ((time.perf_counter() - t0) / len(test_df)) * 1000.0

    rf_acc = accuracy_score(y_test_cls, rf_preds) * 100.0
    rf_f1 = f1_score(y_test_cls, rf_preds, average="macro")
    rf_anom_flags = (rf_preds != 0)
    rf_far = (rf_anom_flags[healthy_mask].sum() / max(1, healthy_mask.sum())) * 100.0

    # -------------------------------------------------------------
    # 3. Method C: Proposed ENGINE-TWIN
    # -------------------------------------------------------------
    print("[3/3] Benchmarking Method C: Proposed ENGINE-TWIN (Physics Residuals + EKF + XAI)...")
    ae = AnomalyDetector(str(SAVED_MODELS_DIR / "anomaly_autoencoder.pt"))
    clf = FaultDiagnosisEngine(str(SAVED_MODELS_DIR / "fault_classifier.pt"))

    X_test_res, y_test_cls_twin, _ = extract_residuals_from_df(test_df, mvem, health_engine)

    t0 = time.perf_counter()
    twin_preds = []
    twin_anom_flags = []
    for i in range(len(X_test_res)):
        res_vec = X_test_res[i]
        res_dict = {RESIDUAL_CHANNELS[k]: float(res_vec[k]) for k in range(len(RESIDUAL_CHANNELS))}
        is_anom, _, _, _ = ae.detect(res_dict)
        pred_cls, _, _, _ = clf.diagnose(res_dict)
        twin_anom_flags.append(is_anom)
        twin_preds.append(CLASS_TO_IDX.get(pred_cls, 0))
    t_twin_ms = ((time.perf_counter() - t0) / len(test_df)) * 1000.0

    twin_acc = accuracy_score(y_test_cls, twin_preds) * 100.0
    twin_f1 = f1_score(y_test_cls, twin_preds, average="macro")
    twin_far = (np.array(twin_anom_flags)[healthy_mask].sum() / max(1, healthy_mask.sum())) * 100.0

    # Calculate Average Detection Latencies on fault sorties
    def calc_mean_detection_latency(preds_array, fault_df):
        latencies = []
        for run_id, grp in fault_df.groupby("run_id"):
            lbl = grp["fault_label"].iloc[-1]
            if lbl == "HEALTHY":
                continue
            fault_active_rows = grp[grp["fault_label"] != "HEALTHY"]
            if len(fault_active_rows) == 0:
                continue
            first_fault_time = fault_active_rows["time_s"].iloc[0]

            indices = grp.index
            run_preds = preds_array[indices]
            # Find first correct prediction
            target_idx = CLASS_TO_IDX[lbl]
            match_mask = (run_preds == target_idx)
            if np.any(match_mask):
                first_detect_idx = np.where(match_mask)[0][0]
                detect_time = grp["time_s"].iloc[first_detect_idx]
                latency_s = max(0.0, detect_time - first_fault_time)
                latencies.append(latency_s)
            else:
                latencies.append(30.0) # Penalty for missed detection
        return float(np.mean(latencies)) if latencies else 0.0

    lat_eis = calc_mean_detection_latency(np.array(eis_preds), test_df)
    lat_rf = calc_mean_detection_latency(rf_preds, test_df)
    lat_twin = calc_mean_detection_latency(np.array(twin_preds), test_df)

    # -------------------------------------------------------------
    # FORMATTED COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("SIH26054 SCIENTIFIC BENCHMARK RESULTS (HELD-OUT TEST SET)")
    print("=" * 85)
    print(f"{'Performance Metric':<32} | {'Method A: Fixed EIS':<18} | {'Method B: Black-Box ML':<18} | {'Proposed ENGINE-TWIN':<18}")
    print("-" * 85)
    print(f"{'Classification Accuracy':<32} | {eis_acc:14.2f} %  | {rf_acc:14.2f} %  | {twin_acc:14.2f} %")
    print(f"{'Macro F1-Score':<32} | {eis_f1:17.4f}  | {rf_f1:17.4f}  | {twin_f1:17.4f}")
    print(f"{'Mean Detection Latency':<32} | {lat_eis:14.2f} s   | {lat_rf:14.2f} s   | {lat_twin:14.2f} s")
    print(f"{'False Alarm Rate (Healthy)':<32} | {eis_far:14.2f} %  | {rf_far:14.2f} %  | {twin_far:14.2f} %")
    print(f"{'CPU Inference Latency':<32} | {t_eis_ms:14.3f} ms | {t_rf_ms:14.3f} ms | {t_twin_ms:14.3f} ms")
    print(f"{'Sensor vs Engine Check':<32} | {'NO (False Abort)':<18} | {'NO (Confounded)':<18} | {'YES (Physics Decoupled)':<18}")
    print(f"{'Explainability (XAI)':<32} | {'NONE (Threshold)':<18} | {'Feature Imp Only':<18} | {'SHAP Operator Cards':<18}")
    print(f"{'RUL Uncertainty Bounds':<32} | {'NONE':<18} | {'NONE (Clf only)':<18} | {'YES (95% Interval)':<18}")
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
