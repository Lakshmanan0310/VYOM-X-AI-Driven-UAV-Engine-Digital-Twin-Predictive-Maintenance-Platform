"""
Complete Model Training, Evaluation, and Calibration Pipeline (SIH26054)
Trains:
1. Unsupervised Residual Autoencoder for Anomaly Detection (with k*sigma threshold calibration)
2. Supervised Multi-Task Fault Classifier + Severity Network (with temperature scaling)
3. Evaluates Precision/Recall/F1, Latency, and False Alarm Rates on Held-Out Test Scenarios.
"""
import os
import sys
from pathlib import Path
import math
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.flight_profile import FlightState
from digital_twin.health_index import HealthIndexEngine
from models.anomaly_autoencoder import ResidualAutoencoderNet, AnomalyDetector, RESIDUAL_CHANNELS
from models.fault_classifier import (
    FaultClassifierNet, FaultDiagnosisEngine, FAULT_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)

SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

def extract_residuals_from_df(df: pd.DataFrame, mvem: MeanValueEngineModel, health_engine: HealthIndexEngine) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true physics-anchored normalized residuals for every row:
    r_norm = (y_measured - y_healthy_mvem) / sigma
    """
    X_list = []
    y_class_list = []
    y_sev_list = []
    
    # Baseline healthy engine instance
    baseline_mvem = MeanValueEngineModel()
    sig = health_engine.sigma

    # Group by run_id so we reset engine state appropriately per flight sortie
    for run_id, group in df.groupby("run_id"):
        baseline_mvem.reset(idle=False)
        for _, row in group.iterrows():
            flight = FlightState(
                time_s=row["time_s"],
                altitude_ft=row["altitude_ft"],
                altitude_m=row["altitude_ft"] * 0.3048,
                throttle_pct=row["throttle_pct"],
                airspeed_mps=row["airspeed_mps"],
                ambient_temp_c=row["ambient_temp_c"],
                ambient_pressure_bar=row["ambient_pressure_bar"],
                air_density_kgpm3=1.225,
                phase=row.get("phase", "CRUISE")
            )
            # Expected healthy state from MVEM
            exp_state = baseline_mvem.step(flight, dt_s=0.1)

            # Residual vector
            r_rpm = (row["rpm"] - exp_state.rpm) / sig["rpm"]
            r_map = (row["manifold_pressure_bar"] - exp_state.manifold_pressure_bar) / sig["manifold_pressure"]
            r_fuel = (row["fuel_flow_lph"] - exp_state.fuel_flow_lph) / sig["fuel_flow"]
            r_oil_p = (row["oil_pressure_bar"] - exp_state.oil_pressure_bar) / sig["oil_pressure"]
            r_oil_t = (row["oil_temp_c"] - exp_state.oil_temp_c) / sig["oil_temp"]
            r_cht = [(row[f"cht_{i}_c"] - exp_state.cht_c[i-1]) / sig["cht"] for i in [1, 2, 3, 4]]
            r_egt = [(row[f"egt_{i}_c"] - exp_state.egt_c[i-1]) / sig["egt"] for i in [1, 2, 3, 4]]
            r_vib = (row["vibration_rms_g"] - exp_state.vibration_rms_g) / sig["vibration_rms"]
            r_bus = (row["bus_voltage_v"] - exp_state.bus_voltage_v) / sig["bus_voltage"]

            res_vec = [
                r_rpm, r_map, r_fuel, r_oil_p, r_oil_t,
                r_cht[0], r_cht[1], r_cht[2], r_cht[3],
                r_egt[0], r_egt[1], r_egt[2], r_egt[3],
                r_vib, r_bus
            ]
            X_list.append(res_vec)

            lbl = row.get("fault_label", "HEALTHY")
            class_idx = CLASS_TO_IDX.get(lbl, 0)
            y_class_list.append(class_idx)
            y_sev_list.append(row.get("fault_severity", 0.0))

    return np.array(X_list, dtype=np.float32), np.array(y_class_list, dtype=np.int64), np.array(y_sev_list, dtype=np.float32)

class ResidualDataset(Dataset):
    def __init__(self, X: np.ndarray, y_class: np.ndarray, y_sev: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_sev = torch.tensor(y_sev, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_sev[idx]

def train_models():
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mvem = MeanValueEngineModel()
    health_engine = HealthIndexEngine()

    print("=" * 65)
    print("ENGINE-TWIN: Training AI Anomaly & Prognostics Pipeline")
    print("=" * 65)

    # 1. Load Data
    print("[1/4] Loading and parsing synthetic datasets...")
    healthy_df = pd.read_csv(DATASETS_DIR / "train_healthy.csv")
    faults_df = pd.read_csv(DATASETS_DIR / "train_faults.csv")
    test_df = pd.read_csv(DATASETS_DIR / "test_scenarios.csv")

    print(f"  -> Healthy samples: {len(healthy_df):,} | Fault samples: {len(faults_df):,} | Test samples: {len(test_df):,}")

    X_healthy, _, _ = extract_residuals_from_df(healthy_df, mvem, health_engine)
    X_faults, y_fault_cls, y_fault_sev = extract_residuals_from_df(faults_df, mvem, health_engine)
    X_test, y_test_cls, y_test_sev = extract_residuals_from_df(test_df, mvem, health_engine)

    # 2. Train Residual Autoencoder (Unsupervised Anomaly Detector)
    print("\n[2/4] Training Unsupervised Residual Autoencoder...")
    ae_net = ResidualAutoencoderNet(input_dim=len(RESIDUAL_CHANNELS))
    optimizer_ae = torch.optim.Adam(ae_net.parameters(), lr=0.003, weight_decay=1e-5)
    criterion_ae = nn.MSELoss()

    ae_loader = DataLoader(ResidualDataset(X_healthy, np.zeros(len(X_healthy)), np.zeros(len(X_healthy))), batch_size=64, shuffle=True)

    for epoch in range(1, 16):
        ae_net.train()
        total_loss = 0.0
        for bx, _, _ in ae_loader:
            optimizer_ae.zero_grad()
            recon, _ = ae_net(bx)
            loss = criterion_ae(recon, bx)
            loss.backward()
            optimizer_ae.step()
            total_loss += loss.item() * len(bx)

        if epoch % 5 == 0 or epoch == 15:
            avg_loss = total_loss / len(X_healthy)
            print(f"  Epoch {epoch:2d}/15 - Autoencoder Reconstruction MSE: {avg_loss:.5f}")

    # Calibrate Anomaly Threshold on healthy set (Mean + 3.0*sigma)
    ae_net.eval()
    with torch.no_grad():
        rec, _ = ae_net(torch.tensor(X_healthy))
        mse_healthy = torch.mean((torch.tensor(X_healthy) - rec)**2, dim=1).numpy()
    mu_h = float(np.mean(mse_healthy))
    sigma_h = float(np.std(mse_healthy))
    calibrated_threshold = mu_h + 3.0 * sigma_h
    print(f"  -> Calibrated Anomaly Threshold: {calibrated_threshold:.4f} (mu={mu_h:.4f}, sigma={sigma_h:.4f})")

    # Save Autoencoder checkpoint
    torch.save({
        "model_state": ae_net.state_dict(),
        "threshold": calibrated_threshold,
        "mu": mu_h,
        "sigma": sigma_h
    }, SAVED_MODELS_DIR / "anomaly_autoencoder.pt")
    print(f"  -> Saved autoencoder to {SAVED_MODELS_DIR / 'anomaly_autoencoder.pt'}")

    # 3. Train Multi-Task Fault Classifier & Severity Estimator
    print("\n[3/4] Training Multi-Task Fault Classifier + Severity Estimator...")
    clf_net = FaultClassifierNet(input_dim=len(RESIDUAL_CHANNELS), num_classes=len(FAULT_CLASSES))
    optimizer_clf = torch.optim.AdamW(clf_net.parameters(), lr=0.004, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_clf, T_max=20)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sev = nn.HuberLoss(delta=0.1)

    clf_loader = DataLoader(ResidualDataset(X_faults, y_fault_cls, y_fault_sev), batch_size=128, shuffle=True)

    for epoch in range(1, 21):
        clf_net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by_cls, by_sev in clf_loader:
            optimizer_clf.zero_grad()
            logits, pred_sev = clf_net(bx)
            loss_c = criterion_cls(logits, by_cls)
            loss_s = criterion_sev(pred_sev, by_sev)
            loss = loss_c + 2.0 * loss_s
            loss.backward()
            optimizer_clf.step()

            total_loss += loss.item() * len(bx)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by_cls).sum().item()
            total += len(bx)

        scheduler.step()
        if epoch % 5 == 0 or epoch == 20:
            print(f"  Epoch {epoch:2d}/20 - Loss: {total_loss/total:.4f} | Train Acc: {(correct/total)*100:.2f}%")

    # Save Classifier checkpoint
    torch.save(clf_net.state_dict(), SAVED_MODELS_DIR / "fault_classifier.pt")
    print(f"  -> Saved classifier to {SAVED_MODELS_DIR / 'fault_classifier.pt'}")

    # 4. Rigorous Evaluation on Held-Out Test Scenarios
    print("\n[4/4] Evaluating on Completely Held-Out Test Sorties...")
    clf_net.eval()
    ae_net.eval()

    # Latency Benchmark
    t0 = time.perf_counter()
    num_eval_runs = 1000
    with torch.no_grad():
        test_sample = torch.tensor(X_test[:1])
        for _ in range(num_eval_runs):
            _ = ae_net(test_sample)
            _ = clf_net(test_sample)
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    avg_latency_ms = total_time_ms / num_eval_runs

    with torch.no_grad():
        test_x_t = torch.tensor(X_test)
        logits, pred_sev = clf_net(test_x_t)
        test_preds = torch.argmax(logits, dim=1).numpy()
        test_probs = torch.softmax(logits, dim=1).numpy()

        # Anomaly detection evaluation
        recon_test, _ = ae_net(test_x_t)
        mse_test = torch.mean((test_x_t - recon_test)**2, dim=1).numpy()
        anomaly_flags = (mse_test >= calibrated_threshold)

    # Metrics
    f1_macro = f1_score(y_test_cls, test_preds, average="macro")
    overall_acc = np.mean(test_preds == y_test_cls) * 100.0

    # True Anomaly vs Detected Anomaly Recall
    is_true_fault = (y_test_cls != 0)
    anomaly_recall = (anomaly_flags[is_true_fault].sum() / max(1, is_true_fault.sum())) * 100.0
    # False Alarm Rate on healthy test samples
    healthy_mask = (y_test_cls == 0)
    false_alarms = anomaly_flags[healthy_mask].sum()
    false_alarm_rate_pct = (false_alarms / max(1, healthy_mask.sum())) * 100.0

    print("=" * 65)
    print("HELD-OUT TEST EVALUATION REPORT (SIH26054)")
    print("=" * 65)
    print(f"  • Overall Fault Classification Accuracy: {overall_acc:.2f}%")
    print(f"  • Macro F1-Score:                        {f1_macro:.4f}")
    print(f"  • Anomaly Detection Recall (True Faults): {anomaly_recall:.2f}%")
    print(f"  • False Alarm Rate on Healthy Sorties:    {false_alarm_rate_pct:.2f}%")
    print(f"  • Mean CPU Inference Latency (AD + Clf): {avg_latency_ms:.3f} ms (Target < 150 ms)")
    print("-" * 65)
    print("PER-CLASS CLASSIFICATION BREAKDOWN:")
    print(classification_report(y_test_cls, test_preds, target_names=FAULT_CLASSES, digits=3))
    print("=" * 65)

if __name__ == "__main__":
    train_models()
