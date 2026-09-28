"""
Synthetic Dataset Generator for ENGINE-TWIN (SIH26054)
Generates multi-run flight datasets using the calibrated Mean Value Engine Model (MVEM)
and datasheet-grounded sensor model.

Outputs:
1. data/datasets/train_healthy.csv   (Unsupervised baseline training for Autoencoder & Sigma calibration)
2. data/datasets/train_faults.csv    (Supervised training for Fault Classifier & RUL Regressor)
3. data/datasets/test_scenarios.csv  (Held-out runs for baseline comparison & ablation benchmarking)
"""
import os
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.flight_profile import FlightProfile, FlightState
from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.sensors import SensorModel, SensorReadings
from simulation.fault_injector import FaultInjector, FaultType, FaultConfig

OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"

def simulate_run(
    run_id: int,
    duration_s: float = 300.0,
    dt_s: float = 0.1,
    delta_t_isa: float = 0.0,
    fault_type: FaultType = FaultType.HEALTHY,
    fault_start_pct: float = 0.35,
    fault_ramp_s: float = 20.0,
    fault_severity: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Executes a single simulation sortie and returns a DataFrame of sensor readings + physics ground truth.
    """
    flight_gen = FlightProfile(delta_t_isa=delta_t_isa)
    mvem = MeanValueEngineModel()
    sensors = SensorModel(seed=seed)
    injector = FaultInjector(mvem, sensors)

    fault_start_s = duration_s * fault_start_pct
    if fault_type != FaultType.HEALTHY:
        cfg = FaultConfig(
            fault_type=fault_type,
            start_time_s=fault_start_s,
            ramp_duration_s=fault_ramp_s,
            severity=fault_severity
        )
        injector.inject_fault(cfg)

    records = []
    num_steps = int(duration_s / dt_s)

    for step in range(num_steps):
        t = round(step * dt_s, 2)
        flight = flight_gen.get_standard_mission_state(t, total_mission_s=duration_s)
        injector.update(t)
        true_engine = mvem.step(flight, dt_s=dt_s)
        meas = sensors.sample(true_engine, flight, dt_s=dt_s)

        # Ground truth label and RUL calculation
        is_fault_active = (fault_type != FaultType.HEALTHY and t >= fault_start_s)
        label = fault_type.value if is_fault_active else "HEALTHY"
        
        # Remaining Useful Life (RUL) in seconds until critical limit reached
        if is_fault_active:
            time_since_onset = t - fault_start_s
            max_tolerable_time = fault_ramp_s * 1.5
            rul_seconds = max(0.0, max_tolerable_time - time_since_onset)
            rul_hours = rul_seconds / 3600.0 * 20.0 # Scale to realistic flight hours
        else:
            rul_hours = 500.0 # Healthy nominal time before overhaul

        row = {
            "run_id": run_id,
            "time_s": meas.time_s,
            "phase": flight.phase,
            "altitude_ft": meas.altitude_ft,
            "ambient_temp_c": meas.ambient_temp_c,
            "ambient_pressure_bar": meas.ambient_pressure_bar,
            "airspeed_mps": meas.airspeed_mps,
            "throttle_pct": meas.throttle_pct,
            # Measured Telemetry
            "rpm": meas.rpm,
            "manifold_pressure_bar": meas.manifold_pressure_bar,
            "fuel_flow_lph": meas.fuel_flow_lph,
            "oil_pressure_bar": meas.oil_pressure_bar,
            "oil_temp_c": meas.oil_temp_c,
            "coolant_temp_c": meas.coolant_temp_c,
            "egt_1_c": meas.egt_c[0],
            "egt_2_c": meas.egt_c[1],
            "egt_3_c": meas.egt_c[2],
            "egt_4_c": meas.egt_c[3],
            "cht_1_c": meas.cht_c[0],
            "cht_2_c": meas.cht_c[1],
            "cht_3_c": meas.cht_c[2],
            "cht_4_c": meas.cht_c[3],
            "vibration_rms_g": meas.vibration_rms_g,
            "vibration_x_g": meas.vibration_x_g,
            "vibration_y_g": meas.vibration_y_g,
            "vibration_z_g": meas.vibration_z_g,
            "bus_voltage_v": meas.bus_voltage_v,
            # True Physics States (for residual calculation and ground-truth validation)
            "true_power_hp": true_engine.power_hp,
            "true_torque_nm": true_engine.torque_nm,
            "true_bsfc_g_kwh": true_engine.bsfc_g_kwh,
            "true_oil_p_bar": true_engine.oil_pressure_bar,
            "true_oil_t_c": true_engine.oil_temp_c,
            "true_map_bar": true_engine.manifold_pressure_bar,
            "true_egt_3_c": true_engine.egt_c[2],
            "true_cht_2_c": true_engine.cht_c[1],
            # Supervision Labels
            "fault_label": label,
            "fault_severity": round(injector.current_severity, 3),
            "rul_hours": round(rul_hours, 2),
            "provenance": "SIMULATED"
        }
        records.append(row)

    return pd.DataFrame(records)

def generate_all_datasets():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ENGINE-TWIN: Generating Calibrated Synthetic Datasets")
    print("=" * 60)

    # 1. Generate Healthy Training Runs (Various atmospheric conditions & profiles)
    print("[1/3] Generating Healthy Training Runs...")
    healthy_dfs = []
    run_idx = 100
    for delta_isa in [-15.0, -5.0, 0.0, 5.0, 15.0]:
        df = simulate_run(
            run_id=run_idx,
            duration_s=240.0,
            dt_s=0.1,
            delta_t_isa=delta_isa,
            fault_type=FaultType.HEALTHY,
            seed=run_idx
        )
        healthy_dfs.append(df)
        run_idx += 1

    train_healthy_df = pd.concat(healthy_dfs, ignore_index=True)
    train_healthy_path = OUTPUT_DIR / "train_healthy.csv"
    train_healthy_df.to_csv(train_healthy_path, index=False)
    print(f"  -> Saved {len(train_healthy_df)} healthy samples to {train_healthy_path.name}")

    # 2. Generate Multi-Fault Training Runs
    print("[2/3] Generating Labeled Fault Training Runs...")
    fault_dfs = []
    all_faults = [
        FaultType.LEAN_MIXTURE_CYL3,
        FaultType.RICH_MIXTURE_CYL1,
        FaultType.COOLING_DEGRADATION_CYL2,
        FaultType.OIL_PRESSURE_LOSS,
        FaultType.TURBO_BOOST_DEFICIENCY,
        FaultType.CYLINDER_MISFIRE_TIMING,
        FaultType.BEARING_WEAR_VIBRATION,
        FaultType.SENSOR_FAULT_EGT3,
        FaultType.ELECTRICAL_VOLTAGE_SAG
    ]

    run_idx = 200
    for f_type in all_faults:
        for sev in [0.75, 1.0]:
            df = simulate_run(
                run_id=run_idx,
                duration_s=240.0,
                dt_s=0.1,
                delta_t_isa=np.random.choice([-10.0, 0.0, 10.0]),
                fault_type=f_type,
                fault_start_pct=0.30,
                fault_ramp_s=15.0,
                fault_severity=sev,
                seed=run_idx
            )
            fault_dfs.append(df)
            run_idx += 1

    train_faults_df = pd.concat(fault_dfs, ignore_index=True)
    train_faults_path = OUTPUT_DIR / "train_faults.csv"
    train_faults_df.to_csv(train_faults_path, index=False)
    print(f"  -> Saved {len(train_faults_df)} labeled fault samples to {train_faults_path.name}")

    # 3. Generate Completely Held-Out Test Sorties
    print("[3/3] Generating Held-Out Test Sorties...")
    test_dfs = []
    run_idx = 500
    # Include 2 healthy test sorties + 1 of each fault type with unique random seeds
    test_fault_list = [FaultType.HEALTHY, FaultType.HEALTHY] + all_faults
    for f_type in test_fault_list:
        df = simulate_run(
            run_id=run_idx,
            duration_s=200.0,
            dt_s=0.1,
            delta_t_isa=float(np.random.uniform(-12.0, 12.0)),
            fault_type=f_type,
            fault_start_pct=float(np.random.uniform(0.25, 0.45)),
            fault_ramp_s=float(np.random.uniform(10.0, 25.0)),
            fault_severity=float(np.random.uniform(0.80, 1.0)),
            seed=run_idx * 7 + 13
        )
        test_dfs.append(df)
        run_idx += 1

    test_df = pd.concat(test_dfs, ignore_index=True)
    test_path = OUTPUT_DIR / "test_scenarios.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  -> Saved {len(test_df)} held-out test samples to {test_path.name}")
    print("=" * 60)
    print("Dataset generation complete. All provenance labels set to 'SIMULATED'.")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_datasets()
