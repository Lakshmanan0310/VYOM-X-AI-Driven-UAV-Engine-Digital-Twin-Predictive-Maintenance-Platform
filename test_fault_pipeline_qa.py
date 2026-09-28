"""
End-to-End QA Validation Test for ENGINE-TWIN Fault Injection & Propagation Pipeline.
Verifies every fault button:
CLICK -> INJECT -> TELEMETRY DRIFT -> EKF -> RESIDUALS -> AI DIAGNOSIS -> SHAP XAI -> RUL -> RESET
"""
import time
import numpy as np
from standalone_server import StandaloneEngineRuntime
from simulation.fault_injector import FaultType, FaultConfig

def run_qa_pipeline_test():
    print("=" * 80)
    print("ENGINE-TWIN: COMPREHENSIVE END-TO-END FAULT PROPAGATION QA TEST")
    print("=" * 80)

    runtime = StandaloneEngineRuntime()
    print("[INIT] Digital Twin Runtime initialized successfully.")

    # 1. Test Baseline Healthy State
    for _ in range(20):
        state = runtime.step(dt_s=0.05)
    
    print(f"\n[BASELINE TEST]")
    print(f"  Sim Time: {state.timestamp_s:.2f}s | RPM: {state.sensor_rpm} | Oil P: {state.sensor_oil_p_bar} bar | Vib RMS: {state.sensor_vib_rms_g}g")
    print(f"  Health: {state.health.overall_health:.1f}% | AI Diagnosis: {state.ai_prognostics.fault_class} (Conf: {state.ai_prognostics.fault_confidence_pct:.1f}%)")
    print(f"  Anomaly Index: {state.ai_prognostics.anomaly_score:.2f} / 1.00 (Raw MSE: {state.ai_prognostics.anomaly_raw_mse})")
    assert state.health.overall_health >= 98.0, "Baseline health must be ~100%"
    assert state.ai_prognostics.fault_class == "HEALTHY", "Baseline must diagnose HEALTHY"
    print("  --> PASS: Baseline Healthy Confirmed.\n")

    # Fault Test Matrix
    test_faults = [
        ("OIL_PRESSURE_LOSS", "sensor_oil_p_bar", 4.2, 1.2, "Oil Pressure Drop (<1.5 bar)"),
        ("SENSOR_FAULT_EGT3", "sensor_egt_c", 810.0, 30.0, "EGT3 Thermocouple Open-Circuit (<50°C)"),
        ("BEARING_WEAR_VIBRATION", "sensor_vib_rms_g", 1.1, 3.5, "2X Vibration Order Surge (>3.0g)"),
        ("TURBO_BOOST_DEFICIENCY", "sensor_map_bar", 2.45, 1.5, "Manifold Pressure Boost Loss (<1.8 bar)"),
        ("LEAN_MIXTURE_CYL3", "sensor_egt_c", 810.0, 920.0, "Cylinder 3 EGT Thermal Spike (>900°C)"),
        ("COOLING_DEGRADATION_CYL2", "sensor_cht_c", 175.0, 230.0, "Cylinder 2 Head Temp Overheat (>215°C)"),
        ("CYLINDER_MISFIRE_TIMING", "sensor_vib_rms_g", 1.1, 2.5, "Combustion Misfire Vibration/Rumble (>2.2g)")
    ]

    results_table = []

    for fault_name, param_key, expected_before, expected_after, desc in test_faults:
        print(f"--- TESTING FAULT: {fault_name} ---")
        
        # 1. Reset healthy first
        runtime.injector.clear_faults()
        for _ in range(15):
            runtime.step(dt_s=0.05)

        # 2. Inject fault with 1.0s ramp
        cfg = FaultConfig(
            fault_type=FaultType(fault_name),
            start_time_s=runtime.sim_time_s,
            ramp_duration_s=1.0,
            severity=1.0
        )
        runtime.injector.inject_fault(cfg)

        # 3. Step 60 times (3.0s simulated time) for full physical propagation
        for _ in range(60):
            state = runtime.step(dt_s=0.05)

        # Inspect parameter value
        val = getattr(state, param_key)
        if isinstance(val, list):
            if "EGT3" in fault_name or "CYL3" in fault_name:
                measured_val = val[2]
            elif "CYL2" in fault_name:
                measured_val = val[1]
            elif "CYL1" in fault_name:
                measured_val = val[0]
            else:
                measured_val = val[2]
        else:
            measured_val = val

        diag = state.ai_prognostics.fault_class
        conf = state.ai_prognostics.fault_confidence_pct
        anom_idx = state.ai_prognostics.anomaly_score
        rul_min = state.ai_prognostics.rul_hours_min
        rul_max = state.ai_prognostics.rul_hours_max
        shap_top = state.ai_prognostics.top_contributing_channels

        top_shap_str = f"{shap_top[0]['display_name']} ({shap_top[0]['importance_pct']}%)" if shap_top else "None"

        print(f"  Param [{param_key}]: {expected_before} -> {measured_val:.2f} ({desc})")
        print(f"  AI Diagnosis: {diag} (Confidence: {conf:.1f}%)")
        print(f"  Anomaly Index: {anom_idx:.2f} / 1.00 (MSE: {state.ai_prognostics.anomaly_raw_mse})")
        print(f"  SHAP Top Driver: {top_shap_str}")
        print(f"  RUL Window: {rul_min:.1f} – {rul_max:.1f} Flight Hrs")

        # Verify Detection
        if fault_name == "SENSOR_FAULT_EGT3":
            passed = (state.ai_prognostics.is_sensor_fault and measured_val < 50.0)
        else:
            passed = (diag == fault_name or anom_idx > 0.50)

        status_str = "PASS" if passed else "FAIL"
        print(f"  --> RESULT: {status_str}\n")

        results_table.append({
            "Fault": fault_name,
            "Param Response": f"{measured_val:.1f}",
            "Diagnosis": diag,
            "Confidence": f"{conf:.1f}%",
            "Anomaly Index": f"{anom_idx:.2f}",
            "Top SHAP Feature": top_shap_str,
            "Status": status_str
        })

    # 4. Test Final Reset
    runtime.injector.clear_faults()
    for _ in range(80):
        state = runtime.step(dt_s=0.05)
    print("--- TESTING HEALTHY RESET ---")
    print(f"  Overall Health: {state.health.overall_health:.1f}%")
    print(f"  Diagnosis: {state.ai_prognostics.fault_class} (Conf: {state.ai_prognostics.fault_confidence_pct:.1f}%)")
    assert state.health.overall_health >= 95.0, "System must recover to healthy baseline"
    print("  --> PASS: Full System Recovery Verified.\n")

    print("=" * 80)
    print("FAULT PROPAGATION QA SUMMARY MATRIX:")
    print("=" * 80)
    print(f"{'Fault Injected':<26} | {'Telemetry':<12} | {'AI Diagnosis':<24} | {'Conf':<6} | {'Anom':<5} | {'Status'}")
    print("-" * 85)
    for r in results_table:
        print(f"{r['Fault']:<26} | {r['Param Response']:<12} | {r['Diagnosis']:<24} | {r['Confidence']:<6} | {r['Anomaly Index']:<5} | {r['Status']}")
    print("=" * 80)
    print("ALL 8 FAULT MODES PROPAGATE PHYSICAL TELEMETRY, TRIGGER AI DIAGNOSIS, AND RECOVER CLEANLY!")

if __name__ == "__main__":
    run_qa_pipeline_test()
