"""
Step 1 Verification Test Suite for ENGINE-TWIN
Tests:
1. ISA Atmospheric calculations across 0–30,000 ft.
2. MVEM altitude-power derating against published DRDO/VRDE numbers + Calibration RMSE.
3. Sensor noise and thermal lag characteristics.
4. Fault injection dynamics and multi-channel physical responses.
"""
import sys
from pathlib import Path
import math
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation.flight_profile import FlightProfile, FlightState
from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.sensors import SensorModel, SensorReadings
from simulation.fault_injector import FaultInjector, FaultType, FaultConfig
from data.generate_dataset import generate_all_datasets, OUTPUT_DIR

def test_isa_atmosphere():
    print("[TEST 1] Testing ISA Atmospheric Model...")
    # Sea level (0m)
    t_0, p_0, rho_0 = FlightProfile.get_isa_atmosphere(0.0)
    assert abs(t_0 - 15.0) < 0.1, f"Expected 15.0 C at sea level, got {t_0}"
    assert abs(p_0 - 1.01325) < 0.001, f"Expected 1.013 bar at sea level, got {p_0}"
    assert abs(rho_0 - 1.225) < 0.01, f"Expected 1.225 kg/m3 at sea level, got {rho_0}"

    # 30,000 ft (9144m)
    h_30k_m = FlightProfile.feet_to_meters(30000.0)
    t_30k, p_30k, rho_30k = FlightProfile.get_isa_atmosphere(h_30k_m)
    # ISA at 9144m: T ~ -44.4 C, P ~ 0.301 bar
    assert abs(t_30k - (-44.4)) < 2.0, f"Expected ~ -44.4 C at 30k ft, got {t_30k}"
    assert abs(p_30k - 0.301) < 0.05, f"Expected ~ 0.301 bar at 30k ft, got {p_30k}"
    print("  -> PASSED: ISA model accurately matches standard aerospace atmospheric tables.")

def test_vrde_altitude_derating():
    print("\n[TEST 2] Testing MVEM Calibration against DRDO/VRDE 2.2L Altitude Power Envelope...")
    # Target published points:
    # 0 ft: 200 hp (MAP ~2.40 bar)
    # 10,000 ft: 200 hp (MAP ~2.45 bar)
    # 20,000 ft: 150 hp (MAP ~2.10 bar)
    # 30,000 ft: 110 hp (MAP ~1.65 bar)
    test_altitudes_ft = [0.0, 10000.0, 20000.0, 30000.0]
    expected_power_hp = [200.0, 200.0, 150.0, 110.0]
    expected_map_bar = [2.40, 2.45, 2.10, 1.65]

    mvem = MeanValueEngineModel()
    measured_powers = []
    measured_maps = []

    print("  Altitude (ft) | Target Power (HP) | Model Power (HP) | Target MAP (bar) | Model MAP (bar)")
    print("  " + "-" * 75)

    for alt_ft in test_altitudes_ft:
        mvem.reset(idle=False)
        alt_m = FlightProfile.feet_to_meters(alt_ft)
        t_c, p_bar, rho = FlightProfile.get_isa_atmosphere(alt_m)
        flight = FlightState(
            time_s=10.0,
            altitude_ft=alt_ft,
            altitude_m=alt_m,
            throttle_pct=100.0,
            airspeed_mps=60.0,
            ambient_temp_c=t_c,
            ambient_pressure_bar=p_bar,
            air_density_kgpm3=rho,
            phase="CLIMB"
        )
        # Settle engine for 15 seconds to reach steady-state
        for _ in range(300):
            state = mvem.step(flight, dt_s=0.05)

        measured_powers.append(state.power_hp)
        measured_maps.append(state.manifold_pressure_bar)
        print(f"  {alt_ft:13.0f} | {expected_power_hp[len(measured_powers)-1]:17.1f} | {state.power_hp:16.1f} | {expected_map_bar[len(measured_maps)-1]:16.2f} | {state.manifold_pressure_bar:15.3f}")

    # Compute Calibration RMSE
    power_rmse = math.sqrt(np.mean([(m - e)**2 for m, e in zip(measured_powers, expected_power_hp)]))
    map_rmse = math.sqrt(np.mean([(m - e)**2 for m, e in zip(measured_maps, expected_map_bar)]))

    print(f"  -> Calibration Power RMSE: {power_rmse:.2f} HP (Fit Error: {(power_rmse/200.0)*100:.1f}%)")
    print(f"  -> Calibration MAP RMSE:   {map_rmse:.3f} bar")
    assert power_rmse < 15.0, f"Power RMSE too high: {power_rmse:.2f} HP"
    print("  -> PASSED: MVEM altitude derating accurately matches DRDO/VRDE 2.2L specifications.")

def test_sensor_model():
    print("\n[TEST 3] Testing Datasheet Sensor Noise, Thermal Lag & Sampling...")
    mvem = MeanValueEngineModel()
    mvem.reset(idle=False)
    sensors = SensorModel(seed=123)
    flight = FlightProfile().get_standard_mission_state(100.0, 3600.0)

    # 1. Warm up engine to steady state
    for _ in range(300):
        true_eng = mvem.step(flight, dt_s=0.05)
        meas = sensors.sample(true_eng, flight, dt_s=0.05)

    # 2. Collect 300 steady-state readings
    egt_residuals = []
    for _ in range(300):
        true_eng = mvem.step(flight, dt_s=0.05)
        meas = sensors.sample(true_eng, flight, dt_s=0.05)
        egt_residuals.append(meas.egt_c[0] - true_eng.egt_c[0])

    measured_sigma = float(np.std(egt_residuals))
    print(f"  -> Target EGT Sensor Noise Sigma: {sensors.sigma_egt:.2f} C | Measured Empirical Sigma: {measured_sigma:.2f} C")
    assert abs(measured_sigma - sensors.sigma_egt) < 0.8, "Sensor noise generator discrepancy"
    print("  -> PASSED: Sensor noise statistics verified against datasheet specification.")

def test_fault_injection():
    print("\n[TEST 4] Testing Fault Injection Dynamic Physics...")
    mvem = MeanValueEngineModel()
    sensors = SensorModel(seed=42)
    injector = FaultInjector(mvem, sensors)
    flight = FlightProfile().get_standard_mission_state(100.0, 3600.0)

    # 1. Test Oil Pressure Loss
    mvem.reset(idle=False)
    # Healthy baseline
    for _ in range(50):
        t_eng = mvem.step(flight, dt_s=0.05)
        m_eng = sensors.sample(t_eng, flight, dt_s=0.05)
    healthy_oil_p = m_eng.oil_pressure_bar

    # Inject oil loss
    injector.inject_fault(FaultConfig(FaultType.OIL_PRESSURE_LOSS, start_time_s=10.0, ramp_duration_s=5.0, severity=1.0))
    for t_step in range(200):
        curr_t = 10.0 + t_step * 0.05
        injector.update(curr_t)
        t_eng = mvem.step(flight, dt_s=0.05)
        m_eng = sensors.sample(t_eng, flight, dt_s=0.05)

    faulty_oil_p = m_eng.oil_pressure_bar
    print(f"  -> Oil Pressure: Healthy = {healthy_oil_p:.2f} bar | Injected Failure = {faulty_oil_p:.2f} bar")
    assert faulty_oil_p < 2.0, "Oil pressure did not drop to critical alarm level"

    # 2. Test Sensor-vs-Engine Fault Distinction (EGT Open Circuit)
    injector.clear_faults()
    injector.inject_fault(FaultConfig(FaultType.SENSOR_FAULT_EGT3, start_time_s=0.0))
    injector.update(1.0)
    t_eng = mvem.step(flight, dt_s=0.05)
    m_eng = sensors.sample(t_eng, flight, dt_s=0.05)

    print(f"  -> EGT Sensor Open-Circuit: Cyl 3 Sensor = {m_eng.egt_c[2]:.1f} C (Reads ambient) vs True Engine = {t_eng.egt_c[2]:.1f} C")
    assert m_eng.egt_c[2] < 50.0, "Sensor open circuit did not drop to ambient"
    assert t_eng.egt_c[2] > 400.0, "True engine EGT should remain at operating temperature"
    print("  -> PASSED: Sensor defect vs Engine fault decoupled successfully.")

if __name__ == "__main__":
    test_isa_atmosphere()
    test_vrde_altitude_derating()
    test_sensor_model()
    test_fault_injection()
    print("\n[TEST 5] Generating Full Datasets...")
    generate_all_datasets()
    print("\n========================================================")
    print("ALL STEP 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("========================================================")
