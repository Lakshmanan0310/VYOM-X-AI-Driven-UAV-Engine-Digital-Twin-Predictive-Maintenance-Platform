"""
Datasheet-Grounded Sensor Model for Aero Piston Engine Instrumentation.
Models physical sensor behavior:
- First-order lag (thermal time constants for thermocouples & RTDs)
- Realistic Gaussian measurement noise
- Finite ADC bit-depth & quantization step sizes
- Multi-rate asynchronous sampling
- Real-world failure modes (dropout, freezing, drift, rail clipping)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import numpy as np

from .mvem import EngineState
from .flight_profile import FlightState

@dataclass
class SensorReadings:
    time_s: float
    # Flight Environment
    altitude_ft: float
    ambient_temp_c: float
    ambient_pressure_bar: float
    airspeed_mps: float
    throttle_pct: float
    # Primary Engine Channels
    rpm: float
    manifold_pressure_bar: float
    manifold_temp_c: float
    fuel_flow_lph: float
    oil_pressure_bar: float
    oil_temp_c: float
    coolant_temp_c: float
    # Multi-cylinder Exhaust & Cylinder Head Temperatures
    egt_c: List[float] # Cylinders 1, 2, 3, 4
    cht_c: List[float] # Cylinders 1, 2, 3, 4
    # Vibration & Electrical
    vibration_rms_g: float
    vibration_x_g: float
    vibration_y_g: float
    vibration_z_g: float
    bus_voltage_v: float
    # Channel Integrity Flags (for sensor-fault identification)
    channel_status: Dict[str, str] = field(default_factory=dict)
    provenance: str = "SIMULATED"

class SensorModel:
    """
    Simulates physical instrumentation with true sensor characteristics.
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

        # Lag States (Internal filter registers for steady-state cruise)
        self._lag_egt = [810.0, 810.0, 810.0, 810.0]
        self._lag_cht = [175.0, 175.0, 175.0, 175.0]
        self._lag_oil_p = 4.2
        self._lag_oil_t = 88.0
        self._lag_coolant_t = 86.0
        self._lag_map = 1.013
        self._lag_rpm = 1400.0
        self._lag_fuel_flow = 10.0
        self._lag_bus_v = 28.0

        # Physical Sensor Parameters (From aviation sensor datasheets)
        # Type-K Thermocouple (EGT): tau ~ 2.0 s, noise ~ 3.5 C, quant 0.5 C
        self.tau_egt = 2.0
        self.sigma_egt = 3.5
        self.quant_egt = 0.5

        # Type-J/Ring Thermocouple (CHT): tau ~ 3.5 s, noise ~ 1.8 C, quant 0.2 C
        self.tau_cht = 3.5
        self.sigma_cht = 1.8
        self.quant_cht = 0.2

        # Piezoresistive Pressure (Oil P): tau ~ 0.3 s, noise ~ 0.06 bar, quant 0.02 bar
        self.tau_oil_p = 0.3
        self.sigma_oil_p = 0.06
        self.quant_oil_p = 0.02

        # Sump RTD (Oil T): tau ~ 4.0 s, noise ~ 0.9 C, quant 0.1 C
        self.tau_oil_t = 4.0
        self.sigma_oil_t = 0.9
        self.quant_oil_t = 0.1

        # MAP Sensor (Piezoresistive): tau ~ 0.08 s, noise ~ 0.02 bar, quant 0.01 bar
        self.tau_map = 0.08
        self.sigma_map = 0.02
        self.quant_map = 0.01

        # Optical/Hall RPM Tachometer: tau ~ 0.05 s, noise ~ 6.0 RPM, quant 1.0 RPM
        self.tau_rpm = 0.05
        self.sigma_rpm = 6.0
        self.quant_rpm = 1.0

        # Turbine Fuel Flow Meter: tau ~ 0.8 s, noise ~ 0.35 LPH, quant 0.05 LPH
        self.tau_fuel = 0.8
        self.sigma_fuel = 0.35
        self.quant_fuel = 0.05

        # IEPE Triaxial Accelerometer: tau ~ 0.02 s, noise ~ 0.08 g, quant 0.01 g
        self.sigma_vib = 0.08
        self.quant_vib = 0.01

        # Bus Voltage ADC: tau ~ 0.1 s, noise ~ 0.12 V, quant 0.05 V
        self.tau_bus_v = 0.1
        self.sigma_bus_v = 0.12
        self.quant_bus_v = 0.05

        # Active Sensor Fault Overrides (for testing & sensor-vs-engine fault discrimination)
        self.fault_overrides: Dict[str, Dict[str, Any]] = {}

    def set_sensor_fault(self, channel: str, fault_type: str, value: Optional[float] = None):
        """
        Injects a physical sensor defect (e.g. 'FROZEN', 'BIAS', 'DROPOUT', 'OPEN_CIRCUIT', 'NOISY').
        """
        self.fault_overrides[channel] = {"type": fault_type, "value": value}

    def clear_sensor_faults(self):
        self.fault_overrides.clear()

    def _quantize(self, val: float, step: float) -> float:
        return round(round(val / step) * step, 4)

    def sample(self, true_engine: EngineState, flight: FlightState, dt_s: float = 0.05) -> SensorReadings:
        """
        Samples the continuous ground-truth physical engine state through the realistic sensor pipeline.
        """
        status_flags = {}

        # 1. Update First-Order Lags
        # x_lag += (x_true - x_lag) * (1 - exp(-dt / tau))
        def update_lag(curr_lag, target, tau):
            alpha = 1.0 - math.exp(-dt_s / max(1e-4, tau))
            return curr_lag + (target - curr_lag) * alpha

        self._lag_rpm = update_lag(self._lag_rpm, true_engine.rpm, self.tau_rpm)
        self._lag_map = update_lag(self._lag_map, true_engine.manifold_pressure_bar, self.tau_map)
        self._lag_oil_p = update_lag(self._lag_oil_p, true_engine.oil_pressure_bar, self.tau_oil_p)
        self._lag_oil_t = update_lag(self._lag_oil_t, true_engine.oil_temp_c, self.tau_oil_t)
        self._lag_coolant_t = update_lag(self._lag_coolant_t, true_engine.coolant_temp_c, 3.0)
        self._lag_fuel_flow = update_lag(self._lag_fuel_flow, true_engine.fuel_flow_lph, self.tau_fuel)
        self._lag_bus_v = update_lag(self._lag_bus_v, true_engine.bus_voltage_v, self.tau_bus_v)

        for i in range(4):
            self._lag_egt[i] = update_lag(self._lag_egt[i], true_engine.egt_c[i], self.tau_egt)
            self._lag_cht[i] = update_lag(self._lag_cht[i], true_engine.cht_c[i], self.tau_cht)

        # 2. Add Measurement Noise & Quantization
        rpm_meas = self._quantize(self._lag_rpm + self.rng.normal(0, self.sigma_rpm), self.quant_rpm)
        map_meas = self._quantize(max(0.0, self._lag_map + self.rng.normal(0, self.sigma_map)), self.quant_map)
        oil_p_meas = self._quantize(max(0.0, self._lag_oil_p + self.rng.normal(0, self.sigma_oil_p)), self.quant_oil_p)
        oil_t_meas = self._quantize(self._lag_oil_t + self.rng.normal(0, self.sigma_oil_t), self.quant_oil_t)
        coolant_t_meas = self._quantize(self._lag_coolant_t + self.rng.normal(0, 0.8), 0.1)
        fuel_meas = self._quantize(max(0.0, self._lag_fuel_flow + self.rng.normal(0, self.sigma_fuel)), self.quant_fuel)
        bus_v_meas = self._quantize(max(0.0, self._lag_bus_v + self.rng.normal(0, self.sigma_bus_v)), self.quant_bus_v)

        egt_meas = []
        cht_meas = []
        for i in range(4):
            e_val = self._quantize(self._lag_egt[i] + self.rng.normal(0, self.sigma_egt), self.quant_egt)
            c_val = self._quantize(self._lag_cht[i] + self.rng.normal(0, self.sigma_cht), self.quant_cht)
            egt_meas.append(e_val)
            cht_meas.append(c_val)

        vib_rms_meas = self._quantize(max(0.0, true_engine.vibration_rms_g + self.rng.normal(0, self.sigma_vib)), self.quant_vib)
        vib_x_meas = self._quantize(true_engine.vibration_x_g + self.rng.normal(0, self.sigma_vib), self.quant_vib)
        vib_y_meas = self._quantize(true_engine.vibration_y_g + self.rng.normal(0, self.sigma_vib), self.quant_vib)
        vib_z_meas = self._quantize(true_engine.vibration_z_g + self.rng.normal(0, self.sigma_vib), self.quant_vib)

        # 3. Apply Active Sensor Faults / Defects if Triggered
        for ch, fault in self.fault_overrides.items():
            f_type = fault["type"]
            if ch == "egt_3":
                if f_type == "OPEN_CIRCUIT":
                    egt_meas[2] = round(flight.ambient_temp_c + self.rng.normal(0, 1.0), 1) # Reads ambient
                    status_flags["egt_3"] = "SENSOR_OPEN_CIRCUIT"
                elif f_type == "FROZEN":
                    egt_meas[2] = 520.0
                    status_flags["egt_3"] = "SENSOR_FROZEN"
                elif f_type == "BIAS":
                    egt_meas[2] += fault.get("value", 120.0)
                    status_flags["egt_3"] = "SENSOR_BIAS"
            elif ch == "oil_pressure":
                if f_type == "DROPOUT":
                    oil_p_meas = 0.0
                    status_flags["oil_pressure"] = "SENSOR_DROPOUT"
                elif f_type == "NOISY":
                    oil_p_meas += self.rng.normal(0, 1.2)
                    status_flags["oil_pressure"] = "SENSOR_EXCESSIVE_NOISE"
            elif ch == "rpm":
                if f_type == "GLITCH":
                    rpm_meas = 0.0 if self.rng.random() < 0.25 else rpm_meas
                    status_flags["rpm"] = "SENSOR_INTERMITTENT_GLITCH"

        return SensorReadings(
            time_s=round(flight.time_s, 2),
            altitude_ft=round(flight.altitude_ft, 1),
            ambient_temp_c=round(flight.ambient_temp_c, 2),
            ambient_pressure_bar=round(flight.ambient_pressure_bar, 4),
            airspeed_mps=round(flight.airspeed_mps, 2),
            throttle_pct=round(flight.throttle_pct, 2),
            rpm=round(rpm_meas, 1),
            manifold_pressure_bar=round(map_meas, 3),
            manifold_temp_c=round(self._lag_map, 1),
            fuel_flow_lph=round(fuel_meas, 2),
            oil_pressure_bar=round(oil_p_meas, 2),
            oil_temp_c=round(oil_t_meas, 1),
            coolant_temp_c=round(coolant_t_meas, 1),
            egt_c=[round(t, 1) for t in egt_meas],
            cht_c=[round(t, 1) for t in cht_meas],
            vibration_rms_g=round(vib_rms_meas, 3),
            vibration_x_g=round(vib_x_meas, 3),
            vibration_y_g=round(vib_y_meas, 3),
            vibration_z_g=round(vib_z_meas, 3),
            bus_voltage_v=round(bus_v_meas, 2),
            channel_status=status_flags,
            provenance="SIMULATED"
        )
