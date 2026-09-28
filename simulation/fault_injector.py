"""
Parametric Fault Injection Engine for Aero Piston Engine Digital Twin.
Implements the 8 core aero piston engine failure modes + sensor failure modes
defined in SIH26054 Proposal Section 26.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

from .mvem import MeanValueEngineModel
from .sensors import SensorModel

class FaultType(str, Enum):
    HEALTHY = "HEALTHY"
    LEAN_MIXTURE_CYL3 = "LEAN_MIXTURE_CYL3"
    RICH_MIXTURE_CYL1 = "RICH_MIXTURE_CYL1"
    COOLING_DEGRADATION_CYL2 = "COOLING_DEGRADATION_CYL2"
    OIL_PRESSURE_LOSS = "OIL_PRESSURE_LOSS"
    TURBO_BOOST_DEFICIENCY = "TURBO_BOOST_DEFICIENCY"
    CYLINDER_MISFIRE_TIMING = "CYLINDER_MISFIRE_TIMING"
    BEARING_WEAR_VIBRATION = "BEARING_WEAR_VIBRATION"
    SENSOR_FAULT_EGT3 = "SENSOR_FAULT_EGT3"
    ELECTRICAL_VOLTAGE_SAG = "ELECTRICAL_VOLTAGE_SAG"

@dataclass
class FaultConfig:
    fault_type: FaultType
    start_time_s: float
    ramp_duration_s: float = 10.0   # Time over which fault evolves (0 for abrupt step)
    severity: float = 1.0           # 0.0 to 1.0 fault severity multiplier
    target_cylinder: int = 2        # 0-indexed cylinder (e.g. 2 = Cyl 3)

class FaultInjector:
    """
    Manages active faults and dynamically modifies MVEM physics parameters
    and sensor properties as a function of simulation time.
    """
    def __init__(self, mvem: MeanValueEngineModel, sensors: SensorModel):
        self.mvem = mvem
        self.sensors = sensors
        self.active_fault: Optional[FaultConfig] = None
        self.current_severity: float = 0.0

    def inject_fault(self, fault: FaultConfig):
        """Activates a specific fault schedule."""
        self.active_fault = fault
        self.current_severity = 0.0

    def clear_faults(self):
        """Resets all injected engine and sensor faults to healthy baseline."""
        self.active_fault = None
        self.current_severity = 0.0
        self.mvem.cylinder_fuel_trim = [1.0, 1.0, 1.0, 1.0]
        self.mvem.cylinder_cooling_trim = [1.0, 1.0, 1.0, 1.0]
        self.mvem.oil_pump_health = 1.0
        self.mvem.turbo_efficiency = 1.0
        self.mvem.bearing_wear_factor = 1.0
        self.mvem.timing_jitter_deg = 0.0
        self.mvem.alternator_health = 1.0
        self.sensors.clear_sensor_faults()

    def update(self, current_time_s: float):
        """
        Evaluates active fault progression and adjusts MVEM & Sensor model states.
        """
        if not self.active_fault or self.active_fault.fault_type == FaultType.HEALTHY:
            self.clear_faults()
            return

        cfg = self.active_fault
        if current_time_s < cfg.start_time_s:
            self.current_severity = 0.0
            return

        # Compute ramp progress (0.0 to cfg.severity)
        if cfg.ramp_duration_s <= 0.0:
            sev = cfg.severity
        else:
            elapsed = current_time_s - cfg.start_time_s
            progress = min(1.0, max(0.0, elapsed / cfg.ramp_duration_s))
            sev = cfg.severity * progress

        self.current_severity = sev

        # Apply specific fault physics modifications
        f_type = cfg.fault_type

        if f_type == FaultType.LEAN_MIXTURE_CYL3:
            # Lean mixture on Cylinder 3 (reduces fuel trim -> raises EGT)
            cyl = cfg.target_cylinder
            self.mvem.cylinder_fuel_trim[cyl] = 1.0 - (0.28 * sev)

        elif f_type == FaultType.RICH_MIXTURE_CYL1:
            # Rich mixture on Cylinder 1 (increases fuel trim -> drops EGT, higher BSFC)
            self.mvem.cylinder_fuel_trim[0] = 1.0 + (0.35 * sev)

        elif f_type == FaultType.COOLING_DEGRADATION_CYL2:
            # Baffle/Airflow restriction on Cylinder 2 (reduces cooling -> CHT climbs)
            self.mvem.cylinder_cooling_trim[1] = 1.0 - (0.60 * sev)

        elif f_type == FaultType.OIL_PRESSURE_LOSS:
            # Oil pressure drop (pump relief valve leak / gallery loss)
            self.mvem.oil_pump_health = 1.0 - (0.70 * sev)

        elif f_type == FaultType.TURBO_BOOST_DEFICIENCY:
            # Turbocharger wastegate stuck or compressor blade erosion
            self.mvem.turbo_efficiency = 1.0 - (0.45 * sev)

        elif f_type == FaultType.CYLINDER_MISFIRE_TIMING:
            # Injection/Ignition timing flutter / valve seating issue
            self.mvem.timing_jitter_deg = 9.0 * sev

        elif f_type == FaultType.BEARING_WEAR_VIBRATION:
            # Mechanical bearing wear (vibration order amplification + increased friction)
            self.mvem.bearing_wear_factor = 1.0 + (1.8 * sev)

        elif f_type == FaultType.SENSOR_FAULT_EGT3:
            # Sensor thermocouple probe disconnection (reads ambient, engine itself is healthy)
            self.sensors.set_sensor_fault("egt_3", "OPEN_CIRCUIT")

        elif f_type == FaultType.ELECTRICAL_VOLTAGE_SAG:
            # Alternator regulator degradation
            self.mvem.alternator_health = 1.0 - (0.32 * sev)
