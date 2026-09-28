"""
5-State Digital Twin Architecture Model (SIH26054 Section 10)
Defines the state representation across the physical, sensor, estimated, virtual, and AI-health layers.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import json

class StateLevel(str, Enum):
    NOMINAL = "NOMINAL"       # Health >= 85%
    ADVISORY = "ADVISORY"     # Health 70-84%
    CAUTION = "CAUTION"       # Health 50-69%
    WARNING = "WARNING"       # Health < 50%
    CRITICAL = "CRITICAL"     # Emergency limit / immediate action needed

@dataclass
class SubsystemHealth:
    overall_health: float              # 0.0 to 100.0%
    cylinder_health: List[float]       # Cylinders 1, 2, 3, 4 [0-100%]
    oil_system_health: float           # 0-100%
    turbo_boost_health: float          # 0-100%
    vibration_health: float            # 0-100%
    electrical_health: float           # 0-100%
    cooling_system_health: float       # 0-100%
    status_level: StateLevel = StateLevel.NOMINAL

@dataclass
class AIHealthState:
    anomaly_detected: bool = False
    anomaly_score: float = 0.0         # Calibrated Anomaly Index (0.00 to 1.00)
    anomaly_raw_mse: float = 0.0       # Raw Autoencoder Reconstruction MSE
    anomaly_threshold: float = 1.0     # Calibrated k*sigma limit
    fault_class: str = "HEALTHY"
    fault_confidence_pct: float = 100.0# Temperature-calibrated softmax probability
    fault_severity: float = 0.0        # 0.0 to 1.0
    is_sensor_fault: bool = False      # True if sensor failure rather than engine failure
    rul_hours_mean: float = 500.0      # RUL point estimate
    rul_hours_min: float = 450.0       # RUL Lower bound (95% confidence interval)
    rul_hours_max: float = 550.0       # RUL Upper bound (95% confidence interval)
    top_contributing_channels: List[Dict[str, Any]] = field(default_factory=list) # SHAP top channels
    recommended_action: str = "Continue nominal mission profile."

@dataclass
class DigitalTwinState:
    """
    Synchronized 5-State Digital Twin snapshot.
    """
    timestamp_s: float
    # Operational Context
    altitude_ft: float
    ambient_temp_c: float
    throttle_pct: float
    airspeed_mps: float
    mission_phase: str
    # 1. Sensor State (Raw incoming telemetry)
    sensor_rpm: float
    sensor_map_bar: float
    sensor_oil_p_bar: float
    sensor_oil_t_c: float
    sensor_coolant_t_c: float
    sensor_fuel_flow_lph: float
    sensor_egt_c: List[float]
    sensor_cht_c: List[float]
    sensor_vib_rms_g: float
    sensor_bus_v: float
    # 2. Physics MVEM State (Expected healthy values for current altitude/throttle/OAT)
    mvem_expected_power_hp: float
    mvem_expected_rpm: float
    mvem_expected_map_bar: float
    mvem_expected_oil_p_bar: float
    mvem_expected_oil_t_c: float
    mvem_expected_egt_c: List[float]
    mvem_expected_cht_c: List[float]
    mvem_expected_fuel_lph: float
    mvem_expected_vib_rms_g: float
    # 3. EKF Estimated State (Filtered optimal true state)
    estimated_rpm: float
    estimated_map_bar: float
    estimated_oil_p_bar: float
    estimated_oil_t_c: float
    estimated_egt_c: List[float]
    estimated_cht_c: List[float]
    # 4. Normalized Residuals: (Sensor - MVEM) / Sigma
    residuals: Dict[str, float] = field(default_factory=dict)
    # 5. Health & Prognostics State
    health: SubsystemHealth = field(default_factory=lambda: SubsystemHealth(100.0, [100.0]*4, 100.0, 100.0, 100.0, 100.0, 100.0))
    ai_prognostics: AIHealthState = field(default_factory=AIHealthState)
    active_fault: str = "HEALTHY"
    provenance: str = "SIMULATED"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes digital twin state to Python dictionary with full field aliases."""
        d = asdict(self)
        d["health"]["status_level"] = self.health.status_level.value
        # Aliases for frontend compatibility
        d["sensor_oil_pressure_bar"] = self.sensor_oil_p_bar
        d["mvem_expected_oil_pressure_bar"] = self.mvem_expected_oil_p_bar
        d["sensor_manifold_pressure_bar"] = self.sensor_map_bar
        d["mvem_expected_manifold_pressure_bar"] = self.mvem_expected_map_bar
        d["sensor_vibration_rms_g"] = self.sensor_vib_rms_g
        d["mvem_expected_vibration_rms_g"] = self.mvem_expected_vib_rms_g
        return d

    def to_json(self) -> str:
        """Serializes digital twin state to JSON string."""
        return json.dumps(self.to_dict())
