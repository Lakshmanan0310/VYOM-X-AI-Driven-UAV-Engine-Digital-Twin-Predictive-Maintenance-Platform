"""
Structured Evidence-Carrying Operator Alert Generator (SIH26054 Section 17)
Formats AI diagnostics, physics evidence, RUL intervals, and sensor integrity into
standardized military aviation decision-support alerts.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import json

@dataclass
class OperatorAlert:
    alert_id: str
    timestamp_s: float
    level: str                  # "ADVISORY", "CAUTION", "WARNING", "CRITICAL"
    subsystem: str
    headline: str
    diagnosed_fault: str
    confidence_pct: float
    sensor_fault_check: str     # "PASSED (ENGINE FAULT CONFIRMED)" or "SENSOR PROBE DEFECT"
    evidence_items: List[str]
    rul_forecast_text: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# Pre-defined domain operational action rules
RECOMMENDED_ACTIONS = {
    "HEALTHY": "Continue nominal mission profile. All engine subsystems healthy.",
    "LEAN_MIXTURE_CYL3": "Reduce throttle to 65% max continuous. Lean mixture detected on Cylinder 3. Monitor CHT3 and schedule injector inspection post-flight.",
    "RICH_MIXTURE_CYL1": "Monitor fuel consumption (+18% above nominal). Inefficient rich burn on Cylinder 1. Recalibrate mixture map at next overhaul.",
    "COOLING_DEGRADATION_CYL2": "CAUTION: Cylinder 2 CHT thermal margin decaying. Increase airspeed by 10 kts to enhance ram-air cooling; if CHT exceeds 220 C, initiate mission abort.",
    "OIL_PRESSURE_LOSS": "CRITICAL EMERGENCY: Immediate oil pressure decay detected. Reduce power to minimum loiter setting and initiate immediate Return-To-Base (RTB).",
    "TURBO_BOOST_DEFICIENCY": "Turbo boost deficiency active. Engine derated at altitude. Limit ceiling to 15,000 ft and abort high-altitude loiter.",
    "CYLINDER_MISFIRE_TIMING": "Crankshaft timing jitter / misfire detected. Restrict rapid throttle transients; inspect ignition/injector timing upon landing.",
    "BEARING_WEAR_VIBRATION": "Mechanical bearing degradation detected (2X order vibration). Inspect crankshaft bearings within 15–25 flight hours.",
    "SENSOR_FAULT_EGT3": "SENSOR DEFECT: Thermocouple on Cylinder 3 failed open. Engine thermodynamics verified normal via CHT and vibration. Replace sensor post-mission; mission may proceed.",
    "ELECTRICAL_VOLTAGE_SAG": "Avionics bus voltage sag under load. Shed non-essential payload radar/sensors to protect flight-critical avionics."
}

class AlertGenerator:
    """
    Constructs explainable operator alerts from Digital Twin diagnostic outputs.
    """
    @staticmethod
    def generate(
        timestamp_s: float,
        fault_class: str,
        confidence_pct: float,
        severity: float,
        is_sensor_fault: bool,
        sensor_check_explanation: str,
        shap_explanations: List[Dict[str, Any]],
        rul_prediction: Any,
        overall_health: float
    ) -> Optional[OperatorAlert]:
        """
        Builds structured operator alert if anomaly or degraded state is active.
        """
        if fault_class == "HEALTHY" and overall_health >= 85.0:
            return None

        # Determine Alert Level
        if overall_health < 40.0 or fault_class in ["OIL_PRESSURE_LOSS"]:
            level = "CRITICAL"
        elif overall_health < 65.0 or severity > 0.6:
            level = "WARNING"
        elif overall_health < 80.0 or is_sensor_fault:
            level = "CAUTION"
        else:
            level = "ADVISORY"

        # Determine target subsystem
        if "CYL" in fault_class or "MIXTURE" in fault_class:
            subsystem = "Cylinders & Combustion"
        elif "OIL" in fault_class:
            subsystem = "Lubrication Subsystem"
        elif "TURBO" in fault_class:
            subsystem = "Turbocharger & Intake"
        elif "VIBRATION" in fault_class or "BEARING" in fault_class:
            subsystem = "Mechanical & Rotating Assembly"
        elif "SENSOR" in fault_class:
            subsystem = "Instrumentation & Sensors"
        elif "ELECTRICAL" in fault_class:
            subsystem = "Electrical & Power Generation"
        else:
            subsystem = "Engine Core"

        # Build evidence items
        evidence_lines = []
        for exp in shap_explanations:
            line = f"{exp['display_name']}: {exp['deviation_text']} (Attribution: {exp['importance_pct']}%)"
            evidence_lines.append(line)

        if not evidence_lines:
            evidence_lines.append("Subsystem residual variance exceeded statistical tolerance threshold.")

        # RUL Forecast text
        if is_sensor_fault:
            rul_text = "N/A (Sensor replacement required; physical engine life unaffected)."
        elif hasattr(rul_prediction, "regime"):
            if rul_prediction.regime == "STABLE":
                rul_text = f"Nominal operation. Scheduled overhaul in ~{rul_prediction.rul_hours_mean:.0f} flight hours."
            elif rul_prediction.regime == "ABRUPT":
                rul_text = "EMERGENCY: Immediate failure threshold reached (<0.5 flight hours)."
            else:
                rul_text = f"Action Window: {rul_prediction.rul_hours_min:.1f} to {rul_prediction.rul_hours_max:.1f} flight hours (Confidence: 95%)."
        else:
            rul_text = "Monitoring degradation trajectory."

        sensor_status = "SENSOR PROBE DEFECT (ENGINE HEALTHY)" if is_sensor_fault else "PASSED (ENGINE FAULT CONFIRMED)"
        action = RECOMMENDED_ACTIONS.get(fault_class, "Inspect indicated engine subsystem.")

        alert_id = f"ALT-{int(timestamp_s * 10):06d}"
        headline = f"{level}: {fault_class.replace('_', ' ')} (Health: {overall_health:.0f}%)"

        return OperatorAlert(
            alert_id=alert_id,
            timestamp_s=round(timestamp_s, 2),
            level=level,
            subsystem=subsystem,
            headline=headline,
            diagnosed_fault=fault_class,
            confidence_pct=round(confidence_pct, 1),
            sensor_fault_check=sensor_status,
            evidence_items=evidence_lines,
            rul_forecast_text=rul_text,
            recommended_action=action
        )
