"""
Physics-Anchored Health Index Engine (SIH26054 Section 11.2)
Calculates continuous 0–100% health scores across engine subsystems based on
normalized residuals against the MVEM expectation:
    Health_sub = 100 * exp(-alpha * |(y_meas - y_mvem) / sigma|)
"""
from typing import Dict, List, Tuple, Any
import math
import numpy as np

from .twin_state import SubsystemHealth, StateLevel

class HealthIndexEngine:
    """
    Subsystem Health Calculation Engine.
    Employs calibrated exponential decay on normalized residuals with safety-critical weighting.
    """
    def __init__(self):
        # Baseline Sensor Sigmas (Calibrated from datasheet model)
        self.sigma = {
            "egt": 3.5,            # Celsius
            "cht": 1.8,            # Celsius
            "oil_pressure": 0.06,  # bar
            "oil_temp": 0.9,       # Celsius
            "manifold_pressure": 0.02, # bar
            "rpm": 8.0,            # RPM
            "fuel_flow": 0.35,     # LPH
            "vibration_rms": 0.08, # g
            "bus_voltage": 0.12    # V
        }

        # Sensitivity Factors (alpha) tuned so fault-onset thresholds map to 50-70% (Caution band)
        self.alpha = {
            "egt": 0.10,
            "cht": 0.15,
            "oil_pressure": 0.18,
            "oil_temp": 0.12,
            "manifold_pressure": 0.14,
            "rpm": 0.10,
            "fuel_flow": 0.10,
            "vibration_rms": 0.20,
            "bus_voltage": 0.12
        }

        # Subsystem Weights for Composite Health Score
        self.weights = {
            "oil_system": 0.25,
            "cylinders": 0.30,
            "turbo_boost": 0.15,
            "vibration": 0.15,
            "electrical": 0.08,
            "cooling": 0.07
        }

    def compute_residuals(
        self,
        sensor_data: Dict[str, Any],
        mvem_expected: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculates normalized residuals: r_norm = (y_sensor - y_mvem) / sigma
        """
        residuals = {}
        # Scalar channels
        for ch in ["oil_pressure", "oil_temp", "manifold_pressure", "rpm", "fuel_flow", "vibration_rms", "bus_voltage"]:
            if ch in sensor_data and ch in mvem_expected:
                diff = sensor_data[ch] - mvem_expected[ch]
                sig = self.sigma.get(ch, 1.0)
                residuals[ch] = round(diff / sig, 3)

        # Multi-cylinder channels
        for i in range(4):
            # EGT
            s_egt = sensor_data["egt_c"][i]
            m_egt = mvem_expected["egt_c"][i]
            residuals[f"egt_cyl_{i+1}"] = round((s_egt - m_egt) / self.sigma["egt"], 3)
            # CHT
            s_cht = sensor_data["cht_c"][i]
            m_cht = mvem_expected["cht_c"][i]
            residuals[f"cht_cyl_{i+1}"] = round((s_cht - m_cht) / self.sigma["cht"], 3)

        return residuals

    def _calc_subscore(self, residual_norm: float, channel_key: str) -> float:
        """
        Computes exponential health sub-score 0-100%.
        Applies a 2.5-sigma noise deadband so normal Gaussian measurement noise
        does not degrade healthy nominal baseline.
        """
        eff_norm = max(0.0, abs(residual_norm) - 2.5)
        a = self.alpha.get(channel_key, 0.15)
        score = 100.0 * math.exp(-a * eff_norm)
        return max(0.0, min(100.0, round(score, 1)))

    def evaluate_health(self, residuals: Dict[str, float]) -> SubsystemHealth:
        """
        Evaluates subsystem and composite engine health from normalized residuals.
        """
        # 1. Oil System Health (fused pressure & temperature)
        oil_p_score = self._calc_subscore(residuals.get("oil_pressure", 0.0), "oil_pressure")
        oil_t_score = self._calc_subscore(residuals.get("oil_temp", 0.0), "oil_temp")
        oil_health = min(oil_p_score, oil_t_score * 1.05)

        # 2. Per-Cylinder Health (Cyl 1 to 4)
        cyl_scores = []
        for i in range(4):
            egt_score = self._calc_subscore(residuals.get(f"egt_cyl_{i+1}", 0.0), "egt")
            cht_score = self._calc_subscore(residuals.get(f"cht_cyl_{i+1}", 0.0), "cht")
            cyl_scores.append(min(egt_score, cht_score))

        # 3. Turbocharger / Manifold Boost Health
        turbo_health = self._calc_subscore(residuals.get("manifold_pressure", 0.0), "manifold_pressure")

        # 4. Mechanical Vibration Health
        vib_health = self._calc_subscore(residuals.get("vibration_rms", 0.0), "vibration_rms")

        # 5. Electrical Bus Health
        elec_health = self._calc_subscore(residuals.get("bus_voltage", 0.0), "bus_voltage")

        # 6. Cooling System Health (driven by CHT average)
        cooling_health = float(np.mean([self._calc_subscore(residuals.get(f"cht_cyl_{i+1}", 0.0), "cht") for i in range(4)]))

        # Composite Engine Health Score (Weighted sum with conservative safety bottleneck)
        avg_cyl_health = float(np.mean(cyl_scores))
        min_cyl_health = float(np.min(cyl_scores))

        weighted_health = (
            oil_health * self.weights["oil_system"] +
            avg_cyl_health * self.weights["cylinders"] +
            turbo_health * self.weights["turbo_boost"] +
            vib_health * self.weights["vibration"] +
            elec_health * self.weights["electrical"] +
            cooling_health * self.weights["cooling"]
        )

        # Safety-Critical Bottleneck Rule: If a single safety subsystem fails critically (e.g. oil loss),
        # overall health cannot pretend to be 90% healthy.
        overall_health = min(weighted_health, min(oil_health, min_cyl_health) * 1.15)
        overall_health = max(0.0, min(100.0, round(overall_health, 1)))

        # Determine Operational Status Level
        if overall_health >= 85.0:
            status = StateLevel.NOMINAL
        elif overall_health >= 70.0:
            status = StateLevel.ADVISORY
        elif overall_health >= 50.0:
            status = StateLevel.CAUTION
        elif overall_health >= 25.0:
            status = StateLevel.WARNING
        else:
            status = StateLevel.CRITICAL

        return SubsystemHealth(
            overall_health=overall_health,
            cylinder_health=[round(c, 1) for c in cyl_scores],
            oil_system_health=round(oil_health, 1),
            turbo_boost_health=round(turbo_health, 1),
            vibration_health=round(vib_health, 1),
            electrical_health=round(elec_health, 1),
            cooling_system_health=round(cooling_health, 1),
            status_level=status
        )
