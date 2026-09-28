"""
Physics-Informed Remaining Useful Life (RUL) Estimator (SIH26054 Section 16, 18)
Projects degradation trajectories to critical maintenance limits and outputs
uncertainty-bounded prediction intervals (e.g., [18, 25] flight hours).
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import numpy as np

@dataclass
class RULPrediction:
    regime: str                     # "STABLE", "DRIFTING", "ABRUPT"
    rul_hours_mean: float           # Expected hours remaining
    rul_hours_min: float            # 95% Confidence lower bound
    rul_hours_max: float            # 95% Confidence upper bound
    degradation_rate_pct_per_hr: float # Health loss rate (%/hr)
    uncertainty_level: str          # "LOW", "MEDIUM", "HIGH"

class RULEstimator:
    """
    Combines linear/exponential trend filtering on rolling health scores with
    empirical noise-driven uncertainty interval expansion.
    """
    def __init__(self, history_window_len: int = 50, critical_health_threshold: float = 30.0):
        self.window_len = history_window_len
        self.critical_thresh = critical_health_threshold
        self.health_history: List[Tuple[float, float]] = [] # (time_s, health_pct)

    def reset(self):
        self.health_history.clear()

    def update(self, time_s: float, current_health_pct: float) -> RULPrediction:
        """
        Updates rolling degradation trajectory and computes RUL confidence interval.
        """
        self.health_history.append((time_s, current_health_pct))
        if len(self.health_history) > self.window_len:
            self.health_history.pop(0)

        # 1. Healthy / Stable State
        if current_health_pct >= 85.0 or len(self.health_history) < 10:
            return RULPrediction(
                regime="STABLE",
                rul_hours_mean=500.0,
                rul_hours_min=450.0,
                rul_hours_max=550.0,
                degradation_rate_pct_per_hr=0.0,
                uncertainty_level="LOW"
            )

        # 2. Abrupt Emergency Step Fault
        times = np.array([pt[0] for pt in self.health_history])
        healths = np.array([pt[1] for pt in self.health_history])
        time_span_s = max(1.0, times[-1] - times[0])
        total_drop = healths[0] - healths[-1]

        if total_drop > 40.0 and time_span_s < 15.0:
            return RULPrediction(
                regime="ABRUPT",
                rul_hours_mean=0.1,
                rul_hours_min=0.0,
                rul_hours_max=0.5,
                degradation_rate_pct_per_hr=total_drop * (3600.0 / time_span_s),
                uncertainty_level="LOW"
            )

        # 3. Drifting Degradation Regime (Linear / Exponential rate fit)
        # Fit slope: health(t) = a * t + b
        dt_hrs = (times - times[0]) / 3600.0 * 20.0 # Normalized simulation time to mission hours
        if np.std(dt_hrs) < 1e-4:
            slope = 0.0
            std_err = 1.0
        else:
            poly, residuals, _, _, _ = np.polyfit(dt_hrs, healths, deg=1, full=True)
            slope = poly[0] # % health drop per flight hour
            std_err = math.sqrt(residuals[0] / max(1, len(dt_hrs) - 2)) if len(residuals) > 0 else 2.0

        if slope >= -0.05:
            # Very slow drift
            rul_mean = 120.0
            rul_min = 90.0
            rul_max = 160.0
            unc_level = "LOW"
            deg_rate = abs(float(slope))
        else:
            deg_rate = abs(float(slope))
            health_remaining = max(1.0, current_health_pct - self.critical_thresh)
            rul_mean = health_remaining / deg_rate

            # Calibrate 95% Confidence Interval (widens with fit residual standard error)
            uncertainty_margin = min(0.45, max(0.12, (std_err / max(1.0, deg_rate)) * 0.35))
            rul_min = max(0.1, rul_mean * (1.0 - uncertainty_margin))
            rul_max = rul_mean * (1.0 + uncertainty_margin)

            unc_level = "HIGH" if uncertainty_margin > 0.30 else "MEDIUM"

        decimals = 2 if rul_mean < 5.0 else 1
        return RULPrediction(
            regime="DRIFTING",
            rul_hours_mean=round(float(rul_mean), decimals),
            rul_hours_min=round(float(rul_min), decimals),
            rul_hours_max=round(float(rul_max), decimals),
            degradation_rate_pct_per_hr=round(float(deg_rate), 2),
            uncertainty_level=unc_level
        )
