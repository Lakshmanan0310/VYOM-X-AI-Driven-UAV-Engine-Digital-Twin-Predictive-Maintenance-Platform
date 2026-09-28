"""
Extended Kalman Filter (EKF) State Estimator for Aero Piston Engine Digital Twin.
Fuses dynamic non-linear MVEM physics predictions with noisy multi-channel telemetry.

State Vector (dim=12):
x = [RPM, MAP, P_oil, T_oil, CHT1, CHT2, CHT3, CHT4, EGT1, EGT2, EGT3, EGT4]^T
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from simulation.mvem import MeanValueEngineModel, EngineState
from simulation.flight_profile import FlightState
from simulation.sensors import SensorReadings

class ExtendedKalmanFilter:
    """
    12-State EKF for Aero Engine State Estimation & Sensor Smoothing.
    """
    def __init__(self):
        self.state_dim = 12
        self.meas_dim = 12

        # State Vector: [RPM, MAP, Oil_P, Oil_T, CHT1..4, EGT1..4]
        self.x = np.array([
            1400.0, 1.013, 4.5, 70.0,
            85.0, 85.0, 85.0, 85.0,
            450.0, 450.0, 450.0, 450.0
        ], dtype=np.float64)

        # State Covariance Matrix P
        self.P = np.eye(self.state_dim, dtype=np.float64) * 5.0

        # Process Noise Covariance Q (Uncertainty in physics model)
        self.Q = np.diag([
            15.0,    # RPM
            0.02,    # MAP
            0.05,    # Oil P
            0.50,    # Oil T
            0.80, 0.80, 0.80, 0.80, # CHT 1-4
            2.50, 2.50, 2.50, 2.50  # EGT 1-4
        ]) ** 2

        # Measurement Noise Covariance R (Datasheet sensor noise)
        self.R = np.diag([
            8.0,     # RPM
            0.02,    # MAP
            0.06,    # Oil P
            0.90,    # Oil T
            1.80, 1.80, 1.80, 1.80, # CHT 1-4
            3.50, 3.50, 3.50, 3.50  # EGT 1-4
        ]) ** 2

        # Measurement Matrix H (Direct observation of all 12 channels)
        self.H = np.eye(self.state_dim, dtype=np.float64)

    def predict(self, mvem_prediction: EngineState, dt_s: float = 0.05):
        """
        EKF Time Update (Prediction Step) using non-linear MVEM state dynamics.
        """
        # Physics state vector from MVEM
        x_mvem = np.array([
            mvem_prediction.rpm,
            mvem_prediction.manifold_pressure_bar,
            mvem_prediction.oil_pressure_bar,
            mvem_prediction.oil_temp_c,
            mvem_prediction.cht_c[0],
            mvem_prediction.cht_c[1],
            mvem_prediction.cht_c[2],
            mvem_prediction.cht_c[3],
            mvem_prediction.egt_c[0],
            mvem_prediction.egt_c[1],
            mvem_prediction.egt_c[2],
            mvem_prediction.egt_c[3],
        ], dtype=np.float64)

        # Non-linear state propagation
        self.x = 0.70 * self.x + 0.30 * x_mvem

        # Jacobian F_k approximated as identity + linear drift
        F = np.eye(self.state_dim)
        self.P = F @ self.P @ F.T + self.Q * dt_s

    def update(self, sensor_readings: SensorReadings):
        """
        EKF Measurement Update (Correction Step) fusing sensor observation vector z.
        """
        z = np.array([
            sensor_readings.rpm,
            sensor_readings.manifold_pressure_bar,
            sensor_readings.oil_pressure_bar,
            sensor_readings.oil_temp_c,
            sensor_readings.cht_c[0],
            sensor_readings.cht_c[1],
            sensor_readings.cht_c[2],
            sensor_readings.cht_c[3],
            sensor_readings.egt_c[0],
            sensor_readings.egt_c[1],
            sensor_readings.egt_c[2],
            sensor_readings.egt_c[3],
        ], dtype=np.float64)

        # Innovation (Residual) y = z - H*x
        y = z - (self.H @ self.x)

        # Innovation Covariance S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain K = P*H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated State Estimate
        self.x = self.x + K @ y

        # Updated Covariance: P = (I - K*H)*P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def get_estimated_state(self) -> Dict[str, Any]:
        """Returns smoothed state dictionary."""
        return {
            "estimated_rpm": round(float(self.x[0]), 1),
            "estimated_map_bar": round(float(self.x[1]), 3),
            "estimated_oil_p_bar": round(float(self.x[2]), 2),
            "estimated_oil_t_c": round(float(self.x[3]), 1),
            "estimated_cht_c": [round(float(self.x[i]), 1) for i in range(4, 8)],
            "estimated_egt_c": [round(float(self.x[i]), 1) for i in range(8, 12)],
        }
