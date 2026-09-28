"""
MALE UAV Mission Profiles & Atmospheric Dynamics (ISA Model)
Generates realistic altitude, ambient temperature, ambient pressure, airspeed, and throttle profiles.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class FlightState:
    time_s: float
    altitude_ft: float
    altitude_m: float
    throttle_pct: float         # 0 to 100%
    airspeed_mps: float         # True Airspeed in m/s
    ambient_temp_c: float       # Outside Air Temp in Celsius
    ambient_pressure_bar: float # Ambient pressure in bar
    air_density_kgpm3: float    # Air density in kg/m^3
    phase: str                  # Mission phase name

class FlightProfile:
    """
    Standard International Atmosphere (ISA) model + MALE UAV Flight Profile Generator.
    Supports structured mission phases (Taxi, Climb to 30k ft, Cruise/Loiter, Descent, Landing)
    as well as continuous customized mission curves.
    """
    # ISA Sea Level Constants
    P0 = 1.01325          # bar (101325 Pa)
    T0 = 288.15           # Kelvin (15 C)
    R_SPEC = 287.058      # J/(kg*K)
    G0 = 9.80665          # m/s^2
    LAPSE_RATE = 0.0065   # K/m up to 11,000 m (Troposphere)
    RHO0 = 1.225          # kg/m^3

    def __init__(self, delta_t_isa: float = 0.0):
        """
        :param delta_t_isa: Temperature offset from standard ISA (e.g. +10 for hot day, -10 for cold)
        """
        self.delta_t_isa = delta_t_isa

    @classmethod
    def get_isa_atmosphere(cls, altitude_m: float, delta_t_isa: float = 0.0):
        """Calculates ambient pressure, temperature and density at given altitude."""
        h = max(0.0, min(altitude_m, 11000.0)) # Troposphere model
        t_kelvin = cls.T0 - cls.LAPSE_RATE * h + delta_t_isa
        p_bar = cls.P0 * ((1.0 - (cls.LAPSE_RATE * h) / cls.T0) ** (cls.G0 / (cls.R_SPEC * cls.LAPSE_RATE)))
        rho = (p_bar * 1e5) / (cls.R_SPEC * t_kelvin)
        t_celsius = t_kelvin - 273.15
        return t_celsius, p_bar, rho

    @staticmethod
    def feet_to_meters(ft: float) -> float:
        return ft * 0.3048

    @staticmethod
    def meters_to_feet(m: float) -> float:
        return m / 0.3048

    def get_standard_mission_state(self, time_s: float, total_mission_s: float = 3600.0) -> FlightState:
        """
        Generates flight parameters for a standard MALE UAV surveillance sortie scaled to total_mission_s.
        Phases:
          0.00 to 0.05: Engine warmup & Taxi
          0.05 to 0.25: Takeoff & Climb to 30,000 ft
          0.25 to 0.75: High-Altitude Cruise / Loiter at 28,000 - 30,000 ft
          0.75 to 0.92: Combat/Surveillance descent to 2,000 ft
          0.92 to 1.00: Approach, Landing & Cooldown
        """
        progress = (time_s % total_mission_s) / total_mission_s

        if progress < 0.05:
            phase = "TAXI_WARMUP"
            alt_ft = 0.0
            throttle = 20.0 + 5.0 * math.sin(time_s * 0.1)
            tas_mps = 5.0
        elif progress < 0.25:
            phase = "CLIMB_TO_ALTITUDE"
            climb_frac = (progress - 0.05) / 0.20
            # Smooth S-curve climb to 30,000 ft
            alt_ft = 30000.0 * (0.5 - 0.5 * math.cos(math.pi * climb_frac))
            throttle = 95.0 - 5.0 * climb_frac
            tas_mps = 35.0 + 25.0 * climb_frac
        elif progress < 0.75:
            phase = "HIGH_ALT_LOITER"
            cruise_frac = (progress - 0.25) / 0.50
            # Loiter at ~29,000 ft with small atmospheric altitude undulations
            alt_ft = 29000.0 + 1000.0 * math.sin(cruise_frac * 8.0 * math.pi)
            throttle = 68.0 + 4.0 * math.sin(cruise_frac * 4.0 * math.pi)
            tas_mps = 55.0 + 3.0 * math.sin(cruise_frac * 6.0 * math.pi)
        elif progress < 0.92:
            phase = "DESCENT"
            descent_frac = (progress - 0.75) / 0.17
            alt_ft = 30000.0 * (1.0 - (0.5 - 0.5 * math.cos(math.pi * descent_frac))) + 1000.0
            throttle = 40.0 - 10.0 * descent_frac
            tas_mps = 55.0 - 15.0 * descent_frac
        else:
            phase = "APPROACH_LANDING"
            alt_ft = 0.0
            throttle = 18.0
            tas_mps = 10.0

        alt_m = self.feet_to_meters(alt_ft)
        t_amb_c, p_amb_bar, rho = self.get_isa_atmosphere(alt_m, self.delta_t_isa)

        return FlightState(
            time_s=time_s,
            altitude_ft=round(alt_ft, 1),
            altitude_m=round(alt_m, 1),
            throttle_pct=round(max(0.0, min(100.0, throttle)), 2),
            airspeed_mps=round(max(0.0, tas_mps), 2),
            ambient_temp_c=round(t_amb_c, 2),
            ambient_pressure_bar=round(p_amb_bar, 4),
            air_density_kgpm3=round(rho, 4),
            phase=phase
        )
