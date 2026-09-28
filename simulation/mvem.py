"""
Mean Value Engine Model (MVEM) — Physics-Based Ground-Truth Simulation
Calibrated for DRDO/VRDE-Class 2.2L 4-Cylinder Turbocharged Aero-Diesel Engine.

Sub-models:
1. Intake Manifold & Turbocharger Aerodynamics (dp_m/dt, Boost derating)
2. Crankshaft Torque Balance & Rotational Dynamics (J*d_omega/dt)
3. Multi-Cylinder Lumped Thermal Dynamics (4x CHT, 4x EGT)
4. Oil Lubrication & Thermal Hydraulics (P_oil, T_oil)
5. Order-Tracked Engine Vibration Generation (1X, 2X, 4X harmonics)
"""
from dataclasses import dataclass, field
import math
from typing import List, Dict, Any, Optional
import numpy as np

from .flight_profile import FlightState

@dataclass
class EngineState:
    time_s: float
    rpm: float
    power_hp: float
    torque_nm: float
    manifold_pressure_bar: float # MAP / Boost
    manifold_temp_c: float
    fuel_flow_lph: float
    bsfc_g_kwh: float            # Brake Specific Fuel Consumption
    oil_pressure_bar: float
    oil_temp_c: float
    coolant_temp_c: float
    egt_c: List[float]           # EGT for Cylinders 1, 2, 3, 4 [Celsius]
    cht_c: List[float]           # CHT for Cylinders 1, 2, 3, 4 [Celsius]
    air_fuel_ratio: List[float]  # Lambda for Cylinders 1, 2, 3, 4
    vibration_x_g: float         # 1X/2X Order-tracked vibration
    vibration_y_g: float
    vibration_z_g: float
    vibration_rms_g: float
    turbo_speed_krpm: float
    bus_voltage_v: float
    provenance: str = "SIMULATED"

class MeanValueEngineModel:
    """
    Control-oriented Physics Engine Simulator.
    Integrates intake ODEs, inertia torque balance, and lumped-parameter thermal heat transfer.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Base Engine Specs (VRDE 2.2L Turbo Aero-Diesel)
        self.displacement_v_d = 2.2e-3    # m^3 (2.2 litres)
        self.cylinders = 4
        self.compression_ratio = 17.5
        self.inertia_j = 0.45            # kg*m^2 (Crankshaft + Flywheel + Propeller)
        self.v_manifold = 0.0035         # m^3
        self.r_air = 287.058             # J/(kg*K)
        self.lhv_fuel = 42.8e6           # J/kg (Diesel fuel lower heating value)
        self.diesel_density = 0.835      # kg/L

        # Thermal Capacities & Conductances
        self.c_head = 450.0              # J/(kg*K) for Aluminum alloy
        self.m_head_per_cyl = 3.2        # kg per cylinder head zone
        self.c_oil_sump = 2100.0         # J/(kg*K)
        self.m_oil_kg = 5.5              # kg oil in system

        # Internal State Variables (Initial Conditions for Cruise)
        self.rpm = 2800.0
        self.p_manifold = 2.45
        self.t_manifold_c = 35.0
        self.t_cht = [175.0, 175.0, 175.0, 175.0]
        self.t_egt = [810.0, 810.0, 810.0, 810.0]
        self.t_oil = 88.0
        self.t_coolant = 86.0
        self.turbo_speed_krpm = 65.0

        # Degradation & Fault Modulation Multipliers (1.0 = perfectly healthy)
        self.cylinder_fuel_trim = [1.0, 1.0, 1.0, 1.0] # Multiplier on fueling per cylinder
        self.cylinder_cooling_trim = [1.0, 1.0, 1.0, 1.0] # Multiplier on cooling airflow per cyl
        self.oil_pump_health = 1.0       # 0 to 1.0 (drops on relief valve / leak)
        self.turbo_efficiency = 1.0      # Multiplier on compressor efficiency
        self.bearing_wear_factor = 1.0   # > 1.0 raises 2X order vibration and friction
        self.timing_jitter_deg = 0.0     # Degrees of timing jitter (misfire proxy)
        self.alternator_health = 1.0     # Multiplier on electrical output

    def reset(self, idle: bool = False):
        """Resets engine to nominal state."""
        self.rpm = 1400.0 if idle else 2800.0
        self.p_manifold = 1.02 if idle else 2.45
        self.t_cht = [85.0] * 4 if idle else [175.0] * 4
        self.t_egt = [450.0] * 4 if idle else [810.0] * 4
        self.t_oil = 70.0 if idle else 88.0
        self.t_coolant = 75.0 if idle else 86.0
        self.t_coolant = 75.0 if idle else 88.0
        self.turbo_speed_krpm = 25.0 if idle else 115.0
        self.cylinder_fuel_trim = [1.0] * 4
        self.cylinder_cooling_trim = [1.0] * 4
        self.oil_pump_health = 1.0
        self.turbo_efficiency = 1.0
        self.bearing_wear_factor = 1.0
        self.timing_jitter_deg = 0.0
        self.alternator_health = 1.0

    def step(self, flight: FlightState, dt_s: float = 0.05) -> EngineState:
        """
        Integrates physics state forward by dt_s given current flight condition.
        """
        throttle = max(0.0, min(100.0, flight.throttle_pct)) / 100.0
        p_amb = flight.ambient_pressure_bar
        t_amb_c = flight.ambient_temp_c
        t_amb_k = t_amb_c + 273.15
        rho_amb = flight.air_density_kgpm3
        airspeed = max(5.0, flight.airspeed_mps)

        # -------------------------------------------------------------
        # 1. Turbocharger & Intake Manifold Dynamics
        # -------------------------------------------------------------
        # VRDE 2.2L Turbo Aero-Diesel Critical Altitude Model:
        # Critical altitude is ~10,000 ft (P_amb ~ 0.697 bar).
        # Below 10k ft, wastegate modulates to hold 2.45 bar max boost.
        # Above 10k ft, compressor speed limit & turbine backpressure cause boost to derate:
        # At 20k ft: MAP ~ 2.10 bar -> Power ~ 150 HP
        # At 30k ft: MAP ~ 1.65 bar -> Power ~ 110 HP
        alt_ft = flight.altitude_ft
        if alt_ft <= 10000.0:
            max_boost_cap = 2.45
        elif alt_ft <= 20000.0:
            frac = (alt_ft - 10000.0) / 10000.0
            max_boost_cap = 2.45 - (2.45 - 2.10) * frac
        else:
            frac = min(1.0, (alt_ft - 20000.0) / 10000.0)
            max_boost_cap = 2.10 - (2.10 - 1.65) * frac

        max_achievable_map = max_boost_cap * self.turbo_efficiency
        target_map = p_amb + (max_achievable_map - p_amb) * (throttle ** 1.3)
        target_map = max(p_amb * 0.95, target_map) # MAP cannot be below idle intake vacuum

        # Turbo inertia lag (time constant ~ 0.35 s)
        self.p_manifold += (target_map - self.p_manifold) * (dt_s / 0.35)
        self.turbo_speed_krpm = 20.0 + (self.p_manifold / 2.45) * 95.0

        # Intercooler thermal effectiveness
        t_compressor_out = t_amb_k * ((self.p_manifold / max(0.2, p_amb)) ** 0.286) - 273.15
        intercooler_eff = 0.85
        self.t_manifold_c = t_compressor_out - intercooler_eff * (t_compressor_out - t_amb_c)

        # Volumetric Efficiency (Speed-density equation)
        speed_ratio = self.rpm / 4200.0
        eta_vol = 0.88 - 0.08 * (speed_ratio ** 2) + 0.05 * (self.p_manifold / 2.45)
        t_man_k = self.t_manifold_c + 273.15
        rho_manifold = (self.p_manifold * 1e5) / (self.r_air * t_man_k)
        air_mass_flow_kgps = eta_vol * (self.displacement_v_d * (self.rpm / 120.0)) * rho_manifold

        # -------------------------------------------------------------
        # 2. Fuel Injection & Indicated Torque Balance
        # -------------------------------------------------------------
        # Diesel stoichiometric AFR ~ 14.5. Nominal lean burn lambda ~ 1.25 to 1.6
        nominal_lambda = 1.35 + 0.40 * (1.0 - throttle)
        base_fuel_kgps = (air_mass_flow_kgps / (14.5 * nominal_lambda))

        # Individual cylinder fueling with trims and timing jitter
        cyl_fuel_kgps = []
        cyl_lambdas = []
        for i in range(self.cylinders):
            f_i = (base_fuel_kgps / 4.0) * self.cylinder_fuel_trim[i]
            # Timing jitter drops combustion efficiency
            jitter_loss = max(0.0, 1.0 - 0.015 * abs(self.timing_jitter_deg))
            cyl_fuel_kgps.append(f_i * jitter_loss)
            # Local AFR lambda
            lam_i = (air_mass_flow_kgps / 4.0) / max(1e-6, f_i * 14.5)
            cyl_lambdas.append(round(float(lam_i), 3))

        total_fuel_kgps = sum(cyl_fuel_kgps)
        fuel_flow_lph = (total_fuel_kgps * 3600.0) / self.diesel_density

        # Indicated Thermal Efficiency (function of compression ratio and load)
        eta_th_ind = 0.42 * (1.0 - 0.08 * (1.0 - throttle))
        indicated_power_watts = total_fuel_kgps * self.lhv_fuel * eta_th_ind

        # Chen-Flynn Mechanical Friction Model + Bearing Wear
        # P_fric ~ (FMEP * V_d * N) / 120
        fmep_bar = (0.45 + 0.00035 * self.rpm + 0.04 * (self.p_manifold / 1.0)) * self.bearing_wear_factor
        friction_power_watts = (fmep_bar * 1e5 * self.displacement_v_d * (self.rpm / 120.0))

        # Propeller Load Torque Absorption: Tau_prop = c_prop * rho * omega^2
        omega = max(10.0, (self.rpm * 2.0 * math.pi) / 60.0)
        c_prop = 0.00078
        propeller_power_watts = c_prop * rho_amb * (omega ** 3.0)

        # Net Crankshaft Power & Torque
        net_power_watts = max(0.0, indicated_power_watts - friction_power_watts)
        net_power_hp = net_power_watts / 745.7
        torque_nm = (net_power_watts / omega) if omega > 0 else 0.0

        # Crankshaft Acceleration: J * d_omega/dt = Tau_net - Tau_prop
        tau_net = torque_nm
        tau_prop = (propeller_power_watts / omega)
        d_omega_dt = (tau_net - tau_prop) / self.inertia_j

        # RPM integration with governor speed stabilization
        target_governor_rpm = 1400.0 + throttle * (4000.0 - 1400.0)
        # Apply altitude derating constraint to net power based on available boost (VRDE 2.2L specs)
        # 2.45 bar -> 200 HP, 2.10 bar -> 150 HP, 1.65 bar -> 110 HP
        if self.p_manifold >= 2.40:
            power_limit_hp = 200.0
        elif self.p_manifold >= 2.10:
            frac = (self.p_manifold - 2.10) / (2.45 - 2.10)
            power_limit_hp = 150.0 + (200.0 - 150.0) * frac
        else:
            frac = max(0.0, (self.p_manifold - 1.65) / (2.10 - 1.65))
            power_limit_hp = 110.0 + (150.0 - 110.0) * frac

        if net_power_hp > power_limit_hp:
            net_power_hp = power_limit_hp

        self.rpm += d_omega_dt * (60.0 / (2.0 * math.pi)) * dt_s
        # Governor feedback stabilization
        self.rpm += (target_governor_rpm - self.rpm) * (dt_s / 0.8)
        self.rpm = max(1200.0, min(4400.0, self.rpm))

        # Brake Specific Fuel Consumption (g/kWh)
        bsfc = ((total_fuel_kgps * 3600.0 * 1000.0) / max(1.0, (net_power_watts / 1000.0)))

        # -------------------------------------------------------------
        # 3. Multi-Cylinder Lumped Thermal Dynamics
        # -------------------------------------------------------------
        # Coolant & Oil Heat Absorption
        self.t_coolant += (85.0 + 15.0 * (net_power_hp / 200.0) - 0.15 * (airspeed - 30.0) - self.t_coolant) * (dt_s / 12.0)
        # Oil Temp: Driven by friction + combustion heat, dissipated by oil cooler
        target_oil_t = 80.0 + 35.0 * (net_power_hp / 200.0) * self.bearing_wear_factor - 0.20 * (airspeed - 30.0) + (t_amb_c - 15.0) * 0.3
        self.t_oil += (target_oil_t - self.t_oil) * (dt_s / 18.0)

        # Oil Pressure Hydraulics: P_oil = P_base(RPM) * viscosity(T_oil) * pump_health
        oil_viscosity_factor = math.exp(-0.015 * (self.t_oil - 90.0))
        base_oil_p = 2.0 + 2.8 * (self.rpm / 4000.0)
        oil_pressure_bar = base_oil_p * oil_viscosity_factor * self.oil_pump_health

        # Per-Cylinder CHT & EGT Equations
        for i in range(self.cylinders):
            fuel_ratio_i = self.cylinder_fuel_trim[i]
            cooling_ratio_i = self.cylinder_cooling_trim[i]

            # EGT: function of air-fuel ratio, fuel trim, throttle, and ignition timing
            fuel_delta_egt = 0.0
            if fuel_ratio_i < 1.0:
                # Lean mixture: delayed late-burning flame raises EGT by up to +180 C
                fuel_delta_egt = (1.0 - fuel_ratio_i) * 260.0
            elif fuel_ratio_i > 1.0:
                # Rich mixture: evaporative cooling from excess fuel drops EGT by up to -140 C
                fuel_delta_egt = (1.0 - fuel_ratio_i) * 180.0

            target_egt = 420.0 + 360.0 * (throttle ** 0.8) + fuel_delta_egt + (abs(self.timing_jitter_deg) * 6.0)
            self.t_egt[i] += (target_egt - self.t_egt[i]) * (dt_s / 0.8)

            # CHT: Cylinder Head Temp (lumped balance with coolant and airspeed)
            target_cht = (110.0 + 85.0 * (net_power_hp / 200.0) * fuel_ratio_i + 
                          (self.t_coolant - 80.0) * 0.65 - 
                          (airspeed - 30.0) * 0.35 * cooling_ratio_i + 
                          (t_amb_c - 15.0) * 0.25)
            # If cooling degradation is active (cooling_ratio_i < 1.0), CHT rises significantly
            if cooling_ratio_i < 1.0:
                target_cht += (1.0 - cooling_ratio_i) * 115.0

            self.t_cht[i] += (target_cht - self.t_cht[i]) * (dt_s / 1.5)

        # -------------------------------------------------------------
        # 4. Vibration Spectrum Generation (Order Tracking)
        # -------------------------------------------------------------
        f_1x = self.rpm / 60.0
        f_2x = 2.0 * f_1x
        
        vib_1x_amp = 0.35 * (self.rpm / 4000.0)
        vib_2x_amp = (0.85 + 2.8 * (self.bearing_wear_factor - 1.0)) * (net_power_hp / 200.0)
        # Timing jitter / misfire injects 0.5X sub-harmonic rumble
        vib_misfire_amp = 0.32 * abs(self.timing_jitter_deg)

        time_t = flight.time_s
        vib_x = vib_1x_amp * math.sin(2.0 * math.pi * f_1x * time_t) + vib_2x_amp * math.sin(2.0 * math.pi * f_2x * time_t)
        vib_y = vib_1x_amp * math.cos(2.0 * math.pi * f_1x * time_t) + 0.8 * vib_2x_amp * math.cos(2.0 * math.pi * f_2x * time_t) + vib_misfire_amp * math.sin(math.pi * f_1x * time_t)
        vib_z = 0.5 * vib_1x_amp + vib_2x_amp * 1.1 + vib_misfire_amp * 0.8 + np.random.normal(0, 0.05)
        vib_rms = math.sqrt((vib_x**2 + vib_y**2 + vib_z**2) / 3.0) + (self.bearing_wear_factor - 1.0) * 1.5 + vib_misfire_amp * 0.5

        # -------------------------------------------------------------
        # 5. Electrical Bus Voltage
        # -------------------------------------------------------------
        # Nominal 28.0 V avionics bus, charges above idle, sags if alternator degraded
        target_v = 28.2 * self.alternator_health if self.rpm > 1800 else 24.5 + 3.7 * (self.rpm / 1800.0)
        bus_voltage_v = target_v - 0.02 * (throttle * 10.0) # minor load drop

        return EngineState(
            time_s=round(flight.time_s, 2),
            rpm=round(self.rpm, 1),
            power_hp=round(net_power_hp, 1),
            torque_nm=round(torque_nm, 1),
            manifold_pressure_bar=round(self.p_manifold, 3),
            manifold_temp_c=round(self.t_manifold_c, 1),
            fuel_flow_lph=round(fuel_flow_lph, 2),
            bsfc_g_kwh=round(bsfc, 1),
            oil_pressure_bar=round(oil_pressure_bar, 2),
            oil_temp_c=round(self.t_oil, 1),
            coolant_temp_c=round(self.t_coolant, 1),
            egt_c=[round(t, 1) for t in self.t_egt],
            cht_c=[round(t, 1) for t in self.t_cht],
            air_fuel_ratio=[round(a, 2) for a in cyl_lambdas],
            vibration_x_g=round(vib_x, 3),
            vibration_y_g=round(vib_y, 3),
            vibration_z_g=round(vib_z, 3),
            vibration_rms_g=round(vib_rms, 3),
            turbo_speed_krpm=round(self.turbo_speed_krpm, 1),
            bus_voltage_v=round(bus_voltage_v, 2),
            provenance="SIMULATED"
        )
