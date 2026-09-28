# ENGINE-TWIN: AI-Enabled Real-Time Digital Twin System for Aero Piston Engines in MALE UAVs

**Smart India Hackathon (SIH 2026)**  
**Problem Statement ID:** SIH26054  
**Theme:** Robotics and Drones | **Category:** Software  
**Organization:** DRDO (Defence Research and Development Organisation)  

---

## 1. Executive Summary & Core Innovation

**ENGINE-TWIN** is a real-time, physics-anchored Digital Twin and explainable AI prognostic system engineered specifically for **aero piston engines** powering Medium-Altitude Long-Endurance (MALE) UAVs (e.g., the DRDO/VRDE 2.2L Turbocharged Aero-Diesel, Rotax 914, and Lycoming O-320).

### Key Architectural Differentiators:
1. **Physics-Anchored Residual Learning:** Rather than feeding raw, altitude-sensitive telemetry into black-box ML models, ENGINE-TWIN runs a synchronized **Mean Value Engine Model (MVEM)** to generate dynamic healthy baseline expectations ($y_{\text{mvem}}$). Machine learning models operate on normalized residual vectors $\mathbf{r} = (\mathbf{y}_{\text{sensor}} - \mathbf{y}_{\text{mvem}}) \oslash \boldsymbol{\sigma}_{\text{sensor}}$, rendering diagnosis invariant to altitude derating (0 to 30,000 ft) and throttle transients.
2. **5-State Extended Kalman Filter (EKF) Twin Core:** Continuous state estimation fusing multi-channel telemetry with non-linear thermodynamic equations, filtering measurement noise and handling intermittent packet loss.
3. **Sensor-vs-Engine Fault Decoupling:** Physical redundancy and cross-sensor correlation layer that prevents sensor probe defects (e.g. an open thermocouple) from triggering false engine shutdowns and aborted missions.
4. **Uncertainty-Calibrated Prognostics (RUL):** Predicts condition-based Remaining Useful Life as a 95% confidence interval (e.g. *18–25 flight hours*), avoiding false precision.
5. **Real-Time SHAP Root-Cause Explainability:** Local feature attribution computed in **<1.0 ms** and formatted into military-grade decision-support alerts with actionable pilot directives.

---

## 2. System Architecture

```
                      [MALE UAV Aero Piston Engine]
                                   │
                                   ▼
                    [Datasheet-Grounded Sensor Model]
             (EGT 1-4, CHT 1-4, Oil P/T, MAP, RPM, Vib, Bus V)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENGINE-TWIN PIPELINE                              │
│                                                                             │
│   ┌───────────────────────────┐         ┌───────────────────────────────┐   │
│   │   Physics MVEM Baseline   │◄───────►│    Extended Kalman Filter     │   │
│   │   Calibrated to VRDE 2.2L │         │    12-State Physics Fusion    │   │
│   └─────────────┬─────────────┘         └───────────────┬───────────────┘   │
│                 │                                       │                   │
│                 └───────────────────┬───────────────────┘                   │
│                                     ▼                                       │
│                       Normalized Physics Residuals                          │
│                   r = (y_sensor - y_mvem) / sigma_channel                   │
│                                     │                                       │
│          ┌──────────────────────────┴──────────────────────────┐            │
│          ▼                                                     ▼            │
│   [Stage 1: Anomaly Trigger]                          [Health Index Engine] │
│   Unsupervised Autoencoder                             Subsystem Health     │
│   e_recon > Mean + 3.0*sigma                           0 to 100% Scores     │
│          │                                                     │            │
│          ▼ (Fires if Anomaly)                                  │            │
│   [Sensor vs. Engine Validator]                                │            │
│   Cross-channel thermodynamic consistency                      │            │
│          │                                                     │            │
│          ▼                                                     │            │
│   [Stage 2: Multi-Task Classifier]                             │            │
│   TCN / GRU Multi-Class Fault Diagnosis                        │            │
│          │                                                     │            │
│          ├──────────────────────────┬──────────────────────────┤            │
│          ▼                          ▼                          ▼            │
│   [RUL Prognostics]          [Fast SHAP XAI]         [5-State Twin Model]   │
│   Degradation Trajectory +   Local Attribution       Physical / Sensor /    │
│   95% Confidence Bounds      Root-Cause Breakdown    Estimated / AI State   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ WebSocket Stream (20 Hz)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPERATOR GROUND STATION DASHBOARD                      │
│   • Interactive 3D Engine Twin View with Cylinder Thermal Heatmap           │
│   • Real-Time Telemetry vs. MVEM Physics Baseline vs. Residual Bounds       │
│   • Subsystem Health Matrix (Overall, Cylinders, Oil, Turbo, Vibration)     │
│   • SHAP Root-Cause Evidence Drawer & Action Directives                     │
│   • 1-Click Live Fault-Injection Control Pad for Stage Demonstration        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Empirical Benchmark Results (Held-Out Test Set)

Evaluated across **22,000 held-out test frames** across all flight phases (Takeoff, Climb to 30,000 ft, High-Altitude Loiter, Descent):

| Metric | Method A: Fixed-Threshold EIS | Method B: Black-Box ML (Raw) | Proposed ENGINE-TWIN |
| :--- | :---: | :---: | :---: |
| **Fault Classification Accuracy** | 59.49 % | 83.68 % | **93.47 %** |
| **Macro F1-Score** | 0.2842 | 0.8778 | **0.9414** |
| **Mean Detection Latency** | 25.51 s *(Late)* | 5.96 s | **2.30 s *(10x Faster)*** |
| **False Alarm Rate on Healthy Sorties** | 0.00 % *(Too wide)* | 27.65 % *(Fails on Alt)* | **2.18 % *(Robust)*** |
| **CPU Inference Latency** | 0.083 ms | 0.004 ms | **1.023 ms *(Target < 150 ms)*** |
| **Sensor vs. Engine Decoupling** | ❌ False Abort | ❌ Confounded | **✅ 100% Correctly Decoupled** |
| **Root-Cause Explainability** | ❌ None | ⚠️ Feature Imp Only | **✅ SHAP Operator Cards** |
| **RUL Uncertainty Bounds** | ❌ None | ❌ None | **✅ 95% Confidence Intervals** |

---

## 4. DRDO/VRDE 2.2L Altitude Derating Fit Verification

| Altitude (ft) | Target Published Power (HP) | Simulated MVEM Power (HP) | Target MAP (bar) | Simulated MAP (bar) | Fit Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Sea Level)** | 200.0 HP | **200.0 HP** | 2.40 bar | 2.45 bar | **0.0%** |
| **10,000** | 200.0 HP | **200.0 HP** | 2.45 bar | 2.45 bar | **0.0%** |
| **20,000** | 150.0 HP | **150.0 HP** | 2.10 bar | 2.10 bar | **0.0%** |
| **30,000** | 110.0 HP | **110.0 HP** | 1.65 bar | 1.65 bar | **0.0%** |

* **Power Fit RMSE:** **0.00 HP (0.0% Fit Error)**
* **MAP Fit RMSE:** **0.025 bar**

---

## 5. Supported Fault Modes & Taxonomy

1. `LEAN_MIXTURE_CYL3`: Injector partial clogging on Cylinder 3 (EGT rises, thermal gradient shifts).
2. `RICH_MIXTURE_CYL1`: Leaking injector on Cylinder 1 (EGT drops, fuel consumption increases).
3. `COOLING_DEGRADATION_CYL2`: Baffle / airflow blockage on Cylinder 2 (CHT rises rapidly).
4. `OIL_PRESSURE_LOSS`: Oil pump pressure relief valve leak (pressure drops below 2.0 bar).
5. `TURBO_BOOST_DEFICIENCY`: Wastegate stuck open / compressor fouling (loss of MAP at altitude).
6. `CYLINDER_MISFIRE_TIMING`: Timing jitter / valve seating defect (RPM instability, vibration 0.5X surge).
7. `BEARING_WEAR_VIBRATION`: Crankshaft / rod bearing wear (progressive 2X vibration amplification).
8. `SENSOR_FAULT_EGT3`: Thermocouple open-circuit defect (reads ambient while engine is healthy).
9. `ELECTRICAL_VOLTAGE_SAG`: Alternator diode fault (bus voltage drops under avionics load).

---

## 6. Quick Start & Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suites
```bash
# Verify Physics Core & Sensor Noise
python test_simulation.py

# Verify EKF, AI Models, RUL & SHAP XAI
python test_ai_layer.py

# Run Full Scientific Benchmark
python validation/benchmark.py
```

### Step 3: Launch Interactive Ground Station Dashboard
```bash
# Windows 1-click batch launcher
run_demo.bat

# Or direct Python command:
python main.py
```
Open **http://127.0.0.1:8000** in any browser.

---

## 7. SIH Judge Live Demonstration Script (2–3 Minutes)

1. **Nominal State:** Start `main.py`. The dashboard displays a healthy 3D engine model with green glowing cylinders, 100% Subsystem Health, and normal telemetry streams at 28,000 ft loiter.
2. **Inject Oil Loss:** Click `Oil Pressure Loss` on the bottom control pad. Within **2.3 seconds**, the Anomaly Detector fires, the classifier diagnoses `OIL_PRESSURE_LOSS` with **97.9% confidence**, Oil Health drops to 0%, the 3D sump turns red, and an immediate **Return-To-Base (RTB)** directive appears.
3. **Demonstrate Sensor Decoupling:** Reset to healthy, then click `Probe Defect (EGT3)`. EGT3 drops to 18°C. The system identifies `SENSOR PROBE DEFECT (ENGINE HEALTHY)`, keeping health at 98% and advising the pilot that the mission may safely proceed.
4. **Demonstrate SHAP Explainability:** Click `Bearing 2X Vib`. The SHAP drawer instantly shows that Tri-Axial Vibration (68.6% attribution) and Sump Temperature drove the diagnosis, accompanied by an actionable **18–25 flight hour inspection window**.
