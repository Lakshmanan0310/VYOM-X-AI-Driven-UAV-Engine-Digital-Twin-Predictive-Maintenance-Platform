# VYOM-X: AI-Powered UAV Engine Digital Twin and Predictive Maintenance System

**Smart India Hackathon (SIH) 2026**  
**Problem Statement ID:** SIH26054  
**Theme:** Robotics and Drones  
**Category:** Software

**Project Name:** VYOM-X  
**Core Platform:** ENGINE-TWIN  
**Website:** [VYOM-X Live Dashboard](https://dashboard-pi-beryl-44.vercel.app)

---

## 1. Project Overview

VYOM-X is an AI-assisted engine monitoring and predictive maintenance platform designed to support the condition assessment of piston engines used in Medium-Altitude Long-Endurance (MALE) Unmanned Aerial Vehicles (UAVs).

The system combines a physics-based engine simulation, sensor telemetry analysis, state estimation, machine learning, and an interactive ground-station interface. Its central component, ENGINE-TWIN, maintains a software representation of engine behaviour and compares expected operating conditions with incoming or simulated sensor measurements.

Rather than evaluating each sensor reading in isolation, the platform considers operating conditions, including engine speed, manifold pressure, exhaust temperature, cylinder head temperature, oil parameters, vibration, and electrical voltage. This approach is intended to help distinguish normal operating variations from patterns that may indicate a developing fault.

### Core capabilities

1. **Physics-informed anomaly detection:** Compares measured telemetry against a model-generated baseline to identify deviations from expected engine behaviour.
2. **Engine state estimation:** Uses an Extended Kalman Filter (EKF) to estimate internal engine states from noisy sensor measurements.
3. **Sensor fault identification:** Examines relationships between multiple measurements to help distinguish instrument faults from possible engine abnormalities.
4. **Predictive maintenance:** Estimates degradation trends and, when sufficient validated data are available, the remaining useful life (RUL) of selected components.
5. **Explainable AI:** Presents model outputs with feature-level explanations to help operators understand the signals associated with a detected anomaly.
6. **Interactive monitoring dashboard:** Displays telemetry, subsystem condition indicators, engine visualisation, diagnostic results, and simulation controls.

The platform is intended as a research and decision-support prototype. Its outputs should not be treated as a replacement for certified aircraft engine instrumentation, approved maintenance procedures, or flight-safety systems.

---

## 2. Technical Approach and Innovation

### 2.1 Physics-informed telemetry analysis

Engine behaviour changes with altitude, throttle setting, rotational speed, ambient conditions, and engine loading. A diagnostic system that ignores these factors may interpret normal operating changes as faults.

ENGINE-TWIN addresses this challenge by comparing sensor measurements with expected values from a Mean Value Engine Model (MVEM).

For sensor channel \(i\), the residual can be expressed as:

\[
r_i(t)=\frac{y_i(t)-\hat{y}_{i,\mathrm{MVEM}}(t)}{\sigma_i}
\]

Where:

- \(y_i(t)\) represents the observed sensor measurement.
- \(\hat{y}_{i,\mathrm{MVEM}}(t)\) represents the corresponding model prediction.
- \(\sigma_i\) represents the selected measurement-noise scale or normalisation factor.
- \(r_i(t)\) represents the normalised deviation from expected behaviour.

These residuals provide the diagnostic layer with operating-context information. Their usefulness depends on the accuracy of the engine model, sensor calibration, and the operating conditions represented in the validation data.

### 2.2 Engine state estimation using EKF

The Extended Kalman Filter combines a nonlinear state-transition model with incoming measurements to estimate engine conditions over time.

The estimator is intended to:

- Reduce the influence of measurement noise.
- Track changing engine operating conditions.
- Estimate states that are not directly measured.
- Support continued estimation during limited or intermittent measurement loss, subject to the estimator design.

The implementation should specify the actual number of estimated states, the state vector, the measurement vector, and the associated process and measurement noise assumptions.

### 2.3 Sensor and engine fault differentiation

An abnormal sensor reading does not necessarily indicate an engine failure. For example, an exhaust gas temperature probe may produce an implausible reading while other engine measurements remain consistent with normal operation.

VYOM-X evaluates cross-channel relationships and physical consistency to help identify whether an anomaly is more consistent with a sensor issue, an engine-related issue, or an uncertain condition.

The result is intended to reduce avoidable false alarms while preserving appropriate warnings when the available evidence is inconclusive.

### 2.4 Remaining Useful Life estimation

The prognostics module analyses degradation patterns to estimate how much useful operating time may remain before a defined maintenance threshold is reached.

Where sufficient degradation histories and validation data are available, the system can report an estimated RUL interval rather than presenting a single value as certain.

For example, an output such as “18–25 operating hours” should be displayed only when supported by the model and its calibration. The interval must be validated against actual outcomes or an appropriate test methodology before being interpreted as a reliable maintenance estimate.

### 2.5 Explainable diagnostic outputs

The explainability layer uses feature-attribution methods, such as SHAP, to help identify which input features contributed to a model prediction.

The dashboard can present:

- Detected anomaly or predicted fault category.
- Relevant sensor deviations.
- Contributing features and their attribution values.
- Estimated confidence or uncertainty, where calibrated.
- Suggested inspection steps based on an approved diagnostic mapping.

Feature attribution explains model behaviour; it does not independently establish the physical cause of a fault.

---

## 3. System Architecture

The platform is organised into five functional layers.

### Layer 1 — Telemetry acquisition

Collects real or simulated engine measurements, including:

- Exhaust Gas Temperature (EGT) across monitored cylinders.
- Cylinder Head Temperature (CHT).
- Oil pressure and oil temperature.
- Manifold Absolute Pressure (MAP).
- Engine revolutions per minute (RPM).
- Engine vibration.
- Electrical bus voltage.

The data-acquisition layer also handles timestamps, missing readings, data validation, and sensor-quality indicators.

### Layer 2 — Physics and state estimation

The engine-model layer calculates expected operating behaviour. The EKF combines model predictions and measurements to produce a filtered estimate of engine state.

### Layer 3 — Diagnostic intelligence

The diagnostic layer processes normalised residuals and other validated features through the following stages:

1. **Anomaly screening:** Identifies deviations from established normal-operation patterns.
2. **Consistency assessment:** Examines relationships between sensor channels and model predictions.
3. **Fault classification:** Estimates a fault category when the classifier supports that conclusion.
4. **Health assessment:** Generates condition indicators for selected engine subsystems.
5. **Prognostics:** Estimates degradation and RUL where sufficient supporting data exist.
6. **Explanation:** Presents feature contributions and relevant diagnostic evidence.

### Layer 4 — Operator dashboard

The ground-station interface provides:

- Interactive 3D engine visualisation.
- Cylinder-level thermal indicators.
- Live or simulated telemetry plots.
- Model predictions and residual comparisons.
- Subsystem health indicators.
- Fault history and diagnostic explanations.
- Controlled fault-injection scenarios for demonstrations.

### Layer 5 — Communication and monitoring

A streaming interface can deliver telemetry and diagnostic updates to the dashboard. A target update frequency of 20 Hz may be used where supported by the data pipeline and measured system performance.

**Architecture summary:**

Telemetry → Physics Model → State Estimation → Residual Analysis → Fault Diagnosis → Prognostics and Explanations → Ground-Station Dashboard

---

## 4. Experimental Evaluation

The proposed evaluation compares three diagnostic approaches:

- **Baseline A:** Fixed-threshold engine indication logic.
- **Baseline B:** Machine learning using raw sensor features.
- **Proposed approach:** Physics-informed residual analysis combined with state estimation and AI-based diagnosis.

The evaluation should use clearly defined train, validation, and held-out test sets. Data from the same simulated or recorded operating sequence should not unintentionally appear across both training and testing partitions.

### Reported benchmark results

The following figures are retained from the current project draft and must be verified against the actual benchmark output before publication.

| Evaluation metric | Fixed-threshold baseline | Raw-feature ML | ENGINE-TWIN |
|---|---:|---:|---:|
| Fault classification accuracy | 59.49% | 83.68% | 93.47% |
| Macro F1-score | 0.2842 | 0.8778 | 0.9414 |
| Mean detection latency | 25.51 s | 5.96 s | 2.30 s |
| False alarm rate on healthy test cases | 0.00% | 27.65% | 2.18% |
| Reported CPU inference latency | 0.083 ms | 0.004 ms | 1.023 ms |

Additional project claims concerning sensor-fault separation, uncertainty-calibrated RUL, and SHAP execution time require separate supporting tests and measurement definitions.

### Recommended validation procedure

- Test each supported fault category against healthy operating cases.
- Include take-off, climb, cruise or loiter, and descent operating profiles where represented by the model.
- Evaluate performance across the full range of tested altitudes and engine loads.
- Report per-class precision, recall, F1-score, and a confusion matrix.
- Measure false alarms per operating hour or sortie, with the denominator clearly stated.
- Measure inference latency on specified hardware and report the statistic used.
- Evaluate RUL interval coverage and interval width against known degradation outcomes.

**Important:** The benchmark values above are reported project figures, not independently verified results. Do not describe them as experimentally established until the corresponding test scripts, datasets, and outputs support them.

---

## 5. Engine Model Calibration

The engine-model calibration module compares simulated outputs against documented reference values or independently measured engine data.

The current project draft contains the following proposed calibration points for a 2.2-litre turbocharged aero-diesel engine.

| Altitude | Reference power | Model power | Reference MAP | Model MAP |
|---|---:|---:|---:|---:|
| Sea level | 200 HP | 200 HP | 2.40 bar | 2.45 bar |
| 10,000 ft | 200 HP | 200 HP | 2.45 bar | 2.45 bar |
| 20,000 ft | 150 HP | 150 HP | 2.10 bar | 2.10 bar |
| 30,000 ft | 110 HP | 110 HP | 1.65 bar | 1.65 bar |

These values should be treated as provisional reference points until their source and applicability to the selected engine configuration are documented.

A rigorous calibration report should identify:

- The engine model and configuration.
- The reference datasheet or measurement source.
- The assumed ambient temperature and pressure.
- Turbocharger and boost-control assumptions.
- The definition of rated power at each operating condition.
- The power and MAP error calculations.

The draft also reports a power RMSE of 0.00 HP and MAP RMSE of 0.025 bar. These figures should be recalculated directly from the complete reference and model-output datasets. A zero power error at selected calibration points does not, by itself, establish zero error across the full flight envelope.

---

## 6. Supported Fault Scenarios

The prototype's proposed fault taxonomy contains nine scenarios.

| Fault identifier | Scenario | Example diagnostic evidence |
|---|---|---|
| `LEAN_MIXTURE_CYL3` | Lean-mixture condition associated with cylinder 3 | Cylinder-specific EGT deviation and thermal imbalance |
| `RICH_MIXTURE_CYL1` | Rich-mixture condition associated with cylinder 1 | EGT deviation and changes in fuel-use indicators, if available |
| `COOLING_DEGRADATION_CYL2` | Reduced cooling effectiveness at cylinder 2 | Elevated CHT relative to operating conditions |
| `OIL_PRESSURE_LOSS` | Abnormal reduction in oil pressure | Pressure deviation and supporting engine-state evidence |
| `TURBO_BOOST_DEFICIENCY` | Reduced boost pressure | MAP deviation under comparable altitude and load conditions |
| `CYLINDER_MISFIRE_TIMING` | Combustion or timing irregularity | RPM variation, vibration changes, and temperature deviations |
| `BEARING_WEAR_VIBRATION` | Bearing-related vibration anomaly | Changes in vibration features and their progression over time |
| `SENSOR_FAULT_EGT3` | Fault in the third EGT measurement channel | Implausible EGT reading with inconsistent cross-channel evidence |
| `ELECTRICAL_VOLTAGE_SAG` | Electrical supply voltage reduction | Bus-voltage deviation and associated electrical indicators |

These scenarios define the intended diagnostic coverage. Each category should be supported by documented test cases, and the system should be allowed to return an uncertain or unknown diagnosis when evidence is insufficient.

---

## 7. Ground-Station Dashboard Features

The operator dashboard is designed to present technical information in a clear, navigable interface.

### Engine visualisation
- Interactive 3D representation of the engine.
- Cylinder-level temperature visualisation.
- Visual indicators for detected abnormalities.

### Live monitoring
- Time-series telemetry charts.
- Measured values compared with model predictions.
- Residual trends and operating-condition information.
- Sensor data-quality indicators.

### Health and diagnostics
- Subsystem condition indicators.
- Fault classification and diagnostic evidence.
- Feature-attribution explanations.
- Historical event tracking.

### Prognostics
- Degradation trends.
- RUL estimates and uncertainty intervals, where validated.
- Maintenance-review indicators.

### Demonstration mode
- Controlled fault-injection scenarios.
- Healthy-versus-faulty operating comparisons.
- Repeatable test cases for demonstrations and evaluation.

Fault injection is intended for controlled simulation and testing, not for introducing faults into an operating aircraft engine.

---

## 8. Installation and Execution

### Prerequisites

- Python version compatible with the project's dependency files.
- Required Python packages.
- The project source code and configuration files.
- A supported operating system and sufficient computing resources.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Run this command from the project root with the intended Python environment activated.

### Step 2 — Run verification tests

```bash
python test_simulation.py
```

Tests the simulation layer and its configured sensor-noise behaviour.

```bash
python test_ai_layer.py
```

Runs the AI-layer checks for the components implemented in the repository.

```bash
python validation/benchmark.py
```

Runs the benchmark pipeline, provided the required data, models, and configuration are available.

### Step 3 — Launch the application

On Windows, run:

```bat
run_demo.bat
```

Alternatively, use:

```bash
python main.py
```

The actual startup command depends on the repository's implementation. Consult the project configuration and README if the application uses a separate frontend, API server, or database service.

---

## 9. Technology and Implementation Overview

The final technology list should match the packages and components actually implemented in the repository.

| Component | Purpose |
|---|---|
| Python | Simulation, data processing, and diagnostic services |
| Mean Value Engine Model | Expected engine-behaviour estimation |
| Extended Kalman Filter | State estimation and measurement fusion |
| Machine learning | Anomaly detection and fault classification |
| SHAP or another validated attribution method | Model-prediction explanations |
| WebSocket or equivalent streaming interface | Dashboard data updates |
| Web-based dashboard | Monitoring, visualisation, and diagnostics |
| 3D rendering framework | Interactive engine representation |

The exact ML architectures, rendering libraries, database technologies, and deployment services should be listed only after confirming their use in the codebase.

---

## 10. Expected Outcomes

VYOM-X aims to demonstrate how physics-based modelling and AI can be combined to improve the interpretation of UAV engine telemetry.

The intended outcomes are:

- Earlier identification of abnormal operating patterns.
- Better contextual interpretation of sensor measurements.
- Clearer differentiation between possible sensor and engine abnormalities.
- More interpretable diagnostic outputs.
- Degradation-aware maintenance support.
- An interactive environment for testing and demonstrating diagnostic scenarios.

The effectiveness of these outcomes depends on model fidelity, representative test data, appropriate calibration, and independent validation.

---

## 11. Website and Project Links

**Live Website:** [VYOM-X Live Dashboard](https://dashboard-pi-beryl-44.vercel.app)

**Backend Health Check:** [API Health Endpoint](https://vyom-x-engine-api.onrender.com/health)

**GitHub Repository:** [https://github.com/Lakshmanan0310/VYOM-X-AI-Driven-UAV-Engine-Digital-Twin-Predictive-Maintenance-Platform]

**Project Demonstration:** [https://youtu.be/W5Lh-i1r_yQ]

**Documentation:** [https://drive.google.com/drive/folders/1wEUOSqkFLh6IIwIudJ3MYVj9DOaRAxul?usp=sharing]

---

## 12. Conclusion

VYOM-X presents a physics-informed approach to AI-assisted engine monitoring for UAV applications. By connecting an engine-behaviour model, state estimation, sensor-consistency checks, machine learning, prognostic analysis, and visual explanations, the platform aims to turn raw telemetry into structured information for maintenance assessment.

Its principal research direction is the integration of physical operating context with data-driven diagnosis. The next steps are to document the engine-model assumptions, validate the diagnostic pipeline using reproducible test datasets, measure actual runtime performance, and evaluate uncertainty estimates before making operational reliability claims.

