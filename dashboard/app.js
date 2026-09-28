/**
 * ENGINE-TWIN: Professional Aerospace Digital Twin Ground Station Controller
 * Robust 20 Hz Telemetry, Multi-Channel Temperature/Pressure Charting,
 * Calibrated Anomaly Indexing, Event Timeline Logging, and Stage Automation.
 * Optimized for Zero-Flicker High-Performance Ground Station Operations.
 */

const MAX_HISTORY = 40;
let ws = null;
let engine3D = null;
let charts = {};

// Audio Synthesizer State
let audioCtx = null;
let isAudioMuted = true;
let lastAudioAlertTime = 0;

// Active System & Fault State (Cached for zero-flicker rendering)
let currentActiveFault = "HEALTHY";
let lastActiveFaultButton = null;
let lastShapSignature = "";
let lastOverallStatus = "";
let eventLogs = [];
let maxEventLogs = 30;

// Auto Demo State Machine
let autoDemoActive = false;
let autoDemoTimer = null;
let autoDemoStep = 0;
let autoDemoTimeRemaining = 0;

// Chart Rendering Throttle State
let lastChartUpdateTime = 0;
const CHART_UPDATE_INTERVAL_MS = 100; // 10 Hz smooth charting (prevents canvas flutter)

// Telemetry History Buffers (Multi-Channel)
const historyData = {
  time: [],
  sensor_egt1: [],
  sensor_egt2: [],
  sensor_egt3: [],
  sensor_egt4: [],
  mvem_egt: [],
  sensor_oil_p: [],
  mvem_oil_p: [],
  sensor_map: [],
  mvem_map: [],
  sensor_vib: [],
  mvem_vib: []
};

// Defensive DOM Helpers
function setText(id, text) {
  const el = document.getElementById(id);
  if (el && el.innerText !== text) {
    el.innerText = text;
  }
}

// ------------------------------------------------------------------
// 1. INITIALIZATION ON DOM READY
// ------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  console.log("ENGINE-TWIN GCS Controller Initialized.");

  // 1. Initialize 3D Engine Model
  if (typeof Engine3DView !== "undefined") {
    try {
      engine3D = new Engine3DView("container-3d");
      window.engine3D = engine3D;
    } catch (e) {
      console.warn("3D View Init Warning:", e);
    }
  }

  // 2. Initialize Telemetry Charts
  try {
    initCharts();
  } catch (e) {
    console.error("Charts Init Error:", e);
  }

  // 3. Connect Telemetry Stream (WebSocket + SSE Fallback)
  connectWebSocket();

  // 4. Log Startup Event
  addEventLog("SYS", "Ground Control Station connected to telemetry stream (20 Hz).");
});

// ------------------------------------------------------------------
// 2. WEB AUDIO API SYNTHESIZER (AEROSPACE COCKPIT ALARMS)
// ------------------------------------------------------------------
function initAudioContext() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) audioCtx = new AudioContext();
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

function toggleAudio() {
  initAudioContext();
  isAudioMuted = !isAudioMuted;
  const icon = document.getElementById("icon-audio");
  const lbl = document.getElementById("lbl-audio");
  const btn = document.getElementById("btn-audio");

  if (isAudioMuted) {
    if (icon) icon.className = "fa-solid fa-volume-xmark";
    if (lbl) lbl.innerText = "MUTED";
    if (btn) btn.className = "px-2.5 py-1 text-xs bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded text-slate-400 transition flex items-center space-x-1.5";
  } else {
    if (icon) icon.className = "fa-solid fa-volume-high text-emerald-400";
    if (lbl) lbl.innerText = "ARMED";
    if (btn) btn.className = "px-2.5 py-1 text-xs bg-emerald-950/80 border border-emerald-700 rounded text-emerald-300 transition flex items-center space-x-1.5";
    playTone(880, 0.1, "sine");
  }
}

function playTone(freq, durationSec, type = "sine") {
  if (isAudioMuted || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + durationSec);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + durationSec);
  } catch (e) {}
}

function playMasterCautionSound() {
  const now = Date.now();
  if (now - lastAudioAlertTime < 2500) return;
  lastAudioAlertTime = now;
  playTone(650, 0.15, "triangle");
  setTimeout(() => playTone(850, 0.20, "triangle"), 160);
}

function playEmergencyAlarmSound() {
  const now = Date.now();
  if (now - lastAudioAlertTime < 2000) return;
  lastAudioAlertTime = now;
  playTone(950, 0.2, "sawtooth");
  setTimeout(() => playTone(750, 0.2, "sawtooth"), 220);
}

// ------------------------------------------------------------------
// 3. TELEMETRY STREAM CONNECTION (WS + SSE FALLBACK)
// ------------------------------------------------------------------
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      updateBadge("badge-connection", "TELEMETRY LIVE (20 Hz WS)", "bg-emerald-950/80", "border-emerald-800", "text-emerald-400");
    };

    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        updateDashboard(state);
      } catch (err) {}
    };

    ws.onerror = () => startSSEStream();
    ws.onclose = () => startSSEStream();
  } catch (e) {
    startSSEStream();
  }
}

let sseSource = null;
let pollInterval = null;

function startSSEStream() {
  if (sseSource || pollInterval) return;

  if (window.EventSource) {
    try {
      sseSource = new EventSource('/api/stream');
      sseSource.onopen = () => {
        updateBadge("badge-connection", "TELEMETRY LIVE (20 Hz SSE)", "bg-emerald-950/80", "border-emerald-800", "text-emerald-400");
      };
      sseSource.onmessage = (e) => {
        try {
          const state = JSON.parse(e.data);
          updateDashboard(state);
        } catch (err) {}
      };
      sseSource.onerror = () => {
        if (sseSource) {
          sseSource.close();
          sseSource = null;
        }
        startPolling();
      };
      return;
    } catch (err) {}
  }
  startPolling();
}

function startPolling() {
  if (pollInterval) return;
  updateBadge("badge-connection", "TELEMETRY LIVE (20 Hz HTTP)", "bg-emerald-950/80", "border-emerald-800", "text-emerald-400");
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/state');
      if (res.ok) {
        const state = await res.json();
        updateDashboard(state);
      }
    } catch (e) {}
  }, 50);
}

// ------------------------------------------------------------------
// 4. REAL-TIME DATA PROPAGATION & COMPONENT SYNCHRONIZATION
// ------------------------------------------------------------------
function updateDashboard(state) {
  if (!state || state.status === "initializing") return;

  // 1. Mission Context Ribbon
  const totalSec = Math.floor(state.timestamp_s || 0);
  const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
  const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
  const secs = String(totalSec % 60).padStart(2, '0');
  setText("txt-time", `${hrs}:${mins}:${secs}`);
  setText("txt-phase", state.mission_phase || "CRUISE / LOITER");
  setText("txt-alt", `${Number(state.altitude_ft || 28500).toLocaleString()} FT`);
  setText("txt-tas", `${(state.airspeed_mps || 55.0).toFixed(1)} M/S`);
  setText("txt-oat", `${(state.ambient_temp_c || -20.0).toFixed(1)} °C`);
  setText("txt-throttle", `${(state.throttle_pct || 72.0).toFixed(1)} %`);
  setText("txt-power-hp", `POWER: ${(state.mvem_expected_power_hp || 150.0).toFixed(1)} HP`);

  // UI-only aerospace performance readouts
  const cht = Array.isArray(state.sensor_cht_c) ? state.sensor_cht_c : [0];
  const egt = Array.isArray(state.sensor_egt_c) ? state.sensor_egt_c : [0];
  setText("txt-rpm", `${Number(state.sensor_rpm || 0).toFixed(0)}`);
  setText("txt-cht", `${Number(cht.reduce((a,b)=>a+Number(b||0),0)/Math.max(cht.length,1)).toFixed(0)} °C`);
  setText("txt-egt", `${Number(egt.reduce((a,b)=>a+Number(b||0),0)/Math.max(egt.length,1)).toFixed(0)} °C`);
  setText("txt-fuel", `${Number(state.sensor_fuel_flow_lph || 0).toFixed(0)} L/H`);
  setText("txt-oil-pressure", `${Number(state.sensor_oil_p_bar || 0).toFixed(2)} BAR`);
  setText("txt-vibration", `${Number(state.sensor_vib_rms_g || 0).toFixed(2)} G`);
  setText("txt-load", `${Number(state.mvem_expected_power_hp || 0).toFixed(1)} HP`);
  setText("txt-twin-sync", "100 %");

  setText("detail-confidence", `${Number(state.ai_prognostics?.fault_confidence_pct ?? 99.8).toFixed(1)} %`);
  setText("detail-rul", `${Number(state.ai_prognostics?.rul_hours_mean ?? 500).toFixed(0)} HRS`);
  setText("detail-action", state.ai_prognostics?.recommended_action || "CONTINUE NOMINAL MISSION PROFILE");

  // Keep the selected component detail synchronized with the same twin state
  // used by the 3D highlight. Cylinder values are component-specific; other
  // selections fall back to their subsystem health and live sensor channels.
  const selectedComponent = window.selectedComponentKey || "";
  const selectedIndex = /^cylinder([1-4])$/.test(selectedComponent)
    ? Number(selectedComponent.slice(-1)) - 1 : -1;
  const selectedHealth = selectedIndex >= 0
    ? Number((state.health?.cylinder_health || [100, 100, 100, 100])[selectedIndex] ?? 100)
    : selectedComponent === "turbo"
      ? Number(state.health?.turbo_boost_health ?? 100)
      : selectedComponent === "crankshaft"
        ? Number(state.health?.vibration_health ?? 100)
        : Number(state.health?.overall_health ?? 100);
  const selectedTemp = selectedIndex >= 0
    ? Number(egt[selectedIndex] || 0)
    : Number(egt.reduce((a, b) => a + Number(b || 0), 0) / Math.max(egt.length, 1));
  setText("detail-health", selectedComponent ? `${selectedHealth.toFixed(1)} %` : "-- %");
  setText("detail-temp", selectedComponent ? `${selectedTemp.toFixed(0)} °C` : "-- °C");
  setText("detail-vibration", selectedComponent ? `${Number(state.sensor_vib_rms_g || 0).toFixed(2)} G` : "-- G");

  // 2. Subsystem Health Matrix
  const h = state.health || {};
  const overall = h.overall_health !== undefined ? h.overall_health : 100.0;
  setText("val-health-overall", `${overall.toFixed(1)} %`);
  setText("val-health-oil", `${(h.oil_system_health !== undefined ? h.oil_system_health : 100).toFixed(1)} %`);
  setText("val-health-turbo", `${(h.turbo_boost_health !== undefined ? h.turbo_boost_health : 100).toFixed(1)} %`);
  setText("val-health-cyls", `${(h.combustion_health !== undefined ? h.combustion_health : 100).toFixed(1)} %`);
  setText("val-health-vib", `${(h.vibration_health !== undefined ? h.vibration_health : 100).toFixed(1)} %`);
  setText("val-health-elec", `${(h.electrical_health !== undefined ? h.electrical_health : 100).toFixed(1)} %`);

  // Status Badge & Alarm Sound (Optimized class caching)
  const currentStatusStr = overall < 50.0 ? "CRITICAL" : (overall < 80.0 ? "WARNING" : "NOMINAL");
  if (currentStatusStr !== lastOverallStatus) {
    lastOverallStatus = currentStatusStr;
    const valOverallEl = document.getElementById("val-health-overall");
    const badgeStatus = document.getElementById("badge-status-level");

    if (currentStatusStr === "CRITICAL") {
      if (valOverallEl) valOverallEl.className = "text-base font-bold text-red-400 mt-0.5";
      if (badgeStatus) {
        badgeStatus.className = "text-[10px] px-2 py-0.5 rounded bg-red-950 border border-red-700 text-red-400 font-bold mono";
        badgeStatus.innerText = "CRITICAL";
      }
      playEmergencyAlarmSound();
    } else if (currentStatusStr === "WARNING") {
      if (valOverallEl) valOverallEl.className = "text-base font-bold text-amber-400 mt-0.5";
      if (badgeStatus) {
        badgeStatus.className = "text-[10px] px-2 py-0.5 rounded bg-amber-950 border border-amber-700 text-amber-400 font-bold mono";
        badgeStatus.innerText = "WARNING";
      }
      playMasterCautionSound();
    } else {
      if (valOverallEl) valOverallEl.className = "text-base font-bold text-emerald-400 mt-0.5";
      if (badgeStatus) {
        badgeStatus.className = "text-[10px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-700 text-emerald-400 font-bold mono";
        badgeStatus.innerText = "NOMINAL";
      }
    }
  }

  const healthClass = (value) => value < 50.0 ? "text-red-400" : (value < 80.0 ? "text-amber-400" : "text-emerald-400");
  [
    ["val-health-overall", overall],
    ["val-health-oil", h.oil_system_health !== undefined ? h.oil_system_health : 100],
    ["val-health-turbo", h.turbo_boost_health !== undefined ? h.turbo_boost_health : 100],
    ["val-health-cyls", h.combustion_health !== undefined ? h.combustion_health : 100],
    ["val-health-vib", h.vibration_health !== undefined ? h.vibration_health : 100],
    ["val-health-elec", h.electrical_health !== undefined ? h.electrical_health : 100]
  ].forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.className = `text-base font-bold ${healthClass(Number(value))} mt-0.5`;
  });

  const alertRingCore = document.getElementById("alert-ring-core");
  const alertRingLabel = document.getElementById("alert-ring-label");
  if (alertRingCore) alertRingCore.className = `gcs-alert-core alert-ring-${currentStatusStr.toLowerCase()}`;
  if (alertRingLabel) {
    alertRingLabel.innerText = `SYSTEM ${currentStatusStr}`;
    alertRingLabel.className = `gcs-alert-label alert-ring-label-${currentStatusStr.toLowerCase()}`;
  }

  // 3. Cylinder-by-Cylinder Quick Stats
  const cylH = h.cylinder_health || [100, 100, 100, 100];
  const egtArr = state.sensor_egt_c || [810, 810, 810, 810];
  for (let i = 0; i < 4; i++) {
    const elStat = document.getElementById(`txt-cyl${i+1}-stat`);
    const elTemp = document.getElementById(`txt-cyl${i+1}-temp`);
    if (elStat) {
      const statText = `${cylH[i].toFixed(0)}%`;
      if (elStat.innerText !== statText) {
        elStat.innerText = statText;
        elStat.className = cylH[i] < 50 ? "text-red-400" : (cylH[i] < 80 ? "text-amber-400" : "text-emerald-400");
      }
    }
    if (elTemp) {
      const tempText = `${egtArr[i].toFixed(0)}°C`;
      if (elTemp.innerText !== tempText) {
        elTemp.innerText = tempText;
      }
    }
  }

  // 4. AI Diagnostics & Anomaly Score
  const ai = state.ai_prognostics || {};
  const isAnom = ai.anomaly_detected || false;
  const anomIndex = ai.anomaly_score !== undefined ? ai.anomaly_score : 0.0;
  const rawMse = ai.anomaly_raw_mse !== undefined ? ai.anomaly_raw_mse : (isAnom ? 120.5 : 0.62);
  const thresh = ai.anomaly_threshold !== undefined ? ai.anomaly_threshold : 26.6;

  const badgeAnom = document.getElementById("badge-anomaly");
  const valAnomIdx = document.getElementById("val-anom-idx");
  const valRawMse = document.getElementById("val-raw-mse");

  if (isAnom || anomIndex > 0.60) {
    if (badgeAnom && badgeAnom.innerText !== "ANOMALY DETECTED") {
      badgeAnom.className = "text-[9px] px-1.5 py-0.5 rounded bg-red-950 border border-red-700 text-red-400 mono font-bold";
      badgeAnom.innerText = "ANOMALY DETECTED";
    }
    if (valAnomIdx && !valAnomIdx.classList.contains("text-red-400")) {
      valAnomIdx.className = "text-red-400 font-bold";
    }
  } else {
    if (badgeAnom && badgeAnom.innerText !== "NOMINAL") {
      badgeAnom.className = "text-[9px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 mono font-bold";
      badgeAnom.innerText = "NOMINAL";
    }
    if (valAnomIdx && !valAnomIdx.classList.contains("text-sky-400")) {
      valAnomIdx.className = "text-sky-400 font-bold";
    }
  }

  setText("val-anom-idx", `${anomIndex.toFixed(2)} / 1.00`);
  setText("val-raw-mse", `MSE: ${rawMse.toFixed(1)} (Thresh: ${thresh.toFixed(1)})`);

  // Diagnosed Fault Label
  const faultClassEl = document.getElementById("lbl-fault-class");
  const faultName = ai.fault_class || "HEALTHY";
  if (faultClassEl) {
    const formattedFault = faultName.replace(/_/g, ' ');
    if (faultClassEl.innerText !== formattedFault) {
      faultClassEl.innerText = formattedFault;
      if (faultName !== "HEALTHY" && faultName !== "SENSOR_FAULT_EGT3") {
        faultClassEl.className = "text-xs font-bold text-red-200 mono bg-red-950/80 px-2.5 py-1.5 rounded border border-red-700 tracking-wide";
      } else if (faultName === "SENSOR_FAULT_EGT3") {
        faultClassEl.className = "text-xs font-bold text-sky-200 mono bg-sky-950/80 px-2.5 py-1.5 rounded border border-sky-700 tracking-wide";
      } else {
        faultClassEl.className = "text-xs font-bold text-slate-200 mono bg-slate-900/90 px-2.5 py-1.5 rounded border border-slate-800 tracking-wide";
      }
    }
  }

  setText("val-conf", `${(ai.fault_confidence_pct || 99.8).toFixed(1)}%`);
  setText("val-sev", (ai.fault_severity || 0.0).toFixed(2));

  // RUL Inspection Window
  const rMin = ai.rul_hours_min !== undefined ? ai.rul_hours_min.toFixed(1) : "450.0";
  const rMax = ai.rul_hours_max !== undefined ? ai.rul_hours_max.toFixed(1) : "550.0";
  setText("txt-rul-interval", `${rMin} – ${rMax} Flight Hrs`);

  // 5. Update SHAP XAI Drawer (Cached)
  updateShapDrawer(ai.top_contributing_channels || []);

  // 6. Highlight Active Fault Button on Bottom Control Panel (Cached)
  const serverActiveFault = state.active_fault || currentActiveFault || "HEALTHY";
  updateActiveFaultButton(serverActiveFault);

  // 7. Update 3D Visualizer
  if (engine3D && typeof engine3D.updateFromTwinState === "function") {
    try {
      engine3D.updateFromTwinState(state);
    } catch (e) {}
  }

  // 8. Update Chart.js Buffers (Throttled for Zero Canvas Flutter)
  updateChartData(state);
}

// ------------------------------------------------------------------
// 5. SHAP XAI ATTRIBUTION DRAWER (ZERO-FLICKER CACHED)
// ------------------------------------------------------------------
function updateShapDrawer(shapItems) {
  const container = document.getElementById("container-shap");
  if (!container) return;

  const currentSignature = JSON.stringify(shapItems || []);
  if (currentSignature === lastShapSignature) return; // Prevent DOM re-creation if unchanged
  lastShapSignature = currentSignature;

  if (!shapItems || shapItems.length === 0) {
    container.innerHTML = `<div class="text-slate-500 text-center py-4 text-xs">All channels within normal physical bounds.</div>`;
    return;
  }

  let html = "";
  shapItems.forEach((item) => {
    const isHigher = item.is_positive !== undefined ? item.is_positive : (item.direction === "ABOVE_BASELINE");
    const barColor = isHigher ? "bg-purple-500" : "bg-sky-500";
    const dirSign = isHigher ? `+${Math.abs(item.residual_sigma || 0).toFixed(1)}σ` : `-${Math.abs(item.residual_sigma || 0).toFixed(1)}σ`;

    html += `
      <div class="bg-slate-900/90 p-1.5 rounded border border-slate-800 space-y-1">
        <div class="flex justify-between items-center text-[11px]">
          <span class="font-medium text-slate-200 truncate pr-2">${item.display_name}</span>
          <span class="font-bold text-slate-300 mono whitespace-nowrap">${item.importance_pct}% (<span class="${isHigher ? 'text-purple-400' : 'text-sky-400'}">${dirSign}</span>)</span>
        </div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div class="${barColor} h-1.5 rounded-full transition-all duration-200" style="width: ${Math.min(100, item.importance_pct)}%"></div>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ------------------------------------------------------------------
// 6. MULTI-CHANNEL CHART.JS INITIALIZATION & UPDATES
// ------------------------------------------------------------------
function initCharts() {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded yet.");
    return;
  }

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    elements: { line: { tension: 0.10, borderWidth: 1.8 }, point: { radius: 0 } },
    scales: {
      x: { display: false },
      y: {
        grid: { color: 'rgba(51, 65, 85, 0.20)' },
        ticks: { color: '#94A3B8', font: { size: 9, family: 'JetBrains Mono' } }
      }
    },
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 10, color: '#CBD5E1', font: { size: 9 } } }
    }
  };

  // Chart 1: Exhaust Gas Temperature (EGT 1-4) & Baseline
  const canvasTemp = document.getElementById('chart-temp');
  if (canvasTemp) {
    charts.temp = new Chart(canvasTemp, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'EGT Cyl 1 (°C)', borderColor: '#FB923C', backgroundColor: 'transparent', data: [] },
          { label: 'EGT Cyl 2 (°C)', borderColor: '#38BDF8', backgroundColor: 'transparent', data: [] },
          { label: 'EGT Cyl 3 (°C)', borderColor: '#34D399', backgroundColor: 'transparent', data: [] },
          { label: 'EGT Cyl 4 (°C)', borderColor: '#FBBF24', backgroundColor: 'transparent', data: [] },
          { label: 'Expected MVEM Baseline', borderColor: '#94A3B8', borderDash: [4, 4], backgroundColor: 'transparent', data: [] }
        ]
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            ...commonOptions.scales.y,
            min: 0,
            max: 1000,
            ticks: {
              color: '#94A3B8',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: (v) => v + '°C'
            }
          }
        }
      }
    });
  }

  // Chart 2: Oil Pressure (bar) & Manifold Absolute Pressure (MAP bar)
  const canvasPressures = document.getElementById('chart-pressures');
  if (canvasPressures) {
    charts.pressures = new Chart(canvasPressures, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Oil Pressure (bar)', borderColor: '#10B981', backgroundColor: 'transparent', data: [] },
          { label: 'Oil P Baseline', borderColor: '#6EE7B7', borderDash: [4, 4], backgroundColor: 'transparent', data: [] },
          { label: 'Manifold MAP (bar)', borderColor: '#06B6D4', backgroundColor: 'transparent', data: [] },
          { label: 'MAP Baseline', borderColor: '#67E8F9', borderDash: [4, 4], backgroundColor: 'transparent', data: [] }
        ]
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            ...commonOptions.scales.y,
            min: 0,
            max: 7.0,
            ticks: {
              color: '#94A3B8',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: (v) => v + ' bar'
            }
          }
        }
      }
    });
  }

  // Chart 3: Vibration Spectrum RMS (g) & Output Power (HP)
  const canvasVib = document.getElementById('chart-vibration');
  if (canvasVib) {
    charts.vibration = new Chart(canvasVib, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Vibration 2X RMS (g)', borderColor: '#C084FC', yAxisID: 'y', backgroundColor: 'transparent', data: [] },
          { label: 'Engine Power (HP)', borderColor: '#38BDF8', borderDash: [3, 3], yAxisID: 'y1', backgroundColor: 'transparent', data: [] }
        ]
      },
      options: {
        ...commonOptions,
        scales: {
          x: commonOptions.scales.x,
          y: {
            ...commonOptions.scales.y,
            min: 0,
            max: 6.0,
            ticks: {
              color: '#C084FC',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: (v) => v + 'g'
            }
          },
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            min: 0,
            max: 220,
            ticks: {
              color: '#38BDF8',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: (v) => v + ' HP'
            }
          }
        }
      }
    });
  }
}

function updateChartData(state) {
  if (!state) return;
  const timeLabel = `${(state.timestamp_s || 0).toFixed(1)}s`;
  historyData.time.push(timeLabel);

  const egts = state.sensor_egt_c || [810, 810, 810, 810];
  const mvemEgt = (state.mvem_expected_egt_c && state.mvem_expected_egt_c[0]) || 810;

  historyData.sensor_egt1.push(egts[0]);
  historyData.sensor_egt2.push(egts[1]);
  historyData.sensor_egt3.push(egts[2]);
  historyData.sensor_egt4.push(egts[3]);
  historyData.mvem_egt.push(mvemEgt);

  historyData.sensor_oil_p.push(state.sensor_oil_pressure_bar !== undefined ? state.sensor_oil_pressure_bar : 4.2);
  historyData.mvem_oil_p.push(state.mvem_expected_oil_pressure_bar || 4.2);
  historyData.sensor_map.push(state.sensor_manifold_pressure_bar !== undefined ? state.sensor_manifold_pressure_bar : 2.45);
  historyData.mvem_map.push(state.mvem_expected_manifold_pressure_bar || 2.45);

  historyData.sensor_vib.push(state.sensor_vibration_rms_g !== undefined ? state.sensor_vibration_rms_g : 0.48);
  historyData.mvem_vib.push(state.mvem_expected_power_hp || 150.0);

  while (historyData.time.length > MAX_HISTORY) {
    historyData.time.shift();
    historyData.sensor_egt1.shift();
    historyData.sensor_egt2.shift();
    historyData.sensor_egt3.shift();
    historyData.sensor_egt4.shift();
    historyData.mvem_egt.shift();
    historyData.sensor_oil_p.shift();
    historyData.mvem_oil_p.shift();
    historyData.sensor_map.shift();
    historyData.mvem_map.shift();
    historyData.sensor_vib.shift();
    historyData.mvem_vib.shift();
  }

  // Smooth Throttled Chart Update (10 Hz)
  const now = Date.now();
  if (now - lastChartUpdateTime >= CHART_UPDATE_INTERVAL_MS) {
    lastChartUpdateTime = now;

    if (charts.temp && charts.temp.data && charts.temp.data.datasets.length >= 5) {
      charts.temp.data.labels = historyData.time;
      charts.temp.data.datasets[0].data = historyData.sensor_egt1;
      charts.temp.data.datasets[1].data = historyData.sensor_egt2;
      charts.temp.data.datasets[2].data = historyData.sensor_egt3;
      charts.temp.data.datasets[3].data = historyData.sensor_egt4;
      charts.temp.data.datasets[4].data = historyData.mvem_egt;
      charts.temp.update('none');
    }

    if (charts.pressures && charts.pressures.data && charts.pressures.data.datasets.length >= 4) {
      charts.pressures.data.labels = historyData.time;
      charts.pressures.data.datasets[0].data = historyData.sensor_oil_p;
      charts.pressures.data.datasets[1].data = historyData.mvem_oil_p;
      charts.pressures.data.datasets[2].data = historyData.sensor_map;
      charts.pressures.data.datasets[3].data = historyData.mvem_map;
      charts.pressures.update('none');
    }

    if (charts.vibration && charts.vibration.data && charts.vibration.data.datasets.length >= 2) {
      charts.vibration.data.labels = historyData.time;
      charts.vibration.data.datasets[0].data = historyData.sensor_vib;
      charts.vibration.data.datasets[1].data = historyData.mvem_vib;
      charts.vibration.update('none');
    }
  }
}

// -------------------------------------------------------------
// FLIGHT SANDBOX & MANUAL OVERRIDE CONTROLS
// -------------------------------------------------------------
let isManualSandbox = false;

function toggleManualSandbox() {
  isManualSandbox = !isManualSandbox;
  const btn = document.getElementById("btn-sandbox-toggle");
  if (btn) {
    if (isManualSandbox) {
      btn.className = "px-2 py-0.5 rounded text-[10px] bg-amber-950 hover:bg-amber-900 text-amber-300 border border-amber-700 font-bold";
      btn.innerText = "[Mode: MANUAL SANDBOX]";
      addEventLog("SYS", "Switched to Manual Flight Sandbox Mode.");
    } else {
      btn.className = "px-2 py-0.5 rounded text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700";
      btn.innerText = "[Mode: AUTO MISSION]";
      addEventLog("SYS", "Restored Auto Standard Mission Flight Profile.");
    }
  }
  onFlightControlChange();
}

function onFlightControlChange() {
  const elThr = document.getElementById("slider-throttle");
  const elAlt = document.getElementById("slider-altitude");
  const elOat = document.getElementById("slider-oat");

  const thr = elThr ? parseFloat(elThr.value) : 72.0;
  const alt = elAlt ? parseFloat(elAlt.value) : 28500.0;
  const oat = elOat ? parseFloat(elOat.value) : -20.0;

  setText("txt-slider-throttle", `${thr.toFixed(0)}%`);
  setText("txt-slider-altitude", `${(alt/1000).toFixed(1)}k FT`);
  setText("txt-slider-oat", `${oat.toFixed(0)}°C`);

  if (isManualSandbox) {
    fetch('/api/flight/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        enabled: true,
        throttle_pct: thr,
        altitude_ft: alt,
        ambient_temp_c: oat
      })
    }).catch(err => console.warn("Flight override error:", err));
  } else {
    fetch('/api/flight/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: false })
    }).catch(err => console.warn("Flight override disable error:", err));
  }
}

// ------------------------------------------------------------------
// 7. FAULT INJECTION & REAL-TIME EVENT HANDLERS
// ------------------------------------------------------------------
async function injectFault(faultType) {
  playTone(700, 0.08, "sine");
  // Immediate optimistic UI highlight
  updateActiveFaultButton(faultType);

  try {
    if (faultType === 'HEALTHY') {
      await fetch('/api/fault/clear', { method: 'POST' });
      addEventLog("FLT", "Engine reset to HEALTHY baseline. All trims normalized.");
    } else {
      const res = await fetch('/api/fault/inject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fault_type: faultType, severity: 1.0, ramp_duration_s: 1.0 })
      });
      const data = await res.json();
      addEventLog("FLT", `Injected: ${faultType.replace(/_/g, ' ')} (Ramp: 1.0s)`);
    }
  } catch (e) {
    console.error("Fault injection error:", e);
    addEventLog("ERR", `Fault injection network error: ${e.message}`);
  }
}

function updateActiveFaultButton(activeFaultName) {
  if (activeFaultName === lastActiveFaultButton) return; // Prevent continuous class manipulation
  lastActiveFaultButton = activeFaultName;
  currentActiveFault = activeFaultName;

  const statusTxt = document.getElementById("txt-active-status");

  // Reset all buttons to neutral state
  const allBtns = document.querySelectorAll(".btn-fault");
  allBtns.forEach(btn => {
    btn.classList.remove("active-fault", "active-sensor", "bg-emerald-950", "border-emerald-700", "text-emerald-300");
    btn.classList.add("bg-slate-900/90", "border-slate-800", "text-slate-300");
  });

  const activeBtn = document.getElementById(`btn-flt-${activeFaultName}`);
  if (activeBtn) {
    activeBtn.classList.remove("bg-slate-900/90", "border-slate-800", "text-slate-300");
    if (activeFaultName === "HEALTHY") {
      activeBtn.classList.add("bg-emerald-950", "border-emerald-700", "text-emerald-300");
      if (statusTxt) {
        statusTxt.className = "text-[10px] mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800";
        statusTxt.innerText = "STATUS: HEALTHY BASELINE";
      }
    } else if (activeFaultName === "SENSOR_FAULT_EGT3") {
      activeBtn.classList.add("active-sensor");
      if (statusTxt) {
        statusTxt.className = "text-[10px] mono text-sky-400 bg-sky-950/80 px-2 py-0.5 rounded border border-sky-800";
        statusTxt.innerText = "STATUS: SENSOR PROBE DEFECT (EGT3) ACTIVE";
      }
    } else {
      activeBtn.classList.add("active-fault");
      if (statusTxt) {
        statusTxt.className = "text-[10px] mono text-red-400 bg-red-950/80 px-2 py-0.5 rounded border border-red-800";
        statusTxt.innerText = `STATUS: ${activeFaultName.replace(/_/g, ' ')} ACTIVE`;
      }
    }
  }
}

// ------------------------------------------------------------------
// 8. EVENT TIMELINE LOGGER
// ------------------------------------------------------------------
function addEventLog(tag, message) {
  const timestamp = new Date().toTimeString().split(' ')[0];
  eventLogs.unshift({ timestamp, tag, message });
  if (eventLogs.length > maxEventLogs) eventLogs.pop();
  window.eventLogs = eventLogs;

  const container = document.getElementById("container-logs");
  const countEl = document.getElementById("txt-log-count");
  if (countEl) countEl.innerText = `${eventLogs.length} events`;

  if (!container) return;
  container.innerHTML = eventLogs.map(log => {
    let tagColor = "text-emerald-400";
    if (log.tag === "FLT") tagColor = "text-amber-400";
    if (log.tag === "ERR") tagColor = "text-red-400";
    if (log.tag === "AI") tagColor = "text-sky-400";
    return `<div><span class="text-slate-500">[${log.timestamp}]</span> <span class="${tagColor} font-bold">[${log.tag}]</span> ${log.message}</div>`;
  }).join('');
}

// ------------------------------------------------------------------
// 9. AUTOMATED 4-PHASE STAGE DEMONSTRATION CONTROLLER
// ------------------------------------------------------------------
const autoDemoPhases = [
  { fault: "HEALTHY", duration: 6, title: "Phase 1/4: Baseline Nominal Cruise Tracking (100% Health)" },
  { fault: "SENSOR_FAULT_EGT3", duration: 8, title: "Phase 2/4: Thermocouple Probe Defect Decoupling (No Abort)" },
  { fault: "OIL_PRESSURE_LOSS", duration: 10, title: "Phase 3/4: Critical Oil Pressure Loss & SHAP Root-Cause" },
  { fault: "HEALTHY", duration: 6, title: "Phase 4/4: Recovery to Nominal Healthy Cruise Baseline" }
];

function toggleAutoDemo() {
  if (autoDemoActive) {
    stopAutoDemo();
  } else {
    startAutoDemo();
  }
}

function startAutoDemo() {
  autoDemoActive = true;
  autoDemoStep = 0;
  const bar = document.getElementById("box-autodemo-status");
  const btn = document.getElementById("btn-autodemo");
  if (bar) { bar.classList.remove("hidden"); bar.classList.add("flex"); }
  if (btn) { btn.classList.add("bg-amber-950", "border-amber-700", "text-amber-300"); }
  addEventLog("SYS", "Automated 4-Phase Stage Showcase started.");
  runNextDemoPhase();
}

function runNextDemoPhase() {
  if (!autoDemoActive) return;
  if (autoDemoStep >= autoDemoPhases.length) {
    stopAutoDemo();
    addEventLog("SYS", "Automated Showcase completed successfully.");
    return;
  }

  const phase = autoDemoPhases[autoDemoStep];
  injectFault(phase.fault);
  autoDemoTimeRemaining = phase.duration;

  const txtPhase = document.getElementById("txt-autodemo-phase");
  if (txtPhase) txtPhase.innerText = `${phase.title} (${autoDemoTimeRemaining}s)`;

  if (autoDemoTimer) clearInterval(autoDemoTimer);
  autoDemoTimer = setInterval(() => {
    autoDemoTimeRemaining--;
    if (txtPhase) txtPhase.innerText = `${phase.title} (${autoDemoTimeRemaining}s)`;
    if (autoDemoTimeRemaining <= 0) {
      clearInterval(autoDemoTimer);
      autoDemoStep++;
      runNextDemoPhase();
    }
  }, 1000);
}

function stopAutoDemo() {
  autoDemoActive = false;
  if (autoDemoTimer) clearInterval(autoDemoTimer);
  const bar = document.getElementById("box-autodemo-status");
  const btn = document.getElementById("btn-autodemo");
  if (bar) { bar.classList.remove("flex"); bar.classList.add("hidden"); }
  if (btn) { btn.classList.remove("bg-amber-950", "border-amber-700", "text-amber-300"); }
  injectFault("HEALTHY");
  addEventLog("SYS", "Automated Showcase stopped.");
}

// ------------------------------------------------------------------
// 10. SIMULATION CONTROLS (SPEED & CAMERA)
// ------------------------------------------------------------------
function setSimSpeed(speed) {
  fetch(`/api/sim/speed?speed=${speed}`, { method: 'POST' }).catch(e => {});
  ['1', '2', '5'].forEach(s => {
    const btn = document.getElementById(`btn-spd-${s}`);
    if (btn) {
      if (parseFloat(s) === speed) {
        btn.className = "px-2 py-0.5 bg-sky-950 border border-sky-700 text-sky-300 rounded font-bold";
      } else {
        btn.className = "px-2 py-0.5 bg-slate-800 border border-slate-700 text-slate-300 rounded";
      }
    }
  });
  addEventLog("SYS", `Simulation speed set to ${speed}x.`);
}

function setCameraPreset(preset) {
  if (engine3D && typeof engine3D.setCameraPreset === "function") {
    engine3D.setCameraPreset(preset);
  }
}

function resetSimulation() {
  fetch('/api/sim/reset', { method: 'POST' }).catch(e => {});
  injectFault('HEALTHY');
  addEventLog("SYS", "Simulation reset to initial flight condition.");
}

function updateBadge(id, text, bgClass, borderClass, textClass) {
  const el = document.getElementById(id);
  if (el) {
    el.className = `flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs mono ${bgClass} border ${borderClass} ${textClass}`;
    const lbl = document.getElementById("lbl-connection");
    if (lbl) lbl.innerText = text;
  }
}
