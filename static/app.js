const $ = id => document.getElementById(id);

const state = {
  camera: false,
  demo: true,
  started: Date.now(),
  fatigue: 27,
  distraction: 12,
  history: Array.from({length: 60}, (_, i) => 22 + Math.sin(i / 7) * 5 + Math.random() * 3)
};

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}
function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}
function updateClock() {
  setText("clock", new Date().toLocaleTimeString([], {hour12:false}));
  const elapsed = Math.floor((Date.now() - state.started) / 1000);
  const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  setText("session-time", `${h}:${m}:${s}`);
}
setInterval(updateClock, 250);
updateClock();

async function startCamera() {
  if (state.camera) {
    video.srcObject?.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    state.camera = false;
    placeholder.style.display = "flex";
    cameraBtn.textContent = "START CAMERA";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: 1280}, height: {ideal: 720}, facingMode: "user"},
      audio: false
    });
    video.srcObject = stream;
    state.camera = true;
    placeholder.style.display = "none";
    cameraBtn.textContent = "STOP CAMERA";
    addLog("CAMERA STREAM ACTIVE", true);
  } catch (error) {
    addLog("CAMERA PERMISSION DENIED", false);
    alert("Camera access was not granted. Allow camera permission and try again.");
  }
}

const video = $("webcam");
const placeholder = $("camera-placeholder");
const cameraBtn = $("camera-btn");
const demoBtn = $("demo-btn");

cameraBtn.addEventListener("click", startCamera);

demoBtn.addEventListener("click", () => {
  state.demo = !state.demo;
  demoBtn.textContent = `DEMO TELEMETRY: ${state.demo ? "ON" : "OFF"}`;
  demoBtn.classList.toggle("active", state.demo);
  addLog(state.demo ? "DEMO TELEMETRY ENABLED" : "DEMO TELEMETRY PAUSED", state.demo);
});

function updateGauge(id, value) {
  const gauge = $(id);
  const degrees = clamp(value, 0, 100) * 3.6;
  const color = gauge.classList.contains("cyan") ? "var(--cyan)" : "var(--green)";
  gauge.style.background = `conic-gradient(${color} 0deg ${degrees}deg, var(--line) ${degrees}deg 360deg)`;
}

function setAlert(id, value, threshold = 60) {
  const box = $(id);
  const stateEl = box.querySelector("strong");
  if (value >= 75) {
    box.className = "alert-box warning";
    stateEl.textContent = "WARNING";
  } else if (value >= threshold) {
    box.className = "alert-box";
    box.style.borderColor = "var(--amber)";
    stateEl.style.color = "var(--amber)";
    stateEl.textContent = "ELEVATED";
  } else {
    box.className = "alert-box";
    box.style.borderColor = "";
    stateEl.style.color = "var(--green)";
    stateEl.textContent = "NORMAL";
  }
}

function updateBar(id, value) {
  const el = $(id);
  if (el) el.style.width = `${clamp(Math.abs(value), 0, 20) / 20 * 100}%`;
}

function randomTelemetry() {
  if (!state.demo) return;

  const t = Date.now() / 1000;

  state.fatigue = clamp(
    state.fatigue + (Math.sin(t / 8) * 0.7) + (Math.random() - 0.5) * 2,
    12, 62
  );

  state.distraction = clamp(
    state.distraction + (Math.sin(t / 5) * 0.8) + (Math.random() - 0.5) * 2,
    4, 42
  );

  const ear = 0.29 + Math.sin(t / 3.2) * 0.025 + Math.random() * 0.012;
  const perclos = 7.2 + Math.sin(t / 6) * 1.8 + Math.random() * 0.7;
  const blink = Math.round(13 + Math.sin(t / 4) * 3 + Math.random() * 2);
  const roll = Math.sin(t / 5) * 4;
  const pitch = Math.cos(t / 6) * 3;
  const yaw = Math.sin(t / 4.2) * 5;
  const mar = 0.16 + Math.abs(Math.sin(t / 7)) * 0.08;
  const gaze = Math.abs(yaw) > 4 ? "LEFT / RIGHT" : "CENTER";
  const gru = 0.35 + Math.sin(t / 9) * 0.13 + Math.random() * 0.04;

  setText("ear", ear.toFixed(2));
  setText("strip-ear", ear.toFixed(2));
  setText("perclos", `${perclos.toFixed(1)}%`);
  setText("blink", `${blink}/m`);
  setText("gaze", gaze);
  setText("strip-gaze", gaze);
  setText("roll", `${roll.toFixed(1)}°`);
  setText("pitch", `${pitch.toFixed(1)}°`);
  setText("yaw", `${yaw.toFixed(1)}°`);
  setText("strip-roll", `${roll.toFixed(1)}°`);
  setText("strip-pitch", `${pitch.toFixed(1)}°`);
  setText("strip-yaw", `${yaw.toFixed(1)}°`);
  setText("mar", mar.toFixed(2));
  setText("gru-z", gru.toFixed(2));

  const fps = 29.4 + Math.random() * 1.2;
  setText("fps", fps.toFixed(1));
  setText("hud-fps", fps.toFixed(1));

  updateBar("roll-bar", Math.abs(roll));
  updateBar("pitch-bar", Math.abs(pitch));
  updateBar("yaw-bar", Math.abs(yaw));

  setText("fatigue-value", `${Math.round(state.fatigue)}%`);
  setText("distraction-value", `${Math.round(state.distraction)}%`);
  updateGauge("fatigue-gauge", state.fatigue);
  updateGauge("distraction-gauge", state.distraction);
  setAlert("fatigue-alert", state.fatigue);
  setAlert("distraction-alert", state.distraction);

  if (state.fatigue >= 60) {
    setText("risk-message", "ELEVATED RISK · ATTENTION REQUIRED");
    $("risk-message").style.color = "var(--amber)";
  } else {
    setText("risk-message", "LOW RISK · CONTINUE MONITORING");
    $("risk-message").style.color = "var(--green)";
  }

  const eventValues = [
    Math.max(0, Math.round(perclos / 4)),
    state.fatigue > 50 ? 1 : 0,
    Math.max(1, Math.round(Math.abs(yaw) / 2)),
    Math.max(0, Math.round(Math.abs(yaw) / 3))
  ];

  ["long-blink","micro","gaze-rate","head-rate"].forEach((id, i) => setText(id, eventValues[i]));
  ["long-blink-bar","micro-bar","gaze-bar","head-bar"].forEach((id, i) => {
    const bar = $(id);
    if (bar) bar.style.width = `${Math.min(eventValues[i] * 20, 100)}%`;
  });

  state.history.push(state.fatigue);
  if (state.history.length > 60) state.history.shift();
  drawChart();
}

function drawChart() {
  const canvas = $("fatigue-chart");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(88,223,255,.08)";
  ctx.lineWidth = 1;

  for (let i = 1; i < 5; i++) {
    const y = (h / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#58dfff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  state.history.forEach((value, i) => {
    const x = (i / (state.history.length - 1)) * w;
    const y = h - (value / 100) * (h - 10) - 5;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();

  const last = state.history[state.history.length - 1];
  const x = w - 2;
  const y = h - (last / 100) * (h - 10) - 5;
  ctx.fillStyle = "#58dfff";
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fill();
}

function addLog(message, ok = false) {
  const log = $("log");
  const row = document.createElement("div");
  const time = new Date().toLocaleTimeString([], {hour12:false});
  row.innerHTML = `<time>${time}</time><span class="${ok ? "ok" : ""}">${message}</span>`;
  log.prepend(row);
  while (log.children.length > 7) log.lastElementChild.remove();
  setText("log-count", log.children.length);
}

$("reset-btn").addEventListener("click", () => {
  state.started = Date.now();
  state.fatigue = 27;
  state.distraction = 12;
  state.history = Array.from({length: 60}, (_, i) => 22 + Math.sin(i / 7) * 5 + Math.random() * 3);
  addLog("DEMO RESET", true);
  drawChart();
});

window.addEventListener("resize", drawChart);
window.addEventListener("beforeunload", () => {
  video.srcObject?.getTracks().forEach(track => track.stop());
});

setText("hud-quality", "GOOD");
setText("hud-face", "LOCKED");
setText("quality", "GOOD");
setText("face", "LOCKED");
updateGauge("fatigue-gauge", state.fatigue);
updateGauge("distraction-gauge", state.distraction);
drawChart();
setInterval(randomTelemetry, 700);

/* Lightweight space-particle field.
   Particles drift slowly and are locally pushed by the cursor. */
const particleCanvas = $("particle-field");
const particleContext = particleCanvas.getContext("2d");
const particles = [];
const pointer = {x: -1000, y: -1000, active: false};

function resizeParticleCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  particleCanvas.width = Math.floor(window.innerWidth * dpr);
  particleCanvas.height = Math.floor(window.innerHeight * dpr);
  particleContext.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function seedParticles() {
  particles.length = 0;
  const count = Math.min(500, Math.max(250, Math.floor((window.innerWidth * window.innerHeight) / 4500)));

  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.12,
      vy: (Math.random() - 0.5) * 0.12,
      size: Math.random() * 1.15 + 0.35,
      alpha: Math.random() * 0.55 + 0.18
    });
  }
}

function animateParticles() {
  const ctx = particleContext;
  const width = window.innerWidth;
  const height = window.innerHeight;

  ctx.clearRect(0, 0, width, height);

  for (const particle of particles) {
    if (pointer.active) {
      const dx = particle.x - pointer.x;
      const dy = particle.y - pointer.y;
      const distance = Math.hypot(dx, dy);
      const radius = 125;

      if (distance < radius && distance > 0.1) {
        const force = (1 - distance / radius) * 0.018;
        particle.vx += (dx / distance) * force;
        particle.vy += (dy / distance) * force;

        const tangent = 0.010 * (1 - distance / radius);
        particle.vx += (-dy / distance) * tangent;
        particle.vy += (dx / distance) * tangent;
      }
    }

    particle.vx += (Math.random() - 0.5) * 0.002;
    particle.vy += (Math.random() - 0.5) * 0.002;
    particle.vx *= 0.992;
    particle.vy *= 0.992;

    particle.x += particle.vx;
    particle.y += particle.vy;

    if (particle.x < -4) particle.x = width + 4;
    if (particle.x > width + 4) particle.x = -4;
    if (particle.y < -4) particle.y = height + 4;
    if (particle.y > height + 4) particle.y = -4;

    const centerX = width / 2;
    const centerY = height * 0.51;
    const distanceFromCenter = Math.hypot(particle.x - centerX, particle.y - centerY);
    const fade = clamp((distanceFromCenter - Math.min(width, height) * 0.16) / (Math.min(width, height) * 0.42), 0, 1);

    ctx.beginPath();
    ctx.fillStyle = `rgba(255,255,255,${particle.alpha * (0.25 + fade * 0.75)})`;
    ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
    ctx.fill();
  }

  requestAnimationFrame(animateParticles);
}

window.addEventListener("pointermove", event => {
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  pointer.active = true;
});

window.addEventListener("pointerleave", () => {
  pointer.active = false;
});

window.addEventListener("resize", () => {
  resizeParticleCanvas();
  seedParticles();
});

resizeParticleCanvas();
seedParticles();
animateParticles();
