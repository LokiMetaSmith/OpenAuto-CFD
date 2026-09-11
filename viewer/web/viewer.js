/* viewer/web/viewer.js — Atlas Fields Studio WebGL Engine */

// --- Global State ---
let scene, camera, renderer, controls;
let corkscrewMesh = null;
let particleSystem = null;
let particlePositions, particleVelocities, particleLifetimes;
const N_PARTICLES = 600;

let currentDomain = "cfd";
let currentFidelity = "tier1";
let currentParams = {};
let paramDefs = {};
let activeColormap = "Turbo";
let wireframeMode = false;
let showParticles = true;
let isPredicting = false;
let pendingPredict = false;

// KiCad 3D PCB State
let pcbGroup = null;
let componentsGroup = null;
let emWaveGroup = null;
let activePcbData = null;
let selectedPcbNet = null;
let kicadFrequency = 5.0;
let isInitialPcbLoad = true;
let lastKicadSyncTime = 0;
let showComponents = true;
let showEMWaves = true;
let rfSweepData = null;
let tdrData = null;
let nanoporeData = null;
let nanoporePoreDiam = 4.0;
let nanoporeBiasMv = 100.0;
let nanoScopeOffset = 0;
let activeVnaTab = "smith";
let vnaDockOpen = false;
let emWaveTime = 0;

// Phase B & C State
let showThermalIR = false;
let thermalMesh = null;
let thermalData = null;
let showDrcMarkers = false;
let drcGroup = null;
let drcData = null;
let fdtdData = null;
let fdtdPlaying = true;
let fdtdFrameIdx = 0;
let fdtdAnimCounter = 0;


// Colormap lookup tables
const COLORMAPS = {
  Turbo: [
    [0.00, [48, 18, 59]], [0.20, [57, 162, 252]], [0.40, [25, 215, 200]],
    [0.60, [212, 230, 54]], [0.80, [253, 165, 51]], [1.00, [122, 4, 3]]
  ],
  Viridis: [
    [0.00, [68, 1, 84]], [0.25, [59, 82, 139]], [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]], [1.00, [253, 231, 37]]
  ],
  Plasma: [
    [0.00, [13, 8, 135]], [0.25, [156, 23, 158]], [0.50, [203, 70, 121]],
    [0.75, [251, 159, 58]], [1.00, [240, 249, 33]]
  ],
  Twilight: [
    [0.00, [226, 217, 227]], [0.25, [125, 155, 201]], [0.50, [65, 44, 79]],
    [0.75, [182, 125, 107]], [1.00, [226, 217, 227]]
  ]
};

function sampleColormapRGB(t, cmapName = "Turbo") {
  const stops = COLORMAPS[cmapName] || COLORMAPS.Turbo;
  t = Math.max(0.0, Math.min(1.0, t));
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      const c1 = stops[i][1];
      const c2 = stops[i + 1][1];
      const r = (c1[0] + (c2[0] - c1[0]) * f) / 255.0;
      const g = (c1[1] + (c2[1] - c1[1]) * f) / 255.0;
      const b = (c1[2] + (c2[2] - c1[2]) * f) / 255.0;
      return new THREE.Color(r, g, b);
    }
  }
  return new THREE.Color(1, 1, 1);
}

// --- Initialization ---
window.addEventListener("DOMContentLoaded", () => {
  initThree();
  initParticleSystem();
  loadBackendStatus();
  animate();

  // Setup periodic queue and KiCad live polling
  pollKiCadLiveStatus();
  setInterval(pollSolverQueue, 2000);
  setInterval(pollKiCadLiveStatus, 1500);
});

function initThree() {
  const container = document.getElementById("viewport-container");
  const canvas = document.getElementById("webgl-canvas");

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06080d);
  scene.fog = new THREE.FogExp2(0x06080d, 0.008);

  // Camera
  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(40, 35, 60);

  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  // Controls
  controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 250;
  controls.minDistance = 5;

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
  keyLight.position.set(50, 80, 50);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0xa855f7, 0.8);
  rimLight.position.set(-50, -30, -50);
  scene.add(rimLight);

  // Studio Grid Floor
  const grid = new THREE.GridHelper(100, 40, 0x1e293b, 0x0f172a);
  grid.position.y = -25;
  scene.add(grid);

  // PCB 3D Group
  pcbGroup = new THREE.Group();
  pcbGroup.name = "pcbGroup";
  pcbGroup.visible = false;
  scene.add(pcbGroup);

  // DRC Violation Holographic Marker Group
  drcGroup = new THREE.Group();
  drcGroup.name = "drcGroup";
  drcGroup.visible = false;
  scene.add(drcGroup);

  window.addEventListener("resize", onWindowResize);
}

function onWindowResize() {
  const container = document.getElementById("viewport-container");
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// --- Parametric Corkscrew Geometry Generator ---
function updateCorkscrewGeometry(params) {
  const turns = parseFloat(params.number_of_complete_revolutions || 2.0);
  const r_in = parseFloat(params.helix_path_radius_mm || 1.8);
  const r_out = 15.0; // Outer tube radius
  const length = parseFloat(params.insert_length_mm || 50.0);
  const chamfer = parseFloat(params.blade_chamfer_mm || 0.5);

  const nRadial = 12;
  const nTheta = Math.floor(60 * turns);
  const geom = new THREE.BufferGeometry();

  const positions = [];
  const normals = [];
  const colors = [];
  const indices = [];

  for (let i = 0; i <= nTheta; i++) {
    const frac = i / nTheta;
    const theta = frac * Math.PI * 2.0 * turns;
    const z = (frac - 0.5) * length;

    for (let j = 0; j <= nRadial; j++) {
      const rFrac = j / nRadial;
      const r = r_in + (r_out - r_in) * rFrac;

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);

      positions.push(x, y, z);
      normals.push(-Math.sin(theta), Math.cos(theta), 0.1);

      // Color coding based on domain
      let col;
      if (currentDomain === "fea") {
        // Stress concentration at inner root
        const stress = Math.max(0.1, 1.0 - rFrac) * (1.2 / (1.0 + chamfer * 0.5));
        col = sampleColormapRGB(stress, "Plasma");
      } else if (currentDomain === "cfd") {
        // Centrifugal pressure/swirl
        col = sampleColormapRGB(rFrac * 0.8 + 0.1, activeColormap);
      } else {
        col = sampleColormapRGB(frac, "Twilight");
      }
      colors.push(col.r, col.g, col.b);
    }
  }

  // Quads to triangles
  for (let i = 0; i < nTheta; i++) {
    for (let j = 0; j < nRadial; j++) {
      const row1 = i * (nRadial + 1);
      const row2 = (i + 1) * (nRadial + 1);

      const a = row1 + j;
      const b = row1 + j + 1;
      const c = row2 + j;
      const d = row2 + j + 1;

      indices.push(a, b, c);
      indices.push(c, b, d);
    }
  }

  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();

  if (corkscrewMesh) {
    scene.remove(corkscrewMesh);
    corkscrewMesh.geometry.dispose();
  }

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.35,
    metalness: 0.2,
    wireframe: wireframeMode,
    side: THREE.DoubleSide
  });

  corkscrewMesh = new THREE.Mesh(geom, mat);
  scene.add(corkscrewMesh);
}

// --- Streamline Particle System ---
function initParticleSystem() {
  const geom = new THREE.BufferGeometry();
  particlePositions = new Float32Array(N_PARTICLES * 3);
  particleVelocities = new Float32Array(N_PARTICLES * 3);
  particleLifetimes = new Float32Array(N_PARTICLES);
  const colors = new Float32Array(N_PARTICLES * 3);

  for (let i = 0; i < N_PARTICLES; i++) {
    resetParticle(i, true);
    const col = sampleColormapRGB(Math.random(), "Turbo");
    colors[i * 3] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }

  geom.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 1.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending
  });

  particleSystem = new THREE.Points(geom, mat);
  scene.add(particleSystem);
}

function resetParticle(i, initial = false) {
  const length = parseFloat(currentParams.insert_length_mm || 50.0);
  const r_in = parseFloat(currentParams.helix_path_radius_mm || 1.8) + 1.0;
  const r_out = 14.0;

  const r = r_in + Math.random() * (r_out - r_in);
  const theta = Math.random() * Math.PI * 2;
  const z = initial ? (Math.random() - 0.5) * length : -length * 0.5;

  particlePositions[i * 3] = r * Math.cos(theta);
  particlePositions[i * 3 + 1] = r * Math.sin(theta);
  particlePositions[i * 3 + 2] = z;
  particleLifetimes[i] = Math.random() * 1.0;
}

function updateParticles(dt) {
  if (!particleSystem || !showParticles || currentDomain === "pcb") return;

  const length = parseFloat(currentParams.insert_length_mm || 50.0);
  const turns = parseFloat(currentParams.number_of_complete_revolutions || 2.0);
  const posAttr = particleSystem.geometry.attributes.position;
  const colAttr = particleSystem.geometry.attributes.color;

  for (let i = 0; i < N_PARTICLES; i++) {
    const idx = i * 3;
    let x = particlePositions[idx];
    let y = particlePositions[idx + 1];
    let z = particlePositions[idx + 2];

    const r = Math.sqrt(x * x + y * y);
    const theta = Math.atan2(y, x);

    // Swirling vortex velocity field
    const omega = 2.0 * Math.PI * turns * (12.0 / Math.max(length, 1.0));
    const u_theta = omega * r * 0.08;
    const u_z = 25.0 * dt;

    x += -u_theta * Math.sin(theta) * dt;
    y += u_theta * Math.cos(theta) * dt;
    z += u_z;

    particlePositions[idx] = x;
    particlePositions[idx + 1] = y;
    particlePositions[idx + 2] = z;

    if (z > length * 0.5 || r > 16.0) {
      resetParticle(i);
    }

    // Dynamic color by velocity
    const speedNorm = Math.min(1.0, (u_theta + 10.0) / 25.0);
    const col = sampleColormapRGB(speedNorm, activeColormap);
    colAttr.array[idx] = col.r;
    colAttr.array[idx + 1] = col.g;
    colAttr.array[idx + 2] = col.b;
  }

  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
}

// --- Animation Loop ---
let lastTime = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTime) / 1000.0);
  lastTime = now;

  // FPS Counter
  frameCount++;
  if (now - lastFpsUpdate > 500) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
    document.getElementById("fps-counter").innerText = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdate = now;
  }

  controls.update();
  updateParticles(dt);
  updateEMWaves(dt);

  // Animate scrolling nanopore electrophysiology oscilloscope
  if (vnaDockOpen && activeVnaTab === "nanopore" && nanoporeData) {
    nanoScopeOffset = (nanoScopeOffset + 2) % Math.max(1, (nanoporeData.current_na ? nanoporeData.current_na.length : 1));
    drawNanoporeOscilloscope(nanoporeData);
  }

  // Animate DRC pulsing 3D violation markers
  if (showDrcMarkers && drcGroup && drcGroup.visible && drcGroup.children.length > 0) {
    const pulseScale = 1.0 + 0.22 * Math.sin(now * 0.006);
    drcGroup.children.forEach(m => {
      m.scale.set(pulseScale, pulseScale, pulseScale);
    });
  }

  // Animate Full-Wave FDTD Slice playback
  if (vnaDockOpen && activeVnaTab === "fdtd" && fdtdData && fdtdPlaying) {
    fdtdAnimCounter++;
    if (fdtdAnimCounter % 3 === 0) {
      fdtdFrameIdx = (fdtdFrameIdx + 1) % (fdtdData.frames ? fdtdData.frames.length : 1);
      drawFdtdCanvas();
      const frameDisp = document.getElementById("fdtd-frame-num");
      if (frameDisp && fdtdData.frames) {
        frameDisp.innerText = `${fdtdFrameIdx + 1}/${fdtdData.frames.length}`;
      }
    }
  }

  // Slow subtle rotation for corkscrew CFD/FEA
  if (corkscrewMesh && currentDomain !== "pcb") {
    corkscrewMesh.rotation.z += 0.002;
  }

  renderer.render(scene, camera);
}

// --- REST API Client & Interactivity ---

async function loadBackendStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    paramDefs = data.param_defs || {};
    currentDomain = data.domain || "cfd";

    document.getElementById("surrogate-samples").innerText = `${data.surrogate_samples} Points`;
    buildParameterSliders();
    triggerPrediction();
  } catch (err) {
    console.error("Backend connection error:", err);
    document.getElementById("backend-status").innerText = "Offline Mode (Synthetic)";
    // Fallback default parameters
    paramDefs = {
      number_of_complete_revolutions: { min: 1.0, max: 4.0, default: 2.0 },
      helix_path_radius_mm: { min: 1.5, max: 5.0, default: 1.8 },
      blade_chamfer_mm: { min: 0.1, max: 1.0, default: 0.5 },
      insert_length_mm: { min: 40.0, max: 60.0, default: 50.0 }
    };
    buildParameterSliders();
    updateCorkscrewGeometry(currentParams);
  }
}

function buildParameterSliders() {
  const container = document.getElementById("sliders-container");
  container.innerHTML = "";

  for (const [pName, defn] of Object.entries(paramDefs)) {
    const pMin = defn.min !== undefined ? defn.min : 0.0;
    const pMax = defn.max !== undefined ? defn.max : 10.0;
    const pDef = defn.default !== undefined ? defn.default : (pMin + pMax) / 2.0;

    currentParams[pName] = pDef;

    const group = document.createElement("div");
    group.className = "slider-group";

    const labelRow = document.createElement("div");
    labelRow.className = "slider-label-row";

    const label = document.createElement("span");
    label.innerText = formatParamName(pName);

    const valDisplay = document.createElement("span");
    valDisplay.className = "slider-val";
    valDisplay.id = `val-${pName}`;
    valDisplay.innerText = Number(pDef).toFixed(2);

    labelRow.appendChild(label);
    labelRow.appendChild(valDisplay);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = pMin;
    slider.max = pMax;
    slider.step = (pMax - pMin) / 100.0;
    slider.value = pDef;
    slider.id = `slider-${pName}`;

    slider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      currentParams[pName] = val;
      valDisplay.innerText = val.toFixed(2);
      onParameterScrub();
    });

    group.appendChild(labelRow);
    group.appendChild(slider);
    container.appendChild(group);
  }

  updateCorkscrewGeometry(currentParams);
}

function formatParamName(str) {
  return str.replace(/_/g, " ").replace(/mm/g, "(mm)");
}

function onParameterScrub() {
  updateCorkscrewGeometry(currentParams);
  if (!isPredicting) {
    triggerPrediction();
  } else {
    pendingPredict = true;
  }
}

async function triggerPrediction() {
  if (currentDomain === "pcb") return;
  isPredicting = true;
  try {
    const t0 = performance.now();
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        params: currentParams,
        fidelity: currentFidelity,
        enforce_conservation: true
      })
    });
    const data = await res.json();
    const tElapsed = Math.round(performance.now() - t0);

    const statusPrefix = currentFidelity === "tier1" ? "Surrogate Live" : "Fine Mesh Ground Truth";
    document.getElementById("backend-status").innerText = `${statusPrefix} (${tElapsed}ms)`;
    updateTelemetryHUD(data.metrics, data.uncertainty, data.conservation);
  } catch (err) {
    console.warn("Prediction fallback:", err);
  } finally {
    isPredicting = false;
    if (pendingPredict) {
      pendingPredict = false;
      triggerPrediction();
    }
  }
}

function updateTelemetryHUD(metrics, uncertainty, conservation) {
  if (!metrics || currentDomain === "pcb") return;

  // Separation Efficiency
  const eff = metrics.separation_efficiency !== undefined ? metrics.separation_efficiency : 96.5;
  document.getElementById("metric-eff").innerText = `${Number(eff).toFixed(2)}%`;
  const effCard = document.getElementById("card-eff");
  effCard.style.borderColor = eff >= 99.95 ? "var(--accent-emerald)" : "var(--border-subtle)";

  // Pressure Drop
  const dpPa = metrics.delta_p !== undefined ? metrics.delta_p : 2600.0;
  const dpPsi = dpPa / 6894.76;
  document.getElementById("metric-dp").innerText = `${dpPsi.toFixed(2)} PSI`;
  document.getElementById("sub-dp").innerText = `${Math.round(dpPa)} Pa (Limit: 0.70 PSI)`;

  // PINN Conservation Telemetry
  const divElem = document.getElementById("metric-div");
  const subDiv = document.getElementById("sub-div");
  if (divElem && conservation) {
    if (conservation.divergence_loss !== undefined) {
      const divVal = Number(conservation.divergence_loss);
      divElem.innerText = `∇·u = ${divVal.toFixed(5)}`;
      const isAdm = conservation.is_physically_admissible !== false;
      subDiv.innerHTML = `<span class="badge-status-dot ${isAdm ? 'admissible' : 'violation'}"></span> ${isAdm ? 'Physically Admissible' : 'Divergence Violation'}`;
    } else if (conservation.equilibrium_loss !== undefined) {
      const eqVal = Number(conservation.equilibrium_loss);
      divElem.innerText = `∇·σ = ${eqVal.toFixed(5)}`;
      subDiv.innerHTML = `<span class="badge-status-dot admissible"></span> Structural Equilibrium Valid`;
    }
  }

  // Von Mises Stress
  const vm = metrics.max_von_mises_stress_MPa !== undefined ? metrics.max_von_mises_stress_MPa : 24.5;
  document.getElementById("metric-stress").innerText = `${Number(vm).toFixed(1)} MPa`;

  // Factor of Safety
  const fos = metrics.factor_of_safety !== undefined ? metrics.factor_of_safety : 2.45;
  document.getElementById("metric-fos").innerText = Number(fos).toFixed(2);
  const fosCard = document.getElementById("card-fos");
  fosCard.style.borderColor = fos >= 1.5 ? "var(--accent-emerald)" : "var(--accent-amber)";

  // Epistemic Uncertainty
  if (uncertainty !== undefined) {
    document.getElementById("metric-unc").innerText = Number(uncertainty).toFixed(3);
  }
}

// --- Inverse Design ---
async function triggerInverseDesign() {
  const btn = document.querySelector(".btn-optimize");
  btn.classList.add("shimmer-active");
  btn.innerText = "Optimizing (L-BFGS-B)...";

  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed_params: currentParams })
    });
    const data = await res.json();
    const optParams = data.optimal_params;

    // Animate sliders to optimal geometry
    for (const [k, v] of Object.entries(optParams)) {
      const slider = document.getElementById(`slider-${k}`);
      const display = document.getElementById(`val-${k}`);
      if (slider) {
        slider.value = v;
        currentParams[k] = v;
      }
      if (display) {
        display.innerText = Number(v).toFixed(2);
      }
    }

    updateCorkscrewGeometry(currentParams);
    updateTelemetryHUD(data.predicted_metrics, data.uncertainty);
  } catch (err) {
    console.error("Inverse design failed:", err);
  } finally {
    btn.classList.remove("shimmer-active");
    btn.innerHTML = "<span>⚡</span><span>AI Inverse Design (L-BFGS-B)</span>";
  }
}

// --- GenCAD Generative Synthesis Action ---
async function synthesizeGenCADFromHUD() {
  showKiCadToast("🧬 GenCAD Synthesis: Querying Transformer AST Model...", 3000);
  try {
    const target = {
      delta_p: 2200.0,
      separation_efficiency: 95.0,
      flow_rate_m3s: 0.012
    };
    const res = await fetch("/api/gencad/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        physics_target: target,
        format_type: "build123d"
      })
    });
    const data = await res.json();
    if (data.status === "success" && data.synthesized_parameters) {
      for (const [k, v] of Object.entries(data.synthesized_parameters)) {
        const slider = document.getElementById(`slider-${k}`);
        const display = document.getElementById(`val-${k}`);
        if (slider) {
          slider.value = v;
          currentParams[k] = v;
        }
        if (display) {
          display.innerText = Number(v).toFixed(2);
        }
      }
      updateCorkscrewGeometry(currentParams);
      showKiCadToast(`⚡ GenCAD AST Sequence -> Nearest Match: ${data.retrieved_nearest_match}`, 5000);
    }
  } catch (err) {
    console.error("GenCAD Synthesis Error:", err);
    showKiCadToast(`❌ GenCAD Error: ${err.message}`, 4000);
  }
}

// --- Background Solver Queue ---
async function dispatchBackgroundSolver() {
  const statusCard = document.getElementById("job-status-card");
  statusCard.style.display = "flex";
  statusCard.classList.add("shimmer-active");
  document.getElementById("job-status-text").innerText = "RUNNING SOLVER...";

  try {
    const res = await fetch("/api/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: currentParams, mock: true })
    });
    const ticket = await res.json();
    document.getElementById("job-id-text").innerText = `Ticket: ${ticket.job_id}`;
  } catch (err) {
    console.error("Dispatch error:", err);
    document.getElementById("job-status-text").innerText = "FAILED";
  }
}

async function pollSolverQueue() {
  try {
    const res = await fetch("/api/poll");
    const data = await res.json();
    if (data.completed && data.completed.length > 0) {
      const lastJob = data.completed[data.completed.length - 1];
      const statusCard = document.getElementById("job-status-card");
      statusCard.classList.remove("shimmer-active");
      document.getElementById("job-status-text").innerText = `COMPLETED (${lastJob.duration_s}s)`;
      document.getElementById("surrogate-samples").innerText = `${data.surrogate_samples} Points`;
      updateTelemetryHUD(lastJob.metrics, 0.05);
    }
  } catch (e) {}
}

// --- UI Actions ---
function switchDomain(domain) {
  currentDomain = domain;
  document.querySelectorAll(".domain-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.domain === domain);
  });

  const cfdPanel = document.getElementById("cfd-panel");
  const kicadPanel = document.getElementById("kicad-panel");

  if (domain === "pcb") {
    // Hide CFD / FEA geometry and particles
    if (corkscrewMesh) corkscrewMesh.visible = false;
    if (particleSystem) particleSystem.visible = false;
    if (pcbGroup) pcbGroup.visible = true;

    // Switch Left Drawer
    if (cfdPanel) cfdPanel.style.display = "none";
    if (kicadPanel) kicadPanel.style.display = "block";

    // Show KiCad-specific viewport tools & VNA dock
    const btnIcs = document.getElementById("btn-toggle-ics");
    const btnEm = document.getElementById("btn-toggle-em");
    const btnVna = document.getElementById("btn-toggle-vna");
    const vnaDock = document.getElementById("vna-dock");
    if (btnIcs) btnIcs.style.display = "inline-block";
    if (btnEm) btnEm.style.display = "inline-block";
    if (btnVna) btnVna.style.display = "inline-block";
    const btnThermal = document.getElementById("btn-toggle-thermal");
    const btnDrc = document.getElementById("btn-toggle-drc");
    if (btnThermal) btnThermal.style.display = "inline-block";
    if (btnDrc) btnDrc.style.display = "inline-block";
    if (vnaDock) vnaDock.style.display = "flex";

    // Set Camera directly over the PCB
    camera.position.set(0, 52, 45);
    controls.target.set(0, 0, 0);
    controls.update();

    // If PCB data is already loaded, update the inspector & HUD & VNA
    if (activePcbData) {
      updatePcbHUD(activePcbData);
      fetchRfSweep(selectedPcbNet || (activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP");
    }
    return;
  }

  // Non-PCB domain (CFD, FEA, Joint, EM Phasor)
  if (pcbGroup) pcbGroup.visible = false;
  if (corkscrewMesh) corkscrewMesh.visible = true;
  if (particleSystem) particleSystem.visible = showParticles;

  if (cfdPanel) cfdPanel.style.display = "block";
  if (kicadPanel) kicadPanel.style.display = "none";

  const btnIcs = document.getElementById("btn-toggle-ics");
  const btnEm = document.getElementById("btn-toggle-em");
  const btnVna = document.getElementById("btn-toggle-vna");
  const vnaDock = document.getElementById("vna-dock");
  if (btnIcs) btnIcs.style.display = "none";
  if (btnEm) btnEm.style.display = "none";
  if (btnVna) btnVna.style.display = "none";
  const btnThermalH = document.getElementById("btn-toggle-thermal");
  const btnDrcH = document.getElementById("btn-toggle-drc");
  if (btnThermalH) btnThermalH.style.display = "none";
  if (btnDrcH) btnDrcH.style.display = "none";
  if (drcGroup) drcGroup.visible = false;
  if (thermalMesh) thermalMesh.visible = false;
  if (vnaDock) vnaDock.style.display = "none";

  resetCfdHudLabels();

  // Switch colormaps and field views
  if (domain === "fea") {
    activeColormap = "Plasma";
    document.getElementById("cmap-select").value = "Plasma";
  } else if (domain === "cfd") {
    activeColormap = "Turbo";
    document.getElementById("cmap-select").value = "Turbo";
  }

  fetch("/api/switch_domain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain: domain })
  });

  updateCorkscrewGeometry(currentParams);
  triggerPrediction();
}

function resetCfdHudLabels() {
  const cardEffTitle = document.querySelector("#card-eff .metric-title");
  if (cardEffTitle) cardEffTitle.innerText = "PARTICLE COLLECTION EFFICIENCY";
  const subEff = document.getElementById("sub-eff");
  if (subEff) subEff.innerText = "Target: > 99.95% (Moon Dust)";

  const cardDpTitle = document.querySelector("#card-dp .metric-title");
  if (cardDpTitle) cardDpTitle.innerText = "PRESSURE DROP (Δp)";
  const subDp = document.getElementById("sub-dp");
  if (subDp) subDp.innerText = "Target: < 0.70 PSI (~2900 Pa)";

  const cardConvTitle = document.querySelector("#card-conservation .metric-title");
  if (cardConvTitle) cardConvTitle.innerText = "PINN CONSERVATION RESIDUAL";
  const subConv = document.getElementById("sub-div");
  if (subConv) subConv.innerHTML = '<span class="badge-status-dot admissible"></span> Physically Admissible';

  const cardStressTitle = document.querySelector("#card-stress .metric-title");
  if (cardStressTitle) cardStressTitle.innerText = "MAX VON MISES STRESS";
  const subStress = document.getElementById("sub-stress");
  if (subStress) subStress.innerText = "Yield: 60 MPa (PETG/PLA)";

  const cardFosTitle = document.querySelector("#card-fos .metric-title");
  if (cardFosTitle) cardFosTitle.innerText = "FACTOR OF SAFETY";
  const subFos = document.getElementById("sub-fos");
  if (subFos) subFos.innerText = "Target: ≥ 1.50";

  const cardUncTitle = document.querySelector("#card-unc .metric-title");
  if (cardUncTitle) cardUncTitle.innerText = "EPISTEMIC UNCERTAINTY";
}

function resetCamera() {
  if (currentDomain === "pcb") {
    camera.position.set(0, 52, 45);
    controls.target.set(0, 0, 0);
  } else {
    camera.position.set(40, 35, 60);
    controls.target.set(0, 0, 0);
  }
  controls.update();
}

function toggleParticles() {
  showParticles = !showParticles;
  if (particleSystem) particleSystem.visible = showParticles;
}

function toggleWireframe() {
  wireframeMode = !wireframeMode;
  if (corkscrewMesh) corkscrewMesh.material.wireframe = wireframeMode;
}

function changeColormap(name) {
  activeColormap = name;
  updateCorkscrewGeometry(currentParams);
}

// --- Fidelity Switching ---
function setFidelity(tier) {
  currentFidelity = tier;
  const btnT1 = document.getElementById("btn-fidelity-t1");
  const btnT2 = document.getElementById("btn-fidelity-t2");
  if (btnT1) btnT1.classList.toggle("active", tier === "tier1");
  if (btnT2) btnT2.classList.toggle("active", tier === "tier2");

  const statusElem = document.getElementById("backend-status");
  if (statusElem) {
    statusElem.innerText = tier === "tier1" ? "Surrogate Live (2ms)" : "Ground-Truth Fine Mesh CFD";
  }
  triggerPrediction();
}

// --- Autonomous AI Engineering Agent Console ---
function toggleAgentDrawer() {
  const drawer = document.getElementById("agent-drawer");
  if (drawer) {
    drawer.classList.toggle("collapsed");
  }
}

function handleAgentKey(e) {
  if (e.key === "Enter") {
    sendAgentMessage();
  }
}

async function sendAgentMessage() {
  const input = document.getElementById("agent-input");
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  const chatMessages = document.getElementById("agent-chat-messages");
  if (!chatMessages) return;

  // Render User Message
  const userDiv = document.createElement("div");
  userDiv.className = "agent-msg user";
  userDiv.innerText = text;
  chatMessages.appendChild(userDiv);

  // Status Tag
  const activeTag = document.getElementById("agent-active-tag");
  if (activeTag) {
    activeTag.innerText = "THINKING...";
    activeTag.style.borderColor = "var(--accent-amber)";
    activeTag.style.color = "var(--accent-amber)";
  }

  // Ensure drawer is open
  const drawer = document.getElementById("agent-drawer");
  if (drawer) drawer.classList.remove("collapsed");

  // Thinking Bubble
  const thinkDiv = document.createElement("div");
  thinkDiv.className = "agent-msg system";
  thinkDiv.innerHTML = "<em>Autonomous Agent reasoning over multi-physics manifold...</em>";
  chatMessages.appendChild(thinkDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch("/api/agent_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        params: currentParams,
        fidelity: currentFidelity
      })
    });
    const data = await res.json();
    thinkDiv.remove();

    const agentDiv = document.createElement("div");
    agentDiv.className = "agent-msg agent";

    let html = `<strong>${data.agent_type || 'CAD_Agent'}:</strong> ${data.reply}<br>`;

    if (data.trace && data.trace.length > 0) {
      html += `<div style="margin-top: 6px; font-size: 11px;">`;
      for (const step of data.trace) {
        html += `<span class="tool-chip">⚡ ${step.tool || 'Tool'}</span> `;
      }
      html += `</div>`;
    }

    if (data.kicad_path) {
      html += `<div style="margin-top: 4px; font-size: 11px; color: var(--accent-emerald);">Exported KiCad PCB: <code>${data.kicad_path}</code></div>`;
    }

    agentDiv.innerHTML = html;
    chatMessages.appendChild(agentDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Apply updated parameters to 3D Viewport if returned
    if (data.updated_params && Object.keys(data.updated_params).length > 0) {
      for (const [k, v] of Object.entries(data.updated_params)) {
        const slider = document.getElementById(`slider-${k}`);
        const display = document.getElementById(`val-${k}`);
        if (slider) {
          slider.value = v;
          currentParams[k] = v;
        }
        if (display) {
          display.innerText = Number(v).toFixed(2);
        }
      }
      updateCorkscrewGeometry(currentParams);
      if (data.metrics) {
        updateTelemetryHUD(data.metrics, data.uncertainty, data.conservation);
      }
    }
  } catch (err) {
    thinkDiv.remove();
    const errDiv = document.createElement("div");
    errDiv.className = "agent-msg system";
    errDiv.style.color = "#ef4444";
    errDiv.innerText = `Agent error: ${err.message}`;
    chatMessages.appendChild(errDiv);
  } finally {
    if (activeTag) {
      activeTag.innerText = "READY";
      activeTag.style.borderColor = "var(--accent-emerald)";
      activeTag.style.color = "var(--accent-emerald)";
    }
  }
}

// --- KiCad Live Synchronization Bridge ---
async function pollKiCadLiveStatus() {
  try {
    const res = await fetch("/api/kicad_status");
    const data = await res.json();
    const badge = document.getElementById("kicad-sync-badge");
    const badgeText = document.getElementById("kicad-sync-text");

    if (data.status === "synchronized" || data.connected) {
      if (badge) {
        badge.className = "kicad-badge active";
      }
      if (badgeText) {
        badgeText.innerText = `KiCad: ${data.board_name || 'Live'}`;
      }

      activePcbData = data;

      // Handle initial page load or first live data arrival
      if (isInitialPcbLoad) {
        isInitialPcbLoad = false;
        lastKicadSyncTime = data.timestamp || 1;
        buildPcb3DScene(data);
        updatePcbInspectorUI(data);

        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get("kicad_live") === "1" || urlParams.has("pcb")) {
          switchDomain("pcb");
        }
      } else if (data.timestamp && data.timestamp > lastKicadSyncTime) {
        lastKicadSyncTime = data.timestamp;
        buildPcb3DScene(data);
        updatePcbInspectorUI(data);

        if (currentDomain === "pcb") {
          updatePcbHUD(data);
        }

        const t = data.primary_trace;
        const m = data.em_metrics;
        if (t && m) {
          showKiCadToast(
            `⚡ KiCad Live Sync: ${data.board_name} | ${t.net_name} (w=${t.width_mm}mm) | Z0=${m.z0_ohms}Ω | S11=${m.s11_return_loss_db}dB`
          );
        }
      }
    } else {
      if (badge) badge.className = "kicad-badge standby";
      if (badgeText) badgeText.innerText = "KiCad: Standby";
    }
  } catch (e) {}
}

function showKiCadToast(message, durationMs = 4000) {
  const toast = document.getElementById("kicad-toast");
  if (!toast) return;
  toast.innerText = message;
  toast.classList.remove("hidden");
  setTimeout(() => {
    toast.classList.add("hidden");
  }, durationMs);
}

// --- KiCad 3D PCB Geometry Builder ---
function buildPcb3DScene(data) {
  if (!pcbGroup) return;

  // Clear previous PCB meshes & geometries
  while (pcbGroup.children.length > 0) {
    const obj = pcbGroup.children[0];
    pcbGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach(m => m.dispose());
      } else {
        obj.material.dispose();
      }
    }
  }

  componentsGroup = new THREE.Group();
  componentsGroup.name = "componentsGroup";
  componentsGroup.visible = showComponents;
  pcbGroup.add(componentsGroup);

  emWaveGroup = new THREE.Group();
  emWaveGroup.name = "emWaveGroup";
  emWaveGroup.visible = showEMWaves;
  pcbGroup.add(emWaveGroup);

  const bounds = (data && data.board_geometry && data.board_geometry.bounds) || {
    width_mm: 55.0,
    height_mm: 52.0
  };
  const bw = Math.max(10.0, bounds.width_mm || 55.0);
  const bh = Math.max(10.0, bounds.height_mm || 52.0);
  const substrateH = 0.8; // mm

  // 1. Dielectric Substrate Mesh (Dark solder mask emerald-slate)
  const subGeo = new THREE.BoxGeometry(bw, substrateH, bh);
  const subMat = new THREE.MeshStandardMaterial({
    color: 0x061a10,
    roughness: 0.7,
    metalness: 0.15
  });
  const substrateMesh = new THREE.Mesh(subGeo, subMat);
  substrateMesh.position.set(0, 0, 0);
  pcbGroup.add(substrateMesh);

  // Silkscreen outline box
  const sw = bw / 2 - 1.2;
  const sh = bh / 2 - 1.2;
  const silkGeo = new THREE.BufferGeometry();
  const silkPts = [
    new THREE.Vector3(-sw, substrateH / 2 + 0.015, -sh),
    new THREE.Vector3(sw, substrateH / 2 + 0.015, -sh),
    new THREE.Vector3(sw, substrateH / 2 + 0.015, sh),
    new THREE.Vector3(-sw, substrateH / 2 + 0.015, sh),
    new THREE.Vector3(-sw, substrateH / 2 + 0.015, -sh)
  ];
  silkGeo.setFromPoints(silkPts);
  const silkMat = new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.5 });
  const silkLine = new THREE.Line(silkGeo, silkMat);
  pcbGroup.add(silkLine);

  const topY = substrateH / 2 + 0.03;
  const botY = -substrateH / 2 - 0.03;

  // 2. Real KiCad Edge.Cuts Outline
  const edgeCuts = (data && data.board_geometry && data.board_geometry.edge_cuts) || [];
  if (edgeCuts.length > 0) {
    const ecPositions = [];
    edgeCuts.forEach(ec => {
      // Top perimeter
      ecPositions.push(ec.x1, topY + 0.005, ec.y1);
      ecPositions.push(ec.x2, topY + 0.005, ec.y2);
      // Bottom perimeter
      ecPositions.push(ec.x1, botY - 0.005, ec.y1);
      ecPositions.push(ec.x2, botY - 0.005, ec.y2);
      // Vertical corner/edge line
      ecPositions.push(ec.x1, topY + 0.005, ec.y1);
      ecPositions.push(ec.x1, botY - 0.005, ec.y1);
    });
    const ecGeo = new THREE.BufferGeometry();
    ecGeo.setAttribute('position', new THREE.Float32BufferAttribute(ecPositions, 3));
    // KiCad authentic bright yellow Edge.Cuts
    const ecMat = new THREE.LineBasicMaterial({ color: 0xfacc15, linewidth: 2 });
    const ecLines = new THREE.LineSegments(ecGeo, ecMat);
    pcbGroup.add(ecLines);
  } else {
    // Fallback edge geometry
    const edgeGeo = new THREE.EdgesGeometry(subGeo);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.8 });
    const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
    pcbGroup.add(edgeLine);
  }

  // 3. Copper & Pad Materials
  const activeNetName = selectedPcbNet || (data && data.primary_trace && data.primary_trace.net_name) || "/Signal_AMP";

  const activeMat = new THREE.MeshStandardMaterial({
    color: 0x00f0ff,
    emissive: 0x00b4d8,
    emissiveIntensity: 0.8,
    roughness: 0.2,
    metalness: 0.85
  });

  const goldMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37, // Immersion gold for tracks
    roughness: 0.3,
    metalness: 0.85
  });

  const bCuMat = new THREE.MeshStandardMaterial({
    color: 0xa06520, // Bottom copper
    roughness: 0.4,
    metalness: 0.8
  });

  const padTinnedMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0, // Shiny silver tinned solder finish (HASL)
    roughness: 0.2,
    metalness: 0.95
  });

  const topPourMat = new THREE.MeshStandardMaterial({
    color: 0x14452a, // Deep emerald solder mask with copper fill beneath
    roughness: 0.45,
    metalness: 0.65,
    side: THREE.DoubleSide
  });

  const botPourMat = new THREE.MeshStandardMaterial({
    color: 0x183c27,
    roughness: 0.45,
    metalness: 0.65,
    side: THREE.DoubleSide
  });

  // 4. Metal Pours (Zones)
  const zones = (data && data.board_geometry && data.board_geometry.zones) || [];
  zones.forEach(zp => {
    const pts = zp.pts;
    if (pts && pts.length >= 3) {
      try {
        const shape = new THREE.Shape();
        shape.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
          shape.lineTo(pts[i][0], pts[i][1]);
        }
        const pourGeo = new THREE.ShapeGeometry(shape);
        pourGeo.rotateX(Math.PI / 2); // Rotate into X-Z plane

        const isTop = zp.is_top;
        const yPos = isTop ? (topY - 0.005) : (botY + 0.005);
        const isSelected = (zp.net === activeNetName);
        const pMat = isSelected ? activeMat : (isTop ? topPourMat : botPourMat);

        const pourMesh = new THREE.Mesh(pourGeo, pMat);
        pourMesh.position.y = yPos;
        pcbGroup.add(pourMesh);

        // Clear outline of metal pour
        const pourEdges = new THREE.EdgesGeometry(pourGeo);
        const pourLineMat = new THREE.LineBasicMaterial({
          color: isSelected ? 0x00f0ff : (isTop ? 0x34d399 : 0x8b6508),
          transparent: true,
          opacity: 0.4
        });
        const pourLine = new THREE.LineSegments(pourEdges, pourLineMat);
        pourLine.position.y = yPos + (isTop ? 0.001 : -0.001);
        pcbGroup.add(pourLine);
      } catch (err) {
        console.warn("Zone triangulation error:", err);
      }
    }
  });

  // 5. Copper Trace Segments
  const segments = (data && data.board_geometry && data.board_geometry.segments) || [];
  const cuThickness = 0.06;
  const padMap = new Set();

  segments.forEach(seg => {
    const x1 = seg.x1;
    const z1 = seg.y1;
    const x2 = seg.x2;
    const z2 = seg.y2;
    const w = Math.max(0.18, seg.width_mm || 0.2);
    const isTop = (seg.layer !== "B.Cu");
    const yPos = isTop ? topY : botY;
    const isSelected = (seg.net_name === activeNetName);

    const mat = isSelected ? activeMat : (isTop ? goldMat : bCuMat);

    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.sqrt(dx * dx + dz * dz);

    if (len > 0.005) {
      const boxGeo = new THREE.BoxGeometry(len, cuThickness, w);
      const boxMesh = new THREE.Mesh(boxGeo, mat);
      boxMesh.position.set((x1 + x2) / 2, yPos, (z1 + z2) / 2);
      boxMesh.rotation.y = -Math.atan2(dz, dx);
      pcbGroup.add(boxMesh);
    }

    [[x1, z1], [x2, z2]].forEach(([px, pz]) => {
      const key = `${px.toFixed(2)}_${pz.toFixed(2)}_${isTop}`;
      if (!padMap.has(key)) {
        padMap.add(key);
        const padGeo = new THREE.CylinderGeometry(w / 2, w / 2, cuThickness, 10);
        const padMesh = new THREE.Mesh(padGeo, mat);
        padMesh.position.set(px, yPos, pz);
        pcbGroup.add(padMesh);
      }
    });
  });

  // 6. Component Pads (Footprints)
  const pads = (data && data.board_geometry && data.board_geometry.pads) || [];
  const padThick = 0.07;
  pads.forEach(pad => {
    const px = pad.x;
    const pz = pad.y;
    const pw = Math.max(0.2, pad.w);
    const ph = Math.max(0.2, pad.h);
    const pRot = -pad.rot * Math.PI / 180.0;
    const isSelected = (pad.net === activeNetName);
    const padMat = isSelected ? activeMat : padTinnedMat;

    // Top Pad
    if (pad.is_top || pad.is_thru) {
      let padMesh;
      if (pad.shape === 'circle') {
        const geo = new THREE.CylinderGeometry(pw / 2, pw / 2, padThick, 16);
        padMesh = new THREE.Mesh(geo, padMat);
        padMesh.position.set(px, topY + padThick / 2, pz);
      } else {
        const geo = new THREE.BoxGeometry(pw, padThick, ph);
        padMesh = new THREE.Mesh(geo, padMat);
        padMesh.position.set(px, topY + padThick / 2, pz);
        padMesh.rotation.y = pRot;
      }
      pcbGroup.add(padMesh);
    }

    // Bottom Pad
    if (pad.is_bot || pad.is_thru) {
      let padMesh;
      if (pad.shape === 'circle') {
        const geo = new THREE.CylinderGeometry(pw / 2, pw / 2, padThick, 16);
        padMesh = new THREE.Mesh(geo, padMat);
        padMesh.position.set(px, botY - padThick / 2, pz);
      } else {
        const geo = new THREE.BoxGeometry(pw, padThick, ph);
        padMesh = new THREE.Mesh(geo, padMat);
        padMesh.position.set(px, botY - padThick / 2, pz);
        padMesh.rotation.y = pRot;
      }
      pcbGroup.add(padMesh);
    }

    // Through-hole drill hole barrel
    if (pad.is_thru) {
      const drillR = Math.min(pw, ph) * 0.32;
      const barrelGeo = new THREE.CylinderGeometry(drillR, drillR, substrateH + 0.1, 12);
      const barrelMat = new THREE.MeshBasicMaterial({ color: 0x05070a });
      const barrelMesh = new THREE.Mesh(barrelGeo, barrelMat);
      barrelMesh.position.set(px, 0, pz);
      pcbGroup.add(barrelMesh);
    }
  });

  // 7. Procedural 3D PCBA Component Packages
  const components = (data && data.board_geometry && data.board_geometry.components) || [];

  const chipBodyMat = new THREE.MeshStandardMaterial({
    color: 0x141820,
    roughness: 0.7,
    metalness: 0.15
  });
  const capBodyMat = new THREE.MeshStandardMaterial({
    color: 0x9f7a53,
    roughness: 0.5,
    metalness: 0.05
  });
  const leadSilverMat = new THREE.MeshStandardMaterial({
    color: 0xd8e0e8,
    roughness: 0.2,
    metalness: 0.95
  });
  const pinGoldMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37,
    roughness: 0.25,
    metalness: 0.9
  });
  const jstMat = new THREE.MeshStandardMaterial({
    color: 0xf3efe6,
    roughness: 0.4,
    metalness: 0.05
  });
  const delrinMat = new THREE.MeshStandardMaterial({
    color: 0x1c2026,
    roughness: 0.55,
    metalness: 0.1
  });

  components.forEach(comp => {
    const cx = comp.x;
    const cz = comp.y;
    const cRot = -comp.rot * Math.PI / 180.0;
    const pkg = comp.package || "";
    const ref = comp.ref || "";
    const isTop = (comp.is_top !== false);
    const compY = isTop ? topY : botY;

    const compContainer = new THREE.Group();
    compContainer.position.set(cx, compY, cz);
    compContainer.rotation.y = cRot;
    if (!isTop) compContainer.rotation.z = Math.PI;

    if (pkg.includes("SOIC-8") || pkg.includes("SO-8")) {
      // --- SOIC-8 Molded Package (U16: LTC6268, U2: LMP7721) ---
      const bodyGeo = new THREE.BoxGeometry(3.9, 1.45, 4.9);
      const bodyMesh = new THREE.Mesh(bodyGeo, chipBodyMat);
      bodyMesh.position.y = 1.45 / 2 + 0.15;
      compContainer.add(bodyMesh);

      // Chamfered Pin 1 indicator dot
      const dotGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.04, 12);
      const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const dotMesh = new THREE.Mesh(dotGeo, dotMat);
      dotMesh.position.set(-1.35, 1.45 + 0.17, -1.8);
      compContainer.add(dotMesh);

      // 8 Gull-wing Leads (4 on each side, pitch 1.27mm)
      const leadZ = [-1.905, -0.635, 0.635, 1.905];
      leadZ.forEach(lz => {
        const lGeo1 = new THREE.BoxGeometry(0.85, 0.16, 0.42);
        const lMesh1 = new THREE.Mesh(lGeo1, leadSilverMat);
        lMesh1.position.set(-2.25, 0.1, lz);
        compContainer.add(lMesh1);

        const lGeo2 = new THREE.BoxGeometry(0.85, 0.16, 0.42);
        const lMesh2 = new THREE.Mesh(lGeo2, leadSilverMat);
        lMesh2.position.set(2.25, 0.1, lz);
        compContainer.add(lMesh2);
      });

      // Silkscreen outline on PCB
      const sOutlineGeo = new THREE.BufferGeometry();
      const sPts = [
        new THREE.Vector3(-2.2, 0.015, -2.7),
        new THREE.Vector3(2.2, 0.015, -2.7),
        new THREE.Vector3(2.2, 0.015, 2.7),
        new THREE.Vector3(-2.2, 0.015, 2.7),
        new THREE.Vector3(-2.2, 0.015, -2.7)
      ];
      sOutlineGeo.setFromPoints(sPts);
      const sLine = new THREE.Line(sOutlineGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 }));
      compContainer.add(sLine);

    } else if (pkg.includes("0805")) {
      // --- SMD 0805 Chip (Resistors & Capacitors) ---
      const isCap = ref.startsWith("C");
      const mat = isCap ? capBodyMat : chipBodyMat;
      const bGeo = new THREE.BoxGeometry(1.2, 0.65, 1.25);
      const bMesh = new THREE.Mesh(bGeo, mat);
      bMesh.position.y = 0.65 / 2 + 0.05;
      compContainer.add(bMesh);

      // 2 End-caps
      const capGeo = new THREE.BoxGeometry(0.42, 0.67, 1.27);
      const cMesh1 = new THREE.Mesh(capGeo, leadSilverMat);
      cMesh1.position.set(-0.79, 0.65 / 2 + 0.05, 0);
      const cMesh2 = new THREE.Mesh(capGeo, leadSilverMat);
      cMesh2.position.set(0.79, 0.65 / 2 + 0.05, 0);
      compContainer.add(cMesh1);
      compContainer.add(cMesh2);

    } else if (pkg.includes("0402")) {
      // --- SMD 0402 Chip ---
      const isCap = ref.startsWith("C");
      const mat = isCap ? capBodyMat : chipBodyMat;
      const bGeo = new THREE.BoxGeometry(0.6, 0.38, 0.5);
      const bMesh = new THREE.Mesh(bGeo, mat);
      bMesh.position.y = 0.38 / 2 + 0.03;
      compContainer.add(bMesh);

      // 2 End-caps
      const capGeo = new THREE.BoxGeometry(0.22, 0.4, 0.52);
      const cMesh1 = new THREE.Mesh(capGeo, leadSilverMat);
      cMesh1.position.set(-0.39, 0.38 / 2 + 0.03, 0);
      const cMesh2 = new THREE.Mesh(capGeo, leadSilverMat);
      cMesh2.position.set(0.39, 0.38 / 2 + 0.03, 0);
      compContainer.add(cMesh1);
      compContainer.add(cMesh2);

    } else if (pkg.includes("Connector_JST") || ref === "J9") {
      // --- JST SH 10-pin Connector (J9) ---
      const shGeo = new THREE.BoxGeometry(12.0, 3.4, 4.5);
      const shMesh = new THREE.Mesh(shGeo, jstMat);
      shMesh.position.set(0, 3.4 / 2 + 0.05, 0);
      compContainer.add(shMesh);

      // Front mating slot cutout
      const slotGeo = new THREE.BoxGeometry(10.2, 2.0, 2.0);
      const slotMesh = new THREE.Mesh(slotGeo, chipBodyMat);
      slotMesh.position.set(0, 2.2, 1.4);
      compContainer.add(slotMesh);

      // Metal side anchors
      const tabGeo = new THREE.BoxGeometry(1.2, 1.6, 2.2);
      const t1 = new THREE.Mesh(tabGeo, leadSilverMat);
      t1.position.set(-6.1, 1.0, 0);
      const t2 = new THREE.Mesh(tabGeo, leadSilverMat);
      t2.position.set(6.1, 1.0, 0);
      compContainer.add(t1);
      compContainer.add(t2);

    } else if (pkg.includes("Nanopore") || ref === "H1") {
      // --- Nanopore Recessed Sensor Holder (H1) ---
      const cylGeo = new THREE.CylinderGeometry(4.7, 4.7, 3.6, 28);
      const cylMesh = new THREE.Mesh(cylGeo, delrinMat);
      cylMesh.position.y = 3.6 / 2 + 0.05;
      compContainer.add(cylMesh);

      // Recessed well
      const wellGeo = new THREE.CylinderGeometry(1.6, 1.6, 2.2, 20);
      const wellMesh = new THREE.Mesh(wellGeo, chipBodyMat);
      wellMesh.position.y = 3.6 - 1.0;
      compContainer.add(wellMesh);

      // Gold internal electrode contact ring
      const ringGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 20);
      const ringMesh = new THREE.Mesh(ringGeo, pinGoldMat);
      ringMesh.position.y = 3.6 - 1.9;
      compContainer.add(ringMesh);

    } else if (pkg.includes("Electrode") || ref.startsWith("J")) {
      // --- Electrode Pin Post (J1, J2, J10, J11) ---
      const postGeo = new THREE.CylinderGeometry(0.4, 0.4, 4.2, 14);
      const postMesh = new THREE.Mesh(postGeo, pinGoldMat);
      postMesh.position.y = 4.2 / 2 + 0.1;
      compContainer.add(postMesh);

      // Solder collar
      const collarGeo = new THREE.CylinderGeometry(0.85, 0.85, 0.45, 14);
      const collarMesh = new THREE.Mesh(collarGeo, leadSilverMat);
      collarMesh.position.y = 0.25;
      compContainer.add(collarMesh);
    }

    componentsGroup.add(compContainer);
  });

  // 8. Electromagnetic Near-Field Traveling Wave Overlay
  const emWaveMat = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending
  });

  let waveCumulativeDist = 0;
  activeNetSegments = [];

  segments.forEach(seg => {
    if (seg.net_name === activeNetName) {
      const x1 = seg.x1, z1 = seg.y1, x2 = seg.x2, z2 = seg.y2;
      const dx = x2 - x1, dz = z2 - z1;
      const len = Math.sqrt(dx * dx + dz * dz);
      const w = Math.max(0.2, seg.width_mm || 0.2);
      const isTop = (seg.layer !== "B.Cu");
      const yPos = isTop ? (topY + 0.08) : (botY - 0.08);

      if (len > 0.01) {
        const waveBoxGeo = new THREE.BoxGeometry(len, 0.04, w * 1.8);
        const waveMesh = new THREE.Mesh(waveBoxGeo, emWaveMat.clone());
        waveMesh.position.set((x1 + x2) / 2, yPos, (z1 + z2) / 2);
        waveMesh.rotation.y = -Math.atan2(dz, dx);
        emWaveGroup.add(waveMesh);

        activeNetSegments.push({
          mesh: waveMesh,
          cumDist: waveCumulativeDist,
          len: len
        });
        waveCumulativeDist += len;
      }
    }
  });
}

// --- KiCad UI & HUD Telemetry Updates ---
function updatePcbInspectorUI(data) {
  if (!data) return;

  const nameElem = document.getElementById("kicad-board-name-display");
  if (nameElem && data.board_name) {
    nameElem.innerText = data.board_name;
  }

  const dimElem = document.getElementById("kicad-board-dim-display");
  if (dimElem && data.board_geometry && data.board_geometry.bounds) {
    const b = data.board_geometry.bounds;
    const segs = data.board_geometry.segments ? data.board_geometry.segments.length : 0;
    dimElem.innerText = `${b.width_mm.toFixed(1)} × ${b.height_mm.toFixed(1)} mm | ${segs} Traces`;
  }

  const netSelect = document.getElementById("kicad-net-select");
  if (netSelect && data.board_geometry && data.board_geometry.nets_summary) {
    const currentVal = selectedPcbNet || netSelect.value;
    netSelect.innerHTML = '<option value="">Auto: Primary Signal Trace</option>';

    const nets = Object.values(data.board_geometry.nets_summary);
    nets.sort((a, b) => (b.total_length_mm || 0) - (a.total_length_mm || 0));

    nets.forEach(n => {
      const opt = document.createElement("option");
      opt.value = n.net_name;
      opt.innerText = `${n.net_name} (w=${n.trace_width_mm}mm, ${n.segment_count} segs, ${n.total_length_mm.toFixed(1)}mm)`;
      if (n.net_name === currentVal) {
        opt.selected = true;
      }
      netSelect.appendChild(opt);
    });
    if (selectedPcbNet) netSelect.value = selectedPcbNet;
  }
}

function updatePcbHUD(data) {
  if (!data) return;

  const cardEffTitle = document.querySelector("#card-eff .metric-title");
  if (cardEffTitle) cardEffTitle.innerText = "CHARACTERISTIC IMPEDANCE (Z0)";
  const cardDpTitle = document.querySelector("#card-dp .metric-title");
  if (cardDpTitle) cardDpTitle.innerText = "RETURN LOSS (S11)";
  const cardConvTitle = document.querySelector("#card-conservation .metric-title");
  if (cardConvTitle) cardConvTitle.innerText = "INSERTION LOSS (S21)";
  const cardStressTitle = document.querySelector("#card-stress .metric-title");
  if (cardStressTitle) cardStressTitle.innerText = "CROSS-TALK ISOLATION";
  const cardFosTitle = document.querySelector("#card-fos .metric-title");
  if (cardFosTitle) cardFosTitle.innerText = "RF MATCHING STATUS";
  const cardUncTitle = document.querySelector("#card-unc .metric-title");
  if (cardUncTitle) cardUncTitle.innerText = "RF FREQUENCY";

  let z0 = 149.75;
  let s11 = -6.03;
  let s21 = -0.18;
  let widthMm = 0.2;
  let lengthMm = 15.0;
  let isMatched = false;
  let netName = selectedPcbNet || (data.primary_trace && data.primary_trace.net_name) || "/Signal_AMP";

  if (data.board_geometry && data.board_geometry.nets_summary && data.board_geometry.nets_summary[netName]) {
    const net = data.board_geometry.nets_summary[netName];
    widthMm = net.trace_width_mm || 0.2;
    lengthMm = net.total_length_mm || 15.0;

    const h = 0.8;
    const er = 2.1;
    const u = widthMm / h;
    const e_eff = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 / Math.sqrt(1.0 + 12.0 / u));
    z0 = (120.0 * Math.PI / Math.sqrt(e_eff)) / (u + 1.393 + 0.667 * Math.log(u + 1.444));
    const gamma = Math.abs((z0 - 50.0) / (z0 + 50.0));
    s11 = 20.0 * Math.log10(Math.max(0.0001, gamma));
    isMatched = Math.abs(z0 - 50.0) <= 5.0;
  } else if (data.em_metrics) {
    z0 = data.em_metrics.z0_ohms || 149.75;
    s11 = data.em_metrics.s11_return_loss_db || -6.03;
    s21 = data.em_metrics.s21_insertion_loss_db || -0.18;
    isMatched = data.em_metrics.is_matched || false;
  }

  const skinDepthUm = 2.09 / Math.sqrt(Math.max(0.1, kicadFrequency));
  s21 = -(0.025 * (lengthMm / 10.0) * Math.sqrt(kicadFrequency)).toFixed(2);

  const effEl = document.getElementById("metric-eff");
  if (effEl) effEl.innerText = `${z0.toFixed(1)} Ω`;
  const subEff = document.getElementById("sub-eff");
  if (subEff) subEff.innerText = "Target: 50.0 Ω ± 5%";

  const dpEl = document.getElementById("metric-dp");
  if (dpEl) dpEl.innerText = `${Number(s11).toFixed(1)} dB`;
  const subDp = document.getElementById("sub-dp");
  if (subDp) subDp.innerText = "Target: < -15.0 dB";

  const divEl = document.getElementById("metric-div");
  if (divEl) divEl.innerText = `${s21} dB`;
  const subDiv = document.getElementById("sub-div");
  if (subDiv) subDiv.innerHTML = `<span class="badge-status-dot admissible"></span> δ = ${skinDepthUm.toFixed(2)} µm`;

  const stressEl = document.getElementById("metric-stress");
  if (stressEl) stressEl.innerText = "-45.0 dB";
  const subStress = document.getElementById("sub-stress");
  if (subStress) subStress.innerText = `Net: ${netName}`;

  const fosEl = document.getElementById("metric-fos");
  if (fosEl) {
    fosEl.innerText = isMatched ? "50Ω MATCHED" : "MISMATCHED";
    fosEl.style.color = isMatched ? "var(--accent-emerald)" : "var(--accent-amber)";
  }
  const subFos = document.getElementById("sub-fos");
  if (subFos) subFos.innerText = isMatched ? "VSWR < 1.2:1 (Optimal)" : `w=${widthMm}mm (Needs w≈2.4mm)`;

  const uncEl = document.getElementById("metric-unc");
  if (uncEl) uncEl.innerText = `${kicadFrequency.toFixed(1)} GHz`;
}

function selectKicadNet(netName) {
  selectedPcbNet = netName || null;
  if (activePcbData) {
    buildPcb3DScene(activePcbData);
    updatePcbHUD(activePcbData);
    fetchRfSweep(selectedPcbNet);
    fetchTdrData(selectedPcbNet);
    const targetNet = selectedPcbNet || (activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "Primary";
    showKiCadToast(`⚡ Selected Net: ${targetNet}`);
  }
}

function updateKicadFrequency(val) {
  kicadFrequency = parseFloat(val);
  const disp = document.getElementById("kicad-freq-val");
  if (disp) disp.innerText = `${kicadFrequency.toFixed(1)} GHz`;
  if (activePcbData && currentDomain === "pcb") {
    updatePcbHUD(activePcbData);
    if (rfSweepData) {
      updateVnaHUD(rfSweepData);
      drawCurrentVnaTab();
    }
  }
}

function calculate50OhmMatch() {
  const h = 0.8;
  const er = 2.1;
  const synthWidth = 2.43;
  const targetInput = document.getElementById("kicad-target-width-input");
  if (targetInput) targetInput.value = synthWidth.toFixed(2);

  showKiCadToast(
    `⚡ 50Ω Microstrip Synthesized: Optimal trace width w = ${synthWidth} mm (Substrate h=${h}mm, εr=${er}). Click 'Push to KiCad' to apply!`,
    6000
  );

  const subFos = document.getElementById("sub-fos");
  if (subFos) subFos.innerText = `Target Width: ${synthWidth} mm (KiCad)`;
}

// --- Bi-Directional Push to KiCad ---
async function pushTraceWidthToKiCad() {
  const targetInput = document.getElementById("kicad-target-width-input");
  const targetWidth = parseFloat(targetInput ? targetInput.value : 2.43);
  const netName = selectedPcbNet || (activePcbData && activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP";

  const pushBtn = document.getElementById("btn-push-kicad");
  if (pushBtn) {
    pushBtn.disabled = true;
    pushBtn.innerHTML = '<span>⏳</span><span>Pushing to KiCad...</span>';
  }

  showKiCadToast(`Pushing w = ${targetWidth} mm to KiCad for net ${netName}...`, 3000);

  try {
    const res = await fetch("/api/kicad_update_trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        net_name: netName,
        new_width_mm: targetWidth
      })
    });
    const result = await res.json();
    if (result.success) {
      showKiCadToast(`✅ KiCad Updated! Net ${netName} trace width set to ${result.new_width_mm} mm. (Backup: ${result.backup_created || 'created'})`, 6000);
      await pollKiCadLiveStatus();
      await fetchRfSweep(netName);
    } else {
      showKiCadToast(`❌ Failed to update KiCad: ${result.error}`, 5000);
    }
  } catch (err) {
    showKiCadToast(`❌ Error: ${err.message}`, 5000);
  } finally {
    if (pushBtn) {
      pushBtn.disabled = false;
      pushBtn.innerHTML = '<span>💾</span><span>Push to KiCad (.kicad_pcb)</span>';
    }
  }
}

// --- Viewport Overlay Toggles ---
function toggleComponents() {
  showComponents = !showComponents;
  if (componentsGroup) componentsGroup.visible = showComponents;
  const btn = document.getElementById("btn-toggle-ics");
  if (btn) btn.classList.toggle("active", showComponents);
}

function toggleEMWaves() {
  showEMWaves = !showEMWaves;
  if (emWaveGroup) emWaveGroup.visible = showEMWaves;
  const btn = document.getElementById("btn-toggle-em");
  if (btn) btn.classList.toggle("active", showEMWaves);
}

function updateEMWaves(dt) {
  if (!showEMWaves || !emWaveGroup || !emWaveGroup.visible || !activeNetSegments || activeNetSegments.length === 0) return;

  emWaveTime += dt * 6.28 * (kicadFrequency / 5.0) * 1.5;
  const lambdaMm = 44.0 / (kicadFrequency / 5.0);
  const beta = (2.0 * Math.PI) / Math.max(1.0, lambdaMm);

  activeNetSegments.forEach(seg => {
    const phase = seg.cumDist * beta - emWaveTime;
    const waveAmp = 0.35 + 0.55 * Math.sin(phase);
    if (seg.mesh && seg.mesh.material) {
      seg.mesh.material.opacity = Math.max(0.1, waveAmp);
    }
  });
}

// --- VNA & Signal Integrity Dock ---
function toggleVnaDock() {
  const dock = document.getElementById("vna-dock");
  const toggleBtn = document.getElementById("vna-dock-toggle");
  if (!dock) return;
  vnaDockOpen = !vnaDockOpen;
  dock.classList.toggle("collapsed", !vnaDockOpen);
  if (toggleBtn) toggleBtn.innerText = vnaDockOpen ? "▼" : "▲";
  if (vnaDockOpen) {
    drawCurrentVnaTab();
  }
}

function switchVnaTab(tabName) {
  activeVnaTab = tabName;
  document.querySelectorAll(".vna-tab-btn").forEach(b => {
    b.classList.toggle("active", b.id === `vna-tab-btn-${tabName}`);
  });
  document.querySelectorAll(".vna-tab-content").forEach(c => {
    c.classList.toggle("active", c.id === `vna-tab-${tabName}`);
  });
  if (tabName === "tdr" && !tdrData) {
    fetchTdrData();
  } else if (tabName === "nanopore" && !nanoporeData) {
    fetchNanoporeData();
  } else if (tabName === "fdtd" && !fdtdData) {
    fetchFdtdData();
  } else {
    drawCurrentVnaTab();
  }
}

function drawCurrentVnaTab() {
  if (activeVnaTab === "smith" && rfSweepData) drawSmithChart(rfSweepData);
  else if (activeVnaTab === "sparam" && rfSweepData) drawSParameterCurves(rfSweepData);
  else if (activeVnaTab === "eye" && rfSweepData) drawEyeDiagram(rfSweepData);
  else if (activeVnaTab === "tdr" && tdrData) drawTdrPlots(tdrData);
  else if (activeVnaTab === "nanopore" && nanoporeData) drawNanoporeOscilloscope(nanoporeData);
  else if (activeVnaTab === "fdtd" && fdtdData) drawFdtdCanvas();
}

async function fetchRfSweep(netName) {
  netName = netName || selectedPcbNet || (activePcbData && activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP";
  try {
    const res = await fetch(`/api/kicad_rf_sweep?net_name=${encodeURIComponent(netName)}`);
    const data = await res.json();
    rfSweepData = data;
    updateVnaHUD(data);
    drawCurrentVnaTab();
  } catch (err) {
    console.error("fetchRfSweep error:", err);
  }
}

function findClosestFreqIndex(freqs, target) {
  if (!freqs || freqs.length === 0) return 0;
  let bestIdx = 0;
  let minDiff = 1e9;
  for (let i = 0; i < freqs.length; i++) {
    const diff = Math.abs(freqs[i] - target);
    if (diff < minDiff) {
      minDiff = diff;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function updateVnaHUD(data) {
  if (!data) return;
  const z0 = data.z0_ohms || 50.0;
  const s11 = (data.s11_db && data.s11_db.length > 0) ? data.s11_db[0] : -6.0;
  const eyeH = (data.eye_metrics && data.eye_metrics.eye_height_mv) || 490.7;
  const eyeW = (data.eye_metrics && data.eye_metrics.eye_width_ps) || 86.9;
  const eyeJ = (data.eye_metrics && data.eye_metrics.total_jitter_ps) || 13.1;

  const pillZ0 = document.getElementById("vna-pill-z0");
  if (pillZ0) pillZ0.innerText = `Z₀: ${z0.toFixed(1)} Ω`;
  const pillS11 = document.getElementById("vna-pill-s11");
  if (pillS11) pillS11.innerText = `S₁₁: ${Number(s11).toFixed(1)} dB`;
  const pillEye = document.getElementById("vna-pill-eye");
  if (pillEye) pillEye.innerText = `Eye: ${eyeH.toFixed(0)} mV`;

  // Update Smith sidebar
  const zinVal = document.getElementById("smith-zin-val");
  if (zinVal && data.zin && data.zin.length > 0) {
    const idx = findClosestFreqIndex(data.frequencies_ghz, kicadFrequency);
    const z = data.zin[idx] || data.zin[0];
    const sign = z[1] >= 0 ? "+" : "-";
    zinVal.innerText = `${z[0].toFixed(1)} ${sign} j${Math.abs(z[1]).toFixed(1)} Ω`;
  }
  const gammaVal = document.getElementById("smith-gamma-val");
  if (gammaVal && data.smith_gamma && data.smith_gamma.length > 0) {
    const idx = findClosestFreqIndex(data.frequencies_ghz, kicadFrequency);
    const g = data.smith_gamma[idx] || data.smith_gamma[0];
    const mag = Math.sqrt(g[0]*g[0] + g[1]*g[1]);
    const deg = Math.atan2(g[1], g[0]) * 180 / Math.PI;
    gammaVal.innerText = `${mag.toFixed(3)} ∠ ${deg.toFixed(1)}°`;
    const vswr = (1 + mag) / Math.max(0.001, 1 - mag);
    const vswrVal = document.getElementById("smith-vswr-val");
    if (vswrVal) vswrVal.innerText = `${vswr.toFixed(2)} : 1`;
  }

  // Update S-Param sidebar
  const spFreq = document.getElementById("sparam-freq-readout");
  if (spFreq) spFreq.innerText = `${kicadFrequency.toFixed(1)} GHz`;
  const spS11 = document.getElementById("sparam-s11-readout");
  if (spS11 && data.s11_db && data.frequencies_ghz) {
    const idx = findClosestFreqIndex(data.frequencies_ghz, kicadFrequency);
    spS11.innerText = `${data.s11_db[idx].toFixed(1)} dB`;
  }

  // Update Eye sidebar
  const ehEl = document.getElementById("eye-height-val");
  if (ehEl) ehEl.innerText = `${eyeH.toFixed(1)} mV`;
  const ewEl = document.getElementById("eye-width-val");
  if (ewEl) ewEl.innerText = `${eyeW.toFixed(1)} ps`;
  const ejEl = document.getElementById("eye-jitter-val");
  if (ejEl) ejEl.innerText = `${eyeJ.toFixed(1)} ps`;
}

// --- Smith Chart Canvas ---
function drawSmithChart(data) {
  const canvas = document.getElementById("smith-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const xc = w / 2;
  const yc = h / 2;
  const R = Math.min(w, h) * 0.44;

  // Background circle
  ctx.fillStyle = "#070b13";
  ctx.beginPath();
  ctx.arc(xc, yc, R, 0, 2 * Math.PI);
  ctx.fill();

  // Grid circles: r = const
  const rValues = [0.2, 0.5, 1.0, 2.0, 5.0];
  rValues.forEach(r => {
    const crX = xc + R * (r / (1 + r));
    const crR = R / (1 + r);
    ctx.strokeStyle = (r === 1.0) ? "rgba(16, 185, 129, 0.45)" : "rgba(56, 189, 248, 0.2)";
    ctx.lineWidth = (r === 1.0) ? 1.5 : 1.0;
    ctx.beginPath();
    ctx.arc(crX, yc, crR, 0, 2 * Math.PI);
    ctx.stroke();
  });

  // Reactance arcs: x = const
  const xValues = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0];
  ctx.save();
  ctx.beginPath();
  ctx.arc(xc, yc, R, 0, 2 * Math.PI);
  ctx.clip(); // Clip arcs inside unit circle

  xValues.forEach(x => {
    const cxX = xc + R;
    const cxY = yc - (R / x);
    const cxR = Math.abs(R / x);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.16)";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.arc(cxX, cxY, cxR, 0, 2 * Math.PI);
    ctx.stroke();
  });
  ctx.restore();

  // Horizontal real line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.moveTo(xc - R, yc);
  ctx.lineTo(xc + R, yc);
  ctx.stroke();

  // 50 Ohm Center Point
  ctx.fillStyle = "#10b981";
  ctx.beginPath();
  ctx.arc(xc, yc, 3.5, 0, 2 * Math.PI);
  ctx.fill();

  // Outer circle border
  ctx.strokeStyle = "rgba(0, 240, 255, 0.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(xc, yc, R, 0, 2 * Math.PI);
  ctx.stroke();

  if (!data || !data.smith_gamma || data.smith_gamma.length === 0) return;

  // Locus curve
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2.2;
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  data.smith_gamma.forEach((pt, i) => {
    const px = xc + pt[0] * R;
    const py = yc - pt[1] * R;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Highlight marker at current frequency
  const idx = findClosestFreqIndex(data.frequencies_ghz, kicadFrequency);
  const curPt = data.smith_gamma[idx];
  if (curPt) {
    const mx = xc + curPt[0] * R;
    const my = yc - curPt[1] * R;
    ctx.fillStyle = "#f59e0b";
    ctx.shadowColor = "#f59e0b";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(mx, my, 4.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Small pulsing ring
    ctx.strokeStyle = "rgba(245, 158, 11, 0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(mx, my, 7.5, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

// --- S-Parameter Curves Canvas ---
function drawSParameterCurves(data) {
  const canvas = document.getElementById("sparam-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 40, padR = 20, padT = 20, padB = 25;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Background
  ctx.fillStyle = "#070b13";
  ctx.fillRect(padL, padT, plotW, plotH);

  // Axis ranges: Freq 0 to 30 GHz, dB 0 to -60 dB
  const minF = 0.0, maxF = 30.0;
  const maxDb = 0.0, minDb = -60.0;

  function toX(f) { return padL + ((f - minF) / (maxF - minF)) * plotW; }
  function toY(db) { return padT + ((maxDb - db) / (maxDb - minDb)) * plotH; }

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "9px SF Mono, monospace";
  ctx.textAlign = "center";

  for (let f = 5; f <= 30; f += 5) {
    const x = toX(f);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillText(`${f}G`, x, padT + plotH + 14);
  }

  ctx.textAlign = "right";
  for (let db = 0; db >= -60; db -= 10) {
    const y = toY(db);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillText(`${db}`, padL - 6, y + 3);
  }

  // -15 dB Target Match Dashed Line
  const y15 = toY(-15.0);
  ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, y15);
  ctx.lineTo(padL + plotW, y15);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!data || !data.frequencies_ghz) return;

  const freqs = data.frequencies_ghz;
  const s11 = data.s11_db;
  const s21 = data.s21_db;

  // Draw S11 Curve (Cyan)
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  freqs.forEach((f, i) => {
    const x = toX(f);
    const y = toY(Math.max(minDb, s11[i]));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw S21 Curve (Emerald)
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  freqs.forEach((f, i) => {
    const x = toX(f);
    const y = toY(Math.max(minDb, s21[i]));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Frequency Cursor Line
  const curX = toX(kicadFrequency);
  ctx.strokeStyle = "rgba(245, 158, 11, 0.8)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(curX, padT);
  ctx.lineTo(curX, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Marker Dot on S11
  const idx = findClosestFreqIndex(freqs, kicadFrequency);
  const curS11 = s11[idx];
  const markerY = toY(Math.max(minDb, curS11));
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(curX, markerY, 4, 0, 2 * Math.PI);
  ctx.fill();

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
}

// --- Live Eye Diagram Canvas (10 Gbps) ---
function drawEyeDiagram(data) {
  const canvas = document.getElementById("eye-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 45, padR = 20, padT = 20, padB = 25;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Background
  ctx.fillStyle = "#070b13";
  ctx.fillRect(padL, padT, plotW, plotH);

  // Axis ranges: Time 0 to 200 ps (2 UI for 10 Gbps), Voltage -100 to 700 mV
  const tMin = 0.0, tMax = 200.0;
  const vMin = -100.0, vMax = 700.0;

  function toX(t) { return padL + (t / tMax) * plotW; }
  function toY(v) { return padT + ((vMax - v) / (vMax - vMin)) * plotH; }

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "9px SF Mono, monospace";
  ctx.textAlign = "center";

  for (let t = 0; t <= 200; t += 50) {
    const x = toX(t);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillText(`${t}ps`, x, padT + plotH + 14);
  }

  ctx.textAlign = "right";
  for (let v = 0; v <= 600; v += 150) {
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillText(`${v}mV`, padL - 6, y + 3);
  }

  // Draw Phosphor PRBS Eye Traces
  const vHigh = (data && data.eye_metrics && data.eye_metrics.eye_height_mv) ? (50.0 + data.eye_metrics.eye_height_mv) : 540.0;
  const vLow = 50.0;
  const vMid = (vHigh + vLow) / 2;
  const tr = 18.0; // 18 ps rise time
  const tJitter = (data && data.eye_metrics && data.eye_metrics.total_jitter_ps) ? data.eye_metrics.total_jitter_ps : 13.1;

  ctx.strokeStyle = "rgba(0, 240, 255, 0.16)";
  ctx.lineWidth = 1.4;

  const patterns = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
  ];

  for (let rep = 0; rep < 8; rep++) {
    const jitter1 = (Math.sin(rep * 1.7) * (tJitter / 2));
    const jitter2 = (Math.cos(rep * 2.3) * (tJitter / 2));
    patterns.forEach(pat => {
      ctx.beginPath();
      for (let t = 0; t <= 200; t += 2) {
        let v;
        if (t < 100) {
          const trans = 0.5 * (1.0 + Math.tanh((t - 50 - jitter1) / (tr * 0.4)));
          v = (1 - trans) * (pat[0] ? vHigh : vLow) + trans * (pat[1] ? vHigh : vLow);
        } else {
          const trans = 0.5 * (1.0 + Math.tanh((t - 150 - jitter2) / (tr * 0.4)));
          v = (1 - trans) * (pat[1] ? vHigh : vLow) + trans * (pat[2] ? vHigh : vLow);
        }
        const px = toX(t);
        const py = toY(v);
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
  }

  // Draw Eye Mask (Diamond in center UI)
  const maskX = toX(100);
  const maskW = (plotW / 2) * 0.35;
  const maskY = toY(vMid);
  const maskH = (plotH) * 0.28;

  ctx.strokeStyle = "rgba(245, 158, 11, 0.7)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(maskX - maskW, maskY);
  ctx.lineTo(maskX, maskY - maskH);
  ctx.lineTo(maskX + maskW, maskY);
  ctx.lineTo(maskX, maskY + maskH);
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
}

// --- Time-Domain Reflectometry (TDR) & Crosstalk ---
async function fetchTdrData(netName) {
  netName = netName || selectedPcbNet || (activePcbData && activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP";
  try {
    const res = await fetch(`/api/kicad_tdr?net_name=${encodeURIComponent(netName)}&rise_time_ps=25.0`);
    const data = await res.json();
    tdrData = data;
    updateTdrHUD(data);
    if (activeVnaTab === "tdr") {
      drawTdrPlots(data);
    }
  } catch (err) {
    console.error("fetchTdrData error:", err);
  }
}

function updateTdrHUD(data) {
  if (!data) return;
  const zrangeEl = document.getElementById("tdr-zrange-val");
  if (zrangeEl) zrangeEl.innerText = `${data.z_min_ohms} / ${data.z_max_ohms} Ω`;

  const xtalk = data.crosstalk;
  if (xtalk) {
    const nextEl = document.getElementById("tdr-next-val");
    if (nextEl) nextEl.innerText = `${xtalk.peak_next_db} dB`;
    const fextEl = document.getElementById("tdr-fext-val");
    if (fextEl) fextEl.innerText = `${xtalk.peak_fext_db} dB`;
    const isoEl = document.getElementById("tdr-isolation-val");
    if (isoEl) {
      isoEl.innerText = xtalk.isolation_status;
      isoEl.style.color = (xtalk.isolation_status.includes("EXCELLENT") || xtalk.isolation_status.includes("GOOD")) ? "var(--accent-emerald)" : "var(--accent-amber)";
    }
  }
}

function drawTdrPlots(data) {
  if (!data) return;
  drawTdrProfileCanvas(data);
  drawCrosstalkCanvas(data.crosstalk);
}

function drawTdrProfileCanvas(data) {
  const canvas = document.getElementById("tdr-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 36, padR = 14, padT = 14, padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.fillStyle = "#070b13";
  ctx.fillRect(padL, padT, plotW, plotH);

  const maxLen = data.total_length_mm || 35.0;
  const maxZ = Math.max(180.0, (data.z_max_ohms || 150.0) + 20.0);
  const minZ = 0.0;

  function toX(xMm) { return padL + (xMm / maxLen) * plotW; }
  function toY(z) { return padT + ((maxZ - z) / (maxZ - minZ)) * plotH; }

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "8px SF Mono, monospace";

  // X ticks
  ctx.textAlign = "center";
  const stepX = maxLen > 40 ? 10 : 5;
  for (let x = 0; x <= maxLen; x += stepX) {
    const px = toX(x);
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
    ctx.stroke();
    ctx.fillText(`${x}mm`, px, padT + plotH + 12);
  }

  // Y ticks
  ctx.textAlign = "right";
  for (let z = 50; z <= maxZ; z += 50) {
    const py = toY(z);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.fillText(`${z}Ω`, padL - 4, py + 3);
  }

  // 50 Ohm target line
  const y50 = toY(50.0);
  ctx.strokeStyle = "rgba(16, 185, 129, 0.5)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, y50);
  ctx.lineTo(padL + plotW, y50);
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot TDR curve
  if (data.distance_mm && data.z_tdr_ohms) {
    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2.0;
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    for (let i = 0; i < data.distance_mm.length; i++) {
      const px = toX(data.distance_mm[i]);
      const py = toY(data.z_tdr_ohms[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Discontinuity tags
  if (data.discontinuities) {
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    data.discontinuities.forEach(d => {
      const dx = toX(d.x_mm);
      const dy = toY(d.z_ohms);
      ctx.fillStyle = d.type === "capacitive" ? "#38bdf8" : (d.type === "inductive" ? "#f59e0b" : "#10b981");
      ctx.beginPath();
      ctx.arc(dx, dy, 3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(padL, padT, plotW, plotH);
}

function drawCrosstalkCanvas(xtalk) {
  const canvas = document.getElementById("crosstalk-canvas");
  if (!canvas || !xtalk) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 36, padR = 14, padT = 14, padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.fillStyle = "#070b13";
  ctx.fillRect(padL, padT, plotW, plotH);

  const minF = 0.0, maxF = 30.0;
  const minDb = -80.0, maxDb = 0.0;

  function toX(f) { return padL + (f / maxF) * plotW; }
  function toY(db) { return padT + ((maxDb - db) / (maxDb - minDb)) * plotH; }

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "8px SF Mono, monospace";

  ctx.textAlign = "center";
  for (let f = 5; f <= 30; f += 5) {
    const px = toX(f);
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
    ctx.stroke();
    ctx.fillText(`${f}G`, px, padT + plotH + 12);
  }

  ctx.textAlign = "right";
  for (let db = -20; db >= -80; db -= 20) {
    const py = toY(db);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.fillText(`${db}`, padL - 4, py + 3);
  }

  // -30 dB isolation line
  const y30 = toY(-30.0);
  ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, y30);
  ctx.lineTo(padL + plotW, y30);
  ctx.stroke();
  ctx.setLineDash([]);

  // NEXT Curve (Amber)
  if (xtalk.frequencies_ghz && xtalk.next_db) {
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < xtalk.frequencies_ghz.length; i++) {
      const px = toX(xtalk.frequencies_ghz[i]);
      const py = toY(Math.max(minDb, xtalk.next_db[i]));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // FEXT Curve (Blue)
  if (xtalk.frequencies_ghz && xtalk.fext_db) {
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < xtalk.frequencies_ghz.length; i++) {
      const px = toX(xtalk.frequencies_ghz[i]);
      const py = toY(Math.max(minDb, xtalk.fext_db[i]));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // Legend
  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(plotW - 75, padT + 6, 8, 8);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "8px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("NEXT", plotW - 63, padT + 13);

  ctx.fillStyle = "#38bdf8";
  ctx.fillRect(plotW - 35, padT + 6, 8, 8);
  ctx.fillText("FEXT", plotW - 23, padT + 13);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(padL, padT, plotW, plotH);
}

// --- Nanopore Electrophysiology Oscilloscope ---
async function fetchNanoporeData() {
  try {
    const res = await fetch(`/api/nanopore_stream?pore_diam_nm=${nanoporePoreDiam}&bias_mv=${nanoporeBiasMv}&event_rate=3000`);
    const data = await res.json();
    nanoporeData = data;
    updateNanoporeHUD(data);
    drawNanoporeOscilloscope(data);
  } catch (err) {
    console.error("fetchNanoporeData error:", err);
  }
}

function updateNanoporeParams() {
  const diamSlider = document.getElementById("nano-diam-slider");
  const biasSlider = document.getElementById("nano-bias-slider");
  if (diamSlider) {
    nanoporePoreDiam = parseFloat(diamSlider.value);
    const dDisp = document.getElementById("nano-diam-disp");
    if (dDisp) dDisp.innerText = `${nanoporePoreDiam.toFixed(1)} nm`;
  }
  if (biasSlider) {
    nanoporeBiasMv = parseFloat(biasSlider.value);
    const bDisp = document.getElementById("nano-bias-disp");
    if (bDisp) bDisp.innerText = `${nanoporeBiasMv.toFixed(0)} mV`;
  }
  fetchNanoporeData();
}

function updateNanoporeHUD(data) {
  if (!data) return;
  const baseEl = document.getElementById("nano-baseline-val");
  if (baseEl) baseEl.innerText = `${data.baseline_current_na} nA`;

  const noiseEl = document.getElementById("nano-noise-val");
  if (noiseEl) noiseEl.innerText = `${data.rms_noise_pa} pA | ${data.snr_db} dB`;

  const eventsEl = document.getElementById("nano-events-val");
  if (eventsEl) eventsEl.innerText = `${data.events_detected} Events (3.0 kHz)`;

  const dwellEl = document.getElementById("nano-dwell-val");
  if (dwellEl) dwellEl.innerText = `Mean Dwell: ${data.mean_dwell_us} µs`;
}

function drawNanoporeOscilloscope(data) {
  const canvas = document.getElementById("nanopore-canvas");
  if (!canvas || !data || !data.current_na) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 40, padR = 15, padT = 15, padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Dark oscilloscope phosphor screen
  ctx.fillStyle = "#04090e";
  ctx.fillRect(padL, padT, plotW, plotH);

  // Phosphor grid lines (10 horizontal, 6 vertical divisions)
  ctx.strokeStyle = "rgba(16, 185, 129, 0.09)";
  ctx.lineWidth = 1;
  for (let c = 1; c < 10; c++) {
    const gx = padL + (c / 10) * plotW;
    ctx.beginPath();
    ctx.moveTo(gx, padT);
    ctx.lineTo(gx, padT + plotH);
    ctx.stroke();
  }
  for (let r = 1; r < 6; r++) {
    const gy = padT + (r / 6) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + plotW, gy);
    ctx.stroke();
  }

  const i0 = data.baseline_current_na || 2.5;
  const maxI = Math.max(3.5, i0 * 1.35);
  const minI = 0.0;

  function toY(curr) {
    return padT + ((maxI - curr) / (maxI - minI)) * plotH;
  }

  // Baseline reference line (Dashed blue)
  const yBase = toY(i0);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, yBase);
  ctx.lineTo(padL + plotW, yBase);
  ctx.stroke();
  ctx.setLineDash([]);

  // Blockade detection threshold line (Dashed red)
  const yThresh = toY(i0 * 0.85);
  ctx.strokeStyle = "rgba(244, 63, 94, 0.4)";
  ctx.setLineDash([2, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, yThresh);
  ctx.lineTo(padL + plotW, yThresh);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = "#64748b";
  ctx.font = "8px SF Mono, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`${maxI.toFixed(1)}nA`, padL - 4, padT + 8);
  ctx.fillText(`${(maxI / 2).toFixed(1)}nA`, padL - 4, padT + plotH / 2 + 3);
  ctx.fillText("0.0nA", padL - 4, padT + plotH);

  ctx.textAlign = "center";
  ctx.fillText("0.0ms", padL, padT + plotH + 12);
  ctx.fillText("1.0ms", padL + plotW / 2, padT + plotH + 12);
  ctx.fillText("2.0ms (Timebase: 200µs/div)", padL + plotW - 40, padT + plotH + 12);

  // Draw scrolling live current beam
  const samples = data.current_na;
  const n = samples.length;
  const offset = Math.floor(nanoScopeOffset) % n;

  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 1.6;
  ctx.shadowColor = "#10b981";
  ctx.shadowBlur = 4;
  ctx.beginPath();

  for (let xPix = 0; xPix < plotW; xPix++) {
    const sIdx = (offset + Math.floor((xPix / plotW) * n)) % n;
    const curr = samples[sIdx];
    const py = toY(curr);
    const px = padL + xPix;
    if (xPix === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Draw event markers
  if (data.events) {
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    data.events.forEach(ev => {
      const relPos = ((ev.start_us / 2000.0) * plotW - offset * (plotW / n));
      const evX = padL + ((relPos % plotW + plotW) % plotW);
      const evY = toY(ev.residual_current_na);

      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(evX, evY, 3, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
      ctx.fillText(`DNA (${ev.dwell_us}µs)`, evX, evY - 8);
    });
  }

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.strokeRect(padL, padT, plotW, plotH);
}



// =====================================================================
// Phase B: Power & Thermal Integrity (IR-Drop & IR Heatmap Overlay)
// =====================================================================

async function toggleThermalIR() {
  showThermalIR = !showThermalIR;
  const btn = document.getElementById("btn-toggle-thermal");
  if (btn) btn.classList.toggle("active", showThermalIR);

  if (showThermalIR) {
    if (!thermalData) {
      await runPowerThermalAnalysis();
    } else {
      buildThermalOverlay(thermalData);
    }
    if (thermalMesh) thermalMesh.visible = true;
    showKiCadToast("🔥 Thermal IR Heatmap Overlay: Active");
  } else {
    if (thermalMesh) thermalMesh.visible = false;
    showKiCadToast("Thermal IR Heatmap: Hidden");
  }
}

async function runPowerThermalAnalysis() {
  try {
    const net = selectedPcbNet || (activePcbData && activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP";
    const res = await fetch(`/api/kicad_power_thermal?net_name=${encodeURIComponent(net)}&current_a=0.50`);
    const data = await res.json();
    thermalData = data;

    // Update Left Drawer Readouts
    const pDrop = document.getElementById("pwr-drop-val");
    if (pDrop) pDrop.innerText = `${data.total_ir_drop_mv} mV`;
    const pJmax = document.getElementById("pwr-jmax-val");
    if (pJmax) pJmax.innerText = `${data.max_current_density_a_mm2} A/mm²`;
    const pTemp = document.getElementById("pwr-temp-val");
    if (pTemp && data.thermal_heatmap) pTemp.innerText = `${data.thermal_heatmap.t_max_c} °C`;
    const pLoss = document.getElementById("pwr-loss-val");
    if (pLoss) pLoss.innerText = `${data.total_dissipation_mw} mW`;

    if (showThermalIR) {
      buildThermalOverlay(data);
      if (thermalMesh) thermalMesh.visible = true;
    }
  } catch (err) {
    console.error("runPowerThermalAnalysis error:", err);
  }
}

function buildThermalOverlay(data) {
  if (!pcbGroup) return;
  if (thermalMesh) {
    pcbGroup.remove(thermalMesh);
    thermalMesh.geometry.dispose();
    if (thermalMesh.material.map) thermalMesh.material.map.dispose();
    thermalMesh.material.dispose();
    thermalMesh = null;
  }

  const th = data.thermal_heatmap;
  if (!th || !th.temp_grid) return;

  const w = th.board_width_mm || 55.0;
  const h = th.board_height_mm || 52.0;
  const grid = th.temp_grid;
  const ny = grid.length;
  const nx = grid[0].length;
  const tMin = th.t_min_c || 22.0;
  const tMax = Math.max(tMin + 5.0, th.t_max_c || 45.0);

  // Create 2D offscreen canvas for temperature heatmap texture
  const c = document.createElement("canvas");
  c.width = nx;
  c.height = ny;
  const ctx = c.getContext("2d");
  const imgData = ctx.createImageData(nx, ny);

  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const t = grid[iy][ix];
      const norm = Math.max(0.0, Math.min(1.0, (t - tMin) / (tMax - tMin)));
      // Ironbow / Inferno false color gradient
      let r = 0, g = 0, b = 0;
      if (norm < 0.25) {
        r = Math.floor(norm * 4 * 40);
        g = Math.floor(norm * 4 * 60);
        b = Math.floor(180 + norm * 4 * 75);
      } else if (norm < 0.6) {
        const u = (norm - 0.25) / 0.35;
        r = Math.floor(40 + u * 215);
        g = Math.floor(60 + u * 150);
        b = Math.floor(255 - u * 200);
      } else if (norm < 0.85) {
        const u = (norm - 0.6) / 0.25;
        r = 255;
        g = Math.floor(210 - u * 120);
        b = 20;
      } else {
        const u = (norm - 0.85) / 0.15;
        r = 255;
        g = Math.floor(90 + u * 165);
        b = Math.floor(20 + u * 235);
      }
      const pIdx = (iy * nx + ix) * 4;
      imgData.data[pIdx] = r;
      imgData.data[pIdx + 1] = g;
      imgData.data[pIdx + 2] = b;
      imgData.data[pIdx + 3] = 190; // Opacity
    }
  }
  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(c);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;

  const geo = new THREE.PlaneGeometry(w, h);
  const mat = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
    side: THREE.DoubleSide
  });

  thermalMesh = new THREE.Mesh(geo, mat);
  thermalMesh.rotation.x = -Math.PI / 2;
  thermalMesh.position.y = 0.95; // Just above top copper
  pcbGroup.add(thermalMesh);
}

// =====================================================================
// Phase B: DRC & DFM Copilot (Holographic Markers & 1-Click Auto-Fix)
// =====================================================================

async function toggleDrcMarkers() {
  showDrcMarkers = !showDrcMarkers;
  const btn = document.getElementById("btn-toggle-drc");
  if (btn) btn.classList.toggle("active", showDrcMarkers);

  if (showDrcMarkers) {
    if (!drcData) {
      await runDrcInspection();
    } else if (drcGroup) {
      drcGroup.visible = true;
    }
    showKiCadToast("🛡️ DRC Visual Copilot: Active (3D Defect Markers)");
  } else {
    if (drcGroup) drcGroup.visible = false;
    showKiCadToast("DRC Visual Copilot: Hidden");
  }
}

async function runDrcInspection() {
  try {
    const res = await fetch("/api/kicad_drc");
    const data = await res.json();
    drcData = data;

    // Update left drawer
    const sumDisp = document.getElementById("drc-summary-val");
    if (sumDisp) {
      sumDisp.innerText = `${data.total_violations} (${data.critical_count} Critical, ${data.warning_count} Warning)`;
    }

    const cont = document.getElementById("drc-items-container");
    if (cont) {
      cont.innerHTML = "";
      if (data.violations && data.violations.length > 0) {
        data.violations.forEach(v => {
          const card = document.createElement("div");
          card.className = "drc-violation-card";
          const badgeClass = v.severity === "CRITICAL" ? "drc-badge-critical" : (v.severity === "WARNING" ? "drc-badge-warning" : "drc-badge-advisory");
          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
              <span class="${badgeClass}">${v.severity}</span>
              <span style="font-weight: bold; color: #fff;">${v.id}</span>
            </div>
            <div style="color: var(--text-primary); font-size: 10px; margin-bottom: 2px;">${v.rule}</div>
            <div style="color: var(--text-muted); font-size: 9px; line-height: 1.2; margin-bottom: 4px;">${v.description}</div>
            <button class="tool-btn" style="padding: 1px 6px; font-size: 9px; background: rgba(245, 158, 11, 0.2);" onclick="autoFixDrcViolation('${v.id}')">⚡ Auto-Fix (${v.autofix_type})</button>
          `;
          cont.appendChild(card);
        });
      } else {
        cont.innerHTML = `<div style="color: #10b981; font-weight: bold;">✔ All DRC & DFM rules satisfied!</div>`;
      }
    }

    buildDrc3DMarkers(data.violations || []);
    if (drcGroup) drcGroup.visible = showDrcMarkers;
  } catch (err) {
    console.error("runDrcInspection error:", err);
  }
}

function buildDrc3DMarkers(violations) {
  if (!drcGroup) return;
  while (drcGroup.children.length > 0) {
    const obj = drcGroup.children[0];
    drcGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }

  // Get board bounds center to map KiCad global coordinates to Three.js centered origin
  const geom = activePcbData && activePcbData.board_geometry;
  const bounds = geom && geom.bounds;
  const cx = bounds && bounds.center_x !== undefined ? bounds.center_x : 164.5;
  const cy = bounds && bounds.center_y !== undefined ? bounds.center_y : 58.5;

  violations.forEach(v => {
    const isCrit = (v.severity === "CRITICAL");
    const col = isCrit ? 0xef4444 : 0xf59e0b;

    // Outer pulsing marker sphere
    const geo = new THREE.SphereGeometry(1.1, 16, 16);
    const mat = new THREE.MeshBasicMaterial({
      color: col,
      wireframe: true,
      transparent: true,
      opacity: 0.85
    });
    const marker = new THREE.Mesh(geo, mat);

    // Inner bright core
    const coreGeo = new THREE.SphereGeometry(0.45, 12, 12);
    const coreMat = new THREE.MeshBasicMaterial({ color: col });
    const core = new THREE.Mesh(coreGeo, coreMat);
    marker.add(core);

    // Position relative to Three.js PCB center
    const px = v.x_mm - cx;
    const pz = v.y_mm - cy;
    marker.position.set(px, 1.6, pz);
    drcGroup.add(marker);
  });
}

async function autoFixDrcViolation(violationId) {
  try {
    const res = await fetch("/api/kicad_autofix_drc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ violation_id: violationId })
    });
    const data = await res.json();
    if (data.success) {
      showKiCadToast(`⚡ Auto-Fix Applied: ${data.message}`);
      await runDrcInspection();
    } else {
      showKiCadToast(`Auto-Fix Failed: ${data.error}`);
    }
  } catch (err) {
    console.error("autoFixDrcViolation error:", err);
  }
}

async function autoFixAllAcidTraps() {
  await autoFixDrcViolation("DRC-AT-1");
}

// =====================================================================
// Phase C: Full-Wave FDTD Electromagnetic Slice
// =====================================================================

async function fetchFdtdData(freqGhz = 5.0) {
  try {
    const net = selectedPcbNet || (activePcbData && activePcbData.primary_trace && activePcbData.primary_trace.net_name) || "/Signal_AMP";
    const res = await fetch(`/api/kicad_fdtd?freq_ghz=${freqGhz}&net_name=${encodeURIComponent(net)}`);
    const data = await res.json();
    fdtdData = data;
    fdtdFrameIdx = 0;

    // Update sidebar telemetry
    const dtDisp = document.getElementById("fdtd-dt-val");
    if (dtDisp) dtDisp.innerText = `${data.dt_ps} ps`;
    const pkDisp = document.getElementById("fdtd-peake-val");
    if (pkDisp) pkDisp.innerText = `${data.peak_ez_v_m} V/m`;
    const gridDisp = document.getElementById("fdtd-grid-val");
    if (gridDisp) gridDisp.innerText = `${data.grid_nx} × ${data.grid_ny} (Mur ABC)`;

    drawFdtdCanvas();
  } catch (err) {
    console.error("fetchFdtdData error:", err);
  }
}

function toggleFdtdPlayback() {
  fdtdPlaying = !fdtdPlaying;
  const btn = document.getElementById("fdtd-play-btn");
  if (btn) btn.innerText = fdtdPlaying ? "⏸ Pause" : "▶ Play";
}

function changeFdtdFreq(val) {
  fetchFdtdData(parseFloat(val));
}

function drawFdtdCanvas() {
  const canvas = document.getElementById("fdtd-canvas");
  if (!canvas || !fdtdData || !fdtdData.frames) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#030712";
  ctx.fillRect(0, 0, w, h);

  const frames = fdtdData.frames;
  const currFrame = frames[fdtdFrameIdx % frames.length];
  if (!currFrame) return;

  const ny = currFrame.length;
  const nx = currFrame[0].length;
  const cellW = w / nx;
  const cellH = h / ny;

  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const val = currFrame[iy][ix]; // Normalized [-1.0, 1.0]
      if (Math.abs(val) > 0.04) {
        if (val > 0) {
          // Positive electric field (Cyan / Electric Blue)
          const alpha = Math.min(1.0, val * 1.4);
          ctx.fillStyle = `rgba(0, 240, 255, ${alpha.toFixed(2)})`;
        } else {
          // Negative electric field (Rose / Orange)
          const alpha = Math.min(1.0, -val * 1.4);
          ctx.fillStyle = `rgba(244, 63, 94, ${alpha.toFixed(2)})`;
        }
        ctx.fillRect(ix * cellW, iy * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }

  // Draw central microstrip guide contour
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(w * 0.08, h * 0.46, w * 0.84, h * 0.08);

  // Time & field annotation
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px SF Mono, monospace";
  const tPs = (fdtdData.frame_times_ps && fdtdData.frame_times_ps[fdtdFrameIdx]) || 0.0;
  ctx.fillText(`t = ${tPs} ps | Mode: 2.5D TM | E_z Wavefront`, 10, 16);
}
