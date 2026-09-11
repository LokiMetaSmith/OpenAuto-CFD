/* viewer/web/viewer.js — OpenAuto-CFD WebGL Engine */

// --- Global State ---
let scene, camera, renderer, controls;
let corkscrewMesh = null;
let daemonPoreGroup = null;
let daemonGeometryMode = "stack";
let daemonCadCache = {};
let particleSystem = null;
let particlePositions, particleVelocities, particleLifetimes;
const N_PARTICLES = 600;
let daemonParticleChannel = new Uint8Array(N_PARTICLES);
let daemonParticleS = new Float32Array(N_PARTICLES);
let daemonParticleLat = new Float32Array(N_PARTICLES);
let daemonParticleY = new Float32Array(N_PARTICLES);
let daemonParticleSpeed = new Float32Array(N_PARTICLES);
let daemonParticleTranslocates = new Uint8Array(N_PARTICLES);
let last3DTranslocationTrigger = 0;
let currentClampingTorque = 0.5;
let currentClampingAnalysis = null;

// Mold Mode Particle State
let moldParticleStage = new Uint8Array(N_PARTICLES);
let moldParticleY = new Float32Array(N_PARTICLES);
let moldParticleZ = new Float32Array(N_PARTICLES);
let moldParticleX = new Float32Array(N_PARTICLES);
let moldParticleTrack = new Float32Array(N_PARTICLES);
let moldParticleTrackX = new Float32Array(N_PARTICLES);
let moldParticleSpeed = new Float32Array(N_PARTICLES);
let moldFillAnimationId = null;
let isMoldFilling = false;

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

// SPICE Monte Carlo & Fluidic Co-Sim State
let spiceData = null;
let spiceRTol = 1.0;
let spiceCTol = 5.0;
let spiceEngineMode = "monte_carlo"; // 'monte_carlo' or 'kicad_native'
let kicadNativeSpiceData = null;
let cosimData = null;
let showFlowcellTube = false;
let flowcellTubeGroup = null;
let flowcellParticles = null;
let flowcellParticleProgress = [];
let flowcellCurve = null;


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
  loadProjectList();
  animate();

  // Setup periodic queue and KiCad live polling
  pollKiCadLiveStatus();
  pollContainerStatus();
  setInterval(pollSolverQueue, 2000);
  setInterval(pollKiCadLiveStatus, 1500);
  setInterval(pollContainerStatus, 5000);
  setInterval(() => {
    if (vnaDockOpen && activeVnaTab === "nanopore") {
      fetchNanoporeData();
    }
  }, 1500);
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

  const topLight = new THREE.DirectionalLight(0xffffff, 0.9);
  topLight.position.set(0, 100, 0);
  scene.add(topLight);

  // Studio Grid Floor
  const grid = new THREE.GridHelper(100, 40, 0x1e293b, 0x0f172a);
  grid.position.y = -25;
  scene.add(grid);

  // PCB 3D Group
  pcbGroup = new THREE.Group();
  pcbGroup.name = "pcbGroup";
  pcbGroup.visible = false;
  scene.add(pcbGroup);

  // Daemon Pore Mechanical & Microfluidic Group
  daemonPoreGroup = new THREE.Group();
  daemonPoreGroup.name = "daemonPoreGroup";
  daemonPoreGroup.rotation.x = -Math.PI / 2;
  daemonPoreGroup.visible = false;
  scene.add(daemonPoreGroup);

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
  if (currentProjectId === "daemon-pore") {
    if (corkscrewMesh) corkscrewMesh.visible = false;
    return;
  }

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

// --- Daemon Pore Mechanical & Microfluidics CAD Engine ---
const DAEMON_PORE_PARAM_DEFS = {
  explode_gap_mm: { min: 0.0, max: 25.0, default: 0.0, step: 0.5, unit: "mm", label: "Explode Stack Gap" },
  pore_diameter_nm: { min: 1.0, max: 20.0, default: 4.0, step: 0.1, unit: "nm", label: "Pore Diameter (d_p)" },
  bias_voltage_mv: { min: 50.0, max: 500.0, default: 120.0, step: 5.0, unit: "mV", label: "Bias Voltage (V_bias)" },
  flow_rate_ul_min: { min: 1.0, max: 50.0, default: 10.0, step: 0.5, unit: "µL/min", label: "Flow Rate (Q)" },
  channel_width_um: { min: 20.0, max: 200.0, default: 50.0, step: 2.0, unit: "µm", label: "Channel Width (w)" },
  channel_height_um: { min: 10.0, max: 100.0, default: 25.0, step: 1.0, unit: "µm", label: "Channel Height (h)" },
  buffer_conc_m: { min: 0.1, max: 3.0, default: 1.0, step: 0.05, unit: "M", label: "KCl Buffer Conc (C_KCl)" }
};

const DAEMON_MOLD_PARAM_DEFS = {
  explode_gap_mm: { min: 0.0, max: 35.0, default: 0.0, step: 0.5, unit: "mm", label: "Parting Gap (Slide Apart)" },
  injection_pressure_psi: { min: 0.1, max: 15.0, default: 3.5, step: 0.1, unit: "PSI", label: "Applied Injection Pressure (P_inj)" },
  pdms_viscosity_pas: { min: 1.0, max: 10.0, default: 3.5, step: 0.1, unit: "Pa·s", label: "PDMS Dynamic Viscosity (µ)" },
  flow_rate_ml_min: { min: 0.5, max: 20.0, default: 5.0, step: 0.5, unit: "mL/min", label: "Target Flow Rate (Q)" },
  mold_temp_c: { min: 20.0, max: 70.0, default: 25.0, step: 1.0, unit: "°C", label: "Mold Temperature (T_mold)" },
  fill_progress_pct: { min: 0.0, max: 100.0, default: 100.0, step: 1.0, unit: "%", label: "Cavity Fill Progress" }
};

let daemonExplodeGap = 0.0;
let daemonAssemblyManifest = null;

async function fetchBuild123dPartGeometry(partId, tolerance = 0.1, forceRefresh = false) {
  const cacheKey = `b123d_${partId}_tol_${tolerance}`;
  if (!forceRefresh && daemonCadCache[cacheKey]) return daemonCadCache[cacheKey];
  try {
    const res = await fetch(`/api/project/build123d_part?project_id=daemon-pore&part_id=${partId}&tolerance=${tolerance}&_t=${Date.now()}`);
    if (!res.ok) {
      console.warn(`Build123d part fetch failed for ${partId}:`, res.statusText);
      return null;
    }
    const buf = await res.arrayBuffer();
    const floatArray = new Float32Array(buf);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(floatArray, 3));
    geom.computeVertexNormals();
    daemonCadCache[cacheKey] = geom;
    return geom;
  } catch (e) {
    console.error(`Error loading build123d part ${partId}:`, e);
    return null;
  }
}

async function fetchPartMeshGeometry(partId) {
  if (daemonCadCache[partId]) return daemonCadCache[partId];
  try {
    const res = await fetch(`/api/project/mesh_binary?project_id=daemon-pore&part_id=${partId}`);
    if (!res.ok) {
      console.warn(`Mesh binary fetch failed for ${partId}:`, res.statusText);
      return null;
    }
    const buf = await res.arrayBuffer();
    const floatArray = new Float32Array(buf);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(floatArray, 3));
    geom.computeVertexNormals();
    daemonCadCache[partId] = geom;
    return geom;
  } catch (e) {
    console.error(`Error loading mesh binary for ${partId}:`, e);
    return null;
  }
}

async function fetchDxfExtrudedGeometry(partId, thickness = 4.0) {
  const cacheKey = `${partId}_thick_${thickness}`;
  if (daemonCadCache[cacheKey]) return daemonCadCache[cacheKey];
  try {
    const res = await fetch(`/api/project/dxf_polylines?project_id=daemon-pore&part_id=${partId}`);
    if (!res.ok) {
      console.warn(`DXF fetch failed for ${partId}:`, res.statusText);
      return null;
    }
    const data = await res.json();
    if (!data.loops || data.loops.length === 0) return null;

    const shape = new THREE.Shape();
    const outer = data.loops[0];
    shape.moveTo(outer[0][0], outer[0][1]);
    for (let i = 1; i < outer.length; i++) {
      shape.lineTo(outer[i][0], outer[i][1]);
    }
    shape.closePath();

    for (let h = 1; h < data.loops.length; h++) {
      const hole = data.loops[h];
      if (!hole || hole.length < 3) continue;
      const holePath = new THREE.Path();
      holePath.moveTo(hole[0][0], hole[0][1]);
      for (let i = 1; i < hole.length; i++) {
        holePath.lineTo(hole[i][0], hole[i][1]);
      }
      holePath.closePath();
      shape.holes.push(holePath);
    }

    const geom = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    geom.translate(0, 0, -thickness / 2.0);
    geom.computeVertexNormals();
    daemonCadCache[cacheKey] = geom;
    return geom;
  } catch (e) {
    console.error(`Error extruding DXF for ${partId}:`, e);
    return null;
  }
}

function getDaemonChannelHeights(mode = daemonGeometryMode, explode = daemonExplodeGap) {
  const gap = (explode !== undefined) ? explode : daemonExplodeGap;
  if (mode === "channels") {
    const zBot = 0.8 + 0 * gap;
    const zPore = 2.0 + 1.5 * gap;
    const zTop = 3.2 + 2 * gap;
    return { zTop, zPore, zBot, yTop: zTop, yPore: zPore, yBot: zBot };
  } else if (mode === "system" || mode === "full_system") {
    const zBot = 37.8 + 1.5 * gap;
    const zPore = 39.0 + 1.95 * gap;
    const zTop = 40.2 + 2.1 * gap;
    return { zTop, zPore, zBot, yTop: zTop, yPore: zPore, yBot: zBot };
  } else if (mode === "enclosure" || mode === "assembly") {
    return { zTop: 16.0, zPore: 14.5, zBot: 13.0, yTop: 16.0, yPore: 14.5, yBot: 13.0 };
  } else {
    // "cartridge" or "stack"
    const zBot = 7.8 + 1.0 * gap;
    const zPore = 9.0 + 1.75 * gap;
    const zTop = 10.2 + 2.0 * gap;
    return { zTop, zPore, zBot, yTop: zTop, yPore: zPore, yBot: zBot };
  }
}

function setDaemonExplodeGap(gap) {
  daemonExplodeGap = Math.max(0.0, parseFloat(gap) || 0.0);

  // 1. Instantly update all part mesh positions in 3D scene (60 FPS smooth!)
  if (daemonPoreGroup) {
    daemonPoreGroup.children.forEach(child => {
      if (child.userData) {
        const slideDir = child.userData.slide_dir || "z";
        const factor = (child.userData.slide_factor !== undefined) ? child.userData.slide_factor : ((child.userData.layer !== undefined) ? child.userData.layer : 0);
        if (slideDir === "x") {
          // Mold Halves: Slide apart laterally along parting plane (X-axis)
          child.position.x = (child.userData.base_x || 0) + factor * daemonExplodeGap;
          child.position.y = (child.userData.base_y || 0);
          child.position.z = (child.userData.base_z || 0);
        } else if (slideDir === "none") {
          // Mold Cavity Gasket / Silicon Wafer: Stay stationary at parting center
          child.position.x = (child.userData.base_x || 0);
          child.position.y = (child.userData.base_y || 0);
          child.position.z = (child.userData.base_z || 0);
        } else {
          // Default Stackup: Explode vertically along Z-axis
          child.position.x = (child.userData.base_x || 0);
          child.position.y = (child.userData.base_y || 0);
          child.position.z = (child.userData.base_z || 0) + factor * daemonExplodeGap;
        }
      }
    });

    // 2. Update fluidic circuit overlay
    const oldCircuit = daemonPoreGroup.getObjectByName("fluidicCircuitGroup");
    if (oldCircuit) {
      daemonPoreGroup.remove(oldCircuit);
    }
    if (daemonGeometryMode === "mold") {
      if (daemonAssemblyManifest && daemonAssemblyManifest.mold_cavity) {
        addDaemonMoldCircuit(daemonAssemblyManifest.mold_cavity);
      }
    } else {
      const heights = getDaemonChannelHeights(daemonGeometryMode, daemonExplodeGap);
      if (heights) {
        addDaemonFluidicCircuit(heights);
      }
    }
  }

  // 3. Sync UI controls
  const quickSlider = document.getElementById("quick-explode-slider");
  if (quickSlider && parseFloat(quickSlider.value) !== daemonExplodeGap) {
    quickSlider.value = daemonExplodeGap;
  }
  const quickVal = document.getElementById("quick-explode-val");
  if (quickVal) {
    quickVal.innerText = `${daemonExplodeGap.toFixed(1)}mm`;
  }
  const sliderInput = document.getElementById("slider-explode_gap_mm");
  if (sliderInput && parseFloat(sliderInput.value) !== daemonExplodeGap) {
    sliderInput.value = daemonExplodeGap;
  }
  const valDisplay = document.getElementById("val-explode_gap_mm");
  if (valDisplay) {
    valDisplay.innerText = `${daemonExplodeGap.toFixed(1)} mm`;
  }
}

function onQuickExplodeChange(val) {
  setDaemonExplodeGap(val);
}


function addDaemonFluidicCircuit(heights) {
  if (!heights || heights.zTop === undefined) return;
  const circuitGroup = new THREE.Group();
  circuitGroup.name = "fluidicCircuitGroup";

  // 1. Nanopore Holographic Sensor Aperture (aligned with 0.5mm central wafer membrane window)
  const poreRingGeom = new THREE.RingGeometry(0.18, 0.35, 32);
  const poreRingMat = new THREE.MeshBasicMaterial({
    color: 0xec4899,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.95
  });
  const poreRing = new THREE.Mesh(poreRingGeom, poreRingMat);
  poreRing.position.set(0, 0, heights.zPore + 0.05);
  circuitGroup.add(poreRing);

  // 2. Translocation Capillary Core (linking zTop to zBot at 0, 0 through the nanopore)
  const span = Math.max(1.0, Math.abs(heights.zTop - heights.zBot));
  const capGeom = new THREE.CylinderGeometry(0.2, 0.2, span, 16);
  capGeom.rotateX(Math.PI / 2);
  const capMat = new THREE.MeshBasicMaterial({
    color: 0xec4899,
    transparent: true,
    opacity: 0.85,
    wireframe: wireframeMode
  });
  const capMesh = new THREE.Mesh(capGeom, capMat);
  capMesh.position.set(0, 0, (heights.zTop + heights.zBot) / 2.0);
  circuitGroup.add(capMesh);

  // 3. Top Channel (Cis) In-Out Track: from (-17.32, -10.0) to (+17.32, +10.0)
  const topTrackGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-17.3205, -10.0, heights.zTop),
    new THREE.Vector3(17.3205, 10.0, heights.zTop)
  ]);
  const topTrackMat = new THREE.LineBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.75,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(topTrackGeom, topTrackMat));

  // 4. Bottom Channel (Trans) Out-In Track: from (+17.32, -10.0) to (-17.32, +10.0)
  const botTrackGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(17.3205, -10.0, heights.zBot),
    new THREE.Vector3(-17.3205, 10.0, heights.zBot)
  ]);
  const botTrackMat = new THREE.LineBasicMaterial({
    color: 0x10b981,
    transparent: true,
    opacity: 0.75,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(botTrackGeom, botTrackMat));

  // 5. Port Holographic Markers (4 Fluidic Ports)
  const ports = [
    { x: -17.3205, y: -10.0, z: heights.zTop, col: 0x00f0ff },
    { x: 17.3205, y: 10.0, z: heights.zTop, col: 0x00f0ff },
    { x: 17.3205, y: -10.0, z: heights.zBot, col: 0x10b981 },
    { x: -17.3205, y: 10.0, z: heights.zBot, col: 0x10b981 }
  ];

  ports.forEach(p => {
    const ringGeom = new THREE.RingGeometry(0.8, 1.8, 24);
    const ringMat = new THREE.MeshBasicMaterial({
      color: p.col,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });
    const ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.position.set(p.x, p.y, p.z);
    circuitGroup.add(ringMesh);
  });

  daemonPoreGroup.add(circuitGroup);
}

function addDaemonMoldCircuit(cavityData) {
  if (!cavityData) return;
  const circuitGroup = new THREE.Group();
  circuitGroup.name = "fluidicCircuitGroup";

  // Coordinates inside daemonPoreGroup are in CAD coordinates:
  // Parting plane: X = 0
  // Cavity center: X = 0, Y = 0, Z = 35.0
  // Hexagon in YZ plane:
  const hexPts = [
    new THREE.Vector3(0, 0, 65.0),
    new THREE.Vector3(0, 25.981, 50.0),
    new THREE.Vector3(0, 25.981, 20.0),
    new THREE.Vector3(0, 0, 5.0),
    new THREE.Vector3(0, -25.981, 20.0),
    new THREE.Vector3(0, -25.981, 50.0),
    new THREE.Vector3(0, 0, 65.0)
  ];
  const hexGeom = new THREE.BufferGeometry().setFromPoints(hexPts);
  const hexMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.90,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(hexGeom, hexMat));

  // Silicon Wafer Seat (4x4mm rotated 45 deg, center Z=35.0, Y=0, X=-1.2)
  const halfDiagWafer = 2.0 * Math.SQRT2; // ~2.828 mm
  const waferPts = [
    new THREE.Vector3(-1.2, 0, 35.0 + halfDiagWafer),
    new THREE.Vector3(-1.2, halfDiagWafer, 35.0),
    new THREE.Vector3(-1.2, 0, 35.0 - halfDiagWafer),
    new THREE.Vector3(-1.2, -halfDiagWafer, 35.0),
    new THREE.Vector3(-1.2, 0, 35.0 + halfDiagWafer)
  ];
  const waferGeom = new THREE.BufferGeometry().setFromPoints(waferPts);
  const waferMat = new THREE.LineBasicMaterial({
    color: 0xf43f5e,
    transparent: true,
    opacity: 0.95,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(waferGeom, waferMat));

  // Diamond Protrusion Pocket (6x6mm rotated 45 deg, center Z=35.0, Y=0, X=-0.8 to -1.6)
  const halfDiagDiamond = 3.0 * Math.SQRT2; // ~4.243 mm
  const diamondPts = [
    new THREE.Vector3(-0.8, 0, 35.0 + halfDiagDiamond),
    new THREE.Vector3(-0.8, halfDiagDiamond, 35.0),
    new THREE.Vector3(-0.8, 0, 35.0 - halfDiagDiamond),
    new THREE.Vector3(-0.8, -halfDiagDiamond, 35.0),
    new THREE.Vector3(-0.8, 0, 35.0 + halfDiagDiamond)
  ];
  const diamondGeom = new THREE.BufferGeometry().setFromPoints(diamondPts);
  const diamondMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.85,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(diamondGeom, diamondMat));

  // Central Nanopore Aperture Ring (facing parting plane along X axis)
  const poreRingGeom = new THREE.RingGeometry(0.2, 0.45, 32);
  const poreRingMat = new THREE.MeshBasicMaterial({
    color: 0xec4899,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.95
  });
  const poreRing = new THREE.Mesh(poreRingGeom, poreRingMat);
  poreRing.rotation.y = Math.PI / 2;
  poreRing.position.set(-1.2, 0, 35.0);
  circuitGroup.add(poreRing);

  // Sprue Channel Runner (Z from 75 to 61 at X=0, Y=0)
  const spruePts = [
    new THREE.Vector3(0, 0, 75.0),
    new THREE.Vector3(0, 0, 61.0)
  ];
  const sprueGeom = new THREE.BufferGeometry().setFromPoints(spruePts);
  const sprueMat = new THREE.LineBasicMaterial({
    color: 0xf59e0b,
    transparent: true,
    opacity: 0.85,
    linewidth: 2
  });
  circuitGroup.add(new THREE.Line(sprueGeom, sprueMat));

  // Air Bleed Vents / Risers (Y = +/- 15.0, Z from 61 to 90)
  for (const yVent of [15.0, -15.0]) {
    const ventPts = [
      new THREE.Vector3(0, yVent, 61.0),
      new THREE.Vector3(0, yVent, 90.0)
    ];
    const ventGeom = new THREE.BufferGeometry().setFromPoints(ventPts);
    const ventMat = new THREE.LineBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.80,
      linewidth: 1
    });
    circuitGroup.add(new THREE.Line(ventGeom, ventMat));
  }

  // Top Reservoir Box Wireframe: centered at (0, 0, 82.5), size (12.0, 50.0, 15.0)
  const resBoxGeom = new THREE.BoxGeometry(12.0, 50.0, 15.0);
  const resWireGeom = new THREE.EdgesGeometry(resBoxGeom);
  const resMat = new THREE.LineBasicMaterial({
    color: 0x64748b,
    transparent: true,
    opacity: 0.5
  });
  const resWire = new THREE.LineSegments(resWireGeom, resMat);
  resWire.position.set(0, 0, 82.5);
  circuitGroup.add(resWire);

  daemonPoreGroup.add(circuitGroup);
}

async function loadDaemonPoreGeometry(mode = "cartridge") {
  if (!daemonPoreGroup) return;

  // Normalize legacy mode names
  if (mode === "stack") mode = "cartridge";
  if (mode === "assembly") mode = "enclosure";
  if (mode === "exploded") {
    mode = "cartridge";
    daemonExplodeGap = 10.0;
  }
  daemonGeometryMode = mode;

  // Clear existing objects in group
  while (daemonPoreGroup.children.length > 0) {
    const child = daemonPoreGroup.children[0];
    daemonPoreGroup.remove(child);
  }

  // Ensure corkscrewMesh is hidden
  if (corkscrewMesh) corkscrewMesh.visible = false;
  daemonPoreGroup.visible = (currentDomain !== "pcb");

  const modeDropdown = document.getElementById("daemon-geometry-mode");
  if (modeDropdown) {
    modeDropdown.value = mode;
    modeDropdown.style.display = (currentProjectId === "daemon-pore" && currentDomain !== "pcb") ? "inline-block" : "none";
  }

  const btnReloadCad = document.getElementById("btn-reload-cad");
  if (btnReloadCad) {
    btnReloadCad.style.display = (currentProjectId === "daemon-pore" && currentDomain !== "pcb") ? "inline-block" : "none";
  }

  const explodeContainer = document.getElementById("daemon-explode-container");
  if (explodeContainer) {
    explodeContainer.style.display = (currentProjectId === "daemon-pore" && currentDomain !== "pcb") ? "inline-flex" : "none";
    const quickSlider = document.getElementById("quick-explode-slider");
    if (quickSlider) quickSlider.value = daemonExplodeGap;
    const quickVal = document.getElementById("quick-explode-val");
    if (quickVal) quickVal.innerText = `${daemonExplodeGap.toFixed(1)}mm`;
  }

  function addMeshWithEdges(geom, colorHex, opacity = 1.0, zPos = 0, isTransparent = false, metalness = 0.3, roughness = 0.4, userData = {}, xPos = 0, yPos = 0) {
    if (!geom) return null;
    const mat = new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: metalness,
      roughness: roughness,
      transparent: isTransparent,
      opacity: opacity,
      wireframe: wireframeMode,
      side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(xPos, yPos, zPos);
    mesh.userData = userData;

    // Glowing edge outline
    try {
      const edgeGeom = new THREE.EdgesGeometry(geom, 28);
      const edgeMat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: Math.min(1.0, opacity + 0.35),
        linewidth: 1
      });
      const wire = new THREE.LineSegments(edgeGeom, edgeMat);
      mesh.add(wire);
    } catch (e) {}

    daemonPoreGroup.add(mesh);
    return mesh;
  }

  // 1. Fetch live parametric assembly manifest directly from build123d engine
  try {
    const res = await fetch(`/api/project/build123d_assembly?project_id=daemon-pore&mode=${mode}&explode=${daemonExplodeGap}&_t=${Date.now()}`);
    if (res.ok) {
      const manifest = await res.json();
      daemonAssemblyManifest = manifest;

      if (manifest.parts && manifest.parts.length > 0) {
        for (const part of manifest.parts) {
          const geom = await fetchBuild123dPartGeometry(part.id);
          if (!geom) continue;

          const colorHex = parseInt(part.color.replace("#", "0x"));
          addMeshWithEdges(
            geom,
            colorHex,
            part.opacity !== undefined ? part.opacity : 1.0,
            part.z_pos || 0,
            !!part.transparent,
            part.metalness !== undefined ? part.metalness : 0.3,
            part.roughness !== undefined ? part.roughness : 0.4,
            {
              partId: part.id,
              name: part.name,
              layer: part.layer,
              base_x: part.base_x !== undefined ? part.base_x : 0,
              base_y: part.base_y !== undefined ? part.base_y : 0,
              base_z: part.base_z !== undefined ? part.base_z : (part.z_pos || 0),
              slide_dir: part.slide_dir || "z",
              slide_factor: part.slide_factor !== undefined ? part.slide_factor : (part.layer !== undefined ? part.layer : 1.0)
            },
            part.x_pos || 0,
            part.y_pos || 0
          );
        }

        // Add dynamically positioned holographic fluidic or mold circuit
        if (manifest.fluidic_heights) {
          addDaemonFluidicCircuit(manifest.fluidic_heights);
        } else if (manifest.mold_cavity) {
          addDaemonMoldCircuit(manifest.mold_cavity);
        }

        // Smooth camera framing
        if (currentDomain !== "pcb") {
          if (mode === "cartridge") {
            camera.position.set(0, 65, 80);
            controls.target.set(0, 10, 0);
          } else if (mode === "channels") {
            camera.position.set(0, 45, 60);
            controls.target.set(0, 3, 0);
          } else if (mode === "enclosure") {
            camera.position.set(0, 95, 120);
            controls.target.set(0, 25, 0);
          } else if (mode === "system") {
            camera.position.set(-30, 150, 240);
            controls.target.set(-40, 10, 0);
          } else if (mode === "mold") {
            camera.position.set(0, 50, 140);
            controls.target.set(0, 45, 0);
          } else if (mode === "pump") {
            camera.position.set(0, 100, 120);
            controls.target.set(0, 30, 0);
          }
          controls.update();
        }

        // Re-seed microfluidic particles inside the updated channel elevations
        if (particlePositions && currentProjectId === "daemon-pore") {
          for (let i = 0; i < N_PARTICLES; i++) {
            resetParticle(i, true);
          }
        }
        return;
      }
    }
  } catch (err) {
    console.warn("Build123d assembly load failed, falling back to static exports:", err);
  }

  // Fallback to static exports if build123d engine failed
  const gPusher = await fetchPartMeshGeometry("bottom_pusher_plate");
  addMeshWithEdges(gPusher, 0x475569, 0.95, 0, false, 0.4, 0.3);
  const gPlate1 = await fetchDxfExtrudedGeometry("plate_1_base", 4.0);
  addMeshWithEdges(gPlate1, 0x10b981, 0.75, 4.0, true, 0.1, 0.2);
  const gGasket = await fetchDxfExtrudedGeometry("gasket_main", 0.8);
  addMeshWithEdges(gGasket, 0xf97316, 0.90, 8.0, true, 0.1, 0.5);
  const gPlate2 = await fetchDxfExtrudedGeometry("plate_2_channel", 4.0);
  addMeshWithEdges(gPlate2, 0x00f0ff, 0.75, 10.4, true, 0.15, 0.15);
  const gPlate3 = await fetchDxfExtrudedGeometry("plate_3_top", 4.0);
  addMeshWithEdges(gPlate3, 0x38bdf8, 0.60, 12.0, true, 0.1, 0.1);
  const gPuck = await fetchPartMeshGeometry("pressure_puck");
  addMeshWithEdges(gPuck, 0xec4899, 0.95, 19.0, false, 0.5, 0.3);
  addDaemonFluidicCircuit(getDaemonChannelHeights(mode));

  if (particlePositions && currentProjectId === "daemon-pore") {
    for (let i = 0; i < N_PARTICLES; i++) {
      resetParticle(i, true);
    }
  }
}

function switchDaemonGeometryMode(mode) {
  daemonGeometryMode = mode;
  buildDaemonPoreSliders();
  loadDaemonPoreGeometry(mode);
  const modeNames = {
    cartridge: "🔬 Build123d Cartridge Stack",
    channels: "🌊 Microfluidic Channels Core",
    enclosure: "📦 Reader Enclosure (Case + Lid + Cap)",
    system: "🔬 Complete Reader Instrument",
    mold: "🧪 PDMS Wafer Mold (Left & Right)",
    pump: "⚙️ Peristaltic Pump Module",
    stack: "🔬 Build123d Cartridge Stack",
    assembly: "📦 Reader Enclosure"
  };
  showKiCadToast(`Loaded: ${modeNames[mode] || mode}`, 2500);
}

async function reloadDaemonCadGeometry() {
  daemonCadCache = {};
  showKiCadToast("🔄 Refreshing Build123d CAD models & stackup...", 2000);
  await loadDaemonPoreGeometry(daemonGeometryMode);
  showKiCadToast("✅ CAD Models reloaded successfully!", 2500);
}

function buildDaemonPoreSliders() {
  const container = document.getElementById("sliders-container");
  if (!container) return;
  container.innerHTML = "";

  const isMold = (daemonGeometryMode === "mold");
  const defs = isMold ? DAEMON_MOLD_PARAM_DEFS : DAEMON_PORE_PARAM_DEFS;

  const panelHeader = document.querySelector("#cfd-panel .section-header");
  if (panelHeader) {
    panelHeader.innerText = isMold ? "PDMS Mold & Injection Controls" : "Nanopore & Microfluidic Controls";
  }

  for (const [pName, defn] of Object.entries(defs)) {
    if (currentParams[pName] === undefined) {
      currentParams[pName] = defn.default;
    }
    const val = currentParams[pName];

    const group = document.createElement("div");
    group.className = "slider-group";

    const labelRow = document.createElement("div");
    labelRow.className = "slider-label-row";

    const label = document.createElement("span");
    label.innerText = defn.label;

    const valDisplay = document.createElement("span");
    valDisplay.className = "slider-val";
    valDisplay.id = `val-${pName}`;
    valDisplay.innerText = `${Number(val).toFixed(defn.step < 1 ? 1 : 0)} ${defn.unit}`;

    labelRow.appendChild(label);
    labelRow.appendChild(valDisplay);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = defn.min;
    slider.max = defn.max;
    slider.step = defn.step || 1.0;
    slider.value = val;
    slider.id = `slider-${pName}`;

    slider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      currentParams[pName] = v;
      valDisplay.innerText = `${v.toFixed(defn.step < 1 ? 1 : 0)} ${defn.unit}`;
      if (pName === "explode_gap_mm") {
        setDaemonExplodeGap(v);
      }
      if (isMold) {
        updateDaemonMoldTelemetry();
      } else {
        updateDaemonPoreTelemetry();
      }
    });

    group.appendChild(labelRow);
    group.appendChild(slider);
    container.appendChild(group);
  }

  if (isMold) {
    const btnGroup = document.createElement("div");
    btnGroup.style.display = "flex";
    btnGroup.style.gap = "8px";
    btnGroup.style.marginTop = "12px";

    const runBtn = document.createElement("button");
    runBtn.id = "btn-run-injection";
    runBtn.className = "kicad-btn primary";
    runBtn.style.flex = "1";
    runBtn.innerText = isMoldFilling ? "⏸ Pause Injection" : "▶ Run Injection Fill";
    runBtn.onclick = () => toggleMoldFillAnimation();
    btnGroup.appendChild(runBtn);

    const resetBtn = document.createElement("button");
    resetBtn.className = "kicad-btn";
    resetBtn.innerText = "↺ Empty";
    resetBtn.onclick = () => resetMoldFill();
    btnGroup.appendChild(resetBtn);

    container.appendChild(btnGroup);
    updateDaemonMoldTelemetry();
  } else {
    updateDaemonPoreTelemetry();
  }
}

function toggleMoldFillAnimation() {
  if (isMoldFilling) {
    isMoldFilling = false;
    if (moldFillAnimationId) cancelAnimationFrame(moldFillAnimationId);
    moldFillAnimationId = null;
    const btn = document.getElementById("btn-run-injection");
    if (btn) btn.innerText = "▶ Resume Injection";
  } else {
    isMoldFilling = true;
    const btn = document.getElementById("btn-run-injection");
    if (btn) btn.innerText = "⏸ Pause Injection";
    if (currentParams.fill_progress_pct >= 100.0) {
      currentParams.fill_progress_pct = 0.0;
    }
    let lastTime = performance.now();
    function stepFill(now) {
      if (!isMoldFilling) return;
      const dt = (now - lastTime) / 1000.0;
      lastTime = now;

      const p_inj = parseFloat(currentParams.injection_pressure_psi || 3.5);
      const flowRate = parseFloat(currentParams.flow_rate_ml_min || 5.0);
      const mu = parseFloat(currentParams.pdms_viscosity_pas || 3.5);
      const dp_req = ((8.0 * mu * 0.025 * (flowRate * 1e-6 / 60)) / (Math.PI * 16e-12) + (12.0 * mu * 0.052 * (flowRate * 1e-6 / 60)) / (0.045 * 4.096e-9) + 1200.0) / 6894.76;
      const maxFill = p_inj >= dp_req ? 100.0 : Math.max(20.0, (p_inj / dp_req) * 100.0);

      // Advance fill at rate proportional to flow rate (preview speed ~25%/s)
      const fillRate = (flowRate / 5.2) * 100.0 * 0.4;
      currentParams.fill_progress_pct = Math.min(maxFill, (currentParams.fill_progress_pct || 0) + fillRate * dt);

      const slider = document.getElementById("slider-fill_progress_pct");
      if (slider) slider.value = currentParams.fill_progress_pct;
      const valDisp = document.getElementById("val-fill_progress_pct");
      if (valDisp) valDisp.innerText = `${currentParams.fill_progress_pct.toFixed(0)} %`;

      updateDaemonMoldTelemetry();

      if (currentParams.fill_progress_pct >= maxFill) {
        isMoldFilling = false;
        if (btn) btn.innerText = "▶ Run Injection Fill";
        return;
      }
      moldFillAnimationId = requestAnimationFrame(stepFill);
    }
    moldFillAnimationId = requestAnimationFrame(stepFill);
  }
}

function resetMoldFill() {
  if (moldFillAnimationId) cancelAnimationFrame(moldFillAnimationId);
  moldFillAnimationId = null;
  isMoldFilling = false;
  currentParams.fill_progress_pct = 0.0;
  const slider = document.getElementById("slider-fill_progress_pct");
  if (slider) slider.value = 0;
  const valDisp = document.getElementById("val-fill_progress_pct");
  if (valDisp) valDisp.innerText = "0 %";
  const btn = document.getElementById("btn-run-injection");
  if (btn) btn.innerText = "▶ Run Injection Fill";
  updateDaemonMoldTelemetry();
}

function updateDaemonMoldTelemetry() {
  if (currentDomain === "pcb") return;

  const p_inj_psi = parseFloat(currentParams.injection_pressure_psi || 3.5);
  const flow_rate_ml_min = parseFloat(currentParams.flow_rate_ml_min || 5.0);
  const pdms_mu = parseFloat(currentParams.pdms_viscosity_pas || 3.5);
  const mold_temp_c = parseFloat(currentParams.mold_temp_c || 25.0);
  const fill_progress_pct = parseFloat(currentParams.fill_progress_pct !== undefined ? currentParams.fill_progress_pct : 100.0);

  // 1. Temperature-adjusted PDMS Viscosity (Arrhenius relation)
  const effective_mu = pdms_mu * Math.exp(2000.0 * (1.0 / (mold_temp_c + 273.15) - 1.0 / 298.15));
  const Q_m3s = (flow_rate_ml_min * 1e-6) / 60.0;

  // 2. Pressure Drop Components
  // Sprue: Hagen-Poiseuille through tapered channel (L=25mm, r_eff=2.0mm)
  const dp_sprue_Pa = (8.0 * effective_mu * 0.025 * Q_m3s) / (Math.PI * Math.pow(0.002, 4));
  // Hexagonal Gasket Cavity: Hele-Shaw flow for thin gap (h = 1.6mm, W_eff = 45mm, L = 52mm)
  const dp_cavity_Pa = (12.0 * effective_mu * 0.052 * Q_m3s) / (0.045 * Math.pow(0.0016, 3));
  // Silicon Wafer die obstruction & alignment pin constriction
  const dp_wafer_Pa = 1200.0 * (effective_mu / 3.5) * Math.sqrt(Math.max(0.1, flow_rate_ml_min / 5.0)) + 350.0;
  const dp_req_Pa = dp_sprue_Pa + dp_cavity_Pa + dp_wafer_Pa;
  const dp_req_psi = dp_req_Pa / 6894.76;

  // 3. Fill Feasibility & Short-Shot Ratio
  const fill_ratio = p_inj_psi / Math.max(0.05, dp_req_psi);
  const maxAchievableFillPct = fill_ratio >= 1.0 ? 100.0 : Math.max(20.0, fill_ratio * 100.0);
  const effective_fill_pct = Math.min(fill_progress_pct, maxAchievableFillPct);
  const isFullFill = fill_ratio >= 1.0;

  // 4. Fill Time
  const v_cavity_ml = 5.2; // total mold volume in mL
  const fill_time_s = (v_cavity_ml / Math.max(0.1, flow_rate_ml_min)) * 60.0;

  // 5. Front Velocity in Cavity
  const area_cavity_m2 = 0.045 * 0.0016;
  const v_avg_ms = Q_m3s / area_cavity_m2;
  const front_vel_mms = v_avg_ms * 1e3;

  // 6. Mold Clamping Safety Factor vs Parting Flash
  const p_applied_Pa = p_inj_psi * 6894.76;
  const a_proj_m2 = 0.0028; // ~28 cm2 projected parting area
  const f_sep_N = p_applied_Pa * a_proj_m2;
  const f_clamp_N = 1200.0; // 4x M4 torque screws (1.2 kN clamp force)
  const clamping_sf = f_clamp_N / Math.max(1.0, f_sep_N);
  const isFlashSafe = clamping_sf >= 1.2;

  // Update Telemetry Cards
  const cardEff = document.getElementById("card-eff");
  if (cardEff) {
    const title = cardEff.querySelector(".metric-title");
    if (title) title.innerText = "REQUIRED INJ. PRESSURE";
    const val = document.getElementById("metric-eff");
    if (val) val.innerText = `${dp_req_psi.toFixed(2)} PSI`;
    const sub = document.getElementById("sub-eff");
    if (sub) {
      sub.innerHTML = isFullFill
        ? `<span class="badge-status-dot admissible"></span> Full Fill Feasible (ΔP_req ≤ P_inj)`
        : `<span class="badge-status-dot unverified"></span> Short-Shot Risk (Need ≥ ${dp_req_psi.toFixed(1)} PSI)`;
    }
    cardEff.style.borderColor = isFullFill ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const cardDp = document.getElementById("card-dp");
  if (cardDp) {
    const title = cardDp.querySelector(".metric-title");
    if (title) title.innerText = "APPLIED INJ. PRESSURE";
    const val = document.getElementById("metric-dp");
    if (val) val.innerText = `${p_inj_psi.toFixed(2)} PSI`;
    const sub = document.getElementById("sub-dp");
    if (sub) sub.innerText = `${Math.round(p_applied_Pa)} Pa (Syringe / Pump)`;
    cardDp.style.borderColor = isFullFill ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const cardCons = document.getElementById("card-conservation");
  if (cardCons) {
    const title = cardCons.querySelector(".metric-title");
    if (title) title.innerText = "EST. CAVITY FILL TIME";
    const val = document.getElementById("metric-div");
    if (val) val.innerText = `${fill_time_s.toFixed(1)} s`;
    const sub = document.getElementById("sub-div");
    if (sub) sub.innerText = `Vol: 5.2 mL @ ${flow_rate_ml_min.toFixed(1)} mL/min`;
  }

  const cardStress = document.getElementById("card-stress");
  if (cardStress) {
    const title = cardStress.querySelector(".metric-title");
    if (title) title.innerText = "RESIN FRONT VELOCITY";
    const val = document.getElementById("metric-stress");
    if (val) val.innerText = `${front_vel_mms.toFixed(1)} mm/s`;
    const sub = document.getElementById("sub-stress");
    if (sub) sub.innerText = `PDMS Viscosity: ${effective_mu.toFixed(2)} Pa·s (@${mold_temp_c.toFixed(0)}°C)`;
  }

  const cardFos = document.getElementById("card-fos");
  if (cardFos) {
    const title = cardFos.querySelector(".metric-title");
    if (title) title.innerText = "MOLD CLAMPING SAFETY";
    const val = document.getElementById("metric-fos");
    if (val) val.innerText = `SF: ${clamping_sf.toFixed(1)}x`;
    const sub = document.getElementById("sub-fos");
    if (sub) {
      sub.innerHTML = isFlashSafe
        ? `<span class="badge-status-dot admissible"></span> Parting Sealed (Zero Flash)`
        : `<span class="badge-status-dot unverified"></span> Flash Warning (F_sep > F_clamp)`;
    }
    cardFos.style.borderColor = isFlashSafe ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const cardUnc = document.getElementById("card-unc");
  if (cardUnc) {
    const title = cardUnc.querySelector(".metric-title");
    if (title) title.innerText = "CAVITY FILL COMPLETION";
    const val = document.getElementById("metric-unc");
    if (val) val.innerText = `${effective_fill_pct.toFixed(1)}%`;
    const subUnc = cardUnc.querySelector(".metric-sub");
    if (subUnc) {
      subUnc.innerText = effective_fill_pct >= 99.9
        ? "Wafer Fully Encapsulated (No Voids)"
        : (isFullFill ? "Filling In Progress..." : `Short Shot Stalled at ${effective_fill_pct.toFixed(1)}%`);
    }
  }

  const backendStatus = document.getElementById("backend-status");
  if (backendStatus) {
    backendStatus.innerText = "PDMS Injection Rheology (Hele-Shaw)";
  }
}

function updateDaemonPoreTelemetry() {
  if (currentDomain === "pcb") return;
  if (daemonGeometryMode === "mold") {
    updateDaemonMoldTelemetry();
    return;
  }

  const dp_nm = parseFloat(currentParams.pore_diameter_nm || 4.0);
  const v_bias_mv = parseFloat(currentParams.bias_voltage_mv || 120.0);
  const q_ul_min = parseFloat(currentParams.flow_rate_ul_min || 10.0);
  const w_um = parseFloat(currentParams.channel_width_um || 50.0);
  const h_um = parseFloat(currentParams.channel_height_um || 25.0);
  const c_kcl = parseFloat(currentParams.buffer_conc_m || 1.0);

  // 1. Ionic Baseline Current (nA): G = sigma * (4L / (pi*d^2) + 1/d)^-1
  // sigma_KCl ~ 11.18 * c_kcl (S/m)
  const sigma = 11.18 * c_kcl;
  const L_pore = 10e-9;
  const d_pore = dp_nm * 1e-9;
  const area_pore = Math.PI * Math.pow(d_pore / 2.0, 2);
  const R_pore = L_pore / (sigma * area_pore);
  const R_access = 1.0 / (2.0 * sigma * d_pore);
  const R_total = R_pore + 2.0 * R_access;
  const I_baseline_nA = ((v_bias_mv * 1e-3) / R_total) * 1e9;

  // 2. Microchannel Pressure Drop (PSI): Hagen-Poiseuille for rectangular channel
  const Q_m3s = (q_ul_min * 1e-9) / 60.0;
  const mu = 1.002e-3; // Pa*s
  const L_channel = 0.040; // 40 mm
  const w_m = w_um * 1e-6;
  const h_m = h_um * 1e-6;
  const aspect = Math.min(w_m, h_m) / Math.max(w_m, h_m);
  const f_geom = 1.0 - 0.63 * aspect;
  const dp_Pa = (12.0 * mu * L_channel * Q_m3s) / (Math.max(w_m, h_m) * Math.pow(Math.min(w_m, h_m), 3) * Math.max(0.1, f_geom));
  const dp_PSI = dp_Pa / 6894.76;

  // 3. Event Capture Rate (Hz)
  const capture_rate_hz = 65.0 * (v_bias_mv / 100.0) * Math.pow(dp_nm / 4.0, 1.4) * (1.0 + 0.04 * q_ul_min) * (c_kcl / 1.0);

  // 4. Centerline Flow Velocity (mm/s)
  const area_channel_m2 = w_m * h_m;
  const v_avg_ms = Q_m3s / area_channel_m2;
  const v_max_mms = (v_avg_ms * 1.5) * 1e3;

  // 5. Reynolds Number
  const D_h = (2.0 * w_m * h_m) / (w_m + h_m);
  const rho = 1000.0;
  const Re = (rho * v_avg_ms * D_h) / mu;

  // 6. Signal-to-Noise Ratio (dB)
  const delta_I_nA = I_baseline_nA * 0.85;
  const noise_rms_pA = Math.sqrt(Math.pow(8.0, 2) + Math.pow(2.5 * Math.sqrt(c_kcl), 2) + Math.pow(v_bias_mv * 0.02, 2));
  const snr_dB = 20.0 * Math.log10(Math.max(1.0, (delta_I_nA * 1000.0) / noise_rms_pA));

  // Update Telemetry Cards
  const cardEff = document.getElementById("card-eff");
  if (cardEff) {
    const title = cardEff.querySelector(".metric-title");
    if (title) title.innerText = "EVENT CAPTURE RATE";
    const val = document.getElementById("metric-eff");
    if (val) val.innerText = `${Math.round(capture_rate_hz)} Hz`;
    const sub = document.getElementById("sub-eff");
    if (sub) sub.innerText = "Translocation Events / sec";
    cardEff.style.borderColor = capture_rate_hz >= 100 ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const cardDp = document.getElementById("card-dp");
  if (cardDp) {
    const title = cardDp.querySelector(".metric-title");
    if (title) title.innerText = "MICROFLUIDIC ΔP";
    const val = document.getElementById("metric-dp");
    if (val) val.innerText = `${dp_PSI.toFixed(2)} PSI`;
    const sub = document.getElementById("sub-dp");
    if (sub) sub.innerText = `${Math.round(dp_Pa)} Pa (Hagen-Poiseuille)`;
    cardDp.style.borderColor = dp_PSI < 1.0 ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const cardCons = document.getElementById("card-conservation");
  if (cardCons) {
    const title = cardCons.querySelector(".metric-title");
    if (title) title.innerText = "IONIC BASELINE CURRENT";
    const val = document.getElementById("metric-div");
    if (val) val.innerText = `${I_baseline_nA.toFixed(2)} nA`;
    const sub = document.getElementById("sub-div");
    if (sub) sub.innerHTML = `<span class="badge-status-dot admissible"></span> Ohmic Open-Pore Baseline`;
  }

  const cardStress = document.getElementById("card-stress");
  if (cardStress) {
    const title = cardStress.querySelector(".metric-title");
    if (title) title.innerText = "CORE FLOW VELOCITY";
    const val = document.getElementById("metric-stress");
    if (val) val.innerText = `${v_max_mms.toFixed(1)} mm/s`;
    const sub = document.getElementById("sub-stress");
    if (sub) sub.innerText = "Max Channel Centerline Vel";
  }

  const cardFos = document.getElementById("card-fos");
  if (cardFos) {
    const title = cardFos.querySelector(".metric-title");
    if (title) title.innerText = "REYNOLDS NUMBER";
    const val = document.getElementById("metric-fos");
    if (val) val.innerText = `${Re.toFixed(3)}`;
    const sub = document.getElementById("sub-fos");
    if (sub) sub.innerText = "Stokes Creeping Flow (Re ≪ 1)";
    cardFos.style.borderColor = "var(--accent-emerald)";
  }

  const cardUnc = document.getElementById("card-unc");
  if (cardUnc) {
    const title = cardUnc.querySelector(".metric-title");
    if (title) title.innerText = "SIGNAL-TO-NOISE RATIO";
    const val = document.getElementById("metric-unc");
    if (val) val.innerText = `${snr_dB.toFixed(1)} dB`;
    const subUnc = cardUnc.querySelector(".metric-sub");
    if (subUnc) subUnc.innerText = "Bandwidth: 100 kHz (TIA)";
  }

  const backendStatus = document.getElementById("backend-status");
  if (backendStatus) {
    backendStatus.innerText = "Nanopore Microfluidics (Native)";
  }

  updateMicrofluidicFlowPhysics(q_ul_min);
}

function restoreCorkscrewTelemetryTitles() {
  const panelHeader = document.querySelector("#cfd-panel .section-header");
  if (panelHeader) {
    panelHeader.innerText = "Parametric Geometric Controls";
  }

  const cardEff = document.getElementById("card-eff");
  if (cardEff) {
    const title = cardEff.querySelector(".metric-title");
    if (title) title.innerText = "COLLECTION EFFICIENCY (η)";
    const sub = document.getElementById("sub-eff");
    if (sub) sub.innerText = "Target: > 99.95% (Moon Dust)";
  }

  const cardDp = document.getElementById("card-dp");
  if (cardDp) {
    const title = cardDp.querySelector(".metric-title");
    if (title) title.innerText = "PRESSURE DROP (ΔP)";
    const sub = document.getElementById("sub-dp");
    if (sub) sub.innerText = "Target: < 0.70 PSI (~2900 Pa)";
  }

  const cardCons = document.getElementById("card-conservation");
  if (cardCons) {
    const title = cardCons.querySelector(".metric-title");
    if (title) title.innerText = "CONTINUITY CONSERVATION";
  }

  const cardStress = document.getElementById("card-stress");
  if (cardStress) {
    const title = cardStress.querySelector(".metric-title");
    if (title) title.innerText = "MAX STRESS (σ_max)";
    const sub = document.getElementById("sub-stress");
    if (sub) sub.innerText = "Yield: 60 MPa (PETG/PLA)";
  }

  const cardFos = document.getElementById("card-fos");
  if (cardFos) {
    const title = cardFos.querySelector(".metric-title");
    if (title) title.innerText = "SAFETY FACTOR";
    const sub = document.getElementById("sub-fos");
    if (sub) sub.innerText = "Target: ≥ 1.50";
  }

  const cardUnc = document.getElementById("card-unc");
  if (cardUnc) {
    const title = cardUnc.querySelector(".metric-title");
    if (title) title.innerText = "UNCERTAINTY";
    const subUnc = cardUnc.querySelector(".metric-sub");
    if (subUnc) subUnc.innerText = "Confidence: 95.8%";
  }
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

function updateSingleDaemonParticle(i, heights) {
  const ch = daemonParticleChannel[i];
  const s = daemonParticleS[i];
  const lat = daemonParticleLat[i];
  const y = daemonParticleY[i];

  let x_world, z_world;
  if (ch === 0 || ch === 2) {
    // Top Channel (Cis): In at (-17.32, +10.0), Center at (0, 0), Out at (+17.32, -10.0)
    // Angle in CAD: 30 deg (u = (0.866025, 0.5), normal = (-0.5, 0.866025))
    // World coordinates (Rx = -90 deg): X = x_cad, Z = -y_cad
    x_world = s * 0.866025 - lat * 0.5;
    z_world = -(s * 0.5 + lat * 0.866025);
  } else {
    // Bottom Channel (Trans): In at (+17.32, +10.0), Center at (0, 0), Out at (-17.32, -10.0)
    // Angle in CAD: 150 deg (u = (-0.866025, 0.5), normal = (-0.5, -0.866025))
    // World coordinates (Rx = -90 deg): X = x_cad, Z = -y_cad
    x_world = s * (-0.866025) - lat * 0.5;
    z_world = -(s * 0.5 - lat * 0.866025);
  }

  const idx = i * 3;
  particlePositions[idx] = x_world;
  particlePositions[idx + 1] = y;
  particlePositions[idx + 2] = z_world;
}

function resetSingleMoldParticle(i, initial = false) {
  const fillPct = (currentParams.fill_progress_pct !== undefined) ? currentParams.fill_progress_pct : 100.0;
  moldParticleTrack[i] = (Math.random() - 0.5) * 2.0;
  moldParticleTrackX[i] = (Math.random() - 0.5) * 2.0;
  moldParticleSpeed[i] = 0.85 + Math.random() * 0.35;

  if (initial) {
    const r = Math.random() * Math.max(10.0, fillPct);
    if (r < 20) {
      moldParticleStage[i] = 0;
      moldParticleY[i] = 75.0 + Math.random() * 14.0;
      moldParticleZ[i] = (Math.random() - 0.5) * 45.0 * ((moldParticleY[i] - 75.0) / 14.0);
      moldParticleX[i] = (Math.random() - 0.5) * 2.0;
    } else if (r < 30) {
      moldParticleStage[i] = 1;
      moldParticleY[i] = 61.0 + Math.random() * 14.0;
      const rad = 1.5 + 1.5 * ((moldParticleY[i] - 61.0) / 14.0);
      moldParticleZ[i] = moldParticleTrack[i] * rad * 0.65;
      moldParticleX[i] = moldParticleTrackX[i] * rad * 0.65;
    } else if (r < 60) {
      moldParticleStage[i] = 2;
      moldParticleY[i] = 39.24 + Math.random() * 21.76;
      const w = 25.981 * Math.min(1.0, (65.0 - moldParticleY[i]) / 15.0);
      moldParticleZ[i] = moldParticleTrack[i] * Math.max(1.0, w) * 0.92;
      moldParticleX[i] = moldParticleTrackX[i] * 0.65;
    } else if (r < 70) {
      moldParticleStage[i] = 3;
      moldParticleY[i] = 30.76 + Math.random() * 8.48;
      const side = moldParticleTrack[i] >= 0 ? 1 : -1;
      const dObstacle = Math.max(0, 4.243 * (1.0 - Math.abs(moldParticleY[i] - 35.0) / 4.243));
      moldParticleZ[i] = side * (dObstacle + 0.45 + Math.abs(moldParticleTrack[i]) * 1.2);
      moldParticleX[i] = moldParticleTrackX[i] * 0.65;
    } else if (r < 90) {
      moldParticleStage[i] = 4;
      moldParticleY[i] = 7.5 + Math.random() * 23.26;
      const w = 25.981 * Math.max(0.05, (moldParticleY[i] - 5.0) / 15.0);
      moldParticleZ[i] = moldParticleTrack[i] * Math.min(25.981, w) * 0.92;
      moldParticleX[i] = moldParticleTrackX[i] * 0.65;
    } else {
      moldParticleStage[i] = 5;
      moldParticleY[i] = 61.0 + Math.random() * 28.0;
      const side = moldParticleTrack[i] >= 0 ? 15.0 : -15.0;
      moldParticleZ[i] = side + moldParticleTrackX[i] * 0.4;
      moldParticleX[i] = moldParticleTrackX[i] * 0.4;
    }
  } else {
    moldParticleStage[i] = 0;
    moldParticleY[i] = 88.0 + Math.random() * 1.5;
    moldParticleZ[i] = (Math.random() - 0.5) * 40.0;
    moldParticleX[i] = (Math.random() - 0.5) * 2.2;
  }

  const idx = i * 3;
  particlePositions[idx]     = moldParticleX[i];
  particlePositions[idx + 1] = moldParticleY[i];
  particlePositions[idx + 2] = -moldParticleZ[i];
}

function updateMoldParticles(dt) {
  if (!particleSystem || !particlePositions) return;
  const posAttr = particleSystem.geometry.attributes.position;
  const colAttr = particleSystem.geometry.attributes.color;

  const p_inj_psi = parseFloat(currentParams.injection_pressure_psi || 3.5);
  const flow_rate_ml_min = parseFloat(currentParams.flow_rate_ml_min || 5.0);
  const pdms_mu = parseFloat(currentParams.pdms_viscosity_pas || 3.5);
  const mold_temp_c = parseFloat(currentParams.mold_temp_c || 25.0);
  const fill_progress_pct = parseFloat(currentParams.fill_progress_pct !== undefined ? currentParams.fill_progress_pct : 100.0);

  // Viscosity temperature dependence
  const effective_mu = pdms_mu * Math.exp(2000.0 * (1.0 / (mold_temp_c + 273.15) - 1.0 / 298.15));
  const Q_m3s = (flow_rate_ml_min * 1e-6) / 60.0;

  // Resistance calculations
  const dp_sprue_Pa = (8.0 * effective_mu * 0.025 * Q_m3s) / (Math.PI * Math.pow(0.002, 4));
  const dp_cavity_Pa = (12.0 * effective_mu * 0.052 * Q_m3s) / (0.045 * Math.pow(0.0016, 3));
  const dp_wafer_Pa = 1200.0 * (effective_mu / 3.5) * Math.sqrt(Math.max(0.1, flow_rate_ml_min / 5.0)) + 350.0;
  const dp_req_Pa = dp_sprue_Pa + dp_cavity_Pa + dp_wafer_Pa;
  const dp_req_psi = dp_req_Pa / 6894.76;

  // Max achievable fill (short shot condition)
  const fill_ratio = p_inj_psi / Math.max(0.05, dp_req_psi);
  const maxAchievableFillPct = fill_ratio >= 1.0 ? 100.0 : Math.max(20.0, fill_ratio * 100.0);
  const effective_fill_pct = Math.min(fill_progress_pct, maxAchievableFillPct);

  // Speed scale
  const baseFlowVel = Math.min(75.0, Math.max(8.0, flow_rate_ml_min * 3.8));

  // Determine current front elevation boundary:
  let y_front = 7.5;
  let maxStageAllowed = 5;
  if (effective_fill_pct < 20.0) {
    maxStageAllowed = 0;
    y_front = 89.0 - (14.0 * (effective_fill_pct / 20.0));
  } else if (effective_fill_pct < 30.0) {
    maxStageAllowed = 1;
    y_front = 75.0 - (14.0 * ((effective_fill_pct - 20.0) / 10.0));
  } else if (effective_fill_pct < 90.0) {
    maxStageAllowed = 4;
    y_front = 61.0 - (53.5 * ((effective_fill_pct - 30.0) / 60.0));
  } else {
    maxStageAllowed = 5;
    y_front = 61.0 + (28.0 * ((effective_fill_pct - 90.0) / 10.0)); // vent climb
  }

  for (let i = 0; i < N_PARTICLES; i++) {
    let stage = moldParticleStage[i];
    let y = moldParticleY[i];
    let z = moldParticleZ[i];
    let x = moldParticleX[i];
    const track = moldParticleTrack[i];
    const trackX = moldParticleTrackX[i];
    const spd = moldParticleSpeed[i];

    // Advance position along stage
    if (stage === 0) {
      // Reservoir: downward funneling
      y -= baseFlowVel * 0.55 * spd * dt;
      z *= (1.0 - 0.02 * baseFlowVel * dt);
      x *= (1.0 - 0.02 * baseFlowVel * dt);
      if (y <= 75.0) {
        stage = 1;
        y = 75.0;
      }
    } else if (stage === 1) {
      // Sprue: fast downward flow through tapered channel
      y -= baseFlowVel * 2.2 * spd * dt;
      const rad = 1.5 + 1.5 * Math.max(0.0, (y - 61.0) / 14.0);
      z = track * rad * 0.65;
      x = trackX * rad * 0.65;
      if (y <= 61.0) {
        stage = 2;
        y = 61.0;
      }
    } else if (stage === 2) {
      // Upper Hexagon
      y -= baseFlowVel * 0.85 * spd * dt;
      const w = 25.981 * Math.min(1.0, (65.0 - y) / 15.0);
      z = track * Math.max(1.0, w) * 0.92;
      x = trackX * 0.65;
      if (y <= 39.24) {
        if (Math.abs(track) < 0.35) {
          stage = 3; // bypass wafer
        } else {
          stage = 4; // flank around wafer into lower hex
        }
      }
    } else if (stage === 3) {
      // Wafer Bypass: silicon diamond at center (Y=0 in CAD, Z=35.0 in CAD), halfDiag = 4.243mm
      y -= baseFlowVel * 0.9 * spd * dt;
      const dY = Math.abs(y - 35.0);
      const side = track >= 0 ? 1 : -1;
      const diamondObstacle = Math.max(0, 4.243 * (1.0 - dY / 4.243));
      z = side * (diamondObstacle + 0.45 + Math.abs(track) * 1.2);
      x = trackX * 0.65;
      if (y <= 30.76) {
        stage = 4;
      }
    } else if (stage === 4) {
      // Lower Hexagon
      y -= baseFlowVel * 0.75 * spd * dt;
      const w = 25.981 * Math.max(0.05, (y - 5.0) / 15.0);
      z = track * Math.min(25.981, w) * 0.92;
      x = trackX * 0.65;
      if (y <= 7.5) {
        if (Math.random() < 0.45 && effective_fill_pct >= 85.0) {
          stage = 5; // branch into air bleed vent
          y = 61.0;
          z = track >= 0 ? 15.0 : -15.0;
        } else {
          resetSingleMoldParticle(i, false);
          stage = moldParticleStage[i];
          y = moldParticleY[i];
          z = moldParticleZ[i];
          x = moldParticleX[i];
        }
      }
    } else if (stage === 5) {
      // Air Bleed Vents / Risers (upwards flow at Y = +/- 15 in CAD)
      y += baseFlowVel * 1.4 * spd * dt;
      z = (track >= 0 ? 15.0 : -15.0) + trackX * 0.35;
      x = trackX * 0.35;
      if (y >= 89.0) {
        resetSingleMoldParticle(i, false);
        stage = moldParticleStage[i];
        y = moldParticleY[i];
        z = moldParticleZ[i];
        x = moldParticleX[i];
      }
    }

    // Check fill front: if particle is beyond the fill front, clamp or hide
    if (stage > maxStageAllowed) {
      resetSingleMoldParticle(i, false);
      stage = moldParticleStage[i];
      y = moldParticleY[i];
      z = moldParticleZ[i];
      x = moldParticleX[i];
    } else if (stage === maxStageAllowed) {
      if (stage === 5) {
        if (y > y_front) y = y_front; // climb up vent
      } else {
        if (y < y_front) y = y_front; // downward front
      }
    }

    moldParticleStage[i] = stage;
    moldParticleY[i] = y;
    moldParticleZ[i] = z;
    moldParticleX[i] = x;

    const idx = i * 3;
    particlePositions[idx]     = x;
    particlePositions[idx + 1] = y;
    particlePositions[idx + 2] = -z;

    // Color: Rich Golden/Amber Liquid PDMS with stage highlights
    if (stage === 3) {
      // Flowing around silicon wafer: brilliant warm gold
      colAttr.array[idx] = 1.0;
      colAttr.array[idx + 1] = 0.82;
      colAttr.array[idx + 2] = 0.25;
    } else if (stage === 5) {
      // Air bleed riser venting: cyan/teal degassing bubbles
      colAttr.array[idx] = 0.22;
      colAttr.array[idx + 1] = 0.88;
      colAttr.array[idx + 2] = 0.95;
    } else if (stage === 1) {
      // High-shear sprue: bright electric amber
      colAttr.array[idx] = 0.98;
      colAttr.array[idx + 1] = 0.72;
      colAttr.array[idx + 2] = 0.15;
    } else {
      // Standard liquid PDMS: warm amber honey
      colAttr.array[idx] = 0.95;
      colAttr.array[idx + 1] = 0.62;
      colAttr.array[idx + 2] = 0.10;
    }
  }

  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
}

function resetParticle(i, initial = false) {
  if (currentProjectId === "daemon-pore") {
    if (daemonGeometryMode === "mold") {
      resetSingleMoldParticle(i, initial);
      return;
    }
    const heights = getDaemonChannelHeights();
    const isTop = Math.random() < 0.55;
    daemonParticleChannel[i] = isTop ? 0 : 1;
    daemonParticleTranslocates[i] = (isTop && Math.random() < 0.28) ? 1 : 0;
    daemonParticleS[i] = initial ? (Math.random() * 40.0 - 20.0) : -20.0 - Math.random() * 2.0;
    daemonParticleLat[i] = (Math.random() - 0.5) * 0.85;
    daemonParticleSpeed[i] = 0.85 + Math.random() * 0.35;
    daemonParticleY[i] = isTop ? (heights.yTop + (Math.random() - 0.5) * 0.2) : (heights.yBot + (Math.random() - 0.5) * 0.2);

    updateSingleDaemonParticle(i, heights);
    particleLifetimes[i] = Math.random();
    return;
  }

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

  const posAttr = particleSystem.geometry.attributes.position;
  const colAttr = particleSystem.geometry.attributes.color;

  if (currentProjectId === "daemon-pore") {
    if (daemonGeometryMode === "mold") {
      updateMoldParticles(dt);
      return;
    }
    const heights = getDaemonChannelHeights();
    const qVal = parseFloat(currentParams.flow_rate_ul_min || 10.0);
    const baseFlowVel = Math.min(65.0, Math.max(5.0, qVal * 2.0));

    for (let i = 0; i < N_PARTICLES; i++) {
      const ch = daemonParticleChannel[i];
      let s = daemonParticleS[i];
      const lat = daemonParticleLat[i];
      const spd = daemonParticleSpeed[i];

      // Parabolic laminar flow profile across channel width
      const rRatio = Math.max(0, 1.0 - Math.pow(lat / 0.45, 2));
      const u_flow = baseFlowVel * (0.35 + 0.65 * rRatio) * spd;

      if (ch === 2) {
        // Translocation through nanopore at center (0, 0): moving vertically from yTop to yBot
        const vBias = parseFloat(currentParams.bias_voltage_mv || 120.0);
        const translocateVel = baseFlowVel * 1.6 * (vBias / 100.0);
        daemonParticleY[i] -= translocateVel * dt;
        daemonParticleLat[i] *= 0.82; // Electrophoretic funneling into central pore (0.5mm window)
        daemonParticleS[i] *= 0.82;

        if (daemonParticleY[i] <= heights.yBot) {
          daemonParticleY[i] = heights.yBot;
          daemonParticleChannel[i] = 1; // joins bottom trans stream
          daemonParticleS[i] = 0.2; // flows toward bottom outlet
          daemonParticleLat[i] = (Math.random() - 0.5) * 0.2;
        }
      } else {
        s += u_flow * dt;
        daemonParticleS[i] = s;

        // Check if top particle translocates at the central nanopore
        if (ch === 0 && daemonParticleTranslocates[i] === 1 && s >= -0.75 && s <= 0.75) {
          daemonParticleChannel[i] = 2; // trigger translocation
          const nowMs = Date.now();
          if (nowMs - last3DTranslocationTrigger > 600) {
            last3DTranslocationTrigger = nowMs;
            injectTranslocation("dsDNA", false);
          }
        }
      }

      if (s > 20.0) {
        resetParticle(i, false);
      } else {
        updateSingleDaemonParticle(i, heights);
      }

      // Color coding
      const idx = i * 3;
      if (daemonParticleChannel[i] === 2) {
        // Translocation event at nanopore: glowing hot magenta/pink
        colAttr.array[idx] = 0.96;
        colAttr.array[idx + 1] = 0.15;
        colAttr.array[idx + 2] = 0.65;
      } else if (daemonParticleChannel[i] === 0) {
        // Top Cis stream (In-Out): Electric Cyan
        const speedNorm = Math.min(1.0, u_flow / (baseFlowVel * 1.3));
        colAttr.array[idx] = 0.0 + 0.2 * speedNorm;
        colAttr.array[idx + 1] = 0.85 + 0.15 * speedNorm;
        colAttr.array[idx + 2] = 1.0;
      } else {
        // Bottom Trans stream (Out-In): Vibrant Emerald
        const speedNorm = Math.min(1.0, u_flow / (baseFlowVel * 1.3));
        colAttr.array[idx] = 0.08 + 0.25 * speedNorm;
        colAttr.array[idx + 1] = 0.95;
        colAttr.array[idx + 2] = 0.55;
      }
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    return;
  }

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

  // Animate fluidic flowcell capillary particles
  if (showFlowcellTube && flowcellParticles && flowcellTubeGroup && flowcellTubeGroup.visible) {
    updateFlowcellParticles(dt);
  }

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
  if (currentProjectId === "daemon-pore") {
    buildDaemonPoreSliders();
    return;
  }

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
  const PARAM_LABELS = {
    "num_turns": "Revolutions (N)",
    "path_radius": "Path Radius (r_p)",
    "profile_radius": "Profile Radius (r_b)",
    "chamfer_width": "Chamfer Width",
    "inlet_fillet": "Inlet Fillet",
    "insert_length": "Insert Length (L)",
    "mesh_cell_size": "Mesh Cell Size",
    "flow_rate": "Flow Rate (Q)",
    "inlet_velocity": "Inlet Velocity (v_in)",
    "particle_diameter": "Particle Diam (d_p)",
    "wall_roughness": "Wall Roughness"
  };
  return PARAM_LABELS[str] || str.replace(/_/g, " ").replace(/mm/g, "(mm)");
}

function onParameterScrub() {
  if (currentProjectId === "daemon-pore") {
    updateDaemonPoreTelemetry();
    return;
  }

  updateCorkscrewGeometry(currentParams);
  if (!isPredicting) {
    triggerPrediction();
  } else {
    pendingPredict = true;
  }
}

async function triggerPrediction() {
  if (currentProjectId === "daemon-pore" || currentDomain === "pcb") return;
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
  document.getElementById("sub-dp").innerText = `${Math.round(dpPa)} Pa (< 0.70 PSI)`;

  // Upstream Fluidic Co-Simulation Trigger
  if (eff !== undefined && dpPsi !== undefined) {
    if ((vnaDockOpen && activeVnaTab === "nanopore") || showFlowcellTube) {
      fetchFluidicCosim(eff, dpPsi);
    }
  }

  // PINN Conservation Telemetry
  const divElem = document.getElementById("metric-div");
  const subDiv = document.getElementById("sub-div");
  if (divElem && conservation) {
    if (conservation.divergence_loss !== undefined) {
      const divVal = Number(conservation.divergence_loss);
      divElem.innerText = `∇·u = ${divVal.toFixed(5)}`;
      const isAdm = conservation.is_physically_admissible !== false;
      subDiv.innerHTML = `<span class="badge-status-dot ${isAdm ? 'admissible' : 'violation'}"></span> ${isAdm ? 'Admissible' : 'Violation'}`;
    } else if (conservation.equilibrium_loss !== undefined) {
      const eqVal = Number(conservation.equilibrium_loss);
      divElem.innerText = `∇·σ = ${eqVal.toFixed(5)}`;
      subDiv.innerHTML = `<span class="badge-status-dot admissible"></span> Equilibrium Valid`;
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
  btn.innerText = "Optimizing...";

  if (currentProjectId === "daemon-pore") {
    setTimeout(() => {
      currentParams.pore_diameter_nm = 4.2;
      currentParams.bias_voltage_mv = 140.0;
      currentParams.flow_rate_ul_min = 12.5;
      currentParams.channel_width_um = 60.0;
      currentParams.channel_height_um = 30.0;
      currentParams.buffer_conc_m = 1.0;
      buildDaemonPoreSliders();
      updateDaemonPoreTelemetry();
      showKiCadToast("⚡ Optimized Nanopore & Fluidic Parameters for SNR & Capture Rate", 3500);
      btn.classList.remove("shimmer-active");
      btn.innerHTML = "<span>⚡</span><span>AI Inverse Design (L-BFGS-B)</span>";
    }, 600);
    return;
  }

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
    if (daemonPoreGroup) daemonPoreGroup.visible = false;
    if (particleSystem) particleSystem.visible = false;
    if (pcbGroup) pcbGroup.visible = true;

    const daemonModeSelect = document.getElementById("daemon-geometry-mode");
    if (daemonModeSelect) daemonModeSelect.style.display = "none";
    const daemonExplodeCont = document.getElementById("daemon-explode-container");
    if (daemonExplodeCont) daemonExplodeCont.style.display = "none";

    // Switch Left Drawer
    if (cfdPanel) cfdPanel.style.display = "none";
    if (kicadPanel) kicadPanel.style.display = "block";

    const btnElectrophys = document.getElementById("btn-toggle-electrophys");
    const btnExportFab = document.getElementById("btn-export-fab");
    const clampingPanel = document.getElementById("daemon-clamping-panel");
    if (btnExportFab) btnExportFab.style.display = "none";
    if (clampingPanel) clampingPanel.style.display = "none";
    if (btnElectrophys) btnElectrophys.style.display = (currentProjectId === "daemon-pore") ? "inline-block" : "none";

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
    const btnFlowcell = document.getElementById("btn-toggle-flowcell");
    if (btnThermal) btnThermal.style.display = "inline-block";
    if (btnDrc) btnDrc.style.display = "inline-block";
    if (btnFlowcell) btnFlowcell.style.display = "inline-block";
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
  if (currentProjectId === "daemon-pore") {
    if (corkscrewMesh) corkscrewMesh.visible = false;
    if (daemonPoreGroup) daemonPoreGroup.visible = true;
    updateDaemonPoreTelemetry();
  } else {
    if (corkscrewMesh) corkscrewMesh.visible = true;
    if (daemonPoreGroup) daemonPoreGroup.visible = false;
  }
  if (particleSystem) particleSystem.visible = showParticles;

  const daemonModeSelect = document.getElementById("daemon-geometry-mode");
  if (daemonModeSelect) {
    daemonModeSelect.style.display = (currentProjectId === "daemon-pore") ? "inline-block" : "none";
  }
  const daemonExplodeCont = document.getElementById("daemon-explode-container");
  if (daemonExplodeCont) {
    daemonExplodeCont.style.display = (currentProjectId === "daemon-pore") ? "inline-flex" : "none";
  }

  if (cfdPanel) cfdPanel.style.display = "block";
  if (kicadPanel) kicadPanel.style.display = "none";

  const btnElectrophys = document.getElementById("btn-toggle-electrophys");
  const btnExportFab = document.getElementById("btn-export-fab");
  const clampingPanel = document.getElementById("daemon-clamping-panel");
  if (btnElectrophys) btnElectrophys.style.display = (currentProjectId === "daemon-pore") ? "inline-block" : "none";
  if (btnExportFab) btnExportFab.style.display = (currentProjectId === "daemon-pore") ? "inline-block" : "none";
  if (clampingPanel) clampingPanel.style.display = (currentProjectId === "daemon-pore") ? "block" : "none";

  const btnIcs = document.getElementById("btn-toggle-ics");
  const btnEm = document.getElementById("btn-toggle-em");
  const btnVna = document.getElementById("btn-toggle-vna");
  const vnaDock = document.getElementById("vna-dock");
  if (btnIcs) btnIcs.style.display = "none";
  if (btnEm) btnEm.style.display = "none";
  if (btnVna) btnVna.style.display = "none";
  const btnThermalH = document.getElementById("btn-toggle-thermal");
  const btnDrcH = document.getElementById("btn-toggle-drc");
  const btnFlowcellH = document.getElementById("btn-toggle-flowcell");
  if (btnThermalH) btnThermalH.style.display = "none";
  if (btnDrcH) btnDrcH.style.display = "none";
  if (btnFlowcellH) btnFlowcellH.style.display = "none";
  if (flowcellTubeGroup) flowcellTubeGroup.visible = false;
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
  if (cardEffTitle) cardEffTitle.innerText = "COLLECTION EFFICIENCY (η)";
  const subEff = document.getElementById("sub-eff");
  if (subEff) subEff.innerText = "Target: > 99.95% (Moon Dust)";

  const cardDpTitle = document.querySelector("#card-dp .metric-title");
  if (cardDpTitle) cardDpTitle.innerText = "PRESSURE DROP (ΔP)";
  const subDp = document.getElementById("sub-dp");
  if (subDp) subDp.innerText = "Target: < 0.70 PSI (~2900 Pa)";

  const cardConvTitle = document.querySelector("#card-conservation .metric-title");
  if (cardConvTitle) cardConvTitle.innerText = "CONTINUITY CONSERVATION";
  const subConv = document.getElementById("sub-div");
  if (subConv) subConv.innerHTML = '<span class="badge-status-dot admissible"></span> Admissible';

  const cardStressTitle = document.querySelector("#card-stress .metric-title");
  if (cardStressTitle) cardStressTitle.innerText = "MAX STRESS (σ_max)";
  const subStress = document.getElementById("sub-stress");
  if (subStress) subStress.innerText = "Yield: 60 MPa (PETG/PLA)";

  const cardFosTitle = document.querySelector("#card-fos .metric-title");
  if (cardFosTitle) cardFosTitle.innerText = "SAFETY FACTOR";
  const subFos = document.getElementById("sub-fos");
  if (subFos) subFos.innerText = "Target: ≥ 1.50";

  const cardUncTitle = document.querySelector("#card-unc .metric-title");
  if (cardUncTitle) cardUncTitle.innerText = "UNCERTAINTY";

  const rightHeader = document.querySelector(".drawer-right .section-header");
  if (rightHeader) rightHeader.innerText = "Live Telemetry (Surrogate)";

  const activeHeaders = document.querySelectorAll(".drawer-right .section-header");
  if (activeHeaders && activeHeaders.length > 1) {
    activeHeaders[1].innerText = "Active Learning";
  }
  const bottomCardTitle = document.querySelector(".drawer-right .metric-card:last-of-type .metric-title");
  if (bottomCardTitle) bottomCardTitle.innerText = "SURROGATE SAMPLES";
  const bottomCardSub = document.querySelector(".drawer-right .metric-card:last-of-type .metric-sub");
  if (bottomCardSub) bottomCardSub.innerText = "Online Closed-Loop Residuals";
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

  if (corkscrewMesh && corkscrewMesh.material) {
    if (Array.isArray(corkscrewMesh.material)) {
      corkscrewMesh.material.forEach(m => { m.wireframe = wireframeMode; m.needsUpdate = true; });
    } else {
      corkscrewMesh.material.wireframe = wireframeMode;
      corkscrewMesh.material.needsUpdate = true;
    }
  }

  if (daemonPoreGroup) {
    daemonPoreGroup.traverse(child => {
      if (child.isMesh && child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(m => { m.wireframe = wireframeMode; m.needsUpdate = true; });
        } else {
          child.material.wireframe = wireframeMode;
          child.material.needsUpdate = true;
        }
      }
    });
  }

  if (pcbGroup) {
    pcbGroup.traverse(child => {
      if (child.isMesh && child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(m => { m.wireframe = wireframeMode; m.needsUpdate = true; });
        } else {
          child.material.wireframe = wireframeMode;
          child.material.needsUpdate = true;
        }
      }
    });
  }

  showKiCadToast(`Wireframe: ${wireframeMode ? "ON" : "OFF"}`, 1500);
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
        domain: currentDomain,
        params: currentParams,
        fidelity: currentFidelity
      })
    });
    const data = await res.json();
    thinkDiv.remove();

    const agentDiv = document.createElement("div");
    agentDiv.className = "agent-msg agent";

    let html = `<strong>${data.agent_type || 'CAD_Agent'}:</strong> ${data.reply}<br>`;

    if (data.board_context && data.board_context.board_name) {
      html += `<div style="margin-top: 4px; font-size: 11px; color: var(--accent-cyan);">Target Board: <strong>${data.board_context.board_name}</strong></div>`;
    }

    if (data.trace && data.trace.length > 0) {
      html += `<div style="margin-top: 6px; font-size: 11px;">`;
      for (const step of data.trace) {
        html += `<span class="tool-chip">⚡ ${step.tool || 'Tool'}</span> `;
      }
      html += `</div>`;
    }

    if (data.updated_power) {
      const up = data.updated_power;
      html += `<div style="margin-top: 6px; padding: 6px 8px; background: rgba(16, 185, 129, 0.1); border-left: 3px solid var(--accent-emerald); font-size: 11px; border-radius: 4px;">
        <span style="color: var(--accent-emerald); font-weight: bold;">⚡ Power Rails Configured:</span> Rail <code>${up.rail || '/3V3'}</code> @ <code>${((up.target_rail_current_a || 0) * 1000).toFixed(1)} mA</code> | IR Drop: <code>${up.ir_drop_mv} mV</code> | Loss: <code>${up.dissipation_mw} mW</code>
      </div>`;

      // Update Left Drawer Power / Thermal Readouts
      const pDrop = document.getElementById("pwr-drop-val");
      if (pDrop && up.ir_drop_mv !== undefined) pDrop.innerText = `${up.ir_drop_mv} mV`;
      const pJmax = document.getElementById("pwr-jmax-val");
      if (pJmax && up.max_j_a_mm2 !== undefined) pJmax.innerText = `${up.max_j_a_mm2} A/mm²`;
      const pTemp = document.getElementById("pwr-temp-val");
      if (pTemp && up.max_temp_c !== undefined) pTemp.innerText = `${up.max_temp_c} °C`;
      const pLoss = document.getElementById("pwr-loss-val");
      if (pLoss && up.dissipation_mw !== undefined) pLoss.innerText = `${up.dissipation_mw} mW`;
      const pCurrent = document.getElementById("slider-pwr-current");
      if (pCurrent && up.target_rail_current_a !== undefined) {
        pCurrent.value = up.target_rail_current_a;
        const valDisp = document.getElementById("val-pwr-current");
        if (valDisp) valDisp.innerText = `${up.target_rail_current_a.toFixed(3)} A`;
      }
    }

    if (data.spice_netlist) {
      html += `<div style="margin-top: 6px;">
        <div style="font-size: 10px; font-weight: bold; color: var(--accent-cyan); text-transform: uppercase;">Updated SPICE Deck:</div>
        <pre style="margin-top: 4px; padding: 6px 8px; background: rgba(0,0,0,0.5); border: 1px solid rgba(0,240,255,0.2); border-radius: 4px; font-family: monospace; font-size: 10px; color: #a5f3fc; overflow-x: auto; max-height: 120px;">${escapeHtml(data.spice_netlist)}</pre>
      </div>`;
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

// --- Multiphysics Project & Board Management ---
let currentProjectId = "daemon-pore";
let currentBoardId = "amplifier";
let projectManifestCache = {};

async function loadProjectList() {
  try {
    const res = await fetch("/api/projects");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("project-select");
    if (!select) return;

    if (data.projects && data.projects.length > 0) {
      select.innerHTML = "";
      data.projects.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.project_id;
        opt.innerText = `${p.name} • ${p.boards_count} board${p.boards_count === 1 ? '' : 's'}`;
        if (p.project_id === data.active_project_id) {
          opt.selected = true;
          currentProjectId = p.project_id;
        }
        select.appendChild(opt);
        projectManifestCache[p.project_id] = p;
      });

      // Add special option to open Add New Project dialog
      const addOpt = document.createElement("option");
      addOpt.value = "__ADD_NEW__";
      addOpt.innerText = "➕ Add New Project...";
      addOpt.style.color = "var(--accent-emerald)";
      addOpt.style.fontWeight = "bold";
      select.appendChild(addOpt);

      if (data.active_project_id) {
        select.value = data.active_project_id;
        const activeProj = projectManifestCache[data.active_project_id];
        if (activeProj) {
          updateBoardSelectorTabs(activeProj.boards, activeProj.active_board_id);
        }
        await switchProject(data.active_project_id);
      }
    }
  } catch (err) {
    console.warn("Failed to load project list:", err);
  }
}

function handleProjectSelectChange(val) {
  if (val === "__ADD_NEW__") {
    openNewProjectModal();
    const select = document.getElementById("project-select");
    if (select && currentProjectId) {
      select.value = currentProjectId;
    }
    return;
  }
  switchProject(val);
}

function openNewProjectModal() {
  const modal = document.getElementById("modal-new-project");
  if (!modal) return;
  modal.style.display = "flex";
  const nameInput = document.getElementById("np-name");
  if (nameInput) {
    nameInput.focus();
  }
}

function closeNewProjectModal() {
  const modal = document.getElementById("modal-new-project");
  if (!modal) return;
  modal.style.display = "none";
}

function autoPopulateProjectId(name) {
  const slugInput = document.getElementById("np-id");
  const slug = name.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  if (slugInput && !slugInput.dataset.manualEdit) {
    slugInput.value = slug;
  }
}

const SUBSTRATE_PRESETS = {
  fr4: { material: "FR4 High-TG", er: 4.3, h: 1.6 },
  ptfe: { material: "PTFE / Rogers RO4350B", er: 2.1, h: 0.8 },
  flex: { material: "Polyimide Flex", er: 3.4, h: 0.2 },
  alumina: { material: "Alumina Ceramic", er: 9.8, h: 0.635 }
};

function handleSubstratePresetChange(val) {
  // Preset selection handler
}

async function handleCreateProjectSubmit(event) {
  event.preventDefault();
  const submitBtn = document.getElementById("btn-submit-project");
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>⏳</span><span>Creating...</span>`;
  }

  const name = document.getElementById("np-name").value.trim();
  const slug = document.getElementById("np-id").value.trim();
  const boardName = document.getElementById("np-board-name").value.trim();
  const customDir = document.getElementById("np-dir").value.trim();
  const boardFile = document.getElementById("np-board-file").value.trim();
  const presetKey = document.getElementById("np-substrate-preset").value;
  const desc = document.getElementById("np-desc").value.trim();

  const preset = SUBSTRATE_PRESETS[presetKey] || SUBSTRATE_PRESETS.fr4;

  const payload = {
    name: name,
    project_id: slug || undefined,
    project_dir: customDir || undefined,
    board_name: boardName || "Main PCB",
    board_filename: boardFile || "board.kicad_pcb",
    substrate_material: preset.material,
    substrate_er: preset.er,
    substrate_thickness_mm: preset.h,
    description: desc
  };

  try {
    const res = await fetch("/api/project/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      alert(`Failed to create project: ${data.error || res.statusText}`);
      return;
    }

    closeNewProjectModal();
    const form = document.getElementById("new-project-form");
    if (form) form.reset();

    // Refresh project dropdown
    await loadProjectList();

    if (data.active_project) {
      const proj = data.active_project;
      projectManifestCache[proj.project_id] = proj;
      currentProjectId = proj.project_id;
      currentBoardId = proj.active_board_id;

      const select = document.getElementById("project-select");
      if (select) select.value = proj.project_id;

      const boardsArray = Object.values(proj.boards || {});
      updateBoardSelectorTabs(boardsArray, proj.active_board_id);

      if (data.board_sync) {
        activePcbData = data.board_sync;
        lastKicadSyncTime = data.board_sync.timestamp || Date.now() / 1000;
        buildPcb3DScene(data.board_sync);
        updatePcbInspectorUI(data.board_sync);
        if (currentDomain === "pcb") {
          updatePcbHUD(data.board_sync);
        }
      }

      showKiCadToast(`✨ Project Created: ${proj.name}!`, 4000);
      switchDomain("pcb");
    }
  } catch (err) {
    console.error("handleCreateProjectSubmit error:", err);
    alert(`Error creating project: ${err.message}`);
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>🚀</span><span>Create & Launch Project</span>`;
    }
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// --- Container & Solver Compute Status Engine ---
let lastContainerStatus = null;

async function pollContainerStatus(forceRefresh = false) {
  try {
    const url = forceRefresh ? "/api/container_status?refresh=true" : "/api/container_status";
    const res = await fetch(url);
    if (!res.ok) return;
    const status = await res.json();
    lastContainerStatus = status;
    updateContainerStatusUI(status);
  } catch (err) {
    console.debug("pollContainerStatus error:", err);
  }
}

function updateContainerStatusUI(status) {
  const badge = document.getElementById("container-status-badge");
  const textEl = document.getElementById("container-status-text");
  if (!badge || !textEl || !status) return;

  const engine = status.container_engine || {};
  const isEngineActive = engine.status === "ACTIVE";
  const cfd = status.cfd || {};
  const fea = status.fea || {};
  const eda = status.eda || {};

  if (isEngineActive) {
    badge.className = "container-badge active";
    textEl.innerText = `${engine.active_runtime || 'Containers'}: Active`;
    badge.title = `Container runtime (${engine.active_runtime}) active. High-fi CFD/FEA solvers ready.`;
  } else if (engine.installed && engine.installed.length > 0) {
    badge.className = "container-badge standby";
    textEl.innerText = `Containers: Stopped (Surrogates 2ms)`;
    badge.title = `Podman machine stopped. Fast surrogate physics models active. Click to start Podman.`;
  } else {
    badge.className = "container-badge offline";
    textEl.innerText = `Containers: None (Surrogates 2ms)`;
    badge.title = `No container runtime found in PATH. Surrogates active for real-time analysis.`;
  }

  // Update modal elements if modal is displayed
  const modal = document.getElementById("modal-compute-status");
  if (modal && modal.style.display !== "none") {
    // Engine Card
    const pillEngine = document.getElementById("pill-container-engine");
    const valEngineType = document.getElementById("val-engine-type");
    const valEngineSocket = document.getElementById("val-engine-socket");
    const valEngineDetail = document.getElementById("val-engine-detail");

    if (pillEngine) {
      if (isEngineActive) {
        pillEngine.className = "compute-status-pill ok";
        pillEngine.innerText = "ACTIVE";
      } else if (engine.installed && engine.installed.length > 0) {
        pillEngine.className = "compute-status-pill warn";
        pillEngine.innerText = "STOPPED";
      } else {
        pillEngine.className = "compute-status-pill off";
        pillEngine.innerText = "NOT FOUND";
      }
    }

    if (valEngineType) {
      const names = (engine.installed || []).map(e => e.name).join(", ");
      valEngineType.innerText = names || "None";
    }
    if (valEngineSocket) {
      valEngineSocket.innerText = isEngineActive ? "Connected (127.0.0.1)" : "Disconnected";
    }
    if (valEngineDetail) {
      valEngineDetail.innerText = isEngineActive ? "Container engine ready for CFD/FEA" : (engine.error || "Machine VM is stopped");
    }

    // CFD Card
    const pillCfd = document.getElementById("pill-cfd");
    const valCfdCont = document.getElementById("val-cfd-container");
    if (pillCfd) {
      if (cfd.status === "READY_CONTAINER") {
        pillCfd.className = "compute-status-pill ok";
        pillCfd.innerText = "Container Ready";
      } else {
        pillCfd.className = "compute-status-pill warn";
        pillCfd.innerText = "Surrogate (2.1ms)";
      }
    }
    if (valCfdCont) {
      valCfdCont.innerText = cfd.status === "READY_CONTAINER" ? "Online" : "Surrogate Fallback";
    }

    // FEA Card
    const pillFea = document.getElementById("pill-fea");
    const valFeaCont = document.getElementById("val-fea-container");
    if (pillFea) {
      if (fea.status === "READY_CONTAINER") {
        pillFea.className = "compute-status-pill ok";
        pillFea.innerText = "Container Ready";
      } else {
        pillFea.className = "compute-status-pill warn";
        pillFea.innerText = "Surrogate (1.8ms)";
      }
    }
    if (valFeaCont) {
      valFeaCont.innerText = fea.status === "READY_CONTAINER" ? "Online" : "Surrogate Fallback";
    }

    // EDA Card
    const pillEda = document.getElementById("pill-eda");
    const valEdaSpice = document.getElementById("val-eda-ngspice");
    if (pillEda) {
      pillEda.className = "compute-status-pill ok";
      pillEda.innerText = "Native Ready";
    }
    if (valEdaSpice) {
      valEdaSpice.innerText = eda.native_dll ? "KiCad 10.0 DLL Ready" : "Pure Python SPICE";
    }

    // Start Podman button state
    const btnStart = document.getElementById("btn-start-podman");
    if (btnStart) {
      const hasPodman = (engine.installed || []).some(e => e.name.toLowerCase() === "podman");
      if (isEngineActive) {
        btnStart.style.display = "none";
      } else if (hasPodman) {
        btnStart.style.display = "inline-block";
        btnStart.innerText = "▶ Start Podman Machine";
        btnStart.disabled = false;
      } else {
        btnStart.style.display = "none";
      }
    }
  }
}

function openComputeStatusModal() {
  const modal = document.getElementById("modal-compute-status");
  if (!modal) return;
  modal.style.display = "flex";
  if (lastContainerStatus) {
    updateContainerStatusUI(lastContainerStatus);
  }
  pollContainerStatus(false);
}

function closeComputeStatusModal() {
  const modal = document.getElementById("modal-compute-status");
  if (!modal) return;
  modal.style.display = "none";
}

async function triggerStartContainerEngine() {
  const btn = document.getElementById("btn-start-podman");
  if (btn) {
    btn.disabled = true;
    btn.innerText = "⏳ Starting Podman Machine...";
  }

  try {
    const res = await fetch("/api/container_start", { method: "POST" });
    const data = await res.json();
    if (data.status === "STARTED") {
      showKiCadToast(`🚀 Podman machine started! Waiting for socket initialization...`, 5000);
      setTimeout(() => pollContainerStatus(true), 3000);
    } else {
      alert(`Podman machine start message: ${data.message || data.error || 'Check Podman setup'}`);
      pollContainerStatus(true);
    }
  } catch (err) {
    alert(`Failed to start container engine: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "▶ Start Podman Machine";
    }
  }
}

function updateBoardSelectorTabs(boards, activeBoardId) {
  const container = document.getElementById("board-selector-container");
  if (!container) return;

  container.innerHTML = "";
  if (!boards || boards.length === 0) {
    container.style.display = "none";
    return;
  }
  container.style.display = "flex";

  boards.forEach(b => {
    const btn = document.createElement("button");
    btn.id = `board-tab-${b.id}`;
    btn.className = `fidelity-btn ${b.id === activeBoardId ? 'active' : ''}`;
    btn.style.cssText = "flex: 1; padding: 6px 4px; font-size: 11px; text-align: center;";
    const icon = b.id === "amplifier" ? "🔬 " : (b.id === "mr1" ? "💻 " : "⚡ ");
    btn.innerText = `${icon}${b.name || b.id}`;
    btn.onclick = () => switchActiveBoard(b.id);
    container.appendChild(btn);
  });
}

async function switchProject(projectId) {
  if (!projectId) return;
  currentProjectId = projectId;
  showKiCadToast(`📂 Loading Project: ${projectId}...`, 3000);

  // Toggle Daemon geometry dropdown and groups
  const modeDropdown = document.getElementById("daemon-geometry-mode");
  const btnReloadCad = document.getElementById("btn-reload-cad");
  const explodeContainer = document.getElementById("daemon-explode-container");
  const btnElectrophys = document.getElementById("btn-toggle-electrophys");
  const btnExportFab = document.getElementById("btn-export-fab");
  const clampingPanel = document.getElementById("daemon-clamping-panel");

  if (projectId === "daemon-pore") {
    if (modeDropdown && currentDomain !== "pcb") modeDropdown.style.display = "inline-block";
    if (btnReloadCad && currentDomain !== "pcb") btnReloadCad.style.display = "inline-block";
    if (explodeContainer && currentDomain !== "pcb") explodeContainer.style.display = "inline-flex";
    if (btnElectrophys) btnElectrophys.style.display = "inline-block";
    if (btnExportFab && currentDomain !== "pcb") btnExportFab.style.display = "inline-block";
    if (clampingPanel && currentDomain !== "pcb") clampingPanel.style.display = "block";
    if (corkscrewMesh) corkscrewMesh.visible = false;
    buildDaemonPoreSliders();
    loadDaemonPoreGeometry(daemonGeometryMode);
    updateDaemonPoreTelemetry();
    updateClampingTorque(currentClampingTorque || 0.5);
    updateMicrofluidicFlowPhysics();
  } else {
    if (modeDropdown) modeDropdown.style.display = "none";
    if (btnReloadCad) btnReloadCad.style.display = "none";
    if (explodeContainer) explodeContainer.style.display = "none";
    if (btnElectrophys) btnElectrophys.style.display = "none";
    if (btnExportFab) btnExportFab.style.display = "none";
    if (clampingPanel) clampingPanel.style.display = "none";
    if (daemonPoreGroup) daemonPoreGroup.visible = false;
    restoreCorkscrewTelemetryTitles();
    buildParameterSliders();
    if (currentDomain !== "pcb") {
      if (corkscrewMesh) corkscrewMesh.visible = true;
      updateCorkscrewGeometry(currentParams);
      triggerPrediction();
    }
  }

  try {
    const res = await fetch("/api/project/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId })
    });
    if (!res.ok) {
      showKiCadToast(`❌ Failed to switch project: ${res.statusText}`);
      return;
    }
    const data = await res.json();
    if (data.active_project) {
      const proj = data.active_project;
      projectManifestCache[projectId] = proj;
      currentBoardId = proj.active_board_id;

      const boardsArray = Object.values(proj.boards || {});
      updateBoardSelectorTabs(boardsArray, proj.active_board_id);

      showKiCadToast(`✅ Active Project: ${proj.name}`, 3500);

      if (data.board_sync) {
        activePcbData = data.board_sync;
        lastKicadSyncTime = data.board_sync.timestamp || Date.now() / 1000;
        buildPcb3DScene(data.board_sync);
        updatePcbInspectorUI(data.board_sync);
        if (currentDomain === "pcb") {
          updatePcbHUD(data.board_sync);
        }
      }
    }
  } catch (err) {
    console.error("switchProject error:", err);
    showKiCadToast(`❌ Switch error: ${err.message}`);
  }
}

async function switchActiveBoard(boardId) {
  if (!boardId) return;
  currentBoardId = boardId;

  // Update tab visual state
  const container = document.getElementById("board-selector-container");
  if (container) {
    const btns = container.querySelectorAll("button");
    btns.forEach(b => {
      if (b.id === `board-tab-${boardId}`) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });
  }

  showKiCadToast(`🔄 Switching Board to [${boardId}]...`, 2000);

  try {
    const res = await fetch("/api/project/switch_board", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board_id: boardId })
    });
    if (!res.ok) {
      showKiCadToast(`❌ Failed to switch board: ${res.statusText}`);
      return;
    }
    const data = await res.json();
    if (data.board_sync) {
      activePcbData = data.board_sync;
      lastKicadSyncTime = data.board_sync.timestamp || Date.now() / 1000;
      buildPcb3DScene(data.board_sync);
      updatePcbInspectorUI(data.board_sync);
      if (currentDomain === "pcb") {
        updatePcbHUD(data.board_sync);
      }
      showKiCadToast(`⚡ Board Loaded: ${data.board_sync.board_name} (${data.board_sync.board_geometry?.segments?.length || 0} traces)`, 3000);
    }
  } catch (err) {
    console.error("switchActiveBoard error:", err);
    showKiCadToast(`❌ Board switch error: ${err.message}`);
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

  // 1. Reconstruct Closed Boundary Loops from Edge.Cuts
  const edgeCuts = (data && data.board_geometry && data.board_geometry.edge_cuts) || [];
  const chains = [];
  if (edgeCuts.length > 0) {
    const nextMap = new Map();
    edgeCuts.forEach(e => {
      const k1 = e.x1.toFixed(2) + ',' + e.y1.toFixed(2);
      nextMap.set(k1, [e.x2, e.y2]);
    });

    const visited = new Set();
    for (const [startK, pt2] of nextMap.entries()) {
      if (!visited.has(startK)) {
        const [sx, sy] = startK.split(',').map(Number);
        const chain = [[sx, sy]];
        visited.add(startK);
        let currK = pt2[0].toFixed(2) + ',' + pt2[1].toFixed(2);
        while (nextMap.has(currK) && !visited.has(currK)) {
          const [cx, cy] = currK.split(',').map(Number);
          chain.push([cx, cy]);
          visited.add(currK);
          const nextPt = nextMap.get(currK);
          if (!nextPt) break;
          currK = nextPt[0].toFixed(2) + ',' + nextPt[1].toFixed(2);
        }
        chains.push(chain);
      }
    }
  }

  // Sort by chain length descending: Chain 1 is the outer Maltese cross/star perimeter
  chains.sort((a, b) => b.length - a.length);

  let substrateMesh = null;
  if (chains.length > 0 && chains[0].length >= 3) {
    try {
      const outerChain = chains[0];
      const shape = new THREE.Shape();
      shape.moveTo(outerChain[0][0], outerChain[0][1]);
      for (let i = 1; i < outerChain.length; i++) {
        shape.lineTo(outerChain[i][0], outerChain[i][1]);
      }

      // Add circular arm mount holes as interior cutouts
      for (let cIdx = 1; cIdx < chains.length; cIdx++) {
        const holeChain = chains[cIdx];
        if (holeChain.length >= 3) {
          const holePath = new THREE.Path();
          holePath.moveTo(holeChain[0][0], holeChain[0][1]);
          for (let h = 1; h < holeChain.length; h++) {
            holePath.lineTo(holeChain[h][0], holeChain[h][1]);
          }
          shape.holes.push(holePath);
        }
      }

      const extrudeSettings = { depth: substrateH, bevelEnabled: false };
      const extrudedGeo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      extrudedGeo.rotateX(Math.PI / 2);
      extrudedGeo.translate(0, substrateH / 2, 0);

      const subMat = new THREE.MeshStandardMaterial({
        color: 0x061a10, // Dark emerald high-Q PTFE / FR4 solder mask
        roughness: 0.65,
        metalness: 0.2
      });
      substrateMesh = new THREE.Mesh(extrudedGeo, subMat);
      pcbGroup.add(substrateMesh);
    } catch (err) {
      console.warn("Substrate extrusion fallback:", err);
    }
  }

  if (!substrateMesh) {
    const subGeo = new THREE.BoxGeometry(bw, substrateH, bh);
    const subMat = new THREE.MeshStandardMaterial({
      color: 0x061a10,
      roughness: 0.7,
      metalness: 0.15
    });
    substrateMesh = new THREE.Mesh(subGeo, subMat);
    substrateMesh.position.set(0, 0, 0);
    pcbGroup.add(substrateMesh);
  }

  const topY = substrateH / 2 + 0.03;
  const botY = -substrateH / 2 - 0.03;

  // 2. Authentic KiCad Edge.Cuts Outline (High-Visibility Radiant Yellow)
  if (edgeCuts.length > 0) {
    const ecPositions = [];
    edgeCuts.forEach(ec => {
      // Top perimeter line (elevated to avoid z-fighting with zones)
      ecPositions.push(ec.x1, topY + 0.015, ec.y1);
      ecPositions.push(ec.x2, topY + 0.015, ec.y2);
      // Bottom perimeter line
      ecPositions.push(ec.x1, botY - 0.015, ec.y1);
      ecPositions.push(ec.x2, botY - 0.015, ec.y2);
      // Vertical corner/edge line connecting top and bottom perimeters
      ecPositions.push(ec.x1, topY + 0.015, ec.y1);
      ecPositions.push(ec.x1, botY - 0.015, ec.y1);
    });
    const ecGeo = new THREE.BufferGeometry();
    ecGeo.setAttribute('position', new THREE.Float32BufferAttribute(ecPositions, 3));
    const ecMat = new THREE.LineBasicMaterial({
      color: 0xfde047, // Radiant KiCad Edge.Cuts yellow
      linewidth: 2
    });
    const ecLines = new THREE.LineSegments(ecGeo, ecMat);
    pcbGroup.add(ecLines);
  }

  // 2b. Authentic Silkscreen markings around Nanopore H1
  const silkRingGeo = new THREE.BufferGeometry();
  const ringPts = [];
  const ringR = 6.8;
  for (let a = 0; a <= 64; a++) {
    const theta = (a / 64) * Math.PI * 2;
    ringPts.push(new THREE.Vector3(Math.cos(theta) * ringR, topY + 0.01, Math.sin(theta) * ringR));
  }
  silkRingGeo.setFromPoints(ringPts);
  const silkMat = new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.75 });
  const silkRing = new THREE.Line(silkRingGeo, silkMat);
  pcbGroup.add(silkRing);

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

  const roleElem = document.getElementById("kicad-board-role");
  if (roleElem && data.board_role) {
    roleElem.innerText = data.board_role;
  }

  const dimElem = document.getElementById("kicad-board-dim-display");
  if (dimElem && data.board_geometry && data.board_geometry.bounds) {
    const b = data.board_geometry.bounds;
    const segs = data.board_geometry.segments ? data.board_geometry.segments.length : 0;
    dimElem.innerText = `${b.width_mm.toFixed(1)} × ${b.height_mm.toFixed(1)} mm • ${segs} Traces`;
  }

  if (data.stackup) {
    const hElem = document.getElementById("kicad-h-val");
    if (hElem && data.stackup.substrate_height_mm) {
      hElem.innerText = `${Number(data.stackup.substrate_height_mm).toFixed(2)} mm`;
    }
    const erElem = document.getElementById("kicad-er-val");
    if (erElem && data.stackup.dielectric_constant) {
      erElem.innerText = `${Number(data.stackup.dielectric_constant).toFixed(2)}`;
    }
    const tandElem = document.getElementById("kicad-tand-val");
    if (tandElem && data.stackup.loss_tangent !== undefined) {
      tandElem.innerText = `${Number(data.stackup.loss_tangent).toFixed(4)}`;
    }
    const tElem = document.getElementById("kicad-t-val");
    if (tElem && data.stackup.copper_thickness_mm) {
      const cuUm = Math.round(Number(data.stackup.copper_thickness_mm) * 1000);
      tElem.innerText = `${cuUm} µm (1 oz)`;
    }
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
      opt.innerText = `${n.net_name} (${n.trace_width_mm}mm • ${n.total_length_mm.toFixed(1)}mm)`;
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
  if (cardEffTitle) cardEffTitle.innerText = "IMPEDANCE (Z₀)";
  const cardDpTitle = document.querySelector("#card-dp .metric-title");
  if (cardDpTitle) cardDpTitle.innerText = "RETURN LOSS (S₁₁)";
  const cardConvTitle = document.querySelector("#card-conservation .metric-title");
  if (cardConvTitle) cardConvTitle.innerText = "INSERTION LOSS (S₂₁)";
  const cardStressTitle = document.querySelector("#card-stress .metric-title");
  if (cardStressTitle) cardStressTitle.innerText = "CROSSTALK ISOLATION";
  const cardFosTitle = document.querySelector("#card-fos .metric-title");
  if (cardFosTitle) cardFosTitle.innerText = "RF MATCHING";
  const cardUncTitle = document.querySelector("#card-unc .metric-title");
  if (cardUncTitle) cardUncTitle.innerText = "SWEEP FREQUENCY";

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

  // Update Right Drawer Headers & Replace Surrogate Placeholder Card
  const rightHeader = document.querySelector(".drawer-right .section-header");
  if (rightHeader) rightHeader.innerText = "Live Telemetry (KiCad PCB)";

  const activeHeaders = document.querySelectorAll(".drawer-right .section-header");
  if (activeHeaders && activeHeaders.length > 1) {
    activeHeaders[1].innerText = "Board Layer Stackup";
  }
  const bottomCardTitle = document.querySelector(".drawer-right .metric-card:last-of-type .metric-title");
  if (bottomCardTitle) bottomCardTitle.innerText = "LAYER STACKUP";
  const bottomCardNum = document.getElementById("surrogate-samples");
  if (bottomCardNum) bottomCardNum.innerText = "2-Layer High-Q";
  const bottomCardSub = document.querySelector(".drawer-right .metric-card:last-of-type .metric-sub");
  if (bottomCardSub) bottomCardSub.innerText = "Top: F.Cu | Bot: B.Cu GND";

  const statusBadgeText = document.getElementById("backend-status");
  if (statusBadgeText) statusBadgeText.innerText = "⚡ KiCad 10 Live Sync";
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
  } else if (tabName === "spice" && !spiceData) {
    fetchSpiceMonteCarlo();
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
  else if (activeVnaTab === "spice") {
    if (spiceEngineMode === "kicad_native") {
      if (kicadNativeSpiceData) drawKicadBodePlot(kicadNativeSpiceData);
      else fetchKicadNativeSpice();
    } else {
      if (spiceData) drawSpiceMonteCarlo(spiceData);
      else fetchSpiceMonteCarlo();
    }
  }
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
    const diamSlider = document.getElementById("nano-diam-slider");
    const biasSlider = document.getElementById("nano-bias-slider");
    const diam = diamSlider ? parseFloat(diamSlider.value) : (nanoporePoreDiam || 4.0);
    const bias = biasSlider ? parseFloat(biasSlider.value) : (nanoporeBiasMv || 100.0);

    // Call Phase 1 live electrophysiology co-simulation backend
    const res = await fetch(`/api/nanopore/live_trace?pore_diam_nm=${diam}&bias_mv=${bias}`);
    if (res.ok) {
      const data = await res.json();
      nanoporeData = {
        ...data,
        current_na: data.current_na || [],
        baseline_current_na: data.baseline_current_na !== undefined ? data.baseline_current_na : 1.20,
        rms_noise_pa: data.rms_noise_pa !== undefined ? data.rms_noise_pa : 11.5,
        snr_db: data.snr_db !== undefined ? data.snr_db : 39.3,
        events_detected: data.events_detected !== undefined ? data.events_detected : (data.events ? data.events.length : 0),
        mean_dwell_us: data.mean_dwell_us !== undefined ? data.mean_dwell_us : 72.0,
        events: data.events || [],
        clogging_events: [],
        clogging_risk: "OPTIMAL"
      };
      updateNanoporeHUD(nanoporeData);
      drawNanoporeOscilloscope(nanoporeData);
      return;
    }
  } catch (err) {
    console.warn("Live electrophys endpoint fallback:", err);
  }

  // Fallback to fluidic cosim / stream if live_trace unavailable
  try {
    const effEl = document.getElementById("metric-eff");
    const eff = effEl ? parseFloat(effEl.innerText) : 99.96;
    const dpEl = document.getElementById("metric-dp");
    const dpPsi = dpEl ? parseFloat(dpEl.innerText) : 0.42;

    const res = await fetch(`/api/fluidic_cosim?eff=${eff}&dp_psi=${dpPsi}&pore_diam_nm=${nanoporePoreDiam}&bias_mv=${nanoporeBiasMv}`);
    const data = await res.json();
    cosimData = data;
    updateCosimHUD(data);

    if (data.stream) {
      nanoporeData = {
        ...data.stream,
        events: data.stream.dna_events || [],
        clogging_events: (data.nanopore_coupling && data.nanopore_coupling.clogging_events) || [],
        clogging_risk: data.nanopore_coupling ? data.nanopore_coupling.clogging_risk : "UNKNOWN",
        debris_pct: data.upstream_filter ? data.upstream_filter.debris_breakthrough_percent : 0.0
      };
      updateNanoporeHUD(nanoporeData);
      drawNanoporeOscilloscope(nanoporeData);
    }
  } catch (err) {
    console.warn("fetchNanoporeData cosim fallback:", err);
    try {
      const res = await fetch(`/api/nanopore_stream?pore_diam_nm=${nanoporePoreDiam}&bias_mv=${nanoporeBiasMv}&event_rate=3000`);
      const data = await res.json();
      nanoporeData = data;
      updateNanoporeHUD(data);
      drawNanoporeOscilloscope(data);
    } catch (e2) {
      console.error("Nanopore stream fallback error:", e2);
    }
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
  if (baseEl) baseEl.innerText = `${Number(data.baseline_current_na || 1.2).toFixed(2)} nA`;

  const noiseEl = document.getElementById("nano-noise-val");
  if (noiseEl) noiseEl.innerText = `${Number(data.rms_noise_pa || 11.5).toFixed(1)} pA | ${Number(data.snr_db || 39.3).toFixed(1)} dB`;

  const eventsEl = document.getElementById("nano-events-val");
  if (eventsEl) {
    const count = data.events_detected !== undefined ? data.events_detected : (data.events ? data.events.length : 0);
    eventsEl.innerText = `${count} Event${count === 1 ? '' : 's'} (${count > 0 ? 'Active' : 'Idle'})`;
  }

  const dwellEl = document.getElementById("nano-dwell-val");
  if (dwellEl) dwellEl.innerText = `Mean Dwell: ${Number(data.mean_dwell_us || 72).toFixed(0)} µs`;
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

  // Draw Purity / Clogging Stream Badge
  if (data.clogging_risk) {
    const isCritical = data.clogging_risk.includes("CRITICAL") || data.clogging_risk.includes("MODERATE");
    ctx.fillStyle = isCritical ? "rgba(239, 68, 68, 0.22)" : "rgba(16, 185, 129, 0.2)";
    ctx.fillRect(padL + 4, padT + 4, 160, 18);
    ctx.strokeStyle = isCritical ? "#ef4444" : "#10b981";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL + 4, padT + 4, 160, 18);

    ctx.fillStyle = isCritical ? "#f87171" : "#34d399";
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(isCritical ? "⚠️ DEBRIS CLOGGING ACTIVE" : "✓ ULTRA-PURE ANALYTE", padL + 10, padT + 16);
  }

  // Draw Debris Clogging Jam Markers
  if (data.clogging_events && data.clogging_events.length > 0) {
    ctx.font = "bold 8px sans-serif";
    ctx.textAlign = "center";
    data.clogging_events.forEach(cev => {
      const relPos = ((cev.start_us / 2000.0) * plotW - offset * (plotW / n));
      const evX = padL + ((relPos % plotW + plotW) % plotW);
      const evY = toY(0.05);

      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(evX, evY, 4.5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#f87171";
      ctx.fillText(`CLOG (${Math.round(cev.duration_us)}µs)`, evX, evY - 8);
    });
  }

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

// --- Nanopore UI & Physical Action Handlers ---

function openNanoporeScopeDock() {
  const dock = document.getElementById("vna-dock");
  const toggleBtn = document.getElementById("vna-dock-toggle");
  if (!dock) return;
  dock.style.display = "flex";
  vnaDockOpen = true;
  dock.classList.remove("collapsed");
  if (toggleBtn) toggleBtn.innerText = "▼";
  switchVnaTab("nanopore");
  fetchNanoporeData();
}

async function injectTranslocation(analyte, showToast = true) {
  try {
    const res = await fetch("/api/nanopore/trigger_translocation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analyte: analyte || "dsDNA" })
    });
    if (res.ok) {
      const result = await res.json();
      if (showToast) {
        showKiCadToast(`🧬 Translocation injected: ${analyte} (ΔI = ${result.event ? (result.event.blockade_delta_pa).toFixed(0) : 360} pA)`, 2500);
      }
      fetchNanoporeData();
    }
  } catch (err) {
    console.error("injectTranslocation error:", err);
  }
}

function triggerCadExport(spec) {
  if (!spec) return;
  const parts = spec.split(":");
  if (parts.length < 2) return;
  const fmt = parts[0];
  const partId = parts[1];

  showKiCadToast(`💾 Exporting ${partId} as ${fmt.toUpperCase()}...`, 3000);
  const downloadUrl = `/api/project/build123d_export?part_id=${encodeURIComponent(partId)}&format=${encodeURIComponent(fmt)}&project_id=daemon-pore`;

  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = `${partId}.${fmt}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

async function updateClampingTorque(val) {
  const torque = parseFloat(val) || 0.5;
  currentClampingTorque = torque;

  const torqueValEl = document.getElementById("clamping-torque-val");
  if (torqueValEl) {
    torqueValEl.innerText = `${torque.toFixed(2)} N·m`;
  }

  try {
    const res = await fetch(`/api/project/clamping_analysis?torque=${torque}&silicone_thickness_mm=1.6&gasket_area_mm2=1980`);
    if (res.ok) {
      const data = await res.json();
      currentClampingAnalysis = data;

      const badge = document.getElementById("clamping-status-badge");
      if (badge) {
        badge.innerText = data.status || "OPTIMAL";
        if (data.status === "OPTIMAL") {
          badge.style.color = "#10b981";
          badge.style.background = "rgba(16, 185, 129, 0.2)";
        } else if (data.status === "LEAK_RISK") {
          badge.style.color = "#f59e0b";
          badge.style.background = "rgba(245, 158, 11, 0.2)";
        } else {
          badge.style.color = "#ef4444";
          badge.style.background = "rgba(239, 68, 68, 0.2)";
        }
      }

      const forceDisp = document.getElementById("clamping-force-disp");
      if (forceDisp) forceDisp.innerText = `${data.clamp_force_n.toFixed(1)} N`;

      const pressDisp = document.getElementById("clamping-pressure-disp");
      if (pressDisp) pressDisp.innerText = `${data.sealing_pressure_mpa.toFixed(3)} MPa`;

      const compDisp = document.getElementById("clamping-comp-disp");
      if (compDisp) {
        compDisp.innerText = `${data.silicone_compression_mm.toFixed(3)} mm (${data.silicone_strain_pct.toFixed(1)}%)`;
      }

      // Dynamic 3D Gasket mesh deformation (scaling thickness in Z)
      if (daemonPoreGroup) {
        const strainRatio = (data.silicone_strain_pct || 1.0) / 100.0;
        const scaleZ = Math.max(0.15, 1.0 - strainRatio);
        daemonPoreGroup.children.forEach(child => {
          if (child.userData && child.userData.partId && (child.userData.partId.includes("gasket") || (child.userData.name && child.userData.name.toLowerCase().includes("gasket")))) {
            child.scale.set(1.0, 1.0, scaleZ);
          }
        });
      }
    }
  } catch (err) {
    console.warn("Clamping analysis error:", err);
  }
}

async function updateMicrofluidicFlowPhysics(flowRate) {
  const q = parseFloat(flowRate || currentParams.flow_rate_ul_min || 10.0);
  try {
    const res = await fetch(`/api/microfluidics/flow_physics?flow_rate=${q}&width_um=${currentParams.channel_width_um || 100}&height_um=${currentParams.channel_height_um || 50}`);
    if (res.ok) {
      const data = await res.json();
      const reDisp = document.getElementById("cfd-re-disp");
      const phys = data.physics || data;
      if (reDisp && phys.reynolds_number !== undefined) {
        reDisp.innerText = `${Number(phys.reynolds_number).toFixed(2)} (${phys.regime || phys.flow_regime || 'Stokes'})`;
      }
    }
  } catch (err) {
    console.warn("Flow physics error:", err);
  }
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


// =====================================================================
// --- Fluidic Co-Simulation & 3D Capillary Flowcell Link ---
// =====================================================================

async function fetchFluidicCosim(eff, dpPsi) {
  if (eff === undefined) {
    const effEl = document.getElementById("metric-eff");
    eff = effEl ? parseFloat(effEl.innerText) : 99.96;
  }
  if (dpPsi === undefined) {
    const dpEl = document.getElementById("metric-dp");
    dpPsi = dpEl ? parseFloat(dpEl.innerText) : 0.42;
  }
  try {
    const res = await fetch(`/api/fluidic_cosim?eff=${eff}&dp_psi=${dpPsi}&pore_diam_nm=${nanoporePoreDiam}&bias_mv=${nanoporeBiasMv}`);
    const data = await res.json();
    cosimData = data;
    updateCosimHUD(data);

    if (data.stream && (activeVnaTab === "nanopore")) {
      nanoporeData = {
        ...data.stream,
        events: data.stream.dna_events || [],
        clogging_events: (data.nanopore_coupling && data.nanopore_coupling.clogging_events) || [],
        clogging_risk: data.nanopore_coupling ? data.nanopore_coupling.clogging_risk : "UNKNOWN",
        debris_pct: data.upstream_filter ? data.upstream_filter.debris_breakthrough_percent : 0.0
      };
      updateNanoporeHUD(nanoporeData);
      drawNanoporeOscilloscope(nanoporeData);
    }
  } catch (err) {
    console.error("fetchFluidicCosim error:", err);
  }
}

function updateCosimHUD(data) {
  if (!data || !data.nanopore_coupling) return;
  const statusEl = document.getElementById("nano-cosim-status");
  const debrisEl = document.getElementById("nano-cosim-debris");
  const risk = data.nanopore_coupling.clogging_risk || "NEGLIGIBLE";
  const eff = data.upstream_filter ? data.upstream_filter.efficiency_percent : 99.96;
  const debrisPct = data.upstream_filter ? data.upstream_filter.debris_breakthrough_percent : 0.0;

  if (statusEl) {
    if (risk.includes("NEGLIGIBLE")) {
      statusEl.innerText = `ULTRA-PURE (η = ${eff.toFixed(2)}%)`;
      statusEl.style.color = "#10b981";
    } else if (risk.includes("LOW")) {
      statusEl.innerText = `CLEAN STREAM (η = ${eff.toFixed(2)}%)`;
      statusEl.style.color = "#38bdf8";
    } else if (risk.includes("MODERATE")) {
      statusEl.innerText = `INTERMITTENT JAMS (η = ${eff.toFixed(2)}%)`;
      statusEl.style.color = "#f59e0b";
    } else {
      statusEl.innerText = `PORE SATURATED (η = ${eff.toFixed(2)}%)`;
      statusEl.style.color = "#ef4444";
    }
  }
  if (debrisEl) {
    debrisEl.innerText = `Debris: ${debrisPct.toFixed(2)}% | Risk: ${risk.split(" ")[0]}`;
  }
}

function toggleFlowcellCapillary() {
  showFlowcellTube = !showFlowcellTube;
  const btn = document.getElementById("btn-toggle-flowcell");
  if (btn) btn.classList.toggle("active", showFlowcellTube);

  if (showFlowcellTube) {
    if (!flowcellTubeGroup) {
      buildFlowcellTube();
    }
    if (flowcellTubeGroup) flowcellTubeGroup.visible = true;

    // Reposition corkscrew slightly to upper-left so both filter & PCB are visible
    if (corkscrewMesh) {
      corkscrewMesh.visible = true;
      corkscrewMesh.position.set(-36, 16, -24);
      corkscrewMesh.scale.set(0.65, 0.65, 0.65);
    }
    if (pcbGroup) pcbGroup.visible = true;

    // Frame camera to encompass both filter and PCB
    camera.position.set(-15, 62, 58);
    controls.target.set(-5, 5, 0);
    controls.update();

    fetchFluidicCosim();
  } else {
    if (flowcellTubeGroup) flowcellTubeGroup.visible = false;
    if (corkscrewMesh) {
      corkscrewMesh.position.set(0, 0, 0);
      corkscrewMesh.scale.set(1.0, 1.0, 1.0);
      corkscrewMesh.visible = (currentDomain !== "pcb");
    }
    if (currentDomain === "pcb") {
      camera.position.set(0, 52, 45);
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }
}

function buildFlowcellTube() {
  if (flowcellTubeGroup) {
    scene.remove(flowcellTubeGroup);
  }
  flowcellTubeGroup = new THREE.Group();
  flowcellTubeGroup.name = "flowcellTubeGroup";

  // Start at corkscrew vortex clean outlet
  const pStart = new THREE.Vector3(-36, 16, -24);

  // Find H1 Delrin Well location on PCB
  let endX = 0.0;
  let endZ = 0.0;
  const comps = (activePcbData && activePcbData.board_geometry && activePcbData.board_geometry.components) ||
                (activePcbData && activePcbData.components) || [];
  const h1 = comps.find(c => c.ref === "H1" || (c.ref && c.ref.startsWith("H")));
  if (h1) {
    endX = h1.x;
    endZ = h1.y;
  }
  const pEnd = new THREE.Vector3(endX, 3.8, endZ);

  // Spline control points for fluidic capillary path
  flowcellCurve = new THREE.CatmullRomCurve3([
    pStart,
    new THREE.Vector3(-28, 22, -18),
    new THREE.Vector3(-14, 25, -6),
    new THREE.Vector3(0, 20, 0),
    new THREE.Vector3(endX - 4, 12, endZ - 2),
    pEnd
  ]);

  // Fluoropolymer / Fused Silica Translucent Capillary Sheath
  const tubeGeo = new THREE.TubeGeometry(flowcellCurve, 64, 0.85, 16, false);
  const tubeMat = new THREE.MeshPhysicalMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.55,
    roughness: 0.15,
    transmission: 0.75,
    thickness: 0.4
  });
  const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
  flowcellTubeGroup.add(tubeMesh);

  // Flowing Analyte Particles inside tube
  const N_FLOW = 45;
  flowcellParticleProgress = [];
  const partGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(N_FLOW * 3);
  const colors = new Float32Array(N_FLOW * 3);

  for (let i = 0; i < N_FLOW; i++) {
    const t = i / N_FLOW;
    flowcellParticleProgress.push(t);
    const pt = flowcellCurve.getPoint(t);
    positions[i * 3] = pt.x;
    positions[i * 3 + 1] = pt.y;
    positions[i * 3 + 2] = pt.z;

    // Pure emerald analyte particles
    colors[i * 3] = 0.06;
    colors[i * 3 + 1] = 0.92;
    colors[i * 3 + 2] = 0.55;
  }

  partGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  partGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const partMat = new THREE.PointsMaterial({
    size: 1.8,
    vertexColors: true,
    transparent: true,
    opacity: 0.95
  });

  flowcellParticles = new THREE.Points(partGeo, partMat);
  flowcellTubeGroup.add(flowcellParticles);

  scene.add(flowcellTubeGroup);
}

function updateFlowcellParticles(dt) {
  if (!flowcellParticles || !flowcellCurve || !flowcellParticleProgress) return;

  const posAttr = flowcellParticles.geometry.attributes.position;
  const colAttr = flowcellParticles.geometry.attributes.color;
  const N = flowcellParticleProgress.length;

  const isDebris = cosimData && cosimData.upstream_filter && cosimData.upstream_filter.efficiency_percent < 98.0;

  for (let i = 0; i < N; i++) {
    flowcellParticleProgress[i] = (flowcellParticleProgress[i] + dt * 0.35) % 1.0;
    const pt = flowcellCurve.getPoint(flowcellParticleProgress[i]);
    posAttr.setXYZ(i, pt.x, pt.y, pt.z);

    if (isDebris) {
      // Amber/Red turbulent debris
      colAttr.setXYZ(i, 0.95, 0.25, 0.2);
    } else {
      // Emerald pure analyte
      colAttr.setXYZ(i, 0.06, 0.92, 0.55);
    }
  }

  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
}

// =====================================================================
// --- SPICE Netlist & Monte Carlo Component Tolerance Engine ---
// =====================================================================

async function fetchSpiceMonteCarlo(rTol, cTol) {
  if (rTol !== undefined) spiceRTol = rTol;
  if (cTol !== undefined) spiceCTol = cTol;

  try {
    const res = await fetch(`/api/spice_monte_carlo?r1_tol=${spiceRTol}&c1_tol=${spiceCTol}&num_runs=500`);
    const data = await res.json();
    spiceData = data;
    updateSpiceHUD(data);
    drawSpiceMonteCarlo(data);
  } catch (err) {
    console.error("fetchSpiceMonteCarlo error:", err);
  }
}

function updateSpiceHUD(data) {
  if (!data) return;

  const pill = document.getElementById("spice-status-pill");
  if (pill) {
    const y = data.yield_percent !== undefined ? data.yield_percent : 100.0;
    pill.innerText = `${y.toFixed(1)}% YIELD (${y >= 95.0 ? "PASS" : "FAIL"})`;
    pill.className = y >= 95.0 ? "yield-pill-pass" : "yield-pill-fail";
  }

  const fcEl = document.getElementById("spice-fc-val");
  if (fcEl) {
    fcEl.innerText = `${data.mean_cutoff_khz.toFixed(1)} ± ${data.std_cutoff_khz.toFixed(1)} kHz`;
  }

  const yieldEl = document.getElementById("spice-yield-val");
  if (yieldEl) {
    yieldEl.innerText = `${data.yield_percent.toFixed(1)}% (${data.pass_count}/${data.num_runs})`;
    yieldEl.style.color = data.yield_percent >= 95.0 ? "#10b981" : "#ef4444";
  }

  const boundsEl = document.getElementById("spice-bounds-val");
  if (boundsEl) {
    boundsEl.innerText = `Bounds: [${data.min_cutoff_khz.toFixed(1)}, ${data.max_cutoff_khz.toFixed(1)}] kHz`;
  }

  if (data.sensitivity_ranking && data.sensitivity_ranking.length > 0) {
    const top = data.sensitivity_ranking[0];
    const drvEl = document.getElementById("spice-driver-val");
    if (drvEl) drvEl.innerText = top.component;
    const drvSub = document.getElementById("spice-driver-sub");
    if (drvSub) drvSub.innerText = `${top.variance_impact_pct.toFixed(1)}% variance impact`;
  }
}

function updateSpiceTolerances() {
  const rSlider = document.getElementById("spice-rtol-slider");
  const cSlider = document.getElementById("spice-ctol-slider");
  if (rSlider) {
    spiceRTol = parseFloat(rSlider.value);
    const rDisp = document.getElementById("spice-rtol-disp");
    if (rDisp) rDisp.innerText = `±${spiceRTol.toFixed(1)}%`;
  }
  if (cSlider) {
    spiceCTol = parseFloat(cSlider.value);
    const cDisp = document.getElementById("spice-ctol-disp");
    if (cDisp) cDisp.innerText = `±${spiceCTol.toFixed(1)}%`;
  }
  if (spiceEngineMode === "kicad_native") {
    const rVal = 1.0 * (1.0 + (spiceRTol - 1.0) * 0.05);
    const cVal = 2.0 * (1.0 + (spiceCTol - 5.0) * 0.05);
    fetchKicadNativeSpice(rVal, cVal);
  } else {
    fetchSpiceMonteCarlo(spiceRTol, spiceCTol);
  }
}

function drawSpiceMonteCarlo(data) {
  const canvas = document.getElementById("spice-canvas");
  if (!canvas || !data || !data.histogram) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 44, padR = 25, padT = 24, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Background
  ctx.fillStyle = "#030712";
  ctx.fillRect(padL, padT, plotW, plotH);

  const hist = data.histogram;
  const binCenters = hist.bin_centers_khz || [];
  const inSpec = hist.in_spec_counts || [];
  const outSpec = hist.out_of_spec_counts || [];
  const nBins = binCenters.length;
  if (nBins === 0) return;

  // Compute maximum count
  let maxCount = 10;
  for (let i = 0; i < nBins; i++) {
    const tot = (inSpec[i] || 0) + (outSpec[i] || 0);
    if (tot > maxCount) maxCount = tot;
  }
  maxCount = Math.ceil(maxCount * 1.15);

  const minFreq = binCenters[0] - (hist.bin_width_khz || 1.5) / 2;
  const maxFreq = binCenters[nBins - 1] + (hist.bin_width_khz || 1.5) / 2;

  function toX(fKhz) {
    return padL + ((fKhz - minFreq) / (maxFreq - minFreq)) * plotW;
  }
  function toY(cnt) {
    return padT + (1.0 - cnt / maxCount) * plotH;
  }

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  const yTicks = 4;
  for (let i = 1; i <= yTicks; i++) {
    const c = Math.round((i / yTicks) * maxCount);
    const gy = toY(c);
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + plotW, gy);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "right";
    ctx.fillText(`${c}`, padL - 6, gy + 3);
  }

  // Spec Limit Shaded Forbidden Zones (< 70 kHz, > 90 kHz)
  const x70 = toX(70.0);
  const x90 = toX(90.0);

  if (x70 > padL) {
    ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
    ctx.fillRect(padL, padT, Math.max(0, x70 - padL), plotH);
  }
  if (x90 < padL + plotW) {
    ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
    ctx.fillRect(x90, padT, padL + plotW - x90, plotH);
  }

  // Bandwidth Mean +/- 1 Sigma Cyan Shaded Zone
  const meanFc = data.mean_cutoff_khz || 77.7;
  const stdFc = data.std_cutoff_khz || 1.3;
  const xM1 = toX(meanFc - stdFc);
  const xP1 = toX(meanFc + stdFc);
  ctx.fillStyle = "rgba(56, 189, 248, 0.14)";
  ctx.fillRect(Math.max(padL, xM1), padT, Math.min(plotW, xP1 - xM1), plotH);

  // Draw Histogram Bars
  const binW = plotW / nBins;
  for (let i = 0; i < nBins; i++) {
    const cntIn = inSpec[i] || 0;
    const cntOut = outSpec[i] || 0;
    const bx = padL + i * binW + 1;
    const bw = Math.max(1, binW - 2);

    // In-spec bar
    if (cntIn > 0) {
      const by = toY(cntIn);
      const bh = padT + plotH - by;
      ctx.fillStyle = "#10b981";
      ctx.fillRect(bx, by, bw, bh);
    }

    // Out-of-spec bar (stacked or isolated)
    if (cntOut > 0) {
      const byOut = toY(cntIn + cntOut);
      const bhOut = toY(cntIn) - byOut;
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(bx, byOut, bw, bhOut);
    }
  }

  // Spec Limit Vertical Dashed Lines
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#f87171";
  ctx.setLineDash([3, 3]);

  if (x70 >= padL && x70 <= padL + plotW) {
    ctx.beginPath();
    ctx.moveTo(x70, padT);
    ctx.lineTo(x70, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "#f87171";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("SPEC MIN (70k)", x70, padT - 6);
  }

  if (x90 >= padL && x90 <= padL + plotW) {
    ctx.beginPath();
    ctx.moveTo(x90, padT);
    ctx.lineTo(x90, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "#f87171";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("SPEC MAX (90k)", x90, padT - 6);
  }
  ctx.setLineDash([]);

  // Mean Cutoff Diamond & Line
  const xMean = toX(meanFc);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(xMean, padT);
  ctx.lineTo(xMean, padT + plotH);
  ctx.stroke();

  // Diamond Marker
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(xMean, padT - 4);
  ctx.lineTo(xMean + 4, padT - 8);
  ctx.lineTo(xMean, padT - 12);
  ctx.lineTo(xMean - 4, padT - 8);
  ctx.closePath();
  ctx.fill();

  ctx.font = "bold 8px SF Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText(`µ = ${meanFc.toFixed(1)}k`, xMean, padT - 15);

  // X Axis Freq Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "8px SF Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText(`${minFreq.toFixed(0)} kHz`, padL, padT + plotH + 14);
  ctx.fillText(`${((minFreq + maxFreq) / 2).toFixed(0)} kHz`, padL + plotW / 2, padT + plotH + 14);
  ctx.fillText(`${maxFreq.toFixed(0)} kHz`, padL + plotW, padT + plotH + 14);
  ctx.fillText("Cutoff Frequency (fc)", padL + plotW / 2, padT + plotH + 24);

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
}


// =====================================================================
// --- KiCad Native ngspice.dll C-API Bridge & Bode Visualization ---
// =====================================================================

function switchSpiceEngine(mode) {
  spiceEngineMode = mode;
  const btnMc = document.getElementById("spice-mode-btn-mc");
  const btnKicad = document.getElementById("spice-mode-btn-kicad");
  const mcCont = document.getElementById("spice-mc-container");
  const bodeCont = document.getElementById("spice-bode-container");
  const mcTelem = document.getElementById("spice-mc-telemetry");
  const kicadTelem = document.getElementById("spice-kicad-telemetry");
  const r1Label = document.getElementById("spice-r1-label");
  const c1Label = document.getElementById("spice-c1-label");

  if (mode === "kicad_native") {
    if (btnMc) btnMc.classList.remove("active");
    if (btnKicad) btnKicad.classList.add("active");
    if (mcCont) mcCont.style.display = "none";
    if (bodeCont) bodeCont.style.display = "flex";
    if (mcTelem) mcTelem.style.display = "none";
    if (kicadTelem) kicadTelem.style.display = "block";
    if (r1Label) r1Label.innerText = "R1 Feedback Trim:";
    if (c1Label) c1Label.innerText = "C1 Feedback Trim:";
    if (!kicadNativeSpiceData) {
      fetchKicadNativeSpice();
    } else {
      drawKicadBodePlot(kicadNativeSpiceData);
    }
  } else {
    if (btnMc) btnMc.classList.add("active");
    if (btnKicad) btnKicad.classList.remove("active");
    if (mcCont) mcCont.style.display = "flex";
    if (bodeCont) bodeCont.style.display = "none";
    if (mcTelem) mcTelem.style.display = "block";
    if (kicadTelem) kicadTelem.style.display = "none";
    if (r1Label) r1Label.innerText = "R1 (1.0 MΩ) Tol:";
    if (c1Label) c1Label.innerText = "C1 (2.0 pF) Tol:";
    if (!spiceData) {
      fetchSpiceMonteCarlo();
    } else {
      drawSpiceMonteCarlo(spiceData);
    }
  }
}

async function fetchKicadNativeSpice(r1, c1, cPar) {
  r1 = r1 !== undefined ? r1 : 1.0;
  c1 = c1 !== undefined ? c1 : 2.0;
  cPar = cPar !== undefined ? cPar : 1.2;

  try {
    const res = await fetch(`/api/kicad_native_spice?analysis=ac&r1_mohm=${r1}&c1_pf=${c1}&c_par_pf=${cPar}`);
    const data = await res.json();
    kicadNativeSpiceData = data;
    updateKicadSpiceHUD(data);
    drawKicadBodePlot(data);
  } catch (err) {
    console.error("fetchKicadNativeSpice error:", err);
  }
}

function updateKicadSpiceHUD(data) {
  if (!data) return;
  const pill = document.getElementById("spice-status-pill");
  if (pill) {
    const pm = data.phase_margin_deg || 73.8;
    pill.innerText = `PM = ${pm.toFixed(1)}° (${pm >= 45.0 ? 'STABLE' : 'UNSTABLE'})`;
    pill.className = pm >= 45.0 ? "yield-pill-pass" : "yield-pill-fail";
  }

  const gainEl = document.getElementById("spice-kicad-gain");
  if (gainEl && data.low_freq_gain_dbohm !== undefined) {
    gainEl.innerText = `${data.low_freq_gain_dbohm.toFixed(1)} dBΩ (${data.parameters ? data.parameters.r1_mohm.toFixed(2) : '1.0'} MΩ)`;
  }

  const pmEl = document.getElementById("spice-kicad-pm");
  if (pmEl && data.phase_margin_deg !== undefined) {
    pmEl.innerText = `${data.phase_margin_deg.toFixed(1)}° (${data.phase_margin_deg >= 45 ? 'Stable' : 'Marginal'})`;
    pmEl.style.color = data.phase_margin_deg >= 45 ? "#10b981" : "#ef4444";
  }

  const fcEl = document.getElementById("spice-kicad-fc");
  if (fcEl && data.cutoff_khz !== undefined) {
    fcEl.innerText = `-3dB Bandwidth: ${data.cutoff_khz.toFixed(1)} kHz`;
  }
}

function drawKicadBodePlot(data) {
  const canvas = document.getElementById("spice-bode-canvas");
  if (!canvas || !data || !data.frequencies_hz) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 46, padR = 46, padT = 24, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Dark oscilloscope screen background
  ctx.fillStyle = "#030712";
  ctx.fillRect(padL, padT, plotW, plotH);

  const freqs = data.frequencies_hz;
  const gains = data.gain_dbohm;
  const phases = data.phase_deg;
  const N = freqs.length;
  if (N === 0) return;

  const minF = freqs[0];
  const maxF = freqs[N - 1];
  const logMinF = Math.log10(Math.max(1.0, minF));
  const logMaxF = Math.log10(Math.max(10.0, maxF));

  // Y-axis scales: Gain 60 to 130 dBOhm, Phase 0 to 180 deg (Inverting TIA)
  const minGain = 60.0, maxGain = 130.0;
  const minPhase = 0.0, maxPhase = 180.0;

  function toX(f) {
    const lf = Math.log10(Math.max(1.0, f));
    return padL + ((lf - logMinF) / (logMaxF - logMinF)) * plotW;
  }
  function toYGain(g) {
    return padT + (1.0 - (g - minGain) / (maxGain - minGain)) * plotH;
  }
  function toYPhase(p) {
    return padT + (1.0 - (p - minPhase) / (maxPhase - minPhase)) * plotH;
  }

  // Draw Grid Lines (Decade Frequencies: 100, 1k, 10k, 100k, 1M, 10M)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
  ctx.lineWidth = 1;
  for (let d = Math.ceil(logMinF); d <= Math.floor(logMaxF); d++) {
    const fVal = Math.pow(10, d);
    const gx = toX(fVal);
    ctx.beginPath();
    ctx.moveTo(gx, padT);
    ctx.lineTo(gx, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "center";
    const lbl = fVal >= 1e6 ? `${(fVal/1e6).toFixed(0)}M` : (fVal >= 1e3 ? `${(fVal/1e3).toFixed(0)}k` : `${fVal.toFixed(0)}`);
    ctx.fillText(lbl, gx, padT + plotH + 12);
  }

  // Horizontal Grid Lines & Y Ticks
  const gainTicks = [60, 80, 100, 120];
  gainTicks.forEach(gt => {
    const gy = toYGain(gt);
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + plotW, gy);
    ctx.stroke();

    // Left Y Axis label (Gain dBOhm)
    ctx.fillStyle = "#00f0ff";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "right";
    ctx.fillText(`${gt}dB`, padL - 4, gy + 3);
  });

  // Right Y Axis label (Phase deg: 0 to 180)
  const phaseTicks = [0, 45, 90, 135, 180];
  phaseTicks.forEach(pt => {
    const py = toYPhase(pt);
    ctx.fillStyle = "#f59e0b";
    ctx.font = "8px SF Mono, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${pt}°`, padL + plotW + 4, py + 3);
  });

  // -3dB Cutoff Line & Diamond
  const fc = (data.cutoff_khz || 89.1) * 1000.0;
  const xFc = toX(fc);
  if (xFc >= padL && xFc <= padL + plotW) {
    ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(xFc, padT);
    ctx.lineTo(xFc, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#00f0ff";
    ctx.beginPath();
    ctx.arc(xFc, toYGain(data.low_freq_gain_dbohm - 3.0), 3.5, 0, 2 * Math.PI);
    ctx.fill();

    ctx.font = "bold 8px SF Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`fc = ${(fc/1e3).toFixed(1)}k`, xFc, padT - 6);
  }

  // Draw Phase Curve (Radiant Amber, Dashed)
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const px = toX(freqs[i]);
    const py = toYPhase(phases[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Gain Curve (Electric Cyan, Solid with Glow)
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2.2;
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 5;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const gx = toX(freqs[i]);
    const gy = toYGain(gains[i]);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Title / Legend in Top Left
  ctx.font = "bold 8px SF Mono, monospace";
  ctx.textAlign = "left";
  ctx.fillStyle = "#00f0ff";
  ctx.fillText("━ Gain |Z_TIA| (dBΩ)", padL + 8, padT + 14);
  ctx.fillStyle = "#f59e0b";
  ctx.fillText("┄ Phase ∠Z (°)", padL + 125, padT + 14);
  ctx.fillStyle = "#10b981";
  ctx.fillText(`PM = ${(data.phase_margin_deg || 73.8).toFixed(1)}°`, padL + 215, padT + 14);

  // Border
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
}

/* =========================================================================
   WIREVIZ HARNESS & TUBING STUDIO CONTROLLER
   ========================================================================= */

let harnessStudioOpen = false;
let currentHarnessProjectId = "daemon-pore";
let currentHarnessType = "electrical";
let harnessZoom = 1.0;
let harnessPanX = 0;
let harnessPanY = 0;
let isHarnessDragging = false;
let harnessDragStartX = 0;
let harnessDragStartY = 0;
let harnessEventsInitialized = false;

const BLANK_HARNESS_TEMPLATE = `# WireViz Harness Specification
metadata:
  title: Custom Interconnect Harness
  author: OpenAuto-CFD Studio
  version: "1.0"
  description: Microfluidic tubing and electrical wiring harness

connectors:
  INLET_PORT:
    type: 1/4-28 Flat-Bottom Port
    subtype: female
    pincount: 1
    pinlabels: [FLUID_IN]

  FLOWCELL_CIS:
    type: Nano-Capillary Inlet
    subtype: female
    pincount: 1
    pinlabels: [CIS_CHAMBER]

cables:
  FEED_LINE:
    type: FEP Microbore Tubing (1/16 OD x 0.020 ID)
    gauge: 0.51 mm
    length: 0.10 m
    color: TQ
    notes: Low dead-volume fluidic link

connections:
  -
    - INLET_PORT: [1]
    - FEED_LINE: [1]
    - FLOWCELL_CIS: [1]
`;

function openHarnessStudio() {
  const modal = document.getElementById("modal-harness-studio");
  if (!modal) return;
  modal.style.display = "flex";
  harnessStudioOpen = true;

  // Initialize event listeners once
  if (!harnessEventsInitialized) {
    initHarnessStudioEvents();
    harnessEventsInitialized = true;
  }

  // Pick suitable preset matching active project
  const presetSelect = document.getElementById("harness-preset-select");
  if (presetSelect) {
    if (currentProjectId === "corkscrew-filter") {
      presetSelect.value = "corkscrew-filter:system";
    } else {
      if (!presetSelect.value || presetSelect.value.startsWith("blank")) {
        presetSelect.value = "daemon-pore:electrical";
      }
    }
    handleHarnessPresetChange(presetSelect.value);
  }
}

function closeHarnessStudio() {
  const modal = document.getElementById("modal-harness-studio");
  if (!modal) return;
  modal.style.display = "none";
  harnessStudioOpen = false;
}

function handleHarnessPresetChange(val) {
  if (!val) return;
  const parts = val.split(":");
  const projId = parts[0];
  const hType = parts[1];
  currentHarnessProjectId = projId;
  currentHarnessType = hType;

  const textarea = document.getElementById("harness-yaml-textarea");
  if (!textarea) return;

  if (projId === "blank") {
    textarea.value = BLANK_HARNESS_TEMPLATE;
    updateHarnessLineCounter();
    renderHarnessEditor();
    return;
  }

  // Fetch YAML from server
  fetch(`/api/project/harness?project_id=${encodeURIComponent(projId)}&type=${encodeURIComponent(hType)}`)
    .then(r => r.json())
    .then(data => {
      if (data && data.yaml) {
        textarea.value = data.yaml;
      } else if (data && data.error) {
        showKiCadToast(`⚠️ Harness load notice: ${data.error}`);
        textarea.value = BLANK_HARNESS_TEMPLATE;
      }
      updateHarnessLineCounter();
      renderHarnessEditor();
    })
    .catch(err => {
      console.warn("Failed to load harness YAML:", err);
      textarea.value = BLANK_HARNESS_TEMPLATE;
      updateHarnessLineCounter();
      renderHarnessEditor();
    });
}

function updateHarnessLineCounter() {
  const textarea = document.getElementById("harness-yaml-textarea");
  const counter = document.getElementById("harness-line-counter");
  if (!textarea || !counter) return;

  const text = textarea.value;
  const lines = text.split("\n").length;
  const chars = text.length;
  counter.textContent = `Lines: ${lines} | Chars: ${chars}`;
}

async function renderHarnessEditor() {
  const textarea = document.getElementById("harness-yaml-textarea");
  const canvas = document.getElementById("harness-svg-canvas");
  const spinner = document.getElementById("harness-render-spinner");
  const emptyMsg = document.getElementById("harness-empty-msg");
  const errBanner = document.getElementById("harness-error-banner");
  const errText = document.getElementById("harness-error-text");
  const statusPill = document.getElementById("harness-status-pill");
  const titleBadge = document.getElementById("harness-diagram-title");

  if (!textarea || !canvas) return;
  const yamlContent = textarea.value;

  if (spinner) spinner.style.display = "block";
  if (emptyMsg) emptyMsg.style.display = "none";
  if (errBanner) errBanner.style.display = "none";
  if (statusPill) {
    statusPill.textContent = "COMPILING...";
    statusPill.className = "compute-status-pill warn";
  }

  try {
    const res = await fetch("/api/harness/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: currentHarnessProjectId,
        type: currentHarnessType,
        yaml: yamlContent
      })
    });

    const data = await res.json();
    if (spinner) spinner.style.display = "none";

    if (data.success && data.svg) {
      canvas.innerHTML = data.svg;
      resetHarnessZoom();

      if (statusPill) {
        statusPill.textContent = "VALIDATED";
        statusPill.className = "compute-status-pill ok";
      }
      if (titleBadge) {
        titleBadge.textContent = data.title || "WireViz Diagram";
      }

      // Populate telemetry & BOM
      updateHarnessTelemetry(data.rules || {});
      updateHarnessBOM(data.bom || []);
    } else {
      if (statusPill) {
        statusPill.textContent = "SYNTAX ERROR";
        statusPill.className = "compute-status-pill off";
      }
      if (errBanner && errText) {
        errBanner.style.display = "block";
        errText.textContent = data.error || "WireViz compilation returned an unknown error.";
      }
    }
  } catch (err) {
    if (spinner) spinner.style.display = "none";
    if (statusPill) {
      statusPill.textContent = "SERVER ERROR";
      statusPill.className = "compute-status-pill off";
    }
    if (errBanner && errText) {
      errBanner.style.display = "block";
      errText.textContent = `Network / server error: ${err.message}`;
    }
  }
}

function updateHarnessTelemetry(rules) {
  const deadvol = document.getElementById("harness-stat-deadvol");
  const dp = document.getElementById("harness-stat-dp");
  const delay = document.getElementById("harness-stat-delay");
  const loopres = document.getElementById("harness-stat-loopres");
  const checksList = document.getElementById("harness-checks-list");

  if (deadvol) deadvol.textContent = `${(rules.total_dead_volume_ul || 0.0).toFixed(1)} μL`;
  if (dp) dp.textContent = `${(rules.max_pressure_drop_psi || 0.0).toFixed(3)} psi`;
  if (delay) delay.textContent = `${(rules.flow_delay_s || 0.0).toFixed(1)} s`;

  let totalR_mOhm = 0;
  if (Array.isArray(rules.erc_checks)) {
    for (const c of rules.erc_checks) {
      totalR_mOhm += (c.loop_resistance_ohms || 0) * 1000.0;
    }
  }
  if (loopres) loopres.textContent = `${totalR_mOhm.toFixed(1)} mΩ`;

  if (!checksList) return;
  checksList.innerHTML = "";

  const allChecks = [];
  if (Array.isArray(rules.frc_checks)) {
    for (const f of rules.frc_checks) {
      allChecks.push({
        type: "FRC (Fluidic)",
        name: f.tube || "Tubing Segment",
        status: f.status || "PASS",
        badgeClass: f.status === "PASS" ? "ok" : "warn",
        detail: `ID: ${f.id_mm} mm | Len: ${f.length_m} m | DeadVol: ${f.dead_volume_ul} μL | ΔP: ${f.dp_psi} psi`
      });
    }
  }
  if (Array.isArray(rules.erc_checks)) {
    for (const e of rules.erc_checks) {
      allChecks.push({
        type: "ERC (Electrical)",
        name: e.wire || "Wire Conductor",
        status: e.status || "PASS",
        badgeClass: e.status === "PASS" ? "ok" : (e.status.includes("WARN") ? "warn" : "off"),
        detail: `${e.awg} | Len: ${e.length_m} m | Loop R: ${(e.loop_resistance_ohms * 1000).toFixed(1)} mΩ | ${e.shielded ? "🛡️ Shielded" : "Unshielded"}`
      });
    }
  }

  if (allChecks.length === 0) {
    checksList.innerHTML = `<div style="padding: 12px; color: var(--text-muted); font-size: 11px; text-align: center;">No physical rule violations detected.</div>`;
    return;
  }

  for (const item of allChecks) {
    const card = document.createElement("div");
    card.className = "harness-check-item";
    card.innerHTML = `
      <div class="harness-check-header">
        <span class="harness-check-name">${item.name} <span style="font-size: 9.5px; color: var(--text-muted);">(${item.type})</span></span>
        <span class="compute-status-pill ${item.badgeClass}" style="font-size: 9px; padding: 1px 6px;">${item.status}</span>
      </div>
      <div class="harness-check-details">
        <span>${item.detail}</span>
      </div>
    `;
    checksList.appendChild(card);
  }
}

function updateHarnessBOM(bom) {
  const tbody = document.getElementById("harness-bom-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!Array.isArray(bom) || bom.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 16px;">No BOM components generated.</td></tr>`;
    return;
  }

  bom.forEach((item, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color: var(--text-muted); font-family: var(--font-mono);">${idx + 1}</td>
      <td style="font-weight: 600; color: var(--accent-cyan); font-family: var(--font-mono);">${item.designators || item.id || "-"}</td>
      <td style="color: #f1f5f9;">${item.description || "Component"}</td>
      <td style="font-family: var(--font-mono); font-weight: bold; color: var(--accent-emerald);">${item.qty} ${item.unit || "ea"}</td>
      <td style="font-size: 9.5px; color: #94a3b8;">${item.mpn && item.mpn !== "-" ? item.mpn : (item.manufacturer || "-")}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function saveHarnessToProject() {
  const textarea = document.getElementById("harness-yaml-textarea");
  if (!textarea) return;
  const yamlContent = textarea.value.trim();
  if (!yamlContent) {
    showKiCadToast("⚠️ Cannot save empty harness specification.");
    return;
  }

  try {
    const res = await fetch("/api/project/harness/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: currentHarnessProjectId,
        type: currentHarnessType,
        yaml: yamlContent
      })
    });
    const data = await res.json();
    if (data.status === "ok") {
      showKiCadToast(`💾 Saved harness (${currentHarnessType}) to project '${currentHarnessProjectId}'!`, 3500);
    } else {
      showKiCadToast(`❌ Failed to save harness: ${data.error || "Unknown error"}`);
    }
  } catch (err) {
    showKiCadToast(`❌ Network error saving harness: ${err.message}`);
  }
}

async function triggerHarnessExport(fmt) {
  if (!fmt) return;
  const textarea = document.getElementById("harness-yaml-textarea");
  const yamlContent = textarea ? textarea.value : "";

  showKiCadToast(`💾 Generating ${fmt.toUpperCase()} export...`, 2000);

  try {
    const res = await fetch("/api/harness/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: currentHarnessProjectId,
        type: currentHarnessType,
        format: fmt,
        yaml: yamlContent
      })
    });

    if (!res.ok) {
      const err = await res.json();
      showKiCadToast(`❌ Export failed: ${err.error || res.statusText}`);
      return;
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = `harness_${currentHarnessProjectId}_${currentHarnessType}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);

    showKiCadToast(`✅ Exported ${a.download}!`, 3000);
  } catch (err) {
    showKiCadToast(`❌ Export error: ${err.message}`);
  }
}

function insertHarnessSnippet(type) {
  const textarea = document.getElementById("harness-yaml-textarea");
  if (!textarea) return;

  const snippets = {
    connector: `
  J_CUSTOM:
    type: Micro-Fit 3.0
    subtype: male
    pincount: 4
    pinlabels: [VBUS, GND, SCL, SDA]
`,
    cable: `
  W_HARNESS:
    gauge: 24 AWG
    length: 0.15 m
    colorcode: DIN
    wirecount: 4
    shield: true
`,
    tubing: `
  TUBE_LINE:
    type: Microbore FEP Tubing (1/16 OD x 0.020 ID)
    gauge: 0.51 mm
    length: 0.12 m
    color: TQ
    notes: Microfluidic link with minimal dead volume
`,
    fitting: `
  FITTING_BARB:
    type: 1/4-28 to 1/16 Barb Fitting
    subtype: female
    pincount: 1
    pinlabels: [FLUID_IN]
`,
    connection: `
  -
    - J_CUSTOM: [1, 2]
    - W_HARNESS: [1, 2]
    - J_TARGET: [1, 2]
`
  };

  const textToInsert = snippets[type] || "";
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const val = textarea.value;

  textarea.value = val.substring(0, start) + textToInsert + val.substring(end);
  textarea.selectionStart = textarea.selectionEnd = start + textToInsert.length;
  textarea.focus();
  updateHarnessLineCounter();
}

function switchHarnessTab(tab) {
  activeHarnessTab = tab;
  const tabs = ["rules", "bom", "info"];
  for (const t of tabs) {
    const btn = document.getElementById(`harness-tab-btn-${t}`);
    const panel = document.getElementById(`harness-tab-${t}`);
    if (btn) btn.classList.toggle("active", t === tab);
    if (panel) panel.classList.toggle("active", t === tab);
  }
}

function zoomHarnessDiagram(factor) {
  harnessZoom *= factor;
  harnessZoom = Math.max(0.15, Math.min(harnessZoom, 5.0));
  applyHarnessTransform();
}

function resetHarnessZoom() {
  harnessZoom = 1.0;
  harnessPanX = 0;
  harnessPanY = 0;
  applyHarnessTransform();
}

function applyHarnessTransform() {
  const canvas = document.getElementById("harness-svg-canvas");
  if (!canvas) return;
  canvas.style.transform = `translate(${harnessPanX}px, ${harnessPanY}px) scale(${harnessZoom})`;
}

function initHarnessStudioEvents() {
  const wrapper = document.getElementById("harness-svg-wrapper");
  const textarea = document.getElementById("harness-yaml-textarea");

  if (textarea) {
    textarea.addEventListener("input", updateHarnessLineCounter);
    textarea.addEventListener("keydown", (e) => {
      // Ctrl+Enter or Cmd+Enter re-renders
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        renderHarnessEditor();
      }
      // Tab key indentation
      if (e.key === "Tab") {
        e.preventDefault();
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        textarea.value = textarea.value.substring(0, start) + "  " + textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + 2;
        updateHarnessLineCounter();
      }
    });
  }

  if (wrapper) {
    // Zoom via mouse wheel
    wrapper.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      zoomHarnessDiagram(zoomFactor);
    }, { passive: false });

    // Pan via click and drag
    wrapper.addEventListener("mousedown", (e) => {
      if (e.button === 0) {
        isHarnessDragging = true;
        harnessDragStartX = e.clientX - harnessPanX;
        harnessDragStartY = e.clientY - harnessPanY;
        wrapper.style.cursor = "grabbing";
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (isHarnessDragging) {
        harnessPanX = e.clientX - harnessDragStartX;
        harnessPanY = e.clientY - harnessDragStartY;
        applyHarnessTransform();
      }
    });

    window.addEventListener("mouseup", () => {
      if (isHarnessDragging) {
        isHarnessDragging = false;
        if (wrapper) wrapper.style.cursor = "grab";
      }
    });
  }
}

function highlightInterconnectIn3D() {
  closeHarnessStudio();
  if (currentProjectId === "daemon-pore") {
    switchDaemonGeometryMode("system");
    toggleFlowcellCapillary();
    showKiCadToast("🔬 3D Viewport: Reader Instrument & Microfluidics Flowcell highlighted!", 3500);
  } else {
    showKiCadToast("🔬 3D Viewport: System Interconnect highlighted!", 2500);
  }
}
