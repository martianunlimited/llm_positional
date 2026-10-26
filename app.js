/**
 * LLM Positional Encodings Lab - Interactive Client-Side Engine
 * Visualizing Permutation Invariance, Sinusoidal Clock Hands,
 * and Modern Rotary Position Embeddings (RoPE).
 */

// Fallback presets
const FALLBACK_PRESETS = [
  {
    "id": "dog-man-permutation",
    "title": "The Bag-of-Words Dilemma: Dog Bites Man",
    "category": "Permutation Dilemma",
    "description": "Demonstrating why Transformers cannot discern word order without PE.",
    "sentence_a": "Dog bites man",
    "sentence_b": "Man bites dog",
    "default_dim": 16,
    "default_base": 10000.0
  },
  {
    "id": "sinusoidal-clock",
    "title": "The Vaswani Clock: Multi-Frequency Harmonic Waves",
    "category": "Sinusoidal Waves",
    "description": "The original 2017 Transformer architecture: Seconds, Minutes, and Hours.",
    "sentence_a": "Attention Is All You Need",
    "default_dim": 16,
    "default_base": 10000.0
  },
  {
    "id": "rope-unit-circle",
    "title": "Modern Rotary Position Embeddings (RoPE)",
    "category": "Rotary Embedding (RoPE)",
    "description": "How LLaMA, Mistral, and modern LLMs rotate vectors in 2D complex planes.",
    "sentence_a": "The quick brown fox jumps over the lazy dog",
    "default_dim": 16,
    "default_base": 10000.0
  },
  {
    "id": "llama3-long-context",
    "title": "LLaMA 3 Context Window Scaling: Base 10,000 vs 500,000",
    "category": "Long-Context Extrapolation",
    "description": "How increasing the RoPE frequency base unlocks 128k context windows.",
    "sentence_a": "The long document begins here and continues for 128k tokens",
    "default_dim": 16,
    "default_base": 500000.0
  }
];

// App State
const state = {
  presets: FALLBACK_PRESETS,
  peEnabled: true,
  maxPos: 32,
  embedDim: 16,
  selectedDim: 0,
  tokenPos: 3,
  selectedPlane: 0,
  ropeBase: 10000.0
};

// ============================================================================
// Core Positional Mathematics
// ============================================================================

function computeSinusoidalPE(maxPos, dim, base = 10000.0) {
  const matrix = [];
  for (let pos = 0; pos < maxPos; pos++) {
    const row = [];
    for (let i = 0; i < dim; i += 2) {
      const denom = Math.pow(base, i / dim);
      row.push(Math.sin(pos / denom));
      if (i + 1 < dim) {
        row.push(Math.cos(pos / denom));
      }
    }
    matrix.push(row);
  }
  return matrix;
}

function computeRoPETheta(subspaceIdx, dim, base = 10000.0) {
  return 1.0 / Math.pow(base, (2.0 * subspaceIdx) / dim);
}

function getWordVector(word, dim = 8) {
  let seed = 0;
  for (let p = 0; p < word.length; p++) {
    seed += word.charCodeAt(p) * (p + 1);
  }
  const vec = [];
  for (let d = 0; d < dim; d++) {
    vec.push(Math.sin(seed + d * 1.5));
  }
  const norm = Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0)) || 1.0;
  return vec.map(x => x / norm);
}

function computePermutationAttention(tokens, usePE, dim = 8) {
  const n = tokens.length;
  const peMatrix = computeSinusoidalPE(n + 5, dim);
  const matrix = [];

  for (let i = 0; i < n; i++) {
    let q = getWordVector(tokens[i], dim);
    if (usePE) {
      q = q.map((val, d) => val + peMatrix[i][d] * 0.6);
    }

    const rowScores = [];
    for (let j = 0; j < n; j++) {
      let k = getWordVector(tokens[j], dim);
      if (usePE) {
        k = k.map((val, d) => val + peMatrix[j][d] * 0.6);
      }
      let dot = 0.0;
      for (let d = 0; d < dim; d++) dot += q[d] * k[d];
      rowScores.push(dot);
    }

    // Softmax
    const maxS = Math.max(...rowScores);
    const exps = rowScores.map(s => Math.exp(s - maxS));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    matrix.push(exps.map(e => e / sumExp));
  }

  return matrix;
}

// ============================================================================
// Visualizer 1: Permutation Invariance Matrix Renderers
// ============================================================================

function renderPermutationMatrices() {
  const sentA = ["Dog", "bites", "man"];
  const sentB = ["Man", "bites", "dog"];

  const matA = computePermutationAttention(sentA, state.peEnabled);
  const matB = computePermutationAttention(sentB, state.peEnabled);

  renderMatrixTable("matrix-sent-a", sentA, matA, "#06b6d4");
  renderMatrixTable("matrix-sent-b", sentB, matB, "#ec4899");

  const badge = document.getElementById("pe-status-text");
  const callout = document.getElementById("permutation-insight-box");

  if (state.peEnabled) {
    badge.textContent = "Status: Positional coordinates active. Word order syntax is preserved!";
    badge.className = "pe-status-badge";
    callout.innerHTML = `
      <strong>💡 Key Insight:</strong> Positional Encoding is <strong>ON</strong>. Unique position vectors are injected into positions 0, 1, and 2. The attention matrix for <em>"Dog bites man"</em> is distinctly asymmetric from <em>"Man bites dog"</em>, enabling the LLM to identify the grammatical subject vs direct object!
    `;
  } else {
    badge.textContent = "Status: Positional Encoding OFF. Bag-of-Words bug active!";
    badge.className = "pe-status-badge off-status";
    callout.innerHTML = `
      <strong>⚠️ Failure Mode:</strong> Positional Encoding is <strong>OFF</strong>. Notice how the attention weights between identical word tokens are completely identical and symmetric! The self-attention mechanism is permutation invariant and cannot tell who bit whom.
    `;
  }
}

function renderMatrixTable(elemId, tokens, matrix, accentColor) {
  const container = document.getElementById(elemId);
  container.innerHTML = "";

  // Corner header
  const corner = document.createElement("div");
  corner.className = "perm-cell header-cell";
  corner.textContent = "Q \\ K";
  container.appendChild(corner);

  // Column headers
  tokens.forEach(tok => {
    const colHeader = document.createElement("div");
    colHeader.className = "perm-cell header-cell";
    colHeader.textContent = tok;
    container.appendChild(colHeader);
  });

  // Rows
  tokens.forEach((tok, i) => {
    const rowHeader = document.createElement("div");
    rowHeader.className = "perm-cell header-cell";
    rowHeader.textContent = tok;
    container.appendChild(rowHeader);

    for (let j = 0; j < tokens.length; j++) {
      const cell = document.createElement("div");
      cell.className = "perm-cell";
      const val = matrix[i][j];
      cell.textContent = val.toFixed(2);

      // Color mapping
      cell.style.backgroundColor = `rgba(245, 158, 11, ${val * 0.75})`;
      cell.style.color = val > 0.4 ? "#fff" : "rgba(255, 255, 255, 0.7)";
      cell.title = `Query '${tok}' [${i}] ➔ Key '${tokens[j]}' [${j}]: ${(val * 100).toFixed(1)}%`;
      container.appendChild(cell);
    }
  });
}

// ============================================================================
// Visualizer 2: Vaswani Sinusoidal Heatmap & Wave Graph
// ============================================================================

function renderSinusoidalHeatmap() {
  const canvas = document.getElementById("sinusoidal-heatmap-canvas");
  const ctx = canvas.getContext("2d");
  const pe = computeSinusoidalPE(state.maxPos, state.embedDim);

  const cellW = canvas.width / state.embedDim;
  const cellH = canvas.height / state.maxPos;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let pos = 0; pos < state.maxPos; pos++) {
    for (let d = 0; d < state.embedDim; d++) {
      const val = pe[pos][d]; // in [-1.0, 1.0]
      // -1.0 -> Blue, 0.0 -> Dark slate, +1.0 -> Amber
      if (val >= 0) {
        ctx.fillStyle = `rgba(245, 158, 11, ${val * 0.9 + 0.1})`;
      } else {
        ctx.fillStyle = `rgba(3, 105, 161, ${-val * 0.9 + 0.1})`;
      }

      ctx.fillRect(d * cellW, pos * cellH, cellW - 1, cellH - 1);

      // Highlight selected dimension column
      if (d === state.selectedDim) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.strokeRect(d * cellW, pos * cellH, cellW - 1, cellH - 1);
      }
    }
  }
}

function renderSinusoidalWave() {
  const canvas = document.getElementById("sinusoidal-wave-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const d = state.selectedDim;
  const isEven = (d % 2 === 0);
  const denom = Math.pow(10000.0, 2.0 * Math.floor(d / 2) / state.embedDim);
  const wavelength = 2.0 * Math.PI * denom;

  // Update readouts
  const titleEl = document.getElementById("active-dim-title");
  titleEl.textContent = `Waveform: Dimension ${d} (${isEven ? "sin" : "cos"})`;
  
  const badgeEl = document.getElementById("active-wavelength-badge");
  badgeEl.textContent = `Wavelength (Period): ${wavelength >= 1000 ? (wavelength / 1000).toFixed(1) + "k" : wavelength.toFixed(2)} tokens`;

  const clockText = document.getElementById("clock-desc-text");
  if (d <= 2) {
    clockText.innerHTML = `<strong>Dimension ${d} (Seconds Hand):</strong> Rapid oscillation with period ~${wavelength.toFixed(1)} tokens. It changes sign frequently, capturing immediate local word order and subject-verb agreements.`;
  } else if (d <= 8) {
    clockText.innerHTML = `<strong>Dimension ${d} (Minutes Hand):</strong> Medium oscillation with period ~${wavelength.toFixed(0)} tokens. It captures sentence clauses and paragraph-level structures.`;
  } else {
    clockText.innerHTML = `<strong>Dimension ${d} (Hours Hand):</strong> Extremely slow oscillation with period ~${(wavelength / 1000).toFixed(1)}k tokens. It provides a steady global coordinate for multi-page documents.`;
  }

  // Draw grid & zero axis
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();

  // Plot continuous curve across positions 0..32
  ctx.strokeStyle = isEven ? "#f59e0b" : "#38bdf8";
  ctx.lineWidth = 3;
  ctx.beginPath();

  const numPoints = 120;
  for (let p = 0; p <= numPoints; p++) {
    const pos = (p / numPoints) * state.maxPos;
    const val = isEven ? Math.sin(pos / denom) : Math.cos(pos / denom);
    const x = (p / numPoints) * canvas.width;
    const y = canvas.height / 2 - val * (canvas.height * 0.4);

    if (p === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Plot discrete integer token dots
  ctx.fillStyle = "#fff";
  for (let pos = 0; pos <= state.maxPos; pos += 2) {
    const val = isEven ? Math.sin(pos / denom) : Math.cos(pos / denom);
    const x = (pos / state.maxPos) * canvas.width;
    const y = canvas.height / 2 - val * (canvas.height * 0.4);

    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function setupDimensionButtons() {
  const ribbon = document.getElementById("dim-selector-ribbon");
  ribbon.innerHTML = "";

  for (let d = 0; d < state.embedDim; d++) {
    const btn = document.createElement("button");
    btn.className = `dim-btn ${d === state.selectedDim ? "active" : ""}`;
    btn.textContent = `d${d} (${d % 2 === 0 ? "sin" : "cos"})`;
    btn.addEventListener("click", () => {
      state.selectedDim = d;
      document.querySelectorAll(".dim-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderSinusoidalHeatmap();
      renderSinusoidalWave();
    });
    ribbon.appendChild(btn);
  }
}

// ============================================================================
// Visualizer 3: Modern RoPE 2D Complex Unit Circle Canvas
// ============================================================================

function renderRoPECircle() {
  const canvas = document.getElementById("rope-circle-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const radius = 140;

  // Draw coordinate axes
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - radius - 20, cy);
  ctx.lineTo(cx + radius + 20, cy);
  ctx.moveTo(cx, cy - radius - 20);
  ctx.lineTo(cx, cy + radius + 20);
  ctx.stroke();

  // Draw Unit Circle
  ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Calculate RoPE Angle
  const theta = computeRoPETheta(state.selectedPlane, state.embedDim, state.ropeBase);
  const angleRad = state.tokenPos * theta;
  const angleDeg = (angleRad * (180.0 / Math.PI)) % 360.0;
  const cosVal = Math.cos(angleRad);
  const sinVal = Math.sin(angleRad);

  // Update readouts
  document.getElementById("readout-angle-deg").textContent = `${angleDeg.toFixed(1)}°`;
  document.getElementById("readout-cos").textContent = cosVal.toFixed(3);
  document.getElementById("readout-sin").textContent = sinVal.toFixed(3);

  // Draw Rotation Arc from 0 to angle
  ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.35, 0, -angleRad, true);
  ctx.stroke();

  // Draw Base Vector at pos 0 (horizontal unit vector)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + radius, cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Rotated RoPE Vector R_m * q
  const vx = cx + cosVal * radius;
  const vy = cy - sinVal * radius; // inverted y in screen coords

  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(vx, vy);
  ctx.stroke();

  // Vector Arrowhead
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(vx, vy, 7, 0, Math.PI * 2);
  ctx.fill();

  // Label
  ctx.font = "bold 14px 'JetBrains Mono'";
  ctx.fillStyle = "#fff";
  ctx.fillText(`R_${state.tokenPos} * q`, vx + 10, vy);
}

function renderRoPEDecay() {
  const canvas = document.getElementById("rope-decay-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const theta = computeRoPETheta(state.selectedPlane, state.embedDim, state.ropeBase);
  const maxDist = 30;

  // Zero axis line
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();

  // Plot cos(delta * theta)
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 3;
  ctx.beginPath();

  for (let dist = 0; dist <= maxDist; dist++) {
    const val = Math.cos(dist * theta);
    const x = (dist / maxDist) * canvas.width;
    const y = canvas.height / 2 - val * (canvas.height * 0.4);

    if (dist === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Highlight current pos distance
  const currentVal = Math.cos(state.tokenPos * theta);
  const curX = (state.tokenPos / maxDist) * canvas.width;
  const curY = canvas.height / 2 - currentVal * (canvas.height * 0.4);

  ctx.fillStyle = "#ec4899";
  ctx.beginPath();
  ctx.arc(curX, curY, 6, 0, Math.PI * 2);
  ctx.fill();

  document.getElementById("decay-val-badge").textContent = `cos(Δθ) = ${currentVal.toFixed(3)}`;
}

// ============================================================================
// Visualizer 4: 128k Context Scaling Comparison Mini-Canvases
// ============================================================================

function renderScalingCanvases() {
  drawMiniCircle("canvas-base-10k", 10000.0, "#ef4444");
  drawMiniCircle("canvas-base-500k", 500000.0, "#10b981");
}

function drawMiniCircle(canvasId, baseVal, strokeColor) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const radius = 95;

  ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Simulate token at position 32,768 in Plane 1
  const pos = 32768;
  const theta = computeRoPETheta(1, 16, baseVal);
  const angleRad = pos * theta;
  const fullRotations = angleRad / (2.0 * Math.PI);

  const cosVal = Math.cos(angleRad);
  const sinVal = Math.sin(angleRad);

  const vx = cx + cosVal * radius;
  const vy = cy - sinVal * radius;

  // Arc
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(vx, vy);
  ctx.stroke();

  ctx.fillStyle = strokeColor;
  ctx.beginPath();
  ctx.arc(vx, vy, 6, 0, Math.PI * 2);
  ctx.fill();

  // Center dot
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fill();
}

// ============================================================================
// Event Listeners & Master Init
// ============================================================================

function setupEventListeners() {
  // Permutation Toggle
  const btnOff = document.getElementById("btn-pe-off");
  const btnOn = document.getElementById("btn-pe-on");

  btnOff.addEventListener("click", () => {
    state.peEnabled = false;
    btnOff.classList.add("active");
    btnOn.classList.remove("active");
    renderPermutationMatrices();
  });

  btnOn.addEventListener("click", () => {
    state.peEnabled = true;
    btnOn.classList.add("active");
    btnOff.classList.remove("active");
    renderPermutationMatrices();
  });

  // Token Pos Slider
  const sliderPos = document.getElementById("slider-token-pos");
  sliderPos.addEventListener("input", (e) => {
    state.tokenPos = parseInt(e.target.value, 10);
    document.getElementById("val-token-pos").textContent = state.tokenPos;
    renderRoPECircle();
    renderRoPEDecay();
  });

  // Subspace Plane Slider
  const sliderPlane = document.getElementById("slider-plane-idx");
  sliderPlane.addEventListener("input", (e) => {
    state.selectedPlane = parseInt(e.target.value, 10);
    const speed = state.selectedPlane === 0 ? "Fastest" : (state.selectedPlane === 3 ? "Slowest" : "Medium");
    document.getElementById("val-plane-idx").textContent = `Plane ${state.selectedPlane} (${speed})`;
    document.getElementById("plane-badge").textContent = `Plane ${state.selectedPlane} (Dims ${state.selectedPlane * 2}, ${state.selectedPlane * 2 + 1})`;
    renderRoPECircle();
    renderRoPEDecay();
  });

  // Heatmap click to select dimension
  const heatCanvas = document.getElementById("sinusoidal-heatmap-canvas");
  heatCanvas.addEventListener("click", (e) => {
    const rect = heatCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickedDim = Math.floor((x / rect.width) * state.embedDim);
    state.selectedDim = Math.max(0, Math.min(state.embedDim - 1, clickedDim));

    document.querySelectorAll(".dim-btn").forEach((b, idx) => {
      b.classList.toggle("active", idx === state.selectedDim);
    });

    renderSinusoidalHeatmap();
    renderSinusoidalWave();
  });
}

function initApp() {
  setupEventListeners();
  setupDimensionButtons();
  renderPermutationMatrices();
  renderSinusoidalHeatmap();
  renderSinusoidalWave();
  renderRoPECircle();
  renderRoPEDecay();
  renderScalingCanvases();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
