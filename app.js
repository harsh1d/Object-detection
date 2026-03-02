/* ─── NeuralLens · app.js ──────────────────────────────────────────────────
   Google Lens–style object detection using:
   • TensorFlow.js  (inference engine)
   • MobileNet v2   (image classification – 1000 classes)
   • COCO-SSD v2    (object detection + bounding boxes – 80 classes)
──────────────────────────────────────────────────────────────────────────── */

'use strict';

// ── DOM references ───────────────────────────────────────────────────────────
const loadingScreen  = document.getElementById('loading-screen');
const loaderBar      = document.getElementById('loader-bar');
const loadingStatus  = document.getElementById('loading-status');
const app            = document.getElementById('app');

const video          = document.getElementById('video');
const canvasOverlay  = document.getElementById('canvas-overlay');
const ctx            = canvasOverlay.getContext('2d');

const btnScan        = document.getElementById('btn-scan');
const btnFlip        = document.getElementById('btn-flip');
const btnAutodetect  = document.getElementById('btn-autodetect');
const btnCloseResult = document.getElementById('btn-close-results');

const detectFlash    = document.getElementById('detect-flash');
const viewfinder     = document.getElementById('viewfinder');
const scanLine       = document.getElementById('scan-line');
const statusBadge    = document.getElementById('status-badge');
const statusText     = document.getElementById('status-text');

const resultsPanel   = document.getElementById('results-panel');
const resultsList    = document.getElementById('results-list');
const noResults      = document.getElementById('no-results');

// ── State ─────────────────────────────────────────────────────────────────────
let mobilenetModel   = null;
let cocoSsdModel     = null;
let facingMode       = 'environment';  // 'environment' = rear, 'user' = front
let autoDetectActive = false;
let autoDetectFrame  = null;
let isDetecting      = false;
let currentStream    = null;

// ── roundRect polyfill (for Firefox < 112 and older browsers) ─────────────────
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    this.moveTo(x + r, y);
    this.lineTo(x + w - r, y);
    this.arcTo(x + w, y, x + w, y + r, r);
    this.lineTo(x + w, y + h - r);
    this.arcTo(x + w, y + h, x + w - r, y + h, r);
    this.lineTo(x + r, y + h);
    this.arcTo(x, y + h, x, y + h - r, r);
    this.lineTo(x, y + r);
    this.arcTo(x, y, x + r, y, r);
    this.closePath();
    return this;
  };
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
(async function init() {
  try {
    await loadModels();
    await startCamera();
    showApp();
  } catch (err) {
    console.error(err);
    showCameraError(err);
  }
})();

// ── Model loading ─────────────────────────────────────────────────────────────
async function loadModels() {
  setLoaderProgress(5, 'Loading TensorFlow.js…');
  await tf.ready();

  setLoaderProgress(25, 'Loading MobileNet…');
  mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });

  setLoaderProgress(60, 'Loading COCO-SSD…');
  cocoSsdModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

  setLoaderProgress(100, 'Models ready!');
  await sleep(400);
}

function setLoaderProgress(pct, msg) {
  loaderBar.style.width = pct + '%';
  if (msg) loadingStatus.textContent = msg;
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  if (currentStream) stopStream();

  const constraints = {
    video: {
      facingMode,
      width:  { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (_) {
    // Fallback: any camera
    currentStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }

  video.srcObject = currentStream;
  // Handle case where metadata is already loaded
  if (video.readyState >= 1) {
    resizeOverlay();
  } else {
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
  }
  await video.play();
  resizeOverlay();
}

function stopStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
}

// Keep canvas overlay in sync with video display dimensions (CSS pixels)
function resizeOverlay() {
  canvasOverlay.width  = video.clientWidth  || video.videoWidth  || 640;
  canvasOverlay.height = video.clientHeight || video.videoHeight || 480;
}
window.addEventListener('resize', resizeOverlay);

// ── Show app after loading ─────────────────────────────────────────────────────
function showApp() {
  loadingScreen.classList.add('fade-out');
  setTimeout(() => { loadingScreen.style.display = 'none'; }, 600);
  app.classList.remove('hidden');
  // Slight delay so CSS transition fires
  requestAnimationFrame(() => requestAnimationFrame(() => {
    app.classList.add('visible');
  }));
  setStatus('ready', '✦ Ready to scan');
  btnScan.classList.add('idle-pulse');
}

// ── Camera error UI ───────────────────────────────────────────────────────────
function showCameraError(err) {
  const isPermission = err && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError');
  loadingStatus.innerHTML = isPermission
    ? '📷 Camera access denied.<br><small>Allow camera permission and reload the page.</small>'
    : `⚠ Error: ${err ? err.message : 'Unknown error'}`;
  loaderBar.style.background = '#ef4444';
  loaderBar.style.width = '100%';
}

// ── Detection ──────────────────────────────────────────────────────────────────
async function runDetection() {
  if (isDetecting) return;
  isDetecting = true;

  // Visual: flash, button spinner, viewfinder detecting state
  triggerFlash();
  triggerHaptic();
  btnScan.classList.remove('idle-pulse');
  btnScan.disabled = true;
  setScanBtnLoading(true);
  viewfinder.classList.add('detecting');
  setStatus('detecting', 'Analyzing…');
  showSkeletonResults();

  try {
    const imageData = captureFrame();
    const [mobileResults, cocoResults] = await Promise.all([
      mobilenetModel.classify(imageData, 7),
      cocoSsdModel.detect(imageData),
    ]);

    drawBoundingBoxes(cocoResults);
    displayResults(mobileResults, cocoResults);
  } catch (err) {
    console.error('Detection error:', err);
    setStatus('ready', '⚠ Detection failed');
    resultsList.innerHTML = '';
    noResults.classList.remove('hidden');
    openResults();
  } finally {
    isDetecting = false;
    viewfinder.classList.remove('detecting');
    setScanBtnLoading(false);
    btnScan.disabled = false;
    if (!autoDetectActive) {
      setStatus('ready', '✦ Ready to scan');
      btnScan.classList.add('idle-pulse');
    }
  }
}

// Capture current video frame at full video resolution (for model accuracy)
function captureFrame() {
  const w = video.videoWidth  || canvasOverlay.width;
  const h = video.videoHeight || canvasOverlay.height;
  const offscreen = document.createElement('canvas');
  offscreen.width  = w;
  offscreen.height = h;
  offscreen.getContext('2d').drawImage(video, 0, 0, w, h);
  return offscreen;
}

// ── Bounding boxes ─────────────────────────────────────────────────────────────
function drawBoundingBoxes(predictions) {
  clearOverlay();
  if (!predictions.length) return;

  // Canvas internal dimensions match the CSS display area (clientWidth x clientHeight).
  // Video content uses object-fit: cover, so we must transform bbox coords from
  // raw video space into the covered/cropped display space.
  const vw = video.videoWidth  || canvasOverlay.width;
  const vh = video.videoHeight || canvasOverlay.height;
  const cw = canvasOverlay.width;
  const ch = canvasOverlay.height;

  const scale = Math.max(cw / vw, ch / vh);
  const offX  = (cw - vw * scale) / 2;
  const offY  = (ch - vh * scale) / 2;

  predictions.forEach(pred => {
    const [bx, by, bw, bh] = pred.bbox;
    const sx = bx * scale + offX;
    const sy = by * scale + offY;
    const sw = bw * scale;
    const sh = bh * scale;

    // Box
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(sx, sy, sw, sh);

    // Label background
    const label = `${pred.class} ${Math.round(pred.score * 100)}%`;
    ctx.font     = 'bold 13px Inter, sans-serif';
    const tw     = ctx.measureText(label).width + 14;
    const th     = 22;
    const ly     = sy > th + 6 ? sy - th - 4 : sy + 4;

    const gradient = ctx.createLinearGradient(sx, 0, sx + tw, 0);
    gradient.addColorStop(0, '#7c3aed');
    gradient.addColorStop(1, '#06b6d4');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(sx, ly, tw, th, 4);
    ctx.fill();

    // Label text
    ctx.fillStyle = '#fff';
    ctx.fillText(label, sx + 7, ly + 15);
  });
}

function clearOverlay() {
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
}

// ── Results panel ──────────────────────────────────────────────────────────────
function displayResults(mobileResults, cocoResults) {
  resultsList.innerHTML = '';
  noResults.classList.add('hidden');

  // Merge: COCO detections come first (they have location info),
  // then MobileNet classifications that aren't already represented.
  const combined = [];
  const seenNames = new Set();

  cocoResults.forEach(p => {
    const name = p.class.toLowerCase();
    if (!seenNames.has(name)) {
      combined.push({ name, score: p.score, source: 'COCO-SSD' });
      seenNames.add(name);
    }
  });

  mobileResults.forEach(p => {
    // MobileNet returns names like "n02123045 tabby, tabby cat" — clean them up
    const raw  = p.className.split(',')[0].replace(/^n\d+ /, '').trim().toLowerCase();
    if (!seenNames.has(raw)) {
      combined.push({ name: raw, score: p.probability, source: 'MobileNet' });
      seenNames.add(raw);
    }
  });

  const top = combined.slice(0, 6);

  if (top.length === 0) {
    noResults.classList.remove('hidden');
  } else {
    top.forEach(item => {
      resultsList.appendChild(buildResultCard(item));
    });
  }

  openResults();
}

function buildResultCard({ name, score, source }) {
  const pct        = Math.round(score * 100);
  const searchName = encodeURIComponent(name);
  const wikiUrl    = `https://en.wikipedia.org/w/index.php?search=${searchName}`;
  const googleUrl  = `https://www.google.com/search?q=${searchName}`;

  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <div class="result-top">
      <span class="result-name">${escapeHtml(name)}</span>
      <span class="result-score">${pct}%</span>
    </div>
    <div class="confidence-track">
      <div class="confidence-fill" style="width:0%"></div>
    </div>
    <div class="result-links">
      <a class="result-link" href="${wikiUrl}" target="_blank" rel="noopener">
        <span class="material-icons-round">menu_book</span>Wikipedia
      </a>
      <a class="result-link" href="${googleUrl}" target="_blank" rel="noopener">
        <span class="material-icons-round">search</span>Google
      </a>
      <span class="source-badge">${source}</span>
    </div>
  `;

  // Animate confidence bar after insert
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      card.querySelector('.confidence-fill').style.width = pct + '%';
    });
  });

  return card;
}

function openResults()  { resultsPanel.classList.add('open'); }
function closeResults() { resultsPanel.classList.remove('open'); }

// ── Auto-detect mode ───────────────────────────────────────────────────────────
function startAutoDetect() {
  autoDetectActive = true;
  btnAutodetect.classList.add('active');
  scanLine.classList.add('active');
  btnScan.classList.remove('idle-pulse');
  setStatus('detecting', 'Auto-detecting…');
  scheduleAutoDetect();
}

function stopAutoDetect() {
  autoDetectActive = false;
  btnAutodetect.classList.remove('active');
  scanLine.classList.remove('active');
  cancelAnimationFrame(autoDetectFrame);
  btnScan.classList.add('idle-pulse');
  setStatus('ready', '✦ Ready to scan');
}

let lastAutoTime = 0;
const AUTO_INTERVAL_MS = 1800; // run detection every ~1.8s in auto mode

function scheduleAutoDetect() {
  if (!autoDetectActive) return;
  autoDetectFrame = requestAnimationFrame(async (ts) => {
    if (ts - lastAutoTime >= AUTO_INTERVAL_MS) {
      lastAutoTime = ts;
      await runDetection();
    }
    scheduleAutoDetect();
  });
}

// ── Status helper ──────────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusText.textContent = text;
  statusBadge.classList.toggle('detecting', state === 'detecting');
  const icon = statusBadge.querySelector('.material-icons-round');
  icon.textContent = state === 'detecting' ? 'radar' : 'visibility';
}

// ── Event listeners ────────────────────────────────────────────────────────────
btnScan.addEventListener('click', (e) => {
  if (!autoDetectActive) {
    addRipple(btnScan, e);
    runDetection();
  }
});

btnFlip.addEventListener('click', async () => {
  stopAutoDetect();
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  clearOverlay();
  closeResults();
  // Spin animation
  btnFlip.classList.add('spinning');
  btnFlip.addEventListener('animationend', () => btnFlip.classList.remove('spinning'), { once: true });
  try {
    await startCamera();
  } catch (err) {
    console.error('Camera flip error:', err);
  }
});

btnAutodetect.addEventListener('click', () => {
  if (autoDetectActive) {
    stopAutoDetect();
  } else {
    startAutoDetect();
  }
});

btnCloseResult.addEventListener('click', () => {
  closeResults();
  clearOverlay();
});

// ── Visual feedback helpers ────────────────────────────────────────────────────

// Brief white flash on the viewfinder
function triggerFlash() {
  detectFlash.classList.add('flash');
  setTimeout(() => detectFlash.classList.remove('flash'), 120);
}

// Mobile haptic feedback (if supported)
function triggerHaptic() {
  if (navigator.vibrate) navigator.vibrate(30);
}

// Ripple effect on a button at pointer position
function addRipple(btn, e) {
  const rect = btn.getBoundingClientRect();
  const rx = ((e.clientX - rect.left) / rect.width * 100).toFixed(1) + '%';
  const ry = ((e.clientY - rect.top)  / rect.height * 100).toFixed(1) + '%';
  btn.style.setProperty('--rx', rx);
  btn.style.setProperty('--ry', ry);
  btn.classList.add('rippling');
  setTimeout(() => btn.classList.remove('rippling'), 550);
}

// Show/hide loading spinner inside the scan button
function setScanBtnLoading(on) {
  const icon = btnScan.querySelector('.material-icons-round');
  const text = btnScan.querySelector('span:last-child');
  if (on) {
    icon.textContent = 'radar';
    icon.style.animation = 'radarSpin 1s linear infinite';
    text.textContent = 'Scanning…';
  } else {
    icon.style.animation = '';
    icon.textContent = 'search';
    text.textContent = 'Scan Object';
  }
}

// Show skeleton placeholder cards while detection runs
function showSkeletonResults() {
  resultsList.innerHTML = '';
  noResults.classList.add('hidden');
  for (let i = 0; i < 3; i++) {
    const s = document.createElement('div');
    s.className = 'skeleton-card';
    s.style.animationDelay = (i * 0.07) + 's';
    s.innerHTML = `
      <div class="skel skel-title"></div>
      <div class="skel skel-bar"></div>
      <div class="skel-tags">
        <div class="skel skel-tag"></div>
        <div class="skel skel-tag"></div>
      </div>`;
    resultsList.appendChild(s);
  }
  openResults();
}

// ── Utilities ──────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
