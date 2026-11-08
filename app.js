/**
 * EmoScan — app.js (FIXED)
 * - Model di-load via fetch dari CDN fallback jika file:// gagal
 * - Tombol kamera aktif meski model belum siap (graceful)
 * - Global upload dari preprocessing.js terhubung ke deteksi emosi
 */

// ── CONFIG ────────────────────────────────────────────────
// Coba load model dari folder lokal './models', fallback ke CDN
const MODEL_URL_LOCAL = './models';
const MODEL_URL_CDN   = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

const EMOTIONS = [
  { key: 'happy',     label: 'Happy',     labelId: 'Senang',   emoji: '😊', color: '#c99b52' },
  { key: 'sad',       label: 'Sad',       labelId: 'Sedih',    emoji: '😢', color: '#7fa3bd' },
  { key: 'angry',     label: 'Angry',     labelId: 'Marah',    emoji: '😠', color: '#c2717a' },
  { key: 'fearful',   label: 'Fear',      labelId: 'Takut',    emoji: '😨', color: '#a396b3' },
  { key: 'disgusted', label: 'Disgust',   labelId: 'Jijik',    emoji: '🤢', color: '#7fae91' },
  { key: 'surprised', label: 'Surprised', labelId: 'Terkejut', emoji: '😲', color: '#c99a63' },
  { key: 'neutral',   label: 'Neutral',   labelId: 'Netral',   emoji: '😐', color: '#9ea3ad' },
];

// ── ELEMENTS ──────────────────────────────────────────────
const video        = document.getElementById('video');
const overlay      = document.getElementById('overlay');
const videoIdle    = document.getElementById('videoIdle');
const videoLoading = document.getElementById('videoLoading');
const liveBadge    = document.getElementById('liveBadge');
const btnStart     = document.getElementById('btnStart');
const btnStop      = document.getElementById('btnStop');
const statusDot    = document.getElementById('statusDot');
const statusText   = document.getElementById('statusText');

const emotionIdleState   = document.getElementById('emotionIdleState');
const emotionActiveState = document.getElementById('emotionActiveState');
const emotionEmoji       = document.getElementById('emotionEmoji');
const emotionLabel       = document.getElementById('emotionLabel');
const emotionLabelId     = document.getElementById('emotionLabelId');
const confidenceBar      = document.getElementById('confidenceBar');
const confidenceText     = document.getElementById('confidenceText');
const scoresGrid         = document.getElementById('scoresGrid');

const uploadPreviewWrap = document.getElementById('uploadPreviewWrap');
const uploadPreview     = document.getElementById('uploadPreview');
const btnDetect         = document.getElementById('btnDetect');
const uploadCanvas      = document.getElementById('uploadCanvas');

// ── STATE ─────────────────────────────────────────────────
let stream         = null;
let animFrame      = null;
let modelReady     = false;   // face detector (tinyFaceDetector) siap
let knnModel       = null;    // model KNN hasil training (dari shared.js loadModel())
let isRunning      = false;
let lastEmotionKey = '';

const noModelBanner = document.getElementById('noModelBanner');

// ── KNN MODEL GATE ────────────────────────────────────────
// Deteksi emosi HANYA boleh jalan kalau model KNN sudah dilatih
// lewat pipeline Dataset Collection -> Training -> Evaluation.
function refreshKnnModelStatus() {
  knnModel = typeof loadModel === 'function' ? loadModel() : null;
  const hasModel = !!(knnModel && knnModel.trainData && knnModel.trainData.length > 0);

  if (noModelBanner) noModelBanner.classList.toggle('hidden', hasModel);
  if (btnDetect) btnDetect.disabled = !hasModel;

  if (!hasModel) {
    if (btnStart) btnStart.disabled = true;
    setStatus('error', 'Belum ada model KNN. Lengkapi Dataset, Training, lalu Evaluation dulu.');
  } else if (modelReady) {
    if (btnStart) btnStart.disabled = false;
    setStatus('ready', `Model KNN siap (${knnModel.nSamples} sampel, K=${knnModel.k}) — klik Mulai Kamera atau upload gambar`);
  }
  return hasModel;
}

// ── INIT ──────────────────────────────────────────────────
buildScoresGrid();

async function init() {
  setStatus('loading', 'Memuat model deteksi wajah dari CDN...');

  try {
    // Try CDN first (works in both file:// and http://)
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL_CDN);
    modelReady = true;
    console.log('Face detector loaded from CDN');
  } catch (err1) {
    console.warn('CDN load failed, trying local...', err1);
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL_LOCAL);
      modelReady = true;
      console.log('Face detector loaded from local');
    } catch (err2) {
      setStatus('error', 'Model deteksi wajah gagal load — jalankan via server lokal (VS Code Live Server)');
      console.error('Both model sources failed:', err2);
      showModelError();
      return;
    }
  }

  // Setelah detektor wajah siap, cek apakah model KNN sudah ada.
  // Kamera/upload TIDAK akan aktif kalau belum ada model hasil training.
  refreshKnnModelStatus();
}

// Re-cek status model setiap kali tab ini kembali aktif (misal user baru
// selesai training di tab lain lalu balik ke halaman ini).
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshKnnModelStatus();
});
window.addEventListener('focus', refreshKnnModelStatus);
window.addEventListener('storage', (e) => {
  if (e.key === 'emoscan_knn_model') refreshKnnModelStatus();
});

function showModelError() {
  const bar = document.getElementById('statusBar');
  bar.style.background = 'rgba(255, 92, 114, 0.15)';
  bar.style.borderBottom = '1px solid #c2717a';
  statusText.innerHTML = `
    Model AI gagal dimuat. Buka folder project dengan <strong>VS Code Live Server</strong> atau jalankan
    <code style="background:#2a2b30;padding:2px 6px;border-radius:4px">python -m http.server 8000</code>
    di folder project, lalu buka <strong>http://localhost:8000</strong>
  `;
}

// ── STATUS ────────────────────────────────────────────────
function setStatus(state, msg) {
  statusText.textContent = msg;
  statusDot.className = 'status-dot ' + state;
}

// ── SCORES GRID ───────────────────────────────────────────
function buildScoresGrid() {
  if (!scoresGrid) return;
  scoresGrid.innerHTML = '';
  EMOTIONS.forEach(em => {
    const row = document.createElement('div');
    row.className = 'score-row';
    row.id = 'row-' + em.key;
    row.innerHTML = `
      <span class="score-emoji">${em.emoji}</span>
      <span class="score-name">${em.label}</span>
      <div class="score-bar-bg">
        <div class="score-bar-fill" id="bar-${em.key}" style="width:0%;background:${em.color}"></div>
      </div>
      <span class="score-pct" id="pct-${em.key}">0%</span>
    `;
    scoresGrid.appendChild(row);
  });
}

// ── UPDATE UI ─────────────────────────────────────────────
function updateEmotionUI(expressions) {
  const sorted   = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
  const [topKey, topScore] = sorted[0];
  const em  = EMOTIONS.find(e => e.key === topKey) || EMOTIONS[6];
  const pct = Math.round(topScore * 100);

  if (emotionActiveState) emotionActiveState.classList.remove('hidden');
  if (emotionIdleState)   emotionIdleState.classList.add('hidden');

  if (topKey !== lastEmotionKey) {
    if (emotionEmoji) {
      emotionEmoji.textContent = em.emoji;
      emotionEmoji.style.animation = 'none';
      requestAnimationFrame(() => { emotionEmoji.style.animation = ''; });
    }
    if (emotionLabel)   emotionLabel.textContent   = em.label;
    if (emotionLabelId) emotionLabelId.textContent = em.labelId;
    lastEmotionKey = topKey;
  }

  if (confidenceBar)  confidenceBar.style.width   = pct + '%';
  if (confidenceText) confidenceText.textContent  = pct + '%';

  EMOTIONS.forEach(e => {
    const score = expressions[e.key] || 0;
    const p     = Math.round(score * 100);
    const bar   = document.getElementById('bar-' + e.key);
    const pctEl = document.getElementById('pct-' + e.key);
    const rowEl = document.getElementById('row-' + e.key);
    if (bar)   bar.style.width    = p + '%';
    if (pctEl) pctEl.textContent  = p + '%';
    if (rowEl) rowEl.classList.toggle('top', e.key === topKey);
  });
}

function resetEmotionUI() {
  if (emotionActiveState) emotionActiveState.classList.add('hidden');
  if (emotionIdleState)   emotionIdleState.classList.remove('hidden');
  lastEmotionKey = '';
  EMOTIONS.forEach(e => {
    const bar = document.getElementById('bar-' + e.key);
    const pct = document.getElementById('pct-' + e.key);
    const row = document.getElementById('row-' + e.key);
    if (bar) bar.style.width     = '0%';
    if (pct) pct.textContent     = '0%';
    if (row) row.classList.remove('top');
  });
  if (confidenceBar)  confidenceBar.style.width  = '0%';
  if (confidenceText) confidenceText.textContent = '0%';
}

// ── CAMERA ────────────────────────────────────────────────
if (btnStart) btnStart.addEventListener('click', startCamera);
if (btnStop)  btnStop.addEventListener('click',  stopCamera);

async function startCamera() {
  if (!refreshKnnModelStatus()) {
    setStatus('error', 'Model KNN belum dilatih. Lengkapi Dataset → Training → Evaluation dulu.');
    return;
  }
  if (!modelReady) {
    setStatus('error', 'Model deteksi wajah belum siap. Buka via server lokal (Live Server / python -m http.server)');
    return;
  }
  if (videoIdle)    videoIdle.classList.add('hidden');
  if (videoLoading) videoLoading.classList.remove('hidden');

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);

    if (videoLoading) videoLoading.classList.add('hidden');
    if (liveBadge)    liveBadge.classList.remove('hidden');
    if (btnStart)     btnStart.classList.add('hidden');
    if (btnStop)      btnStop.classList.remove('hidden');
    isRunning = true;
    setStatus('running', 'Kamera aktif — mendeteksi emosi...');

    overlay.width  = video.videoWidth;
    overlay.height = video.videoHeight;
    detectLoop();
  } catch (err) {
    if (videoLoading) videoLoading.classList.add('hidden');
    if (videoIdle)    videoIdle.classList.remove('hidden');
    setStatus('error', 'Gagal akses kamera: ' + err.message);
  }
}

function stopCamera() {
  isRunning = false;
  cancelAnimationFrame(animFrame);
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (liveBadge) liveBadge.classList.add('hidden');
  if (videoIdle) videoIdle.classList.remove('hidden');
  if (btnStop)   btnStop.classList.add('hidden');
  if (btnStart)  btnStart.classList.remove('hidden');
  setStatus('ready', 'Kamera dihentikan — siap digunakan kembali');
  resetEmotionUI();
}

// ── HELPER: KNN result -> objek "expressions" (kompatibel UI) ─
function knnResultToExpressions(result) {
  // votes/kNearest.length dipakai sebagai pseudo-probabilitas per kelas,
  // supaya bisa langsung dipakai oleh updateEmotionUI()/drawFaceBox()
  const exp = {};
  EMOTIONS.forEach(e => { exp[e.key] = 0; });
  if (result && result.kNearest && result.kNearest.length > 0) {
    Object.entries(result.votes).forEach(([k, v]) => {
      exp[k] = v / result.kNearest.length;
    });
  }
  return exp;
}

// Ambil ImageData dari frame video / elemen gambar (frame penuh,
// konsisten dengan cara Dataset Collection mengambil sampel training)
function captureFullFrameImageData(source, w, h) {
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const ctx = tmp.getContext('2d');
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// ── DETECTION LOOP ────────────────────────────────────────
async function detectLoop() {
  if (!isRunning) return;
  if (!knnModel) { resetEmotionUI(); animFrame = requestAnimationFrame(detectLoop); return; }

  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 });
  try {
    const detections = await faceapi.detectAllFaces(video, opts);
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (detections.length > 0) {
      const imgData  = captureFullFrameImageData(video, video.videoWidth, video.videoHeight);
      const features = extractFeaturesShared(imgData, video.videoWidth, video.videoHeight);
      const result   = knnPredictShared(features, knnModel);
      const expressions = knnResultToExpressions(result);

      detections.forEach(det => drawFaceBox(ctx, det.box, expressions));
      updateEmotionUI(expressions);
    } else {
      resetEmotionUI();
    }
  } catch (e) { /* skip frame */ }
  animFrame = requestAnimationFrame(detectLoop);
}

// ── DRAW FACE BOX ─────────────────────────────────────────
function drawFaceBox(ctx, box, expressions) {
  const { x, y, width, height } = box;
  const topEntry = Object.entries(expressions).sort((a, b) => b[1] - a[1])[0];
  const topKey   = topEntry[0], topScore = topEntry[1];
  const em       = EMOTIONS.find(e => e.key === topKey) || EMOTIONS[6];

  ctx.strokeStyle = em.color; ctx.lineWidth = 2.5;
  ctx.shadowColor = em.color; ctx.shadowBlur = 10;
  ctx.beginPath(); ctx.roundRect(x, y, width, height, 8); ctx.stroke();
  ctx.shadowBlur  = 0;

  const cs = 16;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
  [[x,y],[x+width,y],[x,y+height],[x+width,y+height]].forEach(([cx, cy], i) => {
    ctx.beginPath();
    const sx = (i%2===0?1:-1)*cs, sy = (i<2?1:-1)*cs;
    ctx.moveTo(cx, cy+sy); ctx.lineTo(cx, cy); ctx.lineTo(cx+sx, cy); ctx.stroke();
  });

  const label = `${em.emoji} ${em.label} ${Math.round(topScore*100)}%`;
  ctx.font = 'bold 13px "Space Mono", monospace';
  const tw = ctx.measureText(label).width, pad = 8, lh = 22;
  ctx.fillStyle = em.color;
  ctx.beginPath(); ctx.roundRect(x-1, y-lh-pad, tw+pad*2, lh+pad, 4); ctx.fill();
  ctx.fillStyle = '#000'; ctx.fillText(label, x+pad-1, y-pad+2);
}

// ── DETECT FROM UPLOADED IMAGE ────────────────────────────
// Dipanggil dari preprocessing.js saat global upload berhasil
window.detectEmotionFromImage = async function(imgElement) {
  if (!refreshKnnModelStatus()) {
    setStatus('error', 'Model KNN belum dilatih. Lengkapi Dataset → Training → Evaluation dulu.');
    return;
  }
  if (!modelReady) {
    setStatus('error', 'Model deteksi wajah belum siap. Buka via server lokal.');
    return;
  }
  if (!imgElement) return;

  if (btnDetect) { btnDetect.textContent = 'Mendeteksi...'; btnDetect.disabled = true; }

  try {
    await new Promise(res => { if (imgElement.complete) res(); else imgElement.onload = res; });
    const opts       = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 });
    const detections = await faceapi.detectAllFaces(imgElement, opts);

    if (detections.length === 0) {
      setStatus('error', 'Tidak ada wajah terdeteksi di gambar');
      resetEmotionUI();
    } else {
      const w = imgElement.naturalWidth  || imgElement.width;
      const h = imgElement.naturalHeight || imgElement.height;
      const imgData  = captureFullFrameImageData(imgElement, w, h);
      const features = extractFeaturesShared(imgData, w, h);
      const result   = knnPredictShared(features, knnModel);
      const expressions = knnResultToExpressions(result);

      updateEmotionUI(expressions);
      drawUploadResult(imgElement, detections, expressions);
      setStatus('ready', `${detections.length} wajah terdeteksi — emosi diprediksi via KNN (${result.kNearest.length} tetangga)`);
    }
  } catch (err) {
    setStatus('error', 'Error deteksi: ' + err.message);
    console.error(err);
  }

  if (btnDetect) { btnDetect.textContent = 'Deteksi Gambar'; btnDetect.disabled = false; }
};

if (btnDetect) {
  btnDetect.addEventListener('click', async () => {
    if (uploadPreview && uploadPreview.src && uploadPreview.src !== window.location.href) {
      await window.detectEmotionFromImage(uploadPreview);
    }
  });
}

function drawUploadResult(img, detections, expressions) {
  if (!uploadPreviewWrap) return;
  const canvas = faceapi.createCanvasFromMedia(img);
  canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;border-radius:6px;pointer-events:none;';
  uploadPreviewWrap.style.position = 'relative';
  const old = uploadPreviewWrap.querySelector('canvas');
  if (old) uploadPreviewWrap.removeChild(old);
  uploadPreviewWrap.insertBefore(canvas, img.nextSibling);
  const dims    = faceapi.matchDimensions(canvas, img, true);
  const resized = faceapi.resizeResults(detections, dims);
  const ctx     = canvas.getContext('2d');
  resized.forEach(det => drawFaceBox(ctx, det.box, expressions));
}

// ── BOOT ──────────────────────────────────────────────────
init();
