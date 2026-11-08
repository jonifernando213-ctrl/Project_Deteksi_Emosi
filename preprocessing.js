/**
 * preprocessing.js — GLOBAL UPLOAD VERSION
 * Upload sekali → Data, Technic, Convolution, Morphology, Feature Detection
 * semua otomatis terisi tanpa perlu upload ulang.
 * Universitas Mikroskil — Tugas Pengolahan Citra Digital
 */

// ══════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ══════════════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// ══════════════════════════════════════════════════════════
//  GLOBAL STATE — satu gambar untuk semua tab
// ══════════════════════════════════════════════════════════
let globalImageData = null;   // ImageData object
let globalImgElement = null;  // HTMLImageElement

// ══════════════════════════════════════════════════════════
//  UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════
function loadImageToCanvas(file, canvas, maxW = 400, maxH = 300) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      let w = img.naturalWidth, h = img.naturalHeight;
      if (w > maxW || h > maxH) {
        const scale = Math.min(maxW / w, maxH / h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      if (canvas) {
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        const imageData = ctx.getImageData(0, 0, w, h);
        URL.revokeObjectURL(url);
        resolve({ canvas, ctx, w, h, imageData, img });
      } else {
        const tmp = document.createElement('canvas');
        tmp.width = w; tmp.height = h;
        const ctx = tmp.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        const imageData = ctx.getImageData(0, 0, w, h);
        URL.revokeObjectURL(url);
        resolve({ canvas: tmp, ctx, w, h, imageData, img });
      }
    };
    img.onerror = reject;
    img.src = url;
  });
}

function loadFileAsImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { resolve(img); /* keep url alive */ };
    img.onerror = reject;
    img.src = url;
  });
}

function imageDataFromImg(img, maxW = 400, maxH = 300) {
  let w = img.naturalWidth, h = img.naturalHeight;
  if (w > maxW || h > maxH) {
    const scale = Math.min(maxW / w, maxH / h);
    w = Math.round(w * scale); h = Math.round(h * scale);
  }
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const ctx = tmp.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  return { imageData: ctx.getImageData(0, 0, w, h), w, h, img };
}

function toGrayscaleArray(imageData) {
  const d = imageData.data;
  const gray = new Uint8Array(imageData.width * imageData.height);
  for (let i = 0; i < gray.length; i++) {
    const r = d[i*4], g = d[i*4+1], b = d[i*4+2];
    gray[i] = Math.round(0.299*r + 0.587*g + 0.114*b);
  }
  return gray;
}

function drawGrayToCanvas(gray, w, h, canvas) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  for (let i = 0; i < gray.length; i++) {
    imgData.data[i*4+0] = gray[i];
    imgData.data[i*4+1] = gray[i];
    imgData.data[i*4+2] = gray[i];
    imgData.data[i*4+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

function drawImageDataToCanvas(imgData, canvas) {
  canvas.width = imgData.width; canvas.height = imgData.height;
  canvas.getContext('2d').putImageData(imgData, 0, 0);
}

function clamp(v) { return Math.max(0, Math.min(255, Math.round(v))); }

// ══════════════════════════════════════════════════════════
//  GLOBAL UPLOAD HANDLER — inti perubahan
// ══════════════════════════════════════════════════════════
document.getElementById('globalImgInput').addEventListener('change', async function () {
  const file = this.files[0];
  if (!file) return;

  // Update banner UI
  const gubStatus   = document.getElementById('gubStatus');
  const gubThumb    = document.getElementById('gubThumb');
  const gubFilename = document.getElementById('gubFilename');
  const gubBadge    = document.getElementById('gubBadge');
  const gubHint     = document.getElementById('gubHint');

  gubFilename.textContent = '⏳ Memproses...';

  // Load image
  const img = await loadFileAsImage(file);
  const { imageData, w, h } = imageDataFromImg(img, 400, 300);

  // Store globally
  globalImageData   = imageData;
  globalImgElement  = img;

  // Banner feedback
  gubThumb.src = URL.createObjectURL(file);
  gubThumb.classList.add('visible');
  gubFilename.textContent = file.name + ' (' + w + '×' + h + ' px)';
  gubStatus.classList.add('loaded');
  gubBadge.classList.add('visible');
  gubHint.classList.add('visible');

  // Reset gambar 2 feature matching saat gambar baru diupload
  featImageData2 = null;
  const fi2 = document.getElementById('featImgInput2');
  if (fi2) fi2.value = '';
  const mc = document.getElementById('matchCount');
  if (mc) mc.textContent = '—';
  const fmc = document.getElementById('featMatchCanvas');
  if (fmc) { fmc.width = 0; fmc.height = 0; }

  // === Trigger all tabs automatically ===
  runDataTab(imageData, img, w, h);
  runTechnicTab(imageData, img, w, h);
  runConvolutionTab(imageData, w, h);
  runMorphologyTab(imageData, w, h);
  runFeatureTab(imageData, w, h);

  // ML Tab: kirim imageData ke ml.js
  if (typeof mlReceiveImage === 'function') {
    mlReceiveImage(imageData, w, h);
  }

  // Emotion tab: show preview + auto detect emosi
  const preview = document.getElementById('uploadPreview');
  const previewWrap = document.getElementById('uploadPreviewWrap');
  const hint = document.getElementById('emotionUploadHint');
  preview.src = img.src;
  if (previewWrap) previewWrap.classList.remove('hidden');
  if (hint) hint.style.display = 'none';

  // Auto-trigger deteksi emosi jika model sudah siap
  preview.onload = () => {
    if (typeof window.detectEmotionFromImage === 'function') {
      window.detectEmotionFromImage(preview);
    }
  };
  // Jika gambar sudah complete (cached)
  if (preview.complete && typeof window.detectEmotionFromImage === 'function') {
    window.detectEmotionFromImage(preview);
  }
});

// ══════════════════════════════════════════════════════════
//  TAB 2 — DATA
// ══════════════════════════════════════════════════════════
function runDataTab(imageData, img, w, h) {
  // Draw original
  const origCanvas = document.getElementById('dataOrigCanvas');
  origCanvas.width = w; origCanvas.height = h;
  origCanvas.getContext('2d').drawImage(img, 0, 0, w, h);

  // Info
  document.getElementById('dataWidth').textContent  = w + ' px';
  document.getElementById('dataHeight').textContent = h + ' px';
  document.getElementById('dataTotal').textContent  = (w * h).toLocaleString();

  // Grayscale
  const gray = toGrayscaleArray(imageData);
  drawGrayToCanvas(gray, w, h, document.getElementById('dataGrayCanvas'));

  // Pixel grid
  drawPixelGrid(imageData, w, document.getElementById('pixelGrid'));

  // Histograms
  drawHistogramRGB(imageData, document.getElementById('dataHistCanvas'));
  drawHistogramGray(gray, document.getElementById('dataGrayHistCanvas'));

  // K-Means clustering
  runKMeans(imageData, w, h);
}

function drawPixelGrid(imgData, w, container) {
  container.innerHTML = '';
  const d = imgData.data;
  for (let row = 0; row < 10; row++) {
    for (let col = 0; col < 10; col++) {
      const idx = (row * w + col) * 4;
      const r = d[idx], g = d[idx+1], b = d[idx+2];
      const bright = (r+g+b) / 3;
      const cell = document.createElement('div');
      cell.className = 'pixel-cell';
      cell.style.background = `rgb(${r},${g},${b})`;
      cell.style.color = bright > 128 ? '#000' : '#fff';
      cell.title = `R:${r} G:${g} B:${b}`;
      cell.textContent = Math.round(bright);
      container.appendChild(cell);
    }
  }
}

function drawHistogramRGB(imgData, canvas) {
  const d = imgData.data;
  const rH = new Array(256).fill(0), gH = new Array(256).fill(0), bH = new Array(256).fill(0);
  for (let i = 0; i < d.length; i += 4) { rH[d[i]]++; gH[d[i+1]]++; bH[d[i+2]]++; }
  drawHistCanvas(canvas, [
    { data: rH, color: 'rgba(255,80,80,0.7)' },
    { data: gH, color: 'rgba(80,220,80,0.7)' },
    { data: bH, color: 'rgba(80,120,255,0.7)' }
  ], 'Histogram RGB');
}

function drawHistogramGray(gray, canvas) {
  const h = new Array(256).fill(0);
  gray.forEach(v => h[v]++);
  drawHistCanvas(canvas, [{ data: h, color: 'rgba(200,200,200,0.8)' }], 'Histogram Grayscale');
}

function drawHistCanvas(canvas, series, title) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1b1c20'; ctx.fillRect(0, 0, W, H);
  const maxVal = Math.max(...series.flatMap(s => s.data));
  const bw = W / 256;
  series.forEach(({ data, color }) => {
    ctx.fillStyle = color;
    for (let i = 0; i < 256; i++) {
      const bh = (data[i] / maxVal) * (H - 20);
      ctx.fillRect(i * bw, H - bh - 10, Math.max(bw, 1), bh);
    }
  });
  ctx.fillStyle = '#8a8d95'; ctx.font = '10px Space Mono, monospace';
  ctx.fillText('0', 4, H-2); ctx.fillText('255', W-22, H-2);
  ctx.fillStyle = '#a3a6ad'; ctx.fillText(title, 4, 12);
}

// ══════════════════════════════════════════════════════════
//  TAB 3 — TECHNIC
// ══════════════════════════════════════════════════════════
let techImageData = null, techOrigImg = null;

function runTechnicTab(imageData, img, w, h) {
  techImageData = imageData;
  techOrigImg   = img;

  // Draw original
  const origCanvas = document.getElementById('techOrigCanvas');
  origCanvas.width = w; origCanvas.height = h;
  origCanvas.getContext('2d').drawImage(img, 0, 0, w, h);

  // Auto apply with defaults
  applyTechnic();
}

document.getElementById('techNoiseLevel').addEventListener('input', function () {
  document.getElementById('techNoiseLevelVal').textContent = this.value;
});

document.getElementById('btnApplyTech').addEventListener('click', () => {
  if (!techImageData) { alert('Upload gambar terlebih dahulu via banner di atas.'); return; }
  applyTechnic();
});

function applyTechnic() {
  if (!techImageData) return;
  const size = parseInt(document.getElementById('techSampleSize').value);
  const sampleCanvas = document.getElementById('techSampleCanvas');
  sampleCanvas.width = size; sampleCanvas.height = size;
  const sCtx = sampleCanvas.getContext('2d');
  sCtx.imageSmoothingEnabled = false;
  sCtx.drawImage(techOrigImg, 0, 0, size, size);

  const bits = parseInt(document.getElementById('techQuantBit').value);
  const levels = Math.pow(2, bits), step = 256 / levels;
  const gray = toGrayscaleArray(techImageData);
  const qGray = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++)
    qGray[i] = Math.round(Math.floor(gray[i] / step) * step + step / 2 - 1);
  drawGrayToCanvas(qGray, techImageData.width, techImageData.height, document.getElementById('techQuantCanvas'));

  const noiseType  = document.getElementById('techNoiseType').value;
  const noiseLevel = parseInt(document.getElementById('techNoiseLevel').value);
  const noisyData  = new ImageData(new Uint8ClampedArray(techImageData.data), techImageData.width, techImageData.height);
  applyNoise(noisyData, noiseType, noiseLevel);
  drawImageDataToCanvas(noisyData, document.getElementById('techNoiseCanvas'));
}

function applyNoise(imgData, type, level) {
  const d = imgData.data, n = d.length / 4;
  if (type === 'gaussian') {
    const sigma = level * 0.8;
    for (let i = 0; i < n; i++) {
      const noise = gaussianRandom(0, sigma);
      for (let c = 0; c < 3; c++) d[i*4+c] = clamp(d[i*4+c] + noise);
    }
  } else if (type === 'salt') {
    const prob = level / 1000;
    for (let i = 0; i < n; i++) {
      const r = Math.random();
      if (r < prob) { d[i*4]=d[i*4+1]=d[i*4+2]=255; }
      else if (r < prob * 2) { d[i*4]=d[i*4+1]=d[i*4+2]=0; }
    }
  }
}

function gaussianRandom(mean, std) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ══════════════════════════════════════════════════════════
//  KERNELS
// ══════════════════════════════════════════════════════════
const KERNELS = {
  identity:    { data: [[0,0,0],[0,1,0],[0,0,0]], name: 'Identity', op: 'Tidak ada perubahan' },
  blur3:       { data: [[1,1,1],[1,1,1],[1,1,1]], divisor: 9, name: 'Box Blur 3×3', op: 'Smoothing / rata-rata' },
  blur5:       { data: [[1,2,1],[2,4,2],[1,2,1]], divisor: 16, name: 'Gaussian 3×3 (approx)', op: 'Gaussian smoothing' },
  sharpen:     { data: [[0,-1,0],[-1,5,-1],[0,-1,0]], name: 'Sharpen', op: 'Perjelas tepi' },
  edgeSobel:   { data: [[-1,0,1],[-2,0,2],[-1,0,1]], name: 'Sobel X', op: 'Deteksi tepi horizontal' },
  edgeSobelY:  { data: [[-1,-2,-1],[0,0,0],[1,2,1]], name: 'Sobel Y', op: 'Deteksi tepi vertikal' },
  edgeLaplace: { data: [[0,1,0],[1,-4,1],[0,1,0]], name: 'Laplacian', op: 'Deteksi tepi omnidirectional' },
  emboss:      { data: [[-2,-1,0],[-1,1,1],[0,1,2]], name: 'Emboss', op: 'Efek timbul / reliefs' },
};

// ══════════════════════════════════════════════════════════
//  TAB 4 — CONVOLUTION
// ══════════════════════════════════════════════════════════
let convImageData = null;

function runConvolutionTab(imageData, w, h) {
  convImageData = imageData;
  const origCanvas = document.getElementById('convOrigCanvas');
  origCanvas.width = w; origCanvas.height = h;
  const origCtx = origCanvas.getContext('2d');
  const tmp = new ImageData(new Uint8ClampedArray(imageData.data), w, h);
  origCtx.putImageData(tmp, 0, 0);
  updateKernelDisplay();
  applyConvolution();
}

document.getElementById('convKernelSelect').addEventListener('change', () => {
  updateKernelDisplay();
  if (convImageData) applyConvolution();
});

function updateKernelDisplay() {
  const key = document.getElementById('convKernelSelect').value;
  const k   = KERNELS[key];
  document.getElementById('convKernelSize').textContent = k.data.length + ' × ' + k.data[0].length;
  document.getElementById('convOp').textContent = k.op;
  const container = document.getElementById('kernelDisplay');
  container.style.gridTemplateColumns = `repeat(${k.data[0].length}, 1fr)`;
  container.innerHTML = '';
  k.data.flat().forEach(v => {
    const cell = document.createElement('div');
    cell.className = 'kernel-cell ' + (v > 0 ? 'positive' : v < 0 ? 'negative' : 'zero');
    cell.textContent = v;
    container.appendChild(cell);
  });
}
updateKernelDisplay();

document.getElementById('btnApplyConv').addEventListener('click', () => {
  if (!convImageData) { alert('Upload gambar terlebih dahulu via banner di atas.'); return; }
  applyConvolution();
});

function applyConvolution() {
  if (!convImageData) return;
  const key  = document.getElementById('convKernelSelect').value;
  const kern = KERNELS[key];
  // Update canvas-desc kernel name
  const kNameDesc = document.getElementById('convKernelNameDesc');
  if (kNameDesc) kNameDesc.textContent = kern.name;
  const w = convImageData.width, h = convImageData.height;
  const gray = toGrayscaleArray(convImageData);
  drawGrayToCanvas(gray, w, h, document.getElementById('convGrayCanvas'));
  const result = convolve(gray, w, h, kern.data, kern.divisor || 1);
  drawGrayToCanvas(result, w, h, document.getElementById('convResultCanvas'));
  const sobelX = convolve(gray, w, h, KERNELS.edgeSobel.data, 1, true);
  const sobelY = convolve(gray, w, h, KERNELS.edgeSobelY.data, 1, true);
  const mag = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++)
    mag[i] = clamp(Math.sqrt(sobelX[i] ** 2 + sobelY[i] ** 2));
  drawGrayToCanvas(mag, w, h, document.getElementById('convMagCanvas'));
}

function convolve(gray, w, h, kernel, divisor = 1, signed = false) {
  const kh = kernel.length, kw = kernel[0].length, pad = Math.floor(kh / 2);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = 0; ky < kh; ky++) {
        for (let kx = 0; kx < kw; kx++) {
          const sy = y + ky - pad, sx = x + kx - pad;
          if (sy >= 0 && sy < h && sx >= 0 && sx < w)
            sum += gray[sy * w + sx] * kernel[ky][kx];
        }
      }
      if (divisor !== 1) sum /= divisor;
      out[y * w + x] = signed ? clamp(Math.abs(sum)) : clamp(sum);
    }
  }
  return out;
}

// ══════════════════════════════════════════════════════════
//  TAB 5 — MORPHOLOGY
// ══════════════════════════════════════════════════════════
let morphImageData = null;

function runMorphologyTab(imageData, w, h) {
  morphImageData = imageData;
  document.getElementById('morphThreshVal').textContent = document.getElementById('morphThresh').value;
  applyMorphology();
}

document.getElementById('morphThresh').addEventListener('input', function () {
  document.getElementById('morphThreshVal').textContent = this.value;
  if (morphImageData) applyMorphology();
});

document.getElementById('btnApplyMorph').addEventListener('click', () => {
  if (!morphImageData) { alert('Upload gambar terlebih dahulu via banner di atas.'); return; }
  applyMorphology();
});

function applyMorphology() {
  if (!morphImageData) return;
  const thresh     = parseInt(document.getElementById('morphThresh').value);
  const kernelSize = parseInt(document.getElementById('morphKernelSize').value);
  const w = morphImageData.width, h = morphImageData.height;

  // Update SE label in canvas-desc
  const seLabel = document.getElementById('morphSELabel');
  if (seLabel) seLabel.textContent = kernelSize + '×' + kernelSize;

  const gray = toGrayscaleArray(morphImageData);
  const bin  = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] > thresh ? 1 : 0;

  // 1. Binary
  drawBinaryToCanvas(bin, w, h, document.getElementById('morphBinCanvas'));

  const se      = makeStructuringElement(kernelSize);
  const eroded  = erode(bin, w, h, se);
  const dilated = dilate(bin, w, h, se);

  // 2. Erosi
  drawBinaryToCanvas(eroded, w, h, document.getElementById('morphEroCanvas'));

  // 3. Dilasi
  drawBinaryToCanvas(dilated, w, h, document.getElementById('morphDilCanvas'));

  // 4. Opening = dilasi(erosi) — canvas terpisah
  const opened = dilate(eroded, w, h, se);
  drawBinaryToCanvas(opened, w, h, document.getElementById('morphOpenCanvas'));

  // 5. Closing = erosi(dilasi) — canvas terpisah
  const closed = erode(dilated, w, h, se);
  drawBinaryToCanvas(closed, w, h, document.getElementById('morphCloseCanvas'));

  // 6. Morphological Gradient = dilasi - erosi (per piksel)
  const gradCanvas = document.getElementById('morphGradCanvas');
  gradCanvas.width = w; gradCanvas.height = h;
  const gradCtx = gradCanvas.getContext('2d');
  const gradData = gradCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const val = (dilated[i] - eroded[i]) * 255; // 0 or 255
    // Render sebagai warna ungu agar mudah dibedakan
    gradData.data[i*4+0] = Math.round(val * 0.78);  // R
    gradData.data[i*4+1] = Math.round(val * 0.42);  // G
    gradData.data[i*4+2] = Math.round(val * 1.0);   // B
    gradData.data[i*4+3] = 255;
  }
  gradCtx.putImageData(gradData, 0, 0);
}

function makeStructuringElement(size) {
  const se = [];
  for (let r = 0; r < size; r++) { const row = []; for (let c = 0; c < size; c++) row.push(1); se.push(row); }
  return se;
}

function erode(bin, w, h, se) {
  const pad = Math.floor(se.length / 2), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let fits = true;
      outer: for (let ky = 0; ky < se.length; ky++) {
        for (let kx = 0; kx < se[0].length; kx++) {
          if (se[ky][kx] === 0) continue;
          const sy = y + ky - pad, sx = x + kx - pad;
          if (sy < 0 || sy >= h || sx < 0 || sx >= w || bin[sy * w + sx] === 0) { fits = false; break outer; }
        }
      }
      out[y * w + x] = fits ? 1 : 0;
    }
  }
  return out;
}

function dilate(bin, w, h, se) {
  const pad = Math.floor(se.length / 2), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x] === 0) continue;
      for (let ky = 0; ky < se.length; ky++) {
        for (let kx = 0; kx < se[0].length; kx++) {
          if (se[ky][kx] === 0) continue;
          const sy = y + ky - pad, sx = x + kx - pad;
          if (sy >= 0 && sy < h && sx >= 0 && sx < w) out[sy * w + sx] = 1;
        }
      }
    }
  }
  return out;
}

function drawBinaryToCanvas(bin, w, h, canvas) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(w, h);
  for (let i = 0; i < bin.length; i++) {
    const v = bin[i] * 255;
    imgData.data[i*4+0] = v; imgData.data[i*4+1] = v;
    imgData.data[i*4+2] = v; imgData.data[i*4+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

// ══════════════════════════════════════════════════════════
//  TAB 6 — FEATURE DETECTION & MATCHING
// ══════════════════════════════════════════════════════════
let featImageData1 = null, featImageData2 = null;

function runFeatureTab(imageData, w, h) {
  featImageData1 = imageData;
  applyFeatureDetection();
}

document.getElementById('featThresh').addEventListener('input', function () {
  document.getElementById('featThreshVal').textContent = this.value;
});

document.getElementById('featMethod').addEventListener('change', () => {
  if (featImageData1) applyFeatureDetection();
});

document.getElementById('btnApplyFeat').addEventListener('click', () => {
  if (!featImageData1) { alert('Upload gambar terlebih dahulu via banner di atas.'); return; }
  applyFeatureDetection();
});

function applyFeatureDetection() {
  if (!featImageData1) return;
  const method = document.getElementById('featMethod').value;
  const thresh = parseInt(document.getElementById('featThresh').value);
  const w = featImageData1.width, h = featImageData1.height;
  const gray = toGrayscaleArray(featImageData1);

  document.getElementById('featMethodLabel').textContent =
    method === 'harris' ? 'Harris Corner' : method === 'blob' ? 'Blob Detector' : 'Canny Edge';

  // Update canvas-desc info
  const threshDescEl = document.getElementById('featThreshDesc');
  if (threshDescEl) threshDescEl.textContent = 'sensitivitas ' + thresh + '%';

  if (method === 'harris') {
    const corners = harrisCorners(gray, w, h, thresh);
    drawFeaturesOnCanvas(featImageData1, corners, document.getElementById('featResultCanvas'), '#c99b52', 'circle');
    drawGrayWithFeatures(gray, w, h, corners, document.getElementById('featGrayCanvas'), '#c99b52');
    document.getElementById('featCount').textContent = corners.length;
    const fcd = document.getElementById('featCountDesc'); if (fcd) fcd.textContent = corners.length;
  } else if (method === 'blob') {
    const blobs = simpleBlobs(gray, w, h, thresh);
    drawFeaturesOnCanvas(featImageData1, blobs, document.getElementById('featResultCanvas'), '#83a0ac', 'cross');
    drawGrayWithFeatures(gray, w, h, blobs, document.getElementById('featGrayCanvas'), '#83a0ac');
    document.getElementById('featCount').textContent = blobs.length;
    const fcd2 = document.getElementById('featCountDesc'); if (fcd2) fcd2.textContent = blobs.length;
  } else {
    const sobelX = convolve(gray, w, h, KERNELS.edgeSobel.data, 1, true);
    const sobelY = convolve(gray, w, h, KERNELS.edgeSobelY.data, 1, true);
    const mag = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) mag[i] = clamp(Math.sqrt(sobelX[i]**2 + sobelY[i]**2));
    const edgeThresh = (thresh / 100) * 200;
    const edgePoints = [];
    for (let i = 0; i < mag.length; i++)
      if (mag[i] > edgeThresh) edgePoints.push({ x: i % w, y: Math.floor(i / w) });
    const sampled = edgePoints.filter((_, i) => i % 5 === 0);
    drawFeaturesOnCanvas(featImageData1, sampled, document.getElementById('featResultCanvas'), '#7fae91', 'dot');
    drawGrayWithFeatures(gray, w, h, sampled, document.getElementById('featGrayCanvas'), '#7fae91');
    document.getElementById('featCount').textContent = edgePoints.length + ' edge px';
  }
}

document.getElementById('featImgInput2').addEventListener('change', async function () {
  const file = this.files[0]; if (!file) return;
  const tmpCanvas = document.createElement('canvas');
  const res = await loadImageToCanvas(file, tmpCanvas, 400, 300);
  featImageData2 = res.imageData;
});

document.getElementById('btnMatch').addEventListener('click', () => {
  if (!featImageData1) { alert('Upload Gambar 1 terlebih dahulu dan jalankan Deteksi Fitur.'); return; }
  if (!featImageData2) { alert('Upload Gambar 2 terlebih dahulu.'); return; }
  runMatching();
});

function runMatching() {
  const method = document.getElementById('featMethod').value;
  const thresh = parseInt(document.getElementById('featThresh').value);
  const w1 = featImageData1.width, h1 = featImageData1.height;
  const w2 = featImageData2.width, h2 = featImageData2.height;
  const gray1 = toGrayscaleArray(featImageData1), gray2 = toGrayscaleArray(featImageData2);
  const pts1 = method === 'harris' ? harrisCorners(gray1, w1, h1, thresh) : method === 'blob' ? simpleBlobs(gray1, w1, h1, thresh) : getEdgePoints(gray1, w1, h1, thresh);
  const pts2 = method === 'harris' ? harrisCorners(gray2, w2, h2, thresh) : method === 'blob' ? simpleBlobs(gray2, w2, h2, thresh) : getEdgePoints(gray2, w2, h2, thresh);
  const half = 3;
  function getPatch(gray, w, h, cx, cy) {
    const patch = [];
    for (let dy = -half; dy <= half; dy++) for (let dx = -half; dx <= half; dx++) {
      const y = Math.max(0, Math.min(h-1, cy+dy)), x = Math.max(0, Math.min(w-1, cx+dx));
      patch.push(gray[y*w+x]);
    }
    return patch;
  }
  function patchSAD(p1, p2) { let s = 0; for (let i = 0; i < p1.length; i++) s += Math.abs(p1[i]-p2[i]); return s; }
  const limit1 = pts1.slice(0, 80), limit2 = pts2.slice(0, 80);
  const matches = [];
  limit1.forEach(pt1 => {
    const patch1 = getPatch(gray1, w1, h1, pt1.x, pt1.y);
    let bestDist = Infinity, bestPt2 = null;
    limit2.forEach(pt2 => {
      const dist = patchSAD(patch1, getPatch(gray2, w2, h2, pt2.x, pt2.y));
      if (dist < bestDist) { bestDist = dist; bestPt2 = pt2; }
    });
    if (bestDist < 1200) matches.push({ pt1, pt2: bestPt2, dist: bestDist });
  });
  matches.sort((a,b) => a.dist - b.dist);
  const topMatches = matches.slice(0, 40);
  document.getElementById('matchCount').textContent = topMatches.length + ' dari ' + matches.length;
  const matchCanvas = document.getElementById('featMatchCanvas');
  const totalW = w1 + w2 + 10, totalH = Math.max(h1, h2);
  matchCanvas.width = totalW; matchCanvas.height = totalH;
  const mCtx = matchCanvas.getContext('2d');
  mCtx.fillStyle = '#1b1c20'; mCtx.fillRect(0, 0, totalW, totalH);
  const tmp1 = document.createElement('canvas'); drawImageDataToCanvas(featImageData1, tmp1); mCtx.drawImage(tmp1, 0, 0);
  const tmp2 = document.createElement('canvas'); drawImageDataToCanvas(featImageData2, tmp2); mCtx.drawImage(tmp2, w1+10, 0);
  mCtx.strokeStyle = '#3a3b42'; mCtx.lineWidth = 2;
  mCtx.beginPath(); mCtx.moveTo(w1+5, 0); mCtx.lineTo(w1+5, totalH); mCtx.stroke();
  const colors = ['#c99b52','#83a0ac','#7fae91','#7fa3bd','#a396b3','#c99a63'];
  topMatches.forEach((m, i) => {
    const color = colors[i % colors.length];
    mCtx.strokeStyle = color + '99'; mCtx.lineWidth = 1;
    mCtx.beginPath(); mCtx.moveTo(m.pt1.x, m.pt1.y); mCtx.lineTo(w1+10+m.pt2.x, m.pt2.y); mCtx.stroke();
    mCtx.fillStyle = color;
    mCtx.beginPath(); mCtx.arc(m.pt1.x, m.pt1.y, 3, 0, Math.PI*2); mCtx.fill();
    mCtx.beginPath(); mCtx.arc(w1+10+m.pt2.x, m.pt2.y, 3, 0, Math.PI*2); mCtx.fill();
  });
  mCtx.fillStyle = 'rgba(0,0,0,0.6)'; mCtx.fillRect(0, 0, 80, 18); mCtx.fillRect(w1+10, 0, 80, 18);
  mCtx.fillStyle = '#a3a6ad'; mCtx.font = '10px Space Mono, monospace';
  mCtx.fillText('Gambar 1', 4, 13); mCtx.fillText('Gambar 2', w1+14, 13);
}

// ── Harris Corner ─────────────────────────────────────────
function harrisCorners(gray, w, h, threshPct) {
  const Ix = convolve(gray, w, h, KERNELS.edgeSobel.data, 1, false);
  const Iy = convolve(gray, w, h, KERNELS.edgeSobelY.data, 1, false);
  const k = 0.04, pad = 1;
  const response = new Float32Array(w * h);
  for (let y = pad; y < h-pad; y++) {
    for (let x = pad; x < w-pad; x++) {
      let Mxx=0,Mxy=0,Myy=0;
      for (let wy=-pad;wy<=pad;wy++) for (let wx=-pad;wx<=pad;wx++) {
        const idx=(y+wy)*w+(x+wx), ix=Ix[idx]-128, iy=Iy[idx]-128;
        Mxx+=ix*ix; Mxy+=ix*iy; Myy+=iy*iy;
      }
      const det=Mxx*Myy-Mxy*Mxy, trace=Mxx+Myy;
      response[y*w+x]=det-k*trace*trace;
    }
  }
  let maxR = -Infinity;
  for (let i = 0; i < response.length; i++) { if (response[i] > maxR) maxR = response[i]; }
  const thresh = (threshPct/100)*maxR*0.3;
  const corners=[], minDist=8, marked=new Uint8Array(w*h);
  for (let y=pad;y<h-pad;y++) for (let x=pad;x<w-pad;x++) {
    const r=response[y*w+x];
    if (r>thresh && !marked[y*w+x]) {
      let isMax=true;
      for (let ny=-minDist;ny<=minDist&&isMax;ny++) for (let nx=-minDist;nx<=minDist&&isMax;nx++) {
        const ny2=y+ny,nx2=x+nx;
        if (ny2>=0&&ny2<h&&nx2>=0&&nx2<w&&response[ny2*w+nx2]>r) isMax=false;
      }
      if (isMax) {
        corners.push({x,y,score:r});
        for (let my=-minDist;my<=minDist;my++) for (let mx=-minDist;mx<=minDist;mx++) {
          const my2=y+my,mx2=x+mx;
          if (my2>=0&&my2<h&&mx2>=0&&mx2<w) marked[my2*w+mx2]=1;
        }
      }
    }
  }
  return corners.sort((a,b)=>b.score-a.score).slice(0,200);
}

// ── Blob Detector ─────────────────────────────────────────
function simpleBlobs(gray, w, h, threshPct) {
  const blurred=convolve(gray,w,h,[[1,2,1],[2,4,2],[1,2,1]],16);
  const diff=new Float32Array(gray.length);
  for (let i=0;i<gray.length;i++) diff[i]=Math.abs(gray[i]-blurred[i]);
  let maxD = -Infinity;
  for (let i = 0; i < diff.length; i++) { if (diff[i] > maxD) maxD = diff[i]; }
  const thresh=(threshPct/100)*maxD*0.5;
  const blobs=[], minDist=12, visited=new Uint8Array(gray.length);
  for (let y=2;y<h-2;y++) for (let x=2;x<w-2;x++) {
    const idx=y*w+x;
    if (diff[idx]>thresh&&!visited[idx]) {
      blobs.push({x,y,score:diff[idx]});
      for (let dy=-minDist;dy<=minDist;dy++) for (let dx=-minDist;dx<=minDist;dx++) {
        const ny=y+dy,nx=x+dx;
        if (ny>=0&&ny<h&&nx>=0&&nx<w) visited[ny*w+nx]=1;
      }
    }
  }
  return blobs.sort((a,b)=>b.score-a.score).slice(0,150);
}

function getEdgePoints(gray, w, h, threshPct) {
  const sx=convolve(gray,w,h,KERNELS.edgeSobel.data,1,true);
  const sy=convolve(gray,w,h,KERNELS.edgeSobelY.data,1,true);
  const mag=new Uint8Array(gray.length);
  for (let i=0;i<gray.length;i++) mag[i]=clamp(Math.sqrt(sx[i]**2+sy[i]**2));
  const t=(threshPct/100)*200, pts=[];
  for (let i=0;i<mag.length;i+=4) if (mag[i]>t) pts.push({x:i%w,y:Math.floor(i/w),score:mag[i]});
  return pts.slice(0,200);
}

// ── Draw helpers ──────────────────────────────────────────
function drawFeaturesOnCanvas(imgData, points, canvas, color, shape='circle') {
  drawImageDataToCanvas(imgData, canvas);
  const ctx=canvas.getContext('2d');
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=1.5;
  points.forEach(({x,y}) => {
    if (shape==='circle') { ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.stroke(); }
    else if (shape==='cross') { ctx.beginPath(); ctx.moveTo(x-5,y); ctx.lineTo(x+5,y); ctx.moveTo(x,y-5); ctx.lineTo(x,y+5); ctx.stroke(); }
    else { ctx.beginPath(); ctx.arc(x,y,1.5,0,Math.PI*2); ctx.fill(); }
  });
}

function drawGrayWithFeatures(gray, w, h, points, canvas, color) {
  drawGrayToCanvas(gray, w, h, canvas);
  const ctx=canvas.getContext('2d');
  ctx.strokeStyle=color; ctx.lineWidth=1.5;
  points.forEach(({x,y}) => { ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.stroke(); });
}

// ══════════════════════════════════════════════════════════
//  K-MEANS CLUSTERING (Unsupervised Learning)
// ══════════════════════════════════════════════════════════
function runKMeans(imageData, w, h) {
  const kCanvas = document.getElementById('dataKmeansCanvas');
  const kInfo   = document.getElementById('kmeansInfo');
  if (!kCanvas) return;

  const k = parseInt(document.getElementById('kmeansK')?.value || 4);
  const d = imageData.data;

  // Ambil sampel piksel (maksimal 2000 untuk kecepatan)
  const step = Math.max(1, Math.floor((w * h) / 2000));
  const pixels = [];
  for (let i = 0; i < w * h; i += step) {
    pixels.push([d[i*4], d[i*4+1], d[i*4+2]]);
  }

  // Inisialisasi centroid secara acak dari piksel yang ada
  let centroids = [];
  const used = new Set();
  while (centroids.length < k) {
    const idx = Math.floor(Math.random() * pixels.length);
    if (!used.has(idx)) { used.add(idx); centroids.push([...pixels[idx]]); }
  }

  // Iterasi K-Means (maksimal 15 iterasi)
  let labels = new Array(pixels.length).fill(0);
  for (let iter = 0; iter < 15; iter++) {
    // Assignment step
    let changed = false;
    for (let i = 0; i < pixels.length; i++) {
      let bestDist = Infinity, bestK = 0;
      for (let ki = 0; ki < k; ki++) {
        const dr = pixels[i][0]-centroids[ki][0];
        const dg = pixels[i][1]-centroids[ki][1];
        const db = pixels[i][2]-centroids[ki][2];
        const dist = dr*dr + dg*dg + db*db;
        if (dist < bestDist) { bestDist = dist; bestK = ki; }
      }
      if (labels[i] !== bestK) { labels[i] = bestK; changed = true; }
    }
    if (!changed) break;

    // Update step
    const sums = Array.from({length: k}, () => [0,0,0,0]);
    for (let i = 0; i < pixels.length; i++) {
      sums[labels[i]][0] += pixels[i][0];
      sums[labels[i]][1] += pixels[i][1];
      sums[labels[i]][2] += pixels[i][2];
      sums[labels[i]][3]++;
    }
    for (let ki = 0; ki < k; ki++) {
      if (sums[ki][3] > 0) {
        centroids[ki] = [
          Math.round(sums[ki][0]/sums[ki][3]),
          Math.round(sums[ki][1]/sums[ki][3]),
          Math.round(sums[ki][2]/sums[ki][3])
        ];
      }
    }
  }

  // Render hasil: setiap piksel diganti warna centroid-nya
  kCanvas.width = w; kCanvas.height = h;
  const ctx = kCanvas.getContext('2d');
  const outData = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const si = Math.floor(i / step);
    const ki = labels[Math.min(si, labels.length-1)];
    outData.data[i*4+0] = centroids[ki][0];
    outData.data[i*4+1] = centroids[ki][1];
    outData.data[i*4+2] = centroids[ki][2];
    outData.data[i*4+3] = 255;
  }
  ctx.putImageData(outData, 0, 0);

  // Tampilkan info centroid
  if (kInfo) {
    kInfo.innerHTML = centroids.map((c, i) =>
      `<div class="info-row">
        <span style="display:flex;align-items:center;gap:6px;">
          <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:rgb(${c[0]},${c[1]},${c[2]});border:1px solid #7d818a"></span>
          Cluster ${i+1}
        </span>
        <span style="font-size:0.7rem;color:#8a8d95">rgb(${c[0]},${c[1]},${c[2]})</span>
      </div>`
    ).join('');
  }
}

// Listener untuk slider K-Means
document.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('kmeansK');
  const valEl  = document.getElementById('kmeansKVal');
  const btnRun = document.getElementById('btnRunKmeans');
  if (slider && valEl) {
    slider.addEventListener('input', () => { valEl.textContent = slider.value; });
  }
  if (btnRun) {
    btnRun.addEventListener('click', () => {
      if (!globalImageData) { alert('Upload gambar terlebih dahulu.'); return; }
      runKMeans(globalImageData, globalImageData.width, globalImageData.height);
    });
  }
});
