/**
 * ml.js — Tab Machine Learning (Revisi Tugas 2)
 * Universitas Mikroskil — IF2112 Computer Vision
 * Berisi: Normalisasi, Pooling, KNN, Evaluasi, Confusion Matrix, Export/Import CSV
 */

// ══════════════════════════════════════════════════════════
//  KONSTANTA
// ══════════════════════════════════════════════════════════
const ML_EMOTIONS = ['happy','sad','angry','fearful','disgusted','surprised','neutral'];
const ML_EMOTION_LABELS = {
  happy:'Happy', sad:'Sad', angry:'Angry',
  fearful:'Fear', disgusted:'Disgust', surprised:'Surprised', neutral:'Neutral'
};
const ML_COLORS = {
  happy:'#c99b52', sad:'#7fa3bd', angry:'#c2717a',
  fearful:'#a396b3', disgusted:'#7fae91', surprised:'#c99a63', neutral:'#9ea3ad'
};

// ══════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════
let mlDataset   = [];  // [{features:[], label:''}]
let mlImageData = null;
let mlW = 0, mlH = 0;
let lastEvalResult = null;

// ══════════════════════════════════════════════════════════
//  TERIMA GAMBAR DARI GLOBAL UPLOAD
// ══════════════════════════════════════════════════════════
function mlReceiveImage(imageData, w, h) {
  mlImageData = imageData;
  mlW = w; mlH = h;
  runNormalization(imageData, w, h);
  runPooling(imageData, w, h);
  runUnpooling(imageData, w, h);
  runRegularization(imageData, w, h);
  updateDatasetTable();
}

// ══════════════════════════════════════════════════════════
//  STEP 1 — NORMALISASI (Min-Max)
// ══════════════════════════════════════════════════════════
function runNormalization(imageData, w, h) {
  const normCanvas = document.getElementById('mlNormCanvas');
  const normInfo   = document.getElementById('mlNormInfo');
  if (!normCanvas) return;

  const d = imageData.data;
  normCanvas.width = w; normCanvas.height = h;
  const ctx = normCanvas.getContext('2d');
  const outData = ctx.createImageData(w, h);
  const grayNorm = new Float32Array(w * h);

  let minVal = 1, maxVal = 0, sumVal = 0;
  for (let i = 0; i < w * h; i++) {
    const gn = (0.299*d[i*4] + 0.587*d[i*4+1] + 0.114*d[i*4+2]) / 255;
    grayNorm[i] = gn;
    if (gn < minVal) minVal = gn;
    if (gn > maxVal) maxVal = gn;
    sumVal += gn;
  }
  const range = maxVal - minVal || 1;
  for (let i = 0; i < w * h; i++) {
    const v = Math.round(((grayNorm[i] - minVal) / range) * 255);
    outData.data[i*4] = outData.data[i*4+1] = outData.data[i*4+2] = v;
    outData.data[i*4+3] = 255;
  }
  ctx.putImageData(outData, 0, 0);

  if (normInfo) {
    const mean = (sumVal / (w*h)).toFixed(4);
    normInfo.innerHTML = `
      <div class="info-row"><span>Range sebelum</span><span>0 – 255</span></div>
      <div class="info-row"><span>Range sesudah</span><span>0.0 – 1.0</span></div>
      <div class="info-row"><span>Min terdeteksi</span><span>${minVal.toFixed(4)}</span></div>
      <div class="info-row"><span>Max terdeteksi</span><span>${maxVal.toFixed(4)}</span></div>
      <div class="info-row"><span>Mean</span><span>${mean}</span></div>
      <div class="info-row"><span>Metode</span><span>Min-Max Normalization</span></div>`;
  }
}

// ══════════════════════════════════════════════════════════
//  STEP 2 — POOLING
// ══════════════════════════════════════════════════════════
function runPooling(imageData, w, h) {
  const pSize = parseInt(document.getElementById('mlPoolSize')?.value || 2);
  const gray  = toGrayscaleArrayML(imageData);
  _doMaxPool(gray, w, h, pSize);
  _doAvgPool(gray, w, h, pSize);
  const outW = Math.floor(w/pSize), outH = Math.floor(h/pSize);
  const infoEl = document.getElementById('mlPoolInfo');
  if (infoEl) {
    const reduc = (100 - (outW*outH)/(w*h)*100).toFixed(1);
    infoEl.innerHTML = `
      <div class="info-row"><span>Input</span><span>${w}×${h} = ${(w*h).toLocaleString()} piksel</span></div>
      <div class="info-row"><span>Pool size</span><span>${pSize}×${pSize}</span></div>
      <div class="info-row"><span>Output</span><span>${outW}×${outH} = ${(outW*outH).toLocaleString()} piksel</span></div>
      <div class="info-row"><span>Reduksi data</span><span>${reduc}%</span></div>`;
  }
}

function _doMaxPool(gray, w, h, pSize) {
  const canvas = document.getElementById('mlMaxPoolCanvas');
  if (!canvas) return;
  const outW = Math.floor(w/pSize), outH = Math.floor(h/pSize);
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(outW, outH);
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) {
    let maxV = 0;
    for (let py = 0; py < pSize; py++) for (let px = 0; px < pSize; px++) {
      const sy = y*pSize+py, sx = x*pSize+px;
      if (sy < h && sx < w) { const v = gray[sy*w+sx]; if (v > maxV) maxV = v; }
    }
    const i=(y*outW+x)*4;
    out.data[i]=out.data[i+1]=out.data[i+2]=maxV; out.data[i+3]=255;
  }
  ctx.putImageData(out, 0, 0);
}

function _doAvgPool(gray, w, h, pSize) {
  const canvas = document.getElementById('mlAvgPoolCanvas');
  if (!canvas) return;
  const outW = Math.floor(w/pSize), outH = Math.floor(h/pSize);
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(outW, outH);
  for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) {
    let sum = 0, cnt = 0;
    for (let py = 0; py < pSize; py++) for (let px = 0; px < pSize; px++) {
      const sy = y*pSize+py, sx = x*pSize+px;
      if (sy < h && sx < w) { sum += gray[sy*w+sx]; cnt++; }
    }
    const avg = cnt > 0 ? Math.round(sum/cnt) : 0;
    const i=(y*outW+x)*4;
    out.data[i]=out.data[i+1]=out.data[i+2]=avg; out.data[i+3]=255;
  }
  ctx.putImageData(out, 0, 0);
}

// ══════════════════════════════════════════════════════════
//  STEP 2B — UNPOOLING (Nearest Neighbor Unpooling)
// ══════════════════════════════════════════════════════════
function runUnpooling(imageData, w, h) {
  const pSize = parseInt(document.getElementById('mlPoolSize')?.value || 2);
  const gray  = toGrayscaleArrayML(imageData);

  // 1. Lakukan max pooling dulu untuk dapatkan pooled map
  const outW = Math.floor(w / pSize);
  const outH = Math.floor(h / pSize);
  const pooled    = new Uint8Array(outW * outH);
  const switchMap = new Int32Array(outW * outH); // menyimpan index pixel max

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      let maxV = 0, maxIdx = 0;
      for (let py = 0; py < pSize; py++) {
        for (let px = 0; px < pSize; px++) {
          const sy = y*pSize+py, sx = x*pSize+px;
          if (sy < h && sx < w) {
            const v = gray[sy*w+sx];
            if (v > maxV) { maxV = v; maxIdx = sy*w+sx; }
          }
        }
      }
      pooled[y*outW+x]    = maxV;
      switchMap[y*outW+x] = maxIdx;
    }
  }

  // 2. Unpooling: kembalikan ke ukuran semula
  // Metode: letakkan nilai max kembali ke posisi aslinya (switch unpooling)
  // Posisi lain = 0 (sparse unpooling)
  const unpooled = new Uint8Array(w * h); // semua 0
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const origIdx = switchMap[y*outW+x];
      unpooled[origIdx] = pooled[y*outW+x];
    }
  }

  // 3. Gambar ke canvas — pooled (kecil)
  const poolCanvas = document.getElementById('mlPooledSmallCanvas');
  if (poolCanvas) {
    poolCanvas.width = outW; poolCanvas.height = outH;
    const ctx = poolCanvas.getContext('2d');
    const out = ctx.createImageData(outW, outH);
    for (let i = 0; i < outW*outH; i++) {
      out.data[i*4]=out.data[i*4+1]=out.data[i*4+2]=pooled[i];
      out.data[i*4+3]=255;
    }
    ctx.putImageData(out, 0, 0);
  }

  // 4. Gambar ke canvas — unpooled (kembali ke ukuran semula)
  const unpoolCanvas = document.getElementById('mlUnpoolCanvas');
  if (unpoolCanvas) {
    unpoolCanvas.width = w; unpoolCanvas.height = h;
    const ctx = unpoolCanvas.getContext('2d');
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < w*h; i++) {
      // Warna: piksel yang di-restore = putih terang, sisanya biru gelap
      const v = unpooled[i];
      if (v > 0) {
        out.data[i*4+0] = v;
        out.data[i*4+1] = v;
        out.data[i*4+2] = v;
      } else {
        out.data[i*4+0] = 10;
        out.data[i*4+1] = 10;
        out.data[i*4+2] = 30;
      }
      out.data[i*4+3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }

  // 5. Info
  const infoEl = document.getElementById('mlUnpoolInfo');
  if (infoEl) {
    const nonZero = unpooled.filter(v=>v>0).length;
    const sparsity = ((1 - nonZero/(w*h))*100).toFixed(1);
    infoEl.innerHTML = `
      <div class="info-row"><span>Ukuran setelah pooling</span><span>${outW}×${outH}</span></div>
      <div class="info-row"><span>Ukuran setelah unpooling</span><span>${w}×${h}</span></div>
      <div class="info-row"><span>Metode</span><span>Switch/Max Unpooling</span></div>
      <div class="info-row"><span>Piksel aktif (putih)</span><span>${nonZero.toLocaleString()}</span></div>
      <div class="info-row"><span>Sparsity (piksel nol)</span><span>${sparsity}%</span></div>
      <div class="info-row"><span>Penjelasan</span><span>Nilai max dikembalikan ke posisi asli, sisanya 0</span></div>`;
  }
}

// ══════════════════════════════════════════════════════════
//  STEP 2C — REGULARIZATION (L1, L2, Dropout simulasi)
// ══════════════════════════════════════════════════════════
function runRegularization(imageData, w, h) {
  const method   = document.getElementById('mlRegMethod')?.value || 'l2';
  const lambda   = parseFloat(document.getElementById('mlRegLambda')?.value || 0.01);
  const dropRate = parseFloat(document.getElementById('mlDropRate')?.value || 0.3);

  const gray = toGrayscaleArrayML(imageData);

  // Normalisasi bobot = nilai piksel / 255 (anggap sebagai weight vector)
  const weights = Array.from(gray).map(v => v / 255);

  let regWeights, penalty = 0;

  if (method === 'l1') {
    // L1 Regularization: kurangi setiap bobot dengan lambda * sign(w)
    regWeights = weights.map(w => {
      const shrink = Math.max(0, Math.abs(w) - lambda);
      const result = w >= 0 ? shrink : -shrink;
      penalty += Math.abs(w);
      return result;
    });
    penalty *= lambda;
  } else if (method === 'l2') {
    // L2 Regularization: kurangi setiap bobot dengan lambda * w (weight decay)
    regWeights = weights.map(w => {
      const result = w * (1 - lambda);
      penalty += w * w;
      return result;
    });
    penalty = lambda * 0.5 * penalty;
  } else {
    // Dropout: set beberapa bobot ke 0 secara acak
    regWeights = weights.map(w => Math.random() > dropRate ? w / (1 - dropRate) : 0);
    const zeroed = regWeights.filter(v => v === 0).length;
    penalty = zeroed / regWeights.length;
  }

  // Gambar hasil ke canvas (visualisasi sebagai grayscale)
  const canvas = document.getElementById('mlRegCanvas');
  if (canvas) {
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < w*h; i++) {
      const v = Math.min(255, Math.max(0, Math.round(regWeights[i] * 255)));
      out.data[i*4] = out.data[i*4+1] = out.data[i*4+2] = v;
      out.data[i*4+3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }

  // Gambar perbandingan histogram sebelum vs sesudah
  _drawRegHistogram(weights, regWeights, w, h);

  // Info
  const infoEl = document.getElementById('mlRegInfo');
  if (infoEl) {
    const avgBefore = (weights.reduce((s,v)=>s+v,0)/weights.length).toFixed(4);
    const avgAfter  = (regWeights.reduce((s,v)=>s+v,0)/regWeights.length).toFixed(4);
    const lambdaVal = document.getElementById('mlRegLambdaVal');
    if (lambdaVal) lambdaVal.textContent = lambda.toFixed(3);
    const dropVal = document.getElementById('mlDropRateVal');
    if (dropVal) dropVal.textContent = (dropRate*100).toFixed(0)+'%';

    const methodName = method==='l1'?'L1 (Lasso)':method==='l2'?'L2 (Ridge / Weight Decay)':'Dropout';
    infoEl.innerHTML = `
      <div class="info-row"><span>Metode</span><span style="color:#c99b52">${methodName}</span></div>
      <div class="info-row"><span>Mean bobot sebelum</span><span>${avgBefore}</span></div>
      <div class="info-row"><span>Mean bobot sesudah</span><span>${avgAfter}</span></div>
      ${method==='dropout'
        ? `<div class="info-row"><span>Drop rate</span><span>${(dropRate*100).toFixed(0)}%</span></div>
           <div class="info-row"><span>Piksel di-nolkan</span><span>${(penalty*100).toFixed(1)}%</span></div>`
        : `<div class="info-row"><span>Lambda (λ)</span><span>${lambda.toFixed(3)}</span></div>
           <div class="info-row"><span>Penalty term</span><span>${penalty.toFixed(6)}</span></div>`
      }
      <div class="info-row"><span>Tujuan</span><span>${
        method==='l1'?'Sparsity — dorong bobot ke nol':
        method==='l2'?'Shrinkage — kurangi besar bobot':
        'Robustness — cegah overfitting'
      }</span></div>`;
  }
}

function _drawRegHistogram(before, after, w, h) {
  const canvas = document.getElementById('mlRegHistCanvas');
  if (!canvas) return;
  canvas.width = 400; canvas.height = 160;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#17181c';
  ctx.fillRect(0, 0, 400, 160);

  const bins = 64;
  const histB = new Array(bins).fill(0);
  const histA = new Array(bins).fill(0);
  before.forEach(v => histB[Math.min(bins-1, Math.floor(v*bins))]++);
  after.forEach(v  => histA[Math.min(bins-1, Math.floor(Math.max(0,v)*bins))]++);

  const maxB = Math.max(...histB) || 1;
  const maxA = Math.max(...histA) || 1;
  const maxAll = Math.max(maxB, maxA);
  const bw = 400 / bins;

  for (let i = 0; i < bins; i++) {
    const hB = Math.round((histB[i] / maxAll) * 130);
    const hA = Math.round((histA[i] / maxAll) * 130);
    // Sebelum — biru
    ctx.fillStyle = 'rgba(108,200,255,0.5)';
    ctx.fillRect(i*bw, 155-hB, bw-1, hB);
    // Sesudah — kuning
    ctx.fillStyle = 'rgba(201,155,82,0.8)';
    ctx.fillRect(i*bw, 155-hA, bw-1, hA);
  }

  // Legend
  ctx.font = '10px monospace';
  ctx.fillStyle = 'rgba(108,200,255,0.9)'; ctx.fillRect(8,4,10,8);
  ctx.fillStyle='#a3a6ad'; ctx.fillText('Sebelum regulasi', 22, 12);
  ctx.fillStyle = 'rgba(201,155,82,0.9)'; ctx.fillRect(130,4,10,8);
  ctx.fillStyle='#a3a6ad'; ctx.fillText('Sesudah regulasi', 144, 12);
}

// ══════════════════════════════════════════════════════════
//  STEP 3 — EKSTRAKSI FITUR
// ══════════════════════════════════════════════════════════
function extractFeatures(imageData, w, h) {
  const d    = imageData.data;
  const gray = toGrayscaleArrayML(imageData);

  // 1. Histogram grayscale 64 bin ternormalisasi
  const hist = new Array(64).fill(0);
  for (let i = 0; i < gray.length; i++) hist[Math.floor(gray[i]/4)]++;
  const histNorm = hist.map(v => v / gray.length);

  // 2. Mean & Std Dev (0–1)
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const mean = sum / gray.length / 255;
  let vari = 0;
  for (let i = 0; i < gray.length; i++) {
    const df = (gray[i]/255) - mean; vari += df*df;
  }
  const std = Math.sqrt(vari / gray.length);

  // 3. Edge magnitude rata-rata (Sobel)
  const sx = [[-1,0,1],[-2,0,2],[-1,0,1]];
  const sy = [[-1,-2,-1],[0,0,0],[1,2,1]];
  let edgeSum = 0, edgeCnt = 0;
  for (let y = 1; y < h-1; y+=3) for (let x = 1; x < w-1; x+=3) {
    let gx=0, gy=0;
    for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) {
      const v = gray[(y+ky)*w+(x+kx)];
      gx += v*sx[ky+1][kx+1]; gy += v*sy[ky+1][kx+1];
    }
    edgeSum += Math.sqrt(gx*gx+gy*gy)/(255*Math.SQRT2); edgeCnt++;
  }
  const edgeMean = edgeCnt > 0 ? edgeSum/edgeCnt : 0;

  return [...histNorm, mean, std, edgeMean]; // 64+3 = 67 fitur
}

// ══════════════════════════════════════════════════════════
//  STEP 4 — KNN
// ══════════════════════════════════════════════════════════
function euclideanDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d=a[i]-b[i]; s+=d*d; }
  return Math.sqrt(s);
}

function knnPredict(features, k, dataset) {
  const ds = dataset || mlDataset;
  if (ds.length === 0) return null;
  const dists = ds.map((item,idx) => ({
    dist: euclideanDist(features, item.features),
    label: item.label, idx
  })).sort((a,b) => a.dist-b.dist);
  const kN = dists.slice(0, Math.min(k, dists.length));
  const votes = {};
  ML_EMOTIONS.forEach(e => votes[e]=0);
  kN.forEach(n => votes[n.label]++);
  const sorted = Object.entries(votes).sort((a,b)=>b[1]-a[1]);
  return { prediction: sorted[0][0], votes, kNearest: kN, confidence: sorted[0][1]/kN.length };
}

// ══════════════════════════════════════════════════════════
//  STEP 5 — EVALUASI (Accuracy, Precision, Recall, F1, Confusion Matrix)
// ══════════════════════════════════════════════════════════
function evaluateModel(k, testRatio) {
  if (mlDataset.length < 4) {
    alert('Data training terlalu sedikit untuk evaluasi. Minimal 4 sampel dari 2 kelas berbeda.');
    return;
  }

  // Stratified split: setiap kelas dibagi train/test
  const byLabel = {};
  ML_EMOTIONS.forEach(e => byLabel[e] = []);
  mlDataset.forEach(item => byLabel[item.label].push(item));

  const trainSet = [], testSet = [];
  ML_EMOTIONS.forEach(e => {
    const items = byLabel[e];
    if (items.length === 0) return;
    // Acak
    const shuffled = [...items].sort(() => Math.random()-0.5);
    const nTest = Math.max(1, Math.round(shuffled.length * testRatio));
    testSet.push(...shuffled.slice(0, nTest));
    trainSet.push(...shuffled.slice(nTest));
  });

  if (trainSet.length === 0 || testSet.length === 0) {
    alert('Data tidak cukup untuk dibagi train/test. Tambah lebih banyak sampel.');
    return;
  }

  // Confusion matrix
  const cm = {};
  ML_EMOTIONS.forEach(a => { cm[a]={}; ML_EMOTIONS.forEach(p => cm[a][p]=0); });

  let correct = 0;
  testSet.forEach(item => {
    const result = knnPredict(item.features, k, trainSet);
    if (!result) return;
    const pred = result.prediction;
    cm[item.label][pred]++;
    if (pred === item.label) correct++;
  });

  const accuracy = testSet.length > 0 ? correct / testSet.length : 0;

  // Per-class precision, recall, F1
  const activeLabels = ML_EMOTIONS.filter(e =>
    testSet.some(t => t.label === e) || trainSet.some(t => t.label === e)
  );

  const metrics = {};
  activeLabels.forEach(e => {
    const tp = cm[e][e];
    const fp = ML_EMOTIONS.reduce((s,a) => s + (a!==e ? cm[a][e] : 0), 0);
    const fn = ML_EMOTIONS.reduce((s,p) => s + (p!==e ? cm[e][p] : 0), 0);
    const precision = tp+fp > 0 ? tp/(tp+fp) : 0;
    const recall    = tp+fn > 0 ? tp/(tp+fn) : 0;
    const f1 = precision+recall > 0 ? 2*precision*recall/(precision+recall) : 0;
    metrics[e] = { tp, fp, fn, precision, recall, f1 };
  });

  // Macro average
  const macroP  = activeLabels.reduce((s,e)=>s+metrics[e].precision,0)/activeLabels.length;
  const macroR  = activeLabels.reduce((s,e)=>s+metrics[e].recall,0)/activeLabels.length;
  const macroF1 = activeLabels.reduce((s,e)=>s+metrics[e].f1,0)/activeLabels.length;

  lastEvalResult = { accuracy, macroP, macroR, macroF1, metrics, cm, activeLabels,
    trainSize: trainSet.length, testSize: testSet.length, k, testRatio };

  displayEvaluation(lastEvalResult);
}

function displayEvaluation(ev) {
  const box = document.getElementById('mlEvalResult');
  if (!box) return;

  const pct = v => (v*100).toFixed(1)+'%';

  // Tabel metrik per kelas
  const metricRows = ev.activeLabels.map(e => `
    <tr>
      <td style="color:${ML_COLORS[e]};font-weight:700;padding:5px 8px">${ML_EMOTION_LABELS[e]}</td>
      <td style="text-align:center;padding:5px 8px">${pct(ev.metrics[e].precision)}</td>
      <td style="text-align:center;padding:5px 8px">${pct(ev.metrics[e].recall)}</td>
      <td style="text-align:center;padding:5px 8px">${pct(ev.metrics[e].f1)}</td>
      <td style="text-align:center;padding:5px 8px">${ev.metrics[e].tp}</td>
    </tr>`).join('');

  // Confusion matrix HTML
  const cmHTML = _buildConfusionMatrixHTML(ev.cm, ev.activeLabels);

  box.innerHTML = `
    <!-- Ringkasan -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px;">
      ${[
        ['Accuracy',  pct(ev.accuracy),  '#7fae91'],
        ['Precision', pct(ev.macroP),    '#7fa3bd'],
        ['Recall',    pct(ev.macroR),    '#c99b52'],
        ['F1-Score',  pct(ev.macroF1),   '#a396b3']
      ].map(([label,val,color])=>`
        <div style="background:#17181c;border:1px solid ${color}33;border-radius:8px;padding:10px;text-align:center;">
          <div style="font-size:1.3rem;font-weight:700;color:${color};font-family:'Space Mono',monospace">${val}</div>
          <div style="font-size:0.65rem;color:#9498a1;margin-top:2px">${label}</div>
        </div>`).join('')}
    </div>

    <!-- Info split -->
    <div style="font-size:0.7rem;color:#9498a1;margin-bottom:12px;font-family:'Space Mono',monospace;">
      K=${ev.k} &nbsp;|&nbsp; Train: ${ev.trainSize} sampel (${Math.round((1-ev.testRatio)*100)}%)
      &nbsp;|&nbsp; Test: ${ev.testSize} sampel (${Math.round(ev.testRatio*100)}%)
    </div>

    <!-- Tabel per kelas -->
    <div style="font-size:0.72rem;color:#6f93b8;font-family:'Space Mono',monospace;margin-bottom:6px;">METRIK PER KELAS:</div>
    <div style="overflow-x:auto;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:0.72rem;font-family:'Space Mono',monospace;">
        <thead>
          <tr style="border-bottom:1px solid #3a3b42;">
            <th style="text-align:left;padding:5px 8px;color:#8a8d95">Kelas</th>
            <th style="text-align:center;padding:5px 8px;color:#7fa3bd">Precision</th>
            <th style="text-align:center;padding:5px 8px;color:#c99b52">Recall</th>
            <th style="text-align:center;padding:5px 8px;color:#a396b3">F1-Score</th>
            <th style="text-align:center;padding:5px 8px;color:#7fae91">TP</th>
          </tr>
        </thead>
        <tbody>${metricRows}</tbody>
        <tfoot>
          <tr style="border-top:1px solid #3a3b42;font-weight:700;">
            <td style="padding:5px 8px;color:#fff">Macro Avg</td>
            <td style="text-align:center;padding:5px 8px;color:#7fa3bd">${pct(ev.macroP)}</td>
            <td style="text-align:center;padding:5px 8px;color:#c99b52">${pct(ev.macroR)}</td>
            <td style="text-align:center;padding:5px 8px;color:#a396b3">${pct(ev.macroF1)}</td>
            <td style="text-align:center;padding:5px 8px;color:#8a8d95">—</td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!-- Confusion Matrix -->
    <div style="font-size:0.72rem;color:#6f93b8;font-family:'Space Mono',monospace;margin-bottom:6px;">CONFUSION MATRIX:</div>
    <div style="overflow-x:auto;">${cmHTML}</div>
  `;
}

function _buildConfusionMatrixHTML(cm, labels) {
  const shortLabel = e => e.charAt(0).toUpperCase() + e.slice(1,3);
  // Hitung max untuk warna
  let maxVal = 0;
  labels.forEach(a => labels.forEach(p => { if (cm[a][p] > maxVal) maxVal = cm[a][p]; }));

  const headerCells = labels.map(e =>
    `<th style="text-align:center;padding:4px 6px;font-size:0.65rem;color:${ML_COLORS[e]}">${shortLabel(e)}</th>`
  ).join('');

  const rows = labels.map(actual => {
    const cells = labels.map(pred => {
      const v = cm[actual][pred];
      const isMatch = actual === pred;
      const opacity = maxVal > 0 ? 0.1 + (v/maxVal)*0.7 : 0;
      const bg = isMatch
        ? `rgba(127,174,145,${opacity})`
        : v > 0 ? `rgba(194,113,122,${opacity})` : 'transparent';
      return `<td style="text-align:center;padding:5px 6px;font-size:0.75rem;font-weight:${isMatch?'700':'400'};background:${bg};color:${isMatch?'#7fae91':'#c3c5cb'}">${v}</td>`;
    }).join('');
    return `
      <tr>
        <td style="padding:4px 8px;font-size:0.65rem;color:${ML_COLORS[actual]};white-space:nowrap;font-weight:700">${shortLabel(actual)}</td>
        ${cells}
      </tr>`;
  }).join('');

  return `
    <table style="border-collapse:collapse;font-family:'Space Mono',monospace;font-size:0.72rem;">
      <thead>
        <tr>
          <th style="padding:4px 8px;font-size:0.6rem;color:#7d818a">Actual↓ Pred→</th>
          ${headerCells}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="margin-top:6px;font-size:0.65rem;color:#7d818a">
      Diagonal = Prediksi benar &nbsp;|&nbsp; Luar diagonal = Prediksi salah
    </div>`;
}

// ══════════════════════════════════════════════════════════
//  EXPORT / IMPORT CSV
// ══════════════════════════════════════════════════════════
function exportCSV() {
  if (mlDataset.length === 0) { alert('Tidak ada data untuk diekspor.'); return; }
  const nFeat = mlDataset[0].features.length;
  const header = [...Array(nFeat).keys()].map(i=>`f${i}`).join(',') + ',label';
  const rows = mlDataset.map(item =>
    item.features.map(v=>v.toFixed(6)).join(',') + ',' + item.label
  );
  const csv = [header, ...rows].join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'emoscan_dataset.csv'; a.click();
  URL.revokeObjectURL(url);
}

function importCSV(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const lines = e.target.result.trim().split('\n');
    if (lines.length < 2) { alert('File CSV kosong atau tidak valid.'); return; }
    const header = lines[0].split(',');
    const labelIdx = header.indexOf('label');
    if (labelIdx < 0) { alert('Kolom "label" tidak ditemukan di CSV.'); return; }

    let imported = 0, skipped = 0;
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      if (cols.length < 2) { skipped++; continue; }
      const label = cols[labelIdx].trim();
      if (!ML_EMOTIONS.includes(label)) { skipped++; continue; }
      const features = cols.filter((_,ci)=>ci!==labelIdx).map(Number);
      if (features.some(isNaN)) { skipped++; continue; }
      mlDataset.push({ features, label });
      imported++;
    }
    updateDatasetTable();
    updateTrainStatus();
    alert(`Import selesai: ${imported} sampel berhasil, ${skipped} dilewati.`);
  };
  reader.readAsText(file);
}

// ══════════════════════════════════════════════════════════
//  UI: DATASET TABLE
// ══════════════════════════════════════════════════════════
function updateDatasetTable() {
  const tbody   = document.getElementById('mlDatasetBody');
  const countEl = document.getElementById('mlDatasetCount');
  if (!tbody) return;
  if (countEl) countEl.textContent = mlDataset.length;

  if (mlDataset.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#8a8d95;padding:12px">Belum ada data training</td></tr>';
    return;
  }

  const counts = {}; ML_EMOTIONS.forEach(e=>counts[e]=0);
  mlDataset.forEach(d=>counts[d.label]++);
  const total = mlDataset.length;

  tbody.innerHTML = ML_EMOTIONS.filter(e=>counts[e]>0).map(e=>`
    <tr>
      <td style="color:${ML_COLORS[e]};font-weight:700;padding:5px 8px">${ML_EMOTION_LABELS[e]}</td>
      <td style="text-align:center;padding:5px 8px">${counts[e]}</td>
      <td style="text-align:center;padding:5px 8px;color:#a3a6ad;font-size:0.7rem">${(counts[e]/total*100).toFixed(1)}%</td>
      <td style="text-align:center;padding:5px 8px;font-size:0.7rem;color:#8a8d95">
        ${mlDataset.find(d=>d.label===e)?.features.length ?? '—'} fitur
      </td>
    </tr>`).join('');
}

function updateTrainStatus() {
  const el = document.getElementById('mlTrainStatus');
  if (!el) return;
  const n = mlDataset.length;
  const classes = [...new Set(mlDataset.map(d=>d.label))].length;
  if (n===0) { el.textContent='Belum ada data training'; el.style.color='#8a8d95'; }
  else if (classes<2) { el.textContent=`${n} sampel (butuh ≥2 kelas)`; el.style.color='#c99a63'; }
  else { el.textContent=`${n} sampel, ${classes} kelas — siap prediksi & evaluasi`; el.style.color='#7fae91'; }
}

// ══════════════════════════════════════════════════════════
//  UI: PREDIKSI
// ══════════════════════════════════════════════════════════
function displayPrediction(result, k) {
  const box = document.getElementById('mlPredResult');
  if (!box || !result) return;
  const em    = result.prediction;
  const color = ML_COLORS[em]||'#fff';
  const pct   = Math.round(result.confidence*100);

  const voteBars = ML_EMOTIONS.map(e => {
    const v = result.votes[e]||0;
    const p = result.kNearest.length>0 ? Math.round(v/result.kNearest.length*100) : 0;
    return `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="width:80px;font-size:0.68rem;color:${ML_COLORS[e]}">${ML_EMOTION_LABELS[e].split(' ')[0]}</span>
        <div style="flex:1;height:7px;background:#2a2b30;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:${p}%;background:${ML_COLORS[e]};border-radius:4px"></div>
        </div>
        <span style="font-size:0.68rem;color:#9498a1;width:35px;text-align:right">${v}/${result.kNearest.length}</span>
      </div>`;
  }).join('');

  box.innerHTML = `
    <div style="text-align:center;margin-bottom:14px;">
      <div style="font-size:1.3rem;font-weight:700;color:${color};font-family:'Space Mono',monospace">${em.toUpperCase()}</div>
      <div style="font-size:0.72rem;color:#9498a1;margin-top:4px">KNN (K=${k}) — Confidence: ${pct}%</div>
      <div style="height:5px;background:#2a2b30;border-radius:3px;margin-top:6px;overflow:hidden">
        <div style="height:100%;width:${pct}%;background:${color};border-radius:3px"></div>
      </div>
    </div>
    <div style="border-top:1px solid #2a2b30;padding-top:10px;">
      <div style="font-size:0.68rem;color:#7d818a;margin-bottom:6px;font-family:'Space Mono',monospace">VOTING (${result.kNearest.length} tetangga terdekat):</div>
      ${voteBars}
    </div>`;
}

// ══════════════════════════════════════════════════════════
//  EVENT LISTENERS
// ══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {

  // Pool size — update pooling + unpooling sekaligus
  const poolSel = document.getElementById('mlPoolSize');
  const poolVal = document.getElementById('mlPoolSizeVal');
  if (poolSel) poolSel.addEventListener('change', () => {
    if (poolVal) poolVal.textContent = poolSel.value+'×'+poolSel.value;
    if (mlImageData) {
      runPooling(mlImageData, mlW, mlH);
      runUnpooling(mlImageData, mlW, mlH);
    }
  });

  // Unpooling button
  document.getElementById('btnRunUnpool')?.addEventListener('click', () => {
    if (!mlImageData) { alert('Upload gambar terlebih dahulu.'); return; }
    runUnpooling(mlImageData, mlW, mlH);
  });

  // Regularization controls
  const regMethod = document.getElementById('mlRegMethod');
  if (regMethod) {
    regMethod.addEventListener('change', () => {
      const isDropout = regMethod.value === 'dropout';
      const lambdaRow = document.getElementById('regLambdaRow');
      const dropRow   = document.getElementById('regDropRow');
      if (lambdaRow) lambdaRow.style.display = isDropout ? 'none' : 'flex';
      if (dropRow)   dropRow.style.display   = isDropout ? 'flex' : 'none';
      if (mlImageData) runRegularization(mlImageData, mlW, mlH);
    });
  }

  const regLambda = document.getElementById('mlRegLambda');
  const regLambdaVal = document.getElementById('mlRegLambdaVal');
  if (regLambda) regLambda.addEventListener('input', () => {
    if (regLambdaVal) regLambdaVal.textContent = parseFloat(regLambda.value).toFixed(3);
    if (mlImageData) runRegularization(mlImageData, mlW, mlH);
  });

  const dropRate = document.getElementById('mlDropRate');
  const dropRateVal = document.getElementById('mlDropRateVal');
  if (dropRate) dropRate.addEventListener('input', () => {
    if (dropRateVal) dropRateVal.textContent = (parseFloat(dropRate.value)*100).toFixed(0)+'%';
    if (mlImageData) runRegularization(mlImageData, mlW, mlH);
  });

  document.getElementById('btnRunReg')?.addEventListener('click', () => {
    if (!mlImageData) { alert('Upload gambar terlebih dahulu.'); return; }
    runRegularization(mlImageData, mlW, mlH);
  });

  // K slider
  const kSlider = document.getElementById('mlK');
  const kVal    = document.getElementById('mlKVal');
  if (kSlider && kVal) kSlider.addEventListener('input', ()=>{ kVal.textContent=kSlider.value; });

  // Tambah data training
  document.getElementById('btnAddTraining')?.addEventListener('click', () => {
    if (!mlImageData) { alert('Upload gambar terlebih dahulu.'); return; }
    const label    = document.getElementById('mlTrainLabel').value;
    const features = extractFeatures(mlImageData, mlW, mlH);
    mlDataset.push({ features, label });
    updateDatasetTable();
    updateTrainStatus();
    const btn = document.getElementById('btnAddTraining');
    btn.textContent='Ditambahkan!'; btn.style.background='#7fae91'; btn.style.color='#fff';
    setTimeout(()=>{ btn.textContent='+ Tambah ke Data Training'; btn.style.background=''; btn.style.color=''; }, 1200);
  });

  // Prediksi
  document.getElementById('btnPredict')?.addEventListener('click', () => {
    if (!mlImageData) { alert('Upload gambar terlebih dahulu.'); return; }
    if (mlDataset.length===0) { alert('Tambahkan data training terlebih dahulu.'); return; }
    const k = parseInt(document.getElementById('mlK').value||3);
    const features = extractFeatures(mlImageData, mlW, mlH);
    displayPrediction(knnPredict(features, k), k);
  });

  // Evaluasi model
  document.getElementById('btnEvaluate')?.addEventListener('click', () => {
    const k         = parseInt(document.getElementById('mlK').value||3);
    const testRatio = parseFloat(document.getElementById('mlTestRatio').value||0.2);
    evaluateModel(k, testRatio);
  });

  // Export CSV
  document.getElementById('btnExportCSV')?.addEventListener('click', exportCSV);

  // Import CSV
  document.getElementById('btnImportCSV')?.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) importCSV(file);
    e.target.value='';
  });

  // Hapus dataset
  document.getElementById('btnClearDataset')?.addEventListener('click', () => {
    if (!confirm('Hapus semua data training?')) return;
    mlDataset=[];
    updateDatasetTable(); updateTrainStatus();
    const predBox=document.getElementById('mlPredResult');
    if (predBox) predBox.innerHTML='<p style="color:#8a8d95;font-size:0.78rem">Belum ada prediksi</p>';
    const evalBox=document.getElementById('mlEvalResult');
    if (evalBox) evalBox.innerHTML='<p style="color:#8a8d95;font-size:0.78rem">Belum ada hasil evaluasi</p>';
  });
});

// ══════════════════════════════════════════════════════════
//  HELPER
// ══════════════════════════════════════════════════════════
function toGrayscaleArrayML(imageData) {
  const d=imageData.data, n=imageData.width*imageData.height;
  const gray=new Uint8Array(n);
  for (let i=0;i<n;i++) gray[i]=Math.round(0.299*d[i*4]+0.587*d[i*4+1]+0.114*d[i*4+2]);
  return gray;
}
