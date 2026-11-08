/**
 * shared.js — Fungsi bersama untuk semua halaman EmoScan
 * Universitas Mikroskil — IF2112 Computer Vision — Tugas 2
 *
 * Dipakai oleh: dataset.html, training.html, evaluation.html, index.html
 * Berisi: konstanta emosi, ekstraksi fitur, KNN, localStorage helper
 */

// ══════════════════════════════════════════════════════════
//  KONSTANTA GLOBAL
// ══════════════════════════════════════════════════════════
const ML_PIPELINE_EMOTIONS = ['happy','sad','angry','fearful','disgusted','surprised','neutral'];

const ML_PIPELINE_LABELS = {
  happy:'Happy', sad:'Sad', angry:'Angry',
  fearful:'Fear', disgusted:'Disgust', surprised:'Surprised', neutral:'Neutral'
};

const ML_PIPELINE_EMOJI = {
  happy:'😊', sad:'😢', angry:'😠',
  fearful:'😨', disgusted:'🤢', surprised:'😲', neutral:'😐'
};

const ML_PIPELINE_COLORS = {
  happy:'#c99b52', sad:'#7fa3bd', angry:'#c2717a',
  fearful:'#a396b3', disgusted:'#7fae91', surprised:'#c99a63', neutral:'#9ea3ad'
};

const STORAGE_KEYS = {
  DATASET: 'emoscan_dataset',
  MODEL:   'emoscan_knn_model',
  EVAL:    'emoscan_eval_result'
};

// ══════════════════════════════════════════════════════════
//  LOCALSTORAGE HELPERS
// ══════════════════════════════════════════════════════════
function saveDataset(dataset) {
  try {
    localStorage.setItem(STORAGE_KEYS.DATASET, JSON.stringify(dataset));
    return true;
  } catch (e) {
    console.error('Gagal menyimpan dataset:', e);
    alert('Gagal menyimpan dataset. Mungkin localStorage penuh atau data terlalu besar.');
    return false;
  }
}

function loadDataset() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.DATASET);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Gagal memuat dataset:', e);
    return [];
  }
}

function clearDataset() {
  localStorage.removeItem(STORAGE_KEYS.DATASET);
}

function saveModel(model) {
  try {
    localStorage.setItem(STORAGE_KEYS.MODEL, JSON.stringify(model));
    return true;
  } catch (e) {
    console.error('Gagal menyimpan model:', e);
    return false;
  }
}

function loadModel() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.MODEL);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    console.error('Gagal memuat model:', e);
    return null;
  }
}

function clearModel() {
  localStorage.removeItem(STORAGE_KEYS.MODEL);
}

function saveEvalResult(result) {
  try {
    localStorage.setItem(STORAGE_KEYS.EVAL, JSON.stringify(result));
    return true;
  } catch (e) { return false; }
}

function loadEvalResult() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.EVAL);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

// ══════════════════════════════════════════════════════════
//  EKSTRAKSI FITUR
//  Input : ImageData (dari canvas wajah hasil crop)
//  Output: array 67 fitur [histogram64, mean, std, edgeMean]
// ══════════════════════════════════════════════════════════
function toGrayscaleArray(imageData) {
  const d = imageData.data;
  const n = imageData.width * imageData.height;
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    gray[i] = Math.round(0.299*d[i*4] + 0.587*d[i*4+1] + 0.114*d[i*4+2]);
  }
  return gray;
}

function extractFeaturesShared(imageData, w, h) {
  const gray = toGrayscaleArray(imageData);

  // 1. Histogram grayscale 64 bin (ternormalisasi)
  const hist = new Array(64).fill(0);
  for (let i = 0; i < gray.length; i++) hist[Math.floor(gray[i]/4)]++;
  const histNorm = hist.map(v => v / gray.length);

  // 2. Mean & Std Dev (0-1)
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const mean = sum / gray.length / 255;
  let vari = 0;
  for (let i = 0; i < gray.length; i++) {
    const df = (gray[i]/255) - mean; vari += df*df;
  }
  const std = Math.sqrt(vari / gray.length);

  // 3. Edge magnitude rata-rata (Sobel, step 3)
  const sx = [[-1,0,1],[-2,0,2],[-1,0,1]];
  const sy = [[-1,-2,-1],[0,0,0],[1,2,1]];
  let edgeSum = 0, edgeCnt = 0;
  for (let y = 1; y < h-1; y+=3) {
    for (let x = 1; x < w-1; x+=3) {
      let gx=0, gy=0;
      for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) {
        const v = gray[(y+ky)*w+(x+kx)];
        gx += v*sx[ky+1][kx+1]; gy += v*sy[ky+1][kx+1];
      }
      edgeSum += Math.sqrt(gx*gx+gy*gy)/(255*Math.SQRT2); edgeCnt++;
    }
  }
  const edgeMean = edgeCnt > 0 ? edgeSum/edgeCnt : 0;

  return [...histNorm, mean, std, edgeMean]; // 67 fitur
}

// ══════════════════════════════════════════════════════════
//  KNN CLASSIFIER
// ══════════════════════════════════════════════════════════
function euclideanDistanceShared(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d=a[i]-b[i]; s+=d*d; }
  return Math.sqrt(s);
}

/**
 * Prediksi memakai model KNN yang sudah "dilatih" (tersimpan).
 * model = { trainData: [{features, label}], k: number }
 */
function knnPredictShared(features, model) {
  if (!model || !model.trainData || model.trainData.length === 0) return null;
  const k = model.k || 3;

  const dists = model.trainData.map(item => ({
    dist: euclideanDistanceShared(features, item.features),
    label: item.label
  })).sort((a,b) => a.dist - b.dist);

  const kN = dists.slice(0, Math.min(k, dists.length));
  const votes = {};
  ML_PIPELINE_EMOTIONS.forEach(e => votes[e] = 0);
  kN.forEach(n => votes[n.label]++);

  const sorted = Object.entries(votes).sort((a,b) => b[1]-a[1]);
  return {
    prediction: sorted[0][0],
    votes,
    kNearest: kN,
    confidence: sorted[0][1] / kN.length
  };
}

/**
 * "Training" KNN — secara teknis KNN adalah lazy learner,
 * sehingga "training" berarti menyimpan seluruh dataset sebagai
 * referensi (trainData) yang akan dipakai saat prediksi nanti.
 */
function trainKNN(dataset, k) {
  return {
    trainData: dataset.map(d => ({ features: d.features, label: d.label })),
    k: k,
    trainedAt: new Date().toISOString(),
    nSamples: dataset.length
  };
}

// ══════════════════════════════════════════════════════════
//  EVALUASI MODEL (Accuracy, Precision, Recall, F1, Confusion Matrix)
// ══════════════════════════════════════════════════════════
function evaluateKNN(dataset, k, testRatio) {
  // Stratified split per kelas
  const byLabel = {};
  ML_PIPELINE_EMOTIONS.forEach(e => byLabel[e] = []);
  dataset.forEach(item => byLabel[item.label].push(item));

  const trainSet = [], testSet = [];
  ML_PIPELINE_EMOTIONS.forEach(e => {
    const items = byLabel[e];
    if (items.length === 0) return;
    const shuffled = [...items].sort(() => Math.random()-0.5);
    const nTest = Math.max(1, Math.round(shuffled.length * testRatio));
    testSet.push(...shuffled.slice(0, nTest));
    trainSet.push(...shuffled.slice(nTest));
  });

  if (trainSet.length === 0 || testSet.length === 0) return null;

  const tempModel = { trainData: trainSet, k };

  const cm = {};
  ML_PIPELINE_EMOTIONS.forEach(a => { cm[a]={}; ML_PIPELINE_EMOTIONS.forEach(p => cm[a][p]=0); });

  let correct = 0;
  testSet.forEach(item => {
    const result = knnPredictShared(item.features, tempModel);
    if (!result) return;
    cm[item.label][result.prediction]++;
    if (result.prediction === item.label) correct++;
  });

  const accuracy = testSet.length > 0 ? correct / testSet.length : 0;

  const activeLabels = ML_PIPELINE_EMOTIONS.filter(e =>
    testSet.some(t => t.label === e) || trainSet.some(t => t.label === e)
  );

  const metrics = {};
  activeLabels.forEach(e => {
    const tp = cm[e][e];
    const fp = ML_PIPELINE_EMOTIONS.reduce((s,a) => s + (a!==e ? cm[a][e] : 0), 0);
    const fn = ML_PIPELINE_EMOTIONS.reduce((s,p) => s + (p!==e ? cm[e][p] : 0), 0);
    const precision = tp+fp > 0 ? tp/(tp+fp) : 0;
    const recall    = tp+fn > 0 ? tp/(tp+fn) : 0;
    const f1 = precision+recall > 0 ? 2*precision*recall/(precision+recall) : 0;
    metrics[e] = { tp, fp, fn, precision, recall, f1 };
  });

  const macroP  = activeLabels.reduce((s,e)=>s+metrics[e].precision,0)/activeLabels.length;
  const macroR  = activeLabels.reduce((s,e)=>s+metrics[e].recall,0)/activeLabels.length;
  const macroF1 = activeLabels.reduce((s,e)=>s+metrics[e].f1,0)/activeLabels.length;

  return {
    accuracy, macroP, macroR, macroF1, metrics, cm, activeLabels,
    trainSize: trainSet.length, testSize: testSet.length, k, testRatio,
    evaluatedAt: new Date().toISOString()
  };
}

// ══════════════════════════════════════════════════════════
//  CSV EXPORT / IMPORT
// ══════════════════════════════════════════════════════════
function datasetToCSV(dataset) {
  if (dataset.length === 0) return '';
  const nFeat = dataset[0].features.length;
  const header = [...Array(nFeat).keys()].map(i=>`f${i}`).join(',') + ',label';
  const rows = dataset.map(item =>
    item.features.map(v=>v.toFixed(6)).join(',') + ',' + item.label
  );
  return [header, ...rows].join('\n');
}

function csvToDataset(csvText) {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return { dataset: [], imported: 0, skipped: 0 };

  const header = lines[0].split(',');
  const labelIdx = header.indexOf('label');
  if (labelIdx < 0) return { dataset: [], imported: 0, skipped: lines.length-1 };

  const dataset = [];
  let imported = 0, skipped = 0;

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    if (cols.length < 2) { skipped++; continue; }
    const label = cols[labelIdx].trim();
    if (!ML_PIPELINE_EMOTIONS.includes(label)) { skipped++; continue; }
    const features = cols.filter((_,ci)=>ci!==labelIdx).map(Number);
    if (features.some(isNaN)) { skipped++; continue; }
    dataset.push({ features, label, id: Date.now()+'_'+i });
    imported++;
  }
  return { dataset, imported, skipped };
}

function downloadCSV(csvText, filename) {
  const blob = new Blob([csvText], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════════════════════════════════════════════
//  UTIL: format persen
// ══════════════════════════════════════════════════════════
function fmtPct(v) { return (v*100).toFixed(1)+'%'; }
