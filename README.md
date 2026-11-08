# EmoScan — Deteksi Emosi + Preprocessing Citra Digital


## CARA MENJALANKAN (PENTING!)

Aplikasi ini TIDAK bisa dibuka langsung dengan double-click file `index.html`.
Model AI membutuhkan server lokal. Gunakan salah satu cara berikut:

### Cara 1 — VS Code Live Server 
1. Buka folder project di VS Code
2. Install extension "Live Server" (Ritwick Dey)
3. Klik kanan `index.html` → "Open with Live Server"
4. Browser otomatis terbuka di http://127.0.0.1:5500

### Cara 2 — Python HTTP Server
1. Buka Terminal/CMD di folder project
2. Jalankan: `python -m http.server 8000`
3. Buka browser: http://localhost:8000

### Cara 3 — Node.js
1. `npx serve .`
2. Buka URL yang ditampilkan

## Fitur
- Upload gambar SEKALI di banner atas → semua tab otomatis terisi
- Tab Data: histogram, pixel grid, grayscale
- Tab Technic: sampling, quantization, noise
- Tab Convolution: berbagai kernel filter
- Tab Morphology: erosi, dilasi, opening, closing
- Tab Feature Detection: Harris corner, blob, Canny edge
- Deteksi emosi otomatis dari gambar yang diupload
- Kamera live real-time dengan deteksi emosi

## Model AI
Model di-load otomatis dari CDN (butuh internet) atau dari folder /models (lokal).
