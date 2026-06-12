# Kelompok 8 — Project PCD 2026
Klasifikasi Rimpang (Jahe, Kunyit, Lengkuas) menggunakan GLCM + KNN/SVM/Random Forest

## Dataset
Download dan ekstrak zip berikut ke dalam folder repo:
https://drive.google.com/drive/folders/1prG67Ms5nv39KwF11XKyEZIQ460-OILD?usp=sharing

## Struktur File
- `library.py` — fungsi-fungsi preprocessing dan utils (jangan diubah)
- `baseline.ipynb` — eksperimen tanpa preprocessing (sekaligus template untuk p1-p4)
- `p1_grayscale.ipynb` — Resize + Grayscale
- `p2_clahe.ipynb` — Resize + Grayscale + CLAHE
- `p3_blur_clahe.ipynb` — Resize + Grayscale + Gaussian Blur + CLAHE
- `p4_median_clahe.ipynb` — Resize + Grayscale + Median Blur + CLAHE
- `compare.ipynb` — perbandingan semua hasil (dijalankan terakhir)