# PadiShield

**Aplikasi Deteksi Dini Hama dan Penyakit Padi Berbasis Convolutional Neural Network (CNN) untuk Memperkuat Ketahanan Pangan Nasional** - Statistics Essay Competition, Satria Data 2026.

Repositori ini memuat **pipeline inti model** PadiShield agar hasilnya dapat direproduksi: penyiapan data, pelatihan, dan validasi model deteksi objek YOLO11s untuk enam kategori OPT padi. (Skrip pembuatan figur laporan dan dokumentasi tidak disertakan; repo ini fokus pada reproduksi model.)

## Tautan
- **Aplikasi (live):** https://padishield-sec2026.web.app
- **Dataset (Google Drive):** https://its.id/m/DATASETPADISHIELD  (lihat `data/README.md`)

## Ringkasan model
- Arsitektur: **YOLO11s** (CNN satu tahap, ~9,4 juta parameter).
- Kinerja (data uji): **mAP@50 = 0,788**. Bobot final: `outputs/best_model.pt` (eksperimen exp4b, strategi *tungro union*).
- Enam kategori: Blas, Hawar Daun Bakteri, Tungro, Penggerek Batang Padi, Wereng Batang Cokelat, Tikus.

## Struktur
```
scripts/
  padishield_config.py            konfigurasi & pemetaan 7 label internal -> 6 kategori
  audit_datasets.py               audit & pengecekan label dataset
  build_processed_dataset.py      bangun split train/val/test (70/15/15, anti-kebocoran)
  build_tungro_merged_dataset.py  varian label tungro (region/union) untuk exp4b
  run_overnight_training.py       pelatihan YOLO11s
  eval_best_worst.py              validasi + best/worst prediction per kelas
data/
  data.yaml                       konfigurasi kelas (arahkan ke dataset dari Drive)
  README.md                       petunjuk unduh dataset (citra ada di Google Drive)
outputs/
  best_model.pt                   bobot model final (exp4b, 19 MB)
  eval_exp4b/                     hasil validasi: confusion matrix, kurva P/R/F1, val batches
  best_worst/                     prediksi terbaik & terburuk per kelas + ringkasan CSV
```

## Reproduksi
```bash
pip install -r requirements.txt

# 1) Unduh dataset dari Google Drive -> data/PadiShield_Dataset (lihat data/README.md)

# 2) Validasi model final terhadap test set (cara tercepat memverifikasi hasil):
yolo detect val model=outputs/best_model.pt data=data/data.yaml split=test
#    atau:  python scripts/eval_best_worst.py

# 3) (Opsional) Bangun ulang split dari dataset sumber lalu latih dari awal:
python scripts/audit_datasets.py
python scripts/build_processed_dataset.py
python scripts/build_tungro_merged_dataset.py
python scripts/run_overnight_training.py
```

## Catatan integritas
- Split 70/15/15 terstratifikasi; deduplikasi + *perceptual hashing* mencegah kebocoran data (*data leakage*).
- Keluaran model adalah **penyaringan awal**, bukan diagnosis; keputusan akhir pada penyuluh/POPT.
- Dataset bersumber publik (Kaggle, Roboflow, Mendeley); sitasi lengkap pada laporan.
