# Dataset PadiShield

Citra **tidak** disertakan di repo ini karena berukuran ~1,1 GB. Unduh dari Google Drive:

**[>> TAUTAN GOOGLE DRIVE DATASET <<](https://its.id/m/DATASETPADISHIELD)**

Setelah diunduh, letakkan sehingga strukturnya menjadi:

```
data/PadiShield_Dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
```

## Ringkasan
- Total **9.601** citra: train **6.722**, val **1.437**, test **1.442** (stratified 70/15/15, tanpa kebocoran).
- Format anotasi: YOLO (kotak pembatas). 7 label internal -> 6 kategori aplikasi.

## Pemetaan 7 label internal -> 6 kategori OPT
| Label internal | Kategori aplikasi |
|---|---|
| rice_blast | Blas |
| bacterial_leaf_blight | Hawar Daun Bakteri (HDB) |
| rice_tungro | Tungro |
| stem_borer_insect | Penggerek Batang Padi (PBP) |
| dead_heart | Penggerek Batang Padi (PBP) |
| brown_planthopper | Wereng Batang Cokelat (WBC) |
| rat | Tikus |

Sumber & sitasi tiap dataset: lihat Daftar Pustaka pada laporan/esai.
