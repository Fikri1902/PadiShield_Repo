#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
padishield_config.py — Sumber kebenaran tunggal untuk pemetaan kelas & dataset.

Dipakai oleh audit_datasets.py, build_processed_dataset.py, dan
run_overnight_training.py.

Keputusan pemetaan didokumentasikan di reports/DECISION_LOG.md.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Lokasi
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # ...\NEW_IDEA
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MANIFESTS = PROJECT_ROOT / "data" / "manifests"
REPORTS = PROJECT_ROOT / "reports"

SEED = 42

# --------------------------------------------------------------------------- #
# Tujuh kelas target internal
# --------------------------------------------------------------------------- #
TARGET_CLASSES = [
    "rice_blast",            # 0
    "bacterial_leaf_blight", # 1
    "rice_tungro",           # 2
    "stem_borer_insect",     # 3
    "dead_heart",            # 4
    "brown_planthopper",     # 5
    "rat",                   # 6
]
NAME_TO_ID = {n: i for i, n in enumerate(TARGET_CLASSES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cap jumlah gambar tikus dari dataset 06 untuk mengurangi imbalance & waktu.
RAT_06_CAP = 1500

# --------------------------------------------------------------------------- #
# Spesifikasi sumber dataset
# --------------------------------------------------------------------------- #
# layout:
#   "yolo_split"      : <root>/images/<split>, <root>/labels/<split>
#   "roboflow_split"  : <root>/<split>/images, <root>/<split>/labels
#   "yolo_flat"       : <root>/images, <root>/labels (tanpa split)
#   "folder_class"    : kelas ditentukan oleh NAMA FOLDER (bukan id numerik)
#
# id_map: original_class_id (str) -> target_id. id yang tidak ada di map -> DIBUANG.
# Untuk folder_class, gunakan folder_map: nama_folder -> target_id, dan SEMUA
# box di folder itu diberi target_id tersebut (id numerik di file diabaikan).

DATASET_SPECS = [
    {
        "key": "01_nischallal_rice_disease",
        "root": "01_nischallal_rice_disease",
        "layout": "yolo_split",
        "splits": ["train", "valid", "test"],
        "orig_names": {0: "Brown_Spot", 1: "Rice_Blast", 2: "Bacterial_Blight"},
        "id_map": {"1": 0, "2": 1},          # Rice_Blast->blast, Bacterial->BLB
        "dropped": {"0": "Brown_Spot (bukan kelas target)"},
    },
    {
        "key": "04_rice11_dead_heart",
        "root": "04_rice11_dead_heart",
        "layout": "roboflow_split",
        "splits": ["train", "valid", "test"],
        "orig_names": {0: "bph_damage", 1: "bph_insect", 2: "deadheart",
                       3: "egg_mass", 4: "false_smut", 5: "leaf_blast",
                       6: "neck_blast", 7: "node_blast", 8: "rice_bug",
                       9: "sheath_rot", 10: "stem_borer_larva",
                       11: "stem_borer_moth"},
        "id_map": {
            "5": 0, "6": 0, "7": 0,          # leaf/neck/node blast -> rice_blast
            "1": 5,                          # bph_insect -> brown_planthopper
            "2": 4,                          # deadheart -> dead_heart
            "10": 3, "11": 3,                # stem borer larva/moth -> stem_borer
        },
        "dropped": {
            "0": "bph_damage (gejala, bukan serangga)",
            "3": "egg_mass (bukan target)",
            "4": "false_smut (bukan target)",
            "8": "rice_bug (bukan target)",
            "9": "sheath_rot (bukan target)",
        },
    },
    {
        "key": "02_riceleafdiseasebd_tungro",
        "root": "02_riceleafdiseasebd_tungro",
        "layout": "folder_class",
        # gambar bersih ada di 'Original images/<folder>', label di
        # 'Annotated images ( visual with labels)/<folder>/labels'.
        "clean_images_dir": "Original images",
        "labels_parent_dir": "Annotated images ( visual with labels)",
        "labels_subdir": "labels",
        # id numerik di file TIDAK konsisten (Leaf smut & Rice Tungro sama2 id 3),
        # jadi folder = sumber kebenaran. Hanya Rice Tungro yang dipakai.
        "folder_map": {"Rice Tungro": 2},
        "dropped_folders": {
            "Blast": "anotasi lesi-level; rice_blast sudah dari 01+04",
            "Brown spot": "bukan kelas target",
            "Leaf smut": "bukan kelas target",
            "Sheath blight": "bukan kelas target",
            "Healthy": "tanpa anotasi",
        },
    },
    {
        "key": "06_rat_yolov5",
        "root": "06_rat_yolov5/rat_detection.v4i.yolov5pytorch",
        "layout": "yolo_flat",
        "splits": [""],
        "orig_names": {0: "Rat"},
        "id_map": {"0": 6},                  # Rat -> rat
        "dropped": {},
        "cap_images": RAT_06_CAP,            # batasi jumlah gambar tikus
    },
    {
        "key": "07_rat_rice_field",
        "root": "07_rat_rice_field",
        "layout": "roboflow_split",
        "splits": ["train", "valid", "test"],
        "orig_names": {0: "Rat-in-a-rice-field"},
        "id_map": {"0": 6},                  # Rat -> rat
        "dropped": {},
    },
]

# Dataset yang SENGAJA DIKECUALIKAN dari processed (raw tetap utuh).
EXCLUDED_DATASETS = {
    "03_r2000_stem_borer": (
        "18 kelas tanpa metadata nama (classID_00..17). Tidak ada file names/yaml. "
        "Pemetaan ke stem_borer_insect tidak dapat diandalkan; risiko mislabeling tinggi. "
        "stem_borer_insect sudah tersedia bernama jelas dari dataset 04 "
        "(stem_borer_larva, stem_borer_moth). Raw dipertahankan; kandidat masa depan "
        "bila nama kelas dikonfirmasi."
    ),
    "05_planthopper": (
        "3 kelas tanpa metadata nama (0,1,2). brown_planthopper sudah tersedia bernama "
        "jelas dari dataset 04 (bph_insect, 1141 gambar). Dikecualikan untuk menghindari "
        "mislabeling. Raw dipertahankan; kandidat masa depan bila nama kelas dikonfirmasi."
    ),
}
