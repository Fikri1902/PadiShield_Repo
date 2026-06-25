#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_best_worst.py — Evaluasi model final (exp4b) untuk laporan.

(1) val() pada test set v3 (union tungro) -> metrik + confusion matrix + kurva (plots).
(2) Pilih best & worst prediction PER KATEGORI secara jujur (match pred vs GT pakai IoU),
    simpan gambar beranotasi (GT hijau, pred benar biru, pred salah/FP merah).

TIDAK melatih ulang & TIDAK mengubah data raw / best_model.pt. Hanya inferensi.
Output: outputs/eval_exp4b/ (plots+metrics) dan outputs/best_worst/ (figur + CSV).
"""
import csv, sys
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "outputs" / "best_model.pt"
DSET = ROOT / "data" / "processed_v3_tungro_union"
IMG_DIR = DSET / "images" / "test"
LBL_DIR = DSET / "labels" / "test"
OUT = ROOT / "outputs" / "best_worst"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAMES = ["rice_blast", "bacterial_leaf_blight", "rice_tungro", "stem_borer_insect",
         "dead_heart", "brown_planthopper", "rat"]
CONF = 0.25
IOU_TP = 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw(im, box, color, label):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(im, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(im, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    model = YOLO(str(MODEL))

    # --- (1) val: metrik + plots (confusion matrix, PR/F1 curve) ---
    spec = {"path": str(DSET).replace("\\", "/"), "train": "images/train",
            "val": "images/val", "test": "images/test", "nc": 7,
            "names": {i: n for i, n in enumerate(NAMES)}}
    tmp_yaml = OUT / "eval_data.yaml"
    tmp_yaml.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    try:
        mt = model.val(data=str(tmp_yaml), split="test", imgsz=640, conf=0.001,
                       iou=0.6, plots=True, project=str(ROOT / "outputs"),
                       name="eval_exp4b", exist_ok=True, verbose=False)
        print(f"[val] mAP50={mt.box.map50:.4f} mAP50-95={mt.box.map:.4f} "
              f"mp={mt.box.mp:.4f} mr={mt.box.mr:.4f}")
    except Exception as e:
        print("[val] gagal:", e)

    # --- (2) best/worst per kelas ---
    imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"[predict] {len(imgs)} gambar test ...")
    # per-image cache of GT & preds
    per_img = {}
    for k, ip in enumerate(imgs):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        gts = []
        lp = LBL_DIR / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                s = line.split()
                if len(s) < 5:
                    continue
                c = int(float(s[0])); cx, cy, w, h = map(float, s[1:5])
                gts.append((c, [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]))
        res = model(str(ip), conf=CONF, imgsz=640, verbose=False)[0]
        preds = [(int(b.cls), float(b.conf), [float(v) for v in b.xyxy[0].tolist()])
                 for b in res.boxes]
        per_img[ip] = (W, H, gts, preds)
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(imgs)}")

    rows = []
    for c, name in enumerate(NAMES):
        best = None   # (score, img) score=conf*iou for a correct TP
        worst = None  # (severity, img, kind) higher severity = worse
        for ip, (W, H, gts, preds) in per_img.items():
            gt_c = [g for g in gts if g[0] == c]
            if not gt_c:
                continue
            pred_c = [p for p in preds if p[0] == c]
            # best IoU/conf for each GT of class c
            best_match = 0.0          # best (iou) among matched
            best_cw = 0.0             # best conf*iou
            matched = 0
            for _, gb in gt_c:
                bi, bconf = 0.0, 0.0
                for pc, pconf, pb in pred_c:
                    j = iou(gb, pb)
                    if j > bi:
                        bi, bconf = j, pconf
                if bi >= IOU_TP:
                    matched += 1
                    best_match = max(best_match, bi)
                    best_cw = max(best_cw, bconf * bi)
            n_fp = sum(1 for pc, pconf, pb in preds
                       if pc == c and all(iou(pb, gb) < IOU_TP for _, gb in gt_c))
            missed = len(gt_c) - matched
            # BEST candidate: all GT matched, high conf*iou, no FP
            if matched == len(gt_c) and n_fp == 0:
                score = best_cw
                if best is None or score > best[0]:
                    best = (score, ip, best_match)
            # WORST candidate: prioritise missed GT (FN), then FP, then low IoU
            severity = missed * 2 + (1 if n_fp else 0)
            if severity > 0:
                key = (severity, n_fp, -best_match)
                if worst is None or key > worst[0]:
                    worst = (key, ip, f"missed={missed}, fp={n_fp}")

        for kind, sel in [("best", best), ("worst", worst)]:
            if sel is None:
                rows.append([name, kind, "", "tidak ada kandidat", "", ""])
                continue
            ip = sel[1]
            W, H, gts, preds = per_img[ip]
            im = cv2.imread(str(ip))
            for gc, gb in gts:
                draw(im, gb, (60, 200, 60), f"GT:{NAMES[gc]}")
            for pc, pconf, pb in preds:
                ok = any(pc == gc and iou(pb, gb) >= IOU_TP for gc, gb in gts)
                col = (235, 140, 40) if ok else (40, 40, 235)
                draw(im, pb, col, f"{NAMES[pc]} {pconf:.2f}")
            outp = OUT / f"{kind}_{name}.jpg"
            cv2.imwrite(str(outp), im)
            note = sel[2] if kind == "worst" else f"iou={sel[2]:.2f}"
            rows.append([name, kind, ip.name, note, len(gts), len(preds)])
            print(f"  [{kind}] {name}: {ip.name} ({note})")

    with open(OUT / "best_worst_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "kind", "image", "note", "n_gt", "n_pred"])
        w.writerows(rows)
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
