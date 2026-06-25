#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_datasets.py — Audit seluruh dataset di data/raw (read-only).

Menghasilkan:
    reports/DATASET_AUDIT_REPORT.md
    reports/dataset_statistics.csv
    reports/class_distribution.csv

Memeriksa: gambar, label, format, kelas asli, jumlah box, distribusi kelas,
label kosong, gambar tanpa label, label tanpa gambar, class id di luar rentang,
bbox tidak valid, gambar rusak (header check ringan), struktur split.
Tidak mengubah data raw.
"""

from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path

from padishield_config import (RAW, REPORTS, DATASET_SPECS, EXCLUDED_DATASETS,
                               IMG_EXTS, TARGET_CLASSES)


def list_images(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]


def parse_label(path: Path):
    """Kembalikan (boxes, issues). boxes: list (cid, x, y, w, h)."""
    boxes, issues = [], []
    try:
        text = path.read_text(errors="ignore")
    except OSError as e:
        return boxes, [f"unreadable:{e}"]
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            issues.append(f"L{ln}:too_few_fields")
            continue
        cid = parts[0]
        if not cid.lstrip("-").isdigit():
            issues.append(f"L{ln}:nonnumeric_class({cid})")
            continue
        try:
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            issues.append(f"L{ln}:nonfloat_coords")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1.0001 and 0 < h <= 1.0001):
            issues.append(f"L{ln}:bbox_out_of_range")
        boxes.append((int(cid), x, y, w, h))
    return boxes, issues


def img_ok(path: Path) -> bool:
    """Header check ringan (tanpa decode penuh)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            return head[:2] == b"\xff\xd8"
        if path.suffix.lower() == ".png":
            return head[:8] == b"\x89PNG\r\n\x1a\n"
        if path.suffix.lower() == ".bmp":
            return head[:2] == b"BM"
        if path.suffix.lower() == ".webp":
            return head[:4] == b"RIFF"
        return True
    except OSError:
        return False


def find_label_dirs(spec):
    """Yield (split, images_dir, labels_dir) untuk layout berbasis YOLO."""
    root = RAW / spec["root"]
    layout = spec["layout"]
    if layout == "yolo_split":
        for sp in spec["splits"]:
            yield sp, root / "images" / sp, root / "labels" / sp
    elif layout == "roboflow_split":
        for sp in spec["splits"]:
            d = root / sp
            if d.exists():
                yield sp, d / "images", d / "labels"
    elif layout == "yolo_flat":
        yield "", root / "images", root / "labels"


def audit_yolo(spec):
    """Audit dataset berformat YOLO. Kembalikan dict statistik."""
    orig_names = spec.get("orig_names", {})
    id_map = spec.get("id_map", {})
    n_img = n_lbl = n_box = 0
    empty_labels = imgs_no_label = labels_no_img = 0
    bad_imgs = invalid_box = out_of_range = 0
    orig_box = Counter()         # original id -> box count
    orig_img = Counter()         # original id -> image count
    tgt_box = Counter()          # target id -> box count (after id_map)
    tgt_img = Counter()          # target id -> image count
    splits = []
    sample_issues = []

    for sp, idir, ldir in find_label_dirs(spec):
        if not idir.exists():
            continue
        splits.append(sp or "(flat)")
        imgs = list_images(idir)
        img_stems = {p.stem: p for p in imgs}
        n_img += len(imgs)
        # corrupt check (sample up to 400/split for speed)
        for p in imgs[:400]:
            if not img_ok(p):
                bad_imgs += 1
        lbl_files = [p for p in ldir.glob("*.txt") if p.name.lower() != "classes.txt"] if ldir.exists() else []
        n_lbl += len(lbl_files)
        lbl_stems = {p.stem for p in lbl_files}
        imgs_no_label += sum(1 for st in img_stems if st not in lbl_stems)
        labels_no_img += sum(1 for st in lbl_stems if st not in img_stems)
        for lf in lbl_files:
            boxes, issues = parse_label(lf)
            if not boxes:
                empty_labels += 1
            if issues:
                out_of_range += sum("out_of_range" in i for i in issues)
                invalid_box += sum(("too_few" in i or "nonfloat" in i or "nonnumeric" in i) for i in issues)
                if len(sample_issues) < 15:
                    sample_issues.append(f"{lf.name}: {';'.join(issues[:3])}")
            seen_orig, seen_tgt = set(), set()
            for cid, *_ in boxes:
                n_box += 1
                orig_box[cid] += 1
                seen_orig.add(cid)
                if str(cid) in id_map:
                    t = id_map[str(cid)]
                    tgt_box[t] += 1
                    seen_tgt.add(t)
            for cid in seen_orig:
                orig_img[cid] += 1
            for t in seen_tgt:
                tgt_img[t] += 1
    return {
        "key": spec["key"], "layout": spec["layout"], "splits": ",".join(splits),
        "n_img": n_img, "n_lbl": n_lbl, "n_box": n_box,
        "empty_labels": empty_labels, "imgs_no_label": imgs_no_label,
        "labels_no_img": labels_no_img, "bad_imgs_sampled": bad_imgs,
        "invalid_box": invalid_box, "out_of_range": out_of_range,
        "orig_names": orig_names, "id_map": id_map,
        "orig_box": orig_box, "orig_img": orig_img,
        "cls_box": tgt_box, "cls_img": tgt_img,
        "dropped": spec.get("dropped", {}), "sample_issues": sample_issues,
    }


def audit_folder_class(spec):
    """Audit dataset berbasis folder (dataset 02)."""
    root = RAW / spec["root"]
    clean = root / spec["clean_images_dir"]
    lparent = root / spec["labels_parent_dir"]
    folder_map = spec["folder_map"]
    n_img = n_lbl = n_box = 0
    empty_labels = labels_no_img = 0
    cls_box = Counter(); cls_img = Counter()
    per_folder = {}
    for folder in sorted([d.name for d in lparent.iterdir() if d.is_dir()]):
        ldir = lparent / folder / spec["labels_subdir"]
        cleandir = clean / folder
        if not ldir.exists():
            continue
        lbls = list(ldir.glob("*.txt"))
        clean_stems = {p.stem for p in cleandir.glob("*")} if cleandir.exists() else set()
        fbox = fimg = femp = 0
        for lf in lbls:
            boxes, _ = parse_label(lf)
            if not boxes:
                femp += 1
            if lf.stem not in clean_stems:
                labels_no_img += 1
            if boxes:
                fimg += 1
            fbox += len(boxes)
        per_folder[folder] = {"labels": len(lbls), "boxes": fbox,
                              "clean_imgs": len(clean_stems),
                              "target": folder_map.get(folder)}
        n_lbl += len(lbls); n_box += fbox; empty_labels += femp
        if folder in folder_map:
            tid = folder_map[folder]
            cls_box[tid] += fbox; cls_img[tid] += fimg; n_img += fimg
    return {
        "key": spec["key"], "layout": spec["layout"],
        "splits": "(folder-based, no split)",
        "n_img": n_img, "n_lbl": n_lbl, "n_box": n_box,
        "empty_labels": empty_labels, "imgs_no_label": 0,
        "labels_no_img": labels_no_img, "bad_imgs_sampled": 0,
        "invalid_box": 0, "out_of_range": 0,
        "per_folder": per_folder, "folder_map": folder_map,
        "dropped": spec.get("dropped_folders", {}),
        "cls_box": cls_box, "cls_img": cls_img, "sample_issues": [],
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in DATASET_SPECS:
        print(f"Auditing {spec['key']} ...")
        if spec["layout"] == "folder_class":
            results.append(audit_folder_class(spec))
        else:
            results.append(audit_yolo(spec))

    # ---- dataset_statistics.csv ----
    with open(REPORTS / "dataset_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "layout", "splits", "images_used", "label_files",
                    "boxes", "empty_labels", "imgs_without_label",
                    "labels_without_img", "corrupt_imgs_sampled",
                    "invalid_boxes", "boxes_out_of_range"])
        for r in results:
            w.writerow([r["key"], r["layout"], r["splits"], r["n_img"], r["n_lbl"],
                        r["n_box"], r["empty_labels"], r["imgs_no_label"],
                        r["labels_no_img"], r["bad_imgs_sampled"],
                        r["invalid_box"], r["out_of_range"]])

    # ---- class_distribution.csv (target-class level) ----
    tgt_box = Counter(); tgt_img = Counter()
    rows_cd = []
    for r in results:
        for cid, n in r["cls_box"].items():
            tgt_box[cid] += n
        for cid, n in r["cls_img"].items():
            tgt_img[cid] += n
        # per-dataset contribution to target classes
        for cid in sorted(r["cls_box"]):
            rows_cd.append([r["key"], cid, TARGET_CLASSES[cid],
                            r["cls_img"].get(cid, 0), r["cls_box"].get(cid, 0)])
    with open(REPORTS / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "target_class_id", "target_class_name",
                    "images_with_class", "boxes"])
        for row in rows_cd:
            w.writerow(row)
        w.writerow([])
        w.writerow(["TOTAL", "target_class_id", "target_class_name",
                    "images_with_class", "boxes"])
        for cid in range(len(TARGET_CLASSES)):
            w.writerow(["TOTAL", cid, TARGET_CLASSES[cid],
                        tgt_img.get(cid, 0), tgt_box.get(cid, 0)])

    # ---- DATASET_AUDIT_REPORT.md ----
    L = ["# DATASET_AUDIT_REPORT — PadiShield", "",
         "Audit otomatis terhadap `data/raw` (read-only). "
         "Tujuh kelas target internal:", "",
         "```", *[f"{i} {n}" for i, n in enumerate(TARGET_CLASSES)], "```", "",
         "## Ringkasan per dataset (yang dipakai)", "",
         "| Dataset | Layout | Splits | Gambar dipakai | Label | Box | "
         "Label kosong | Img tanpa label | Label tanpa img | Corrupt(sampel) | Box invalid | Box out-of-range |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append("| {k} | {ly} | {sp} | {ni} | {nl} | {nb} | {el} | {inl} | {lni} | {bi} | {ib} | {oor} |".format(
            k=r["key"], ly=r["layout"], sp=r["splits"], ni=r["n_img"], nl=r["n_lbl"],
            nb=r["n_box"], el=r["empty_labels"], inl=r["imgs_no_label"],
            lni=r["labels_no_img"], bi=r["bad_imgs_sampled"], ib=r["invalid_box"],
            oor=r["out_of_range"]))
    L += ["", "## Pemetaan kelas asli -> target per dataset", ""]
    for r in results:
        L.append(f"### {r['key']}")
        L.append("")
        if r["layout"] == "folder_class":
            L.append("Kelas ditentukan oleh **nama folder** (id numerik tidak konsisten).")
            L.append("")
            L.append("| Folder | Label files | Boxes | Clean imgs | -> Target |")
            L.append("|---|---|---|---|---|")
            for folder, d in r["per_folder"].items():
                tgt = TARGET_CLASSES[d["target"]] if d["target"] is not None else "DIBUANG"
                L.append(f"| {folder} | {d['labels']} | {d['boxes']} | {d['clean_imgs']} | {tgt} |")
        else:
            L.append("| Orig ID | Orig name | Img w/ class | Boxes | -> Target |")
            L.append("|---|---|---|---|---|")
            on = r["orig_names"]; idm = r["id_map"]
            ob = r.get("orig_box", {}); oi = r.get("orig_img", {})
            for oid in sorted(on.keys()):
                tgt = TARGET_CLASSES[idm[str(oid)]] if str(oid) in idm else "DIBUANG"
                L.append(f"| {oid} | {on[oid]} | {oi.get(oid,0)} | {ob.get(oid,0)} | {tgt} |")
        if r["dropped"]:
            L.append("")
            L.append("Dibuang: " + "; ".join(f"`{k}` ({v})" for k, v in r["dropped"].items()))
        if r.get("sample_issues"):
            L.append("")
            L.append("Contoh isu label: " + " / ".join(r["sample_issues"][:5]))
        L.append("")

    L += ["## Distribusi kelas target (gabungan akhir)", "",
          "| ID | Kelas | Gambar (≈) | Boxes |", "|---|---|---|---|"]
    for cid in range(len(TARGET_CLASSES)):
        L.append(f"| {cid} | {TARGET_CLASSES[cid]} | {tgt_img.get(cid,0)} | {tgt_box.get(cid,0)} |")
    L += ["", "_Catatan: 'Gambar (≈)' menjumlahkan gambar-yang-memuat-kelas antar "
          "dataset; gambar multi-kelas (dataset 04) dapat terhitung di >1 kelas._", ""]

    L += ["## Dataset yang DIKECUALIKAN dari processed (raw tetap utuh)", ""]
    for k, why in EXCLUDED_DATASETS.items():
        L.append(f"- **{k}** — {why}")
    L.append("")
    L += ["## Catatan kualitas data penting", "",
          "- Dataset 02: numerik class-id TIDAK konsisten antar folder "
          "(Leaf smut & Rice Tungro sama-sama id 3) -> gunakan nama folder.",
          "- Dataset 02: gambar bersih di `Original images/`, overlay visual di "
          "`.../visuals/` (tidak dipakai untuk training).",
          "- Dataset 06: `classes.txt` harus diabaikan saat membaca label.",
          "- bacterial_leaf_blight adalah kelas paling langka (hanya dari dataset 01).",
          ""]
    (REPORTS / "DATASET_AUDIT_REPORT.md").write_text("\n".join(L), encoding="utf-8")

    print("Audit selesai. Output di reports/.")
    print(f"  Total box per target: {dict(sorted(tgt_box.items()))}")


if __name__ == "__main__":
    main()
