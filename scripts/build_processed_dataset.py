#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_processed_dataset.py — Bangun dataset gabungan YOLO di data/processed.

Tidak mengubah data/raw. Gambar di-hardlink (fallback copy) dengan nama unik;
label ditulis ulang dengan class-id target. Mencegah leakage via:
  * exact-duplicate (md5) -> kelompok sama,
  * near-duplicate average-hash (aHash 8x8) -> kelompok sama,
  * seluruh anggota kelompok masuk split yang sama.
Split 70/15/15 distratifikasi pada kelas paling langka di tiap kelompok agar
semua kelas hadir di train/val/test.

Output:
  data/processed/images/{train,val,test}, labels/{train,val,test}
  data/processed/padishield.yaml
  data/manifests/master_manifest.csv
  data/manifests/class_mapping.csv
"""

from __future__ import annotations
import csv, hashlib, os, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from padishield_config import (RAW, PROCESSED, MANIFESTS, TARGET_CLASSES,
                               NAME_TO_ID, IMG_EXTS, DATASET_SPECS, SEED)

SPLIT_RATIO = (0.70, 0.15, 0.15)   # train, val, test


# --------------------------------------------------------------------------- #
# Pengumpulan item sumber
# --------------------------------------------------------------------------- #

def parse_boxes(path: Path):
    out = []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return out
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
            continue
        try:
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        # clamp ke [0,1] untuk keamanan (tanpa mengubah raw)
        x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0); h = min(max(h, 0.0), 1.0)
        if w <= 0 or h <= 0:
            continue
        out.append((int(parts[0]), x, y, w, h))
    return out


def find_image_for(stem: str, image_dir: Path):
    for ext in IMG_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # case-insensitive fallback
    for p in image_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def collect_yolo(spec):
    """Kumpulkan item dari dataset berformat YOLO. Yield dict per gambar."""
    root = RAW / spec["root"]
    id_map = spec["id_map"]
    layout = spec["layout"]
    pairs = []  # (split, image_dir, label_dir)
    if layout == "yolo_split":
        for sp in spec["splits"]:
            pairs.append((sp, root / "images" / sp, root / "labels" / sp))
    elif layout == "roboflow_split":
        for sp in spec["splits"]:
            d = root / sp
            if d.exists():
                pairs.append((sp, d / "images", d / "labels"))
    elif layout == "yolo_flat":
        pairs.append(("", root / "images", root / "labels"))

    items = []
    for sp, idir, ldir in pairs:
        if not ldir.exists():
            continue
        for lf in ldir.glob("*.txt"):
            if lf.name.lower() == "classes.txt":
                continue
            boxes = parse_boxes(lf)
            kept = [(id_map[str(c)], x, y, w, h) for (c, x, y, w, h) in boxes
                    if str(c) in id_map]
            if not kept:
                continue  # lewati background / hanya kelas dibuang
            img = find_image_for(lf.stem, idir)
            if img is None:
                continue
            items.append({
                "source_dataset": spec["key"], "original_split": sp or "flat",
                "source_image": img, "source_label": lf,
                "boxes": kept,
                "orig_ids": sorted({c for c, *_ in boxes}),
            })
    return items


def collect_folder_class(spec):
    """Dataset 02: kelas = nama folder; gambar bersih dari Original images/."""
    root = RAW / spec["root"]
    clean_parent = root / spec["clean_images_dir"]
    lbl_parent = root / spec["labels_parent_dir"]
    items = []
    for folder, tid in spec["folder_map"].items():
        ldir = lbl_parent / folder / spec["labels_subdir"]
        cdir = clean_parent / folder
        if not ldir.exists() or not cdir.exists():
            continue
        for lf in ldir.glob("*.txt"):
            boxes = parse_boxes(lf)
            if not boxes:
                continue
            kept = [(tid, x, y, w, h) for (_, x, y, w, h) in boxes]  # folder = truth
            img = find_image_for(lf.stem, cdir)
            if img is None:
                continue
            items.append({
                "source_dataset": spec["key"], "original_split": folder,
                "source_image": img, "source_label": lf,
                "boxes": kept, "orig_ids": ["folder:" + folder],
            })
    return items


# --------------------------------------------------------------------------- #
# Hashing untuk dedup / leakage
# --------------------------------------------------------------------------- #

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash_of(path: Path) -> str:
    try:
        im = Image.open(path).convert("L").resize((8, 8))
        px = list(im.getdata())
        avg = sum(px) / len(px)
        bits = "".join("1" if p >= avg else "0" for p in px)
        return f"{int(bits, 2):016x}"
    except Exception:
        return "na"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def safe_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    rng = random.Random(SEED)
    print("Mengumpulkan item sumber ...")
    items = []
    for spec in DATASET_SPECS:
        if spec["layout"] == "folder_class":
            got = collect_folder_class(spec)
        else:
            got = collect_yolo(spec)
        # cap (mis. rat 06)
        cap = spec.get("cap_images")
        if cap and len(got) > cap:
            rng.shuffle(got)
            dropped = len(got) - cap
            got = got[:cap]
            print(f"  {spec['key']}: cap {cap} (buang {dropped} gambar)")
        print(f"  {spec['key']}: {len(got)} gambar dengan kelas target")
        items.extend(got)

    print(f"Total kandidat gambar: {len(items)}")

    print("Menghitung hash (md5 + aHash) untuk dedup & anti-leakage ...")
    for it in items:
        it["md5"] = md5_of(it["source_image"])
        it["ahash"] = ahash_of(it["source_image"])

    # Exact dedup by md5 (keep first), tetap catat yang dibuang.
    seen_md5 = {}
    deduped, exact_dups = [], 0
    for it in items:
        if it["md5"] in seen_md5:
            exact_dups += 1
            continue
        seen_md5[it["md5"]] = it
        deduped.append(it)
    print(f"Exact duplicates dibuang: {exact_dups}; tersisa {len(deduped)}")

    # Kelompok anti-leakage: group key = aHash (near-dup). md5 jadi duplicate_group.
    groups = defaultdict(list)
    for it in deduped:
        key = it["ahash"] if it["ahash"] != "na" else "md5:" + it["md5"]
        groups[key].append(it)

    # Frekuensi kelas (per gambar) untuk menentukan kelas terlangka.
    img_class_freq = Counter()
    for it in deduped:
        for c in {b[0] for b in it["boxes"]}:
            img_class_freq[c] += 1
    rarity = {c: img_class_freq[c] for c in range(len(TARGET_CLASSES))}

    def group_stratum(grp):
        present = set()
        for it in grp:
            present |= {b[0] for b in it["boxes"]}
        # kelas paling langka yang hadir
        return min(present, key=lambda c: rarity.get(c, 1 << 30))

    # Bagi per stratum agar tiap kelas terdistribusi.
    strata = defaultdict(list)
    for key, grp in groups.items():
        strata[group_stratum(grp)].append((key, grp))

    assignment = {}  # group key -> split
    tr, va, te = SPLIT_RATIO
    for cls, glist in strata.items():
        rng.shuffle(glist)
        n = len(glist)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        for i, (key, _) in enumerate(glist):
            if i < n_tr:
                assignment[key] = "train"
            elif i < n_tr + n_va:
                assignment[key] = "val"
            else:
                assignment[key] = "test"

    # Tulis gambar+label, bangun manifest.
    print("Menulis processed dataset ...")
    for sp in ("train", "val", "test"):
        (PROCESSED / "images" / sp).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / sp).mkdir(parents=True, exist_ok=True)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    man_path = MANIFESTS / "master_manifest.csv"
    counts = defaultdict(lambda: Counter())   # split -> class -> boxes
    img_counts = Counter()                     # split -> images
    name_collide = Counter()
    link_modes = Counter()

    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["processed_image", "processed_label", "source_dataset",
                    "source_image", "source_label", "original_split",
                    "original_class_ids", "target_class_ids", "target_class_names",
                    "group_id", "duplicate_group", "final_split"])
        for it in deduped:
            key = it["ahash"] if it["ahash"] != "na" else "md5:" + it["md5"]
            split = assignment[key]
            # nama unik: <dataset>__<stem><ext>
            stem = it["source_image"].stem.replace(" ", "_")
            ext = it["source_image"].suffix.lower()
            base = f"{it['source_dataset']}__{stem}"
            name = base + ext
            if name_collide[name]:
                name = f"{base}_{name_collide[name]}{ext}"
            name_collide[base + ext] += 1

            dst_img = PROCESSED / "images" / split / name
            dst_lbl = PROCESSED / "labels" / split / (Path(name).stem + ".txt")
            link_modes[safe_link(it["source_image"], dst_img)] += 1

            tgt_ids = sorted({b[0] for b in it["boxes"]})
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (c, x, y, w, h) in it["boxes"]]
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            for c, *_ in it["boxes"]:
                counts[split][c] += 1
            img_counts[split] += 1

            w.writerow([
                f"images/{split}/{name}", f"labels/{split}/{dst_lbl.name}",
                it["source_dataset"], str(it["source_image"]), str(it["source_label"]),
                it["original_split"], "|".join(map(str, it["orig_ids"])),
                "|".join(map(str, tgt_ids)),
                "|".join(TARGET_CLASSES[c] for c in tgt_ids),
                key, it["md5"], split,
            ])

    # class_mapping.csv
    with open(MANIFESTS / "class_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_class_id", "target_class_name", "sources"])
        srcmap = {
            0: "01:Rice_Blast; 04:leaf/neck/node_blast",
            1: "01:Bacterial_Blight",
            2: "02:Rice Tungro folder",
            3: "04:stem_borer_larva/moth",
            4: "04:deadheart",
            5: "04:bph_insect",
            6: "06:Rat; 07:Rat-in-a-rice-field",
        }
        for cid, name in enumerate(TARGET_CLASSES):
            w.writerow([cid, name, srcmap.get(cid, "")])

    # padishield.yaml
    yaml = [
        "# PadiShield merged dataset (auto-generated, do not edit raw).",
        f"path: {PROCESSED.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(TARGET_CLASSES)}",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(TARGET_CLASSES)],
        "",
    ]
    (PROCESSED / "padishield.yaml").write_text("\n".join(yaml), encoding="utf-8")

    # Ringkasan
    print("\n=== RINGKASAN PROCESSED ===")
    print(f"Link modes: {dict(link_modes)}")
    total_imgs = sum(img_counts.values())
    print(f"Total images: {total_imgs}  (train {img_counts['train']}, "
          f"val {img_counts['val']}, test {img_counts['test']})")
    print(f"{'class':24s} {'train':>7s}{'val':>7s}{'test':>7s}  (boxes)")
    for c in range(len(TARGET_CLASSES)):
        print(f"{TARGET_CLASSES[c]:24s} {counts['train'][c]:7d}{counts['val'][c]:7d}"
              f"{counts['test'][c]:7d}")
    # images per class per split
    print("\nImages-with-class per split:")
    img_cls = {sp: Counter() for sp in ("train", "val", "test")}
    with open(man_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sp = row["final_split"]
            for nm in row["target_class_names"].split("|"):
                if nm:
                    img_cls[sp][nm] += 1
    print(f"{'class':24s} {'train':>7s}{'val':>7s}{'test':>7s}")
    for c in range(len(TARGET_CLASSES)):
        nm = TARGET_CLASSES[c]
        print(f"{nm:24s} {img_cls['train'][nm]:7d}{img_cls['val'][nm]:7d}{img_cls['test'][nm]:7d}")
    print("\nYAML:", PROCESSED / "padishield.yaml")
    print("Manifest:", man_path)


if __name__ == "__main__":
    main()
