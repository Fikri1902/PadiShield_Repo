#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tungro_merged_dataset.py — Dataset turunan exp4.

Menggabungkan banyak box lesi tungro (class 2) yang kecil & padat menjadi box
"region tungro" per-klaster yang lebih besar (lebih mudah dideteksi). Kelas lain
tidak diubah. SPLIT SAMA PERSIS dengan data/processed (baca dari label yang sudah
ter-split) → TIDAK ada leakage; hanya bentuk label tungro yang berubah.

Output: data/processed_v2_tungro_merged/{images,labels}/{train,val,test} +
padishield_v2.yaml

Catatan integritas: ini redefinisi tugas tungro menjadi deteksi region/area,
didokumentasikan di reports/DECISION_LOG.md. Bukan leakage, bukan manipulasi
split.
"""

from __future__ import annotations
import os, shutil
from collections import Counter
from pathlib import Path

from padishield_config import PROCESSED, PROJECT_ROOT, TARGET_CLASSES, IMG_EXTS

TUNGRO_ID = 2
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed_v2_tungro_merged"
PAD = 0.10             # margin inflasi (frac gambar): gabung lesi jadi ~1 box region/daun
MAX_ITER = 60


def parse(path: Path):
    rows = []
    for ln in path.read_text(errors="ignore").splitlines():
        p = ln.split()
        if len(p) < 5 or not p[0].lstrip("-").isdigit():
            continue
        try:
            c = int(p[0]); x, y, w, h = map(float, p[1:5])
        except ValueError:
            continue
        rows.append((c, x, y, w, h))
    return rows


def merge_cluster(boxes, pad=PAD, max_iter=MAX_ITER):
    """boxes: list (xc,yc,w,h) -> list of merged (xc,yc,w,h) per klaster."""
    rects = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for (x, y, w, h) in boxes]
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        used = [False] * len(rects)
        out = []
        for i in range(len(rects)):
            if used[i]:
                continue
            a = list(rects[i]); used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                b = rects[j]
                # interseksi setelah masing-masing diinflasi sebesar pad
                if (a[0] - pad < b[2] + pad and b[0] - pad < a[2] + pad and
                        a[1] - pad < b[3] + pad and b[1] - pad < a[3] + pad):
                    a = [min(a[0], b[0]), min(a[1], b[1]),
                         max(a[2], b[2]), max(a[3], b[3])]
                    used[j] = True; changed = True
            out.append(a)
        rects = out
    res = []
    for x1, y1, x2, y2 in rects:
        x1 = min(max(x1, 0.0), 1.0); y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0); y2 = min(max(y2, 0.0), 1.0)
        w = max(x2 - x1, 1e-4); h = max(y2 - y1, 1e-4)
        res.append(((x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return res


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=PAD,
                    help="margin inflasi; besar => box lebih besar")
    ap.add_argument("--union", action="store_true",
                    help="satu box per gambar (union semua lesi)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="folder output dataset")
    args = ap.parse_args()
    pad = 5.0 if args.union else args.pad      # pad sangat besar = union seluruh gambar
    OUT = Path(args.out)
    mode = "UNION (1 box/gambar)" if args.union else f"pad={args.pad}"
    print(f"Build tungro-merged: {mode} -> {OUT.name}")

    before = Counter(); after = Counter()
    img_sizes = []  # area fraction of merged tungro boxes (sanity)
    n_imgs = 0
    for split in ("train", "val", "test"):
        lbl_dir = PROCESSED / "labels" / split
        img_dir = PROCESSED / "images" / split
        for lf in lbl_dir.glob("*.txt"):
            rows = parse(lf)
            tung = [(x, y, w, h) for (c, x, y, w, h) in rows if c == TUNGRO_ID]
            other = [(c, x, y, w, h) for (c, x, y, w, h) in rows if c != TUNGRO_ID]
            before[split] += len(tung)
            out_lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (c, x, y, w, h) in other]
            if tung:
                merged = merge_cluster(tung, pad=pad)
                after[split] += len(merged)
                for (x, y, w, h) in merged:
                    out_lines.append(f"{TUNGRO_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    img_sizes.append(w * h)
            # tulis label baru
            (OUT / "labels" / split / lf.name).parent.mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split / lf.name).write_text("\n".join(out_lines) + "\n",
                                                          encoding="utf-8")
            # hardlink gambar pasangannya
            img = None
            for ext in IMG_EXTS:
                cand = img_dir / f"{lf.stem}{ext}"
                if cand.exists():
                    img = cand; break
            if img is None:
                for c in img_dir.glob(f"{lf.stem}.*"):
                    if c.suffix.lower() in IMG_EXTS:
                        img = c; break
            if img:
                link(img, OUT / "images" / split / img.name)
                n_imgs += 1

    # yaml
    yaml = [
        "# PadiShield v2 — tungro lesion boxes merged into region boxes (exp4).",
        "# Split identik dengan data/processed (no leakage). Hanya label tungro diubah.",
        f"path: {OUT.as_posix()}",
        "train: images/train", "val: images/val", "test: images/test", "",
        f"nc: {len(TARGET_CLASSES)}", "names:",
        *[f"  {i}: {n}" for i, n in enumerate(TARGET_CLASSES)], "",
    ]
    (OUT / "padishield_v2.yaml").write_text("\n".join(yaml), encoding="utf-8")

    print("=== TUNGRO MERGE SUMMARY ===")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: tungro boxes {before[sp]:6d} -> {after[sp]:5d} "
              f"(reduksi {100*(1-after[sp]/max(before[sp],1)):.1f}%)")
    if img_sizes:
        import statistics as st
        print(f"  merged tungro box area (frac gambar): median "
              f"{st.median(img_sizes):.3f}, max {max(img_sizes):.3f}")
        big = sum(1 for a in img_sizes if a > 0.6) / len(img_sizes)
        print(f"  fraksi box 'raksasa' (>60% gambar): {big:.2f}")
    print(f"  images linked: {n_imgs}")
    print(f"  yaml: {OUT/'padishield_v2.yaml'}")


if __name__ == "__main__":
    main()
