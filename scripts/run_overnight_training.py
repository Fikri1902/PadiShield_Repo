#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_overnight_training.py — Pipeline training + evaluasi PadiShield (YOLO).

Subcommands:
    --smoke       smoke test cepat (fraction kecil, sedikit epoch) untuk validasi pipeline
    --baseline    training baseline penuh dengan early stopping + fallback OOM
    --evaluate W  evaluasi checkpoint W (default: best.pt baseline) pada test split
    --all         smoke -> baseline -> evaluate

Robust: logging stdout/stderr ke runs/logs/, resume, fallback batch/imgsz saat OOM,
seed deterministik, tidak butuh input interaktif.

Artefak evaluasi -> reports/ dan outputs/.
"""

from __future__ import annotations
import argparse, csv, datetime as dt, json, shutil, sys, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from padishield_config import PROCESSED, REPORTS, TARGET_CLASSES, SEED  # noqa: E402

DATA_YAML = PROCESSED / "padishield.yaml"
DATA_YAML_V2 = PROJECT_ROOT / "data" / "processed_v2_tungro_merged" / "padishield_v2.yaml"
PROCESSED_V2 = PROJECT_ROOT / "data" / "processed_v2_tungro_merged"
DATA_YAML_V3 = PROJECT_ROOT / "data" / "processed_v3_tungro_union" / "padishield_v2.yaml"
PROCESSED_V3 = PROJECT_ROOT / "data" / "processed_v3_tungro_union"
RUNS = PROJECT_ROOT / "runs"
LOGS = RUNS / "logs"
OUTPUTS = PROJECT_ROOT / "outputs"
DETECT_PROJECT = RUNS / "detect"

BASELINE_NAME = "baseline_yolo11s"

# Konfigurasi baseline (dipilih untuk RTX 4070 Ti 12GB, hasil dalam semalam).
BASE_MODEL = "yolo11s.pt"
FALLBACK_MODEL = "yolov8s.pt"
IMGSZ = 640
EPOCHS = 100
PATIENCE = 20
BATCH = 16
WORKERS = 8
DEVICE = 0


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Tee:
    """Tulis ke stdout asli sekaligus file log."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.stdout = sys.__stdout__
    def write(self, s):
        self.f.write(s); self.f.flush()
        try:
            self.stdout.write(s)
        except Exception:
            pass
    def flush(self):
        self.f.flush()


def banner(msg):
    print("\n" + "=" * 70)
    print(f"[{now()}] {msg}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def run_smoke():
    from ultralytics import YOLO
    banner("SMOKE TEST — validasi pipeline (fraction 0.05, 3 epoch)")
    model_name = BASE_MODEL
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Gagal load {model_name}: {e}; fallback {FALLBACK_MODEL}")
        model_name = FALLBACK_MODEL
        model = YOLO(model_name)
    try:
        model.train(
            data=str(DATA_YAML), epochs=3, imgsz=IMGSZ, batch=8, fraction=0.05,
            workers=4, device=DEVICE, seed=SEED, deterministic=True, cache=False,
            project=str(DETECT_PROJECT), name="smoke", exist_ok=True, verbose=True,
            plots=False, val=True,
        )
        print(f"\n[OK] Smoke test selesai dengan {model_name}. Pipeline valid.")
        return True, model_name
    except Exception as e:
        print(f"[FAIL] Smoke test gagal: {e}")
        traceback.print_exc()
        return False, model_name


# --------------------------------------------------------------------------- #
# Baseline dengan fallback OOM
# --------------------------------------------------------------------------- #

def run_baseline(model_name=BASE_MODEL, resume=False):
    from ultralytics import YOLO
    banner(f"BASELINE TRAINING — {model_name}, imgsz={IMGSZ}, epochs={EPOCHS}")
    save_dir = DETECT_PROJECT / BASELINE_NAME
    last_ckpt = save_dir / "weights" / "last.pt"

    # Resume bila diminta & checkpoint ada.
    if resume and last_ckpt.exists():
        print(f"Resume dari {last_ckpt}")
        model = YOLO(str(last_ckpt))
        try:
            model.train(resume=True)
            return save_dir
        except Exception as e:
            print(f"Resume gagal ({e}); mulai baru.")

    attempts = [
        dict(batch=BATCH, imgsz=IMGSZ, cache=False),
        dict(batch=8, imgsz=IMGSZ, cache=False),
        dict(batch=8, imgsz=512, cache=False),
        dict(batch=4, imgsz=512, cache=False),
    ]
    last_err = None
    for i, cfg in enumerate(attempts):
        try:
            print(f"\n--- Attempt {i+1}: {cfg} ---")
            model = YOLO(model_name)
            model.train(
                data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE,
                imgsz=cfg["imgsz"], batch=cfg["batch"], workers=WORKERS,
                device=DEVICE, seed=SEED, deterministic=True, cache=cfg["cache"],
                optimizer="auto", close_mosaic=10, fliplr=0.5, flipud=0.0,
                project=str(DETECT_PROJECT), name=BASELINE_NAME, exist_ok=True,
                verbose=True, plots=True, val=True,
            )
            print(f"\n[OK] Baseline selesai (attempt {i+1}, {cfg}).")
            return save_dir
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"[OOM/CUDA] attempt {i+1} gagal: {e}\nTurunkan konfigurasi ...")
                import torch, gc
                gc.collect(); torch.cuda.empty_cache()
                continue
            print(f"[ERROR] attempt {i+1}: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            last_err = e
            print(f"[ERROR] attempt {i+1}: {e}")
            traceback.print_exc()
            # coba fallback model jika model utama bermasalah
            if model_name != FALLBACK_MODEL and i == 0:
                print(f"Beralih ke fallback model {FALLBACK_MODEL}")
                model_name = FALLBACK_MODEL
            continue
    raise RuntimeError(f"Semua attempt baseline gagal. Error terakhir: {last_err}")


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #

def _test_counts(test_labels_dir=None):
    """Jumlah box & gambar per kelas di test split."""
    from collections import Counter
    box = Counter(); img = Counter()
    tdir = test_labels_dir or (PROCESSED / "labels" / "test")
    for t in Path(tdir).glob("*.txt"):
        seen = set()
        for ln in t.read_text(errors="ignore").splitlines():
            ln = ln.strip()
            if ln:
                c = int(ln.split()[0]); box[c] += 1; seen.add(c)
        for c in seen:
            img[c] += 1
    return box, img


def run_hires_experiment(imgsz=960, batch=8, epochs=80, patience=20,
                         name="exp2_yolo11s_hires"):
    """Eksperimen resolusi tinggi untuk menyelamatkan deteksi lesi kecil tungro
    (recall baseline sangat rendah karena box ~9px@640). Fallback OOM otomatis."""
    from ultralytics import YOLO
    banner(f"HI-RES EXPERIMENT '{name}' — {BASE_MODEL}, imgsz={imgsz}, "
           f"batch={batch}, epochs={epochs}")
    save_dir = DETECT_PROJECT / name
    # rantai fallback OOM: turunkan batch, lalu resolusi
    attempts = [
        dict(batch=batch, imgsz=imgsz),
        dict(batch=max(batch // 2, 2), imgsz=imgsz),
        dict(batch=2, imgsz=imgsz),
        dict(batch=4, imgsz=int(imgsz * 0.8) // 32 * 32),
    ]
    last_err = None
    for i, cfg in enumerate(attempts):
        try:
            print(f"\n--- {name} attempt {i+1}: {cfg} ---")
            model = YOLO(BASE_MODEL)
            model.train(
                data=str(DATA_YAML), epochs=epochs, patience=patience,
                imgsz=cfg["imgsz"], batch=cfg["batch"], workers=WORKERS,
                device=DEVICE, seed=SEED, deterministic=True, cache=False,
                optimizer="auto", close_mosaic=10, fliplr=0.5, flipud=0.0,
                project=str(DETECT_PROJECT), name=name, exist_ok=True,
                verbose=True, plots=True, val=True,
            )
            print(f"\n[OK] {name} selesai (attempt {i+1}, {cfg}).")
            return save_dir, cfg["imgsz"]
        except RuntimeError as e:
            last_err = e
            print(f"[OOM/CUDA] {name} attempt {i+1}: {e}; turunkan konfigurasi ...")
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
            continue
    raise RuntimeError(f"{name} gagal semua attempt. Error terakhir: {last_err}")


def run_experiment4(epochs=100, patience=20, data_yaml=DATA_YAML_V2,
                    name="exp4_tungro_merged"):
    """Eksperimen 4/4b: latih ulang pada dataset tungro yang labelnya digabung
    (region-level). Konfigurasi = baseline @640 (stabil, terbukti)."""
    from ultralytics import YOLO
    banner(f"EXPERIMENT '{name}' — {BASE_MODEL} @640 di {Path(data_yaml).parent.name} "
           f"(epochs={epochs})")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset belum dibuat: {data_yaml}. "
                                f"Jalankan scripts/build_tungro_merged_dataset.py")
    save_dir = DETECT_PROJECT / name
    for i, cfg in enumerate([dict(batch=BATCH, imgsz=IMGSZ),
                             dict(batch=8, imgsz=IMGSZ)]):
        try:
            print(f"\n--- {name} attempt {i+1}: {cfg} ---")
            model = YOLO(BASE_MODEL)
            model.train(
                data=str(data_yaml), epochs=epochs, patience=patience,
                imgsz=cfg["imgsz"], batch=cfg["batch"], workers=WORKERS,
                device=DEVICE, seed=SEED, deterministic=True, cache=False,
                optimizer="auto", close_mosaic=10, fliplr=0.5, flipud=0.0,
                project=str(DETECT_PROJECT), name=name, exist_ok=True,
                verbose=True, plots=True, val=True,
            )
            print(f"\n[OK] {name} selesai (attempt {i+1}, {cfg}).")
            return save_dir
        except RuntimeError as e:
            msg = str(e).lower()
            print(f"[ERROR] {name} attempt {i+1}: {e}")
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
            if "already mapped" in msg or "device-side assert" in msg:
                raise RuntimeError("CUDA context rusak; butuh proses baru.") from e
            continue
    raise RuntimeError(f"{name} gagal.")


def run_evaluate(weights: Path, imgsz=IMGSZ, eval_name="eval_test", tag="",
                 data_yaml=DATA_YAML, test_labels_dir=None):
    from ultralytics import YOLO
    banner(f"EVALUASI pada TEST split — {weights} (imgsz={imgsz}, data={Path(data_yaml).name})")
    if not Path(weights).exists():
        raise FileNotFoundError(f"Checkpoint tidak ada: {weights}")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=imgsz, batch=16, device=DEVICE,
        workers=WORKERS, plots=True, project=str(DETECT_PROJECT), name=eval_name,
        exist_ok=True, verbose=True,
    )
    save_dir = Path(metrics.save_dir)

    box = metrics.box
    # metrik keseluruhan
    mp = float(box.mp); mr = float(box.mr)
    map50 = float(box.map50); map5095 = float(box.map)
    f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    # per kelas
    test_box, test_img = _test_counts(test_labels_dir)
    ap_idx = list(box.ap_class_index)
    per_class = []
    for i, ci in enumerate(ap_idx):
        p = float(box.p[i]); r = float(box.r[i])
        ap50 = float(box.ap50[i]); ap = float(box.ap[i])
        cls_f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class.append({
            "class_id": int(ci), "class_name": TARGET_CLASSES[int(ci)],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(cls_f1, 4),
            "ap50": round(ap50, 4), "ap50_95": round(ap, 4),
            "test_boxes": test_box.get(int(ci), 0),
            "test_images": test_img.get(int(ci), 0),
        })

    # best confidence by F1 (dari kurva jika tersedia)
    best_conf = None
    try:
        # curves_results: list of [x, y, xlabel, ylabel]; cari F1 curve
        for x, y, xl, yl in box.curves_results:
            if "F1" in yl or "f1" in yl:
                import numpy as np
                yv = np.array(y)
                mean_f1 = yv.mean(axis=0) if yv.ndim > 1 else yv
                bi = int(mean_f1.argmax())
                best_conf = round(float(np.array(x)[bi]), 4)
                break
    except Exception:
        best_conf = None

    # speed
    speed = getattr(metrics, "speed", {}) or {}
    n_params = None
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        pass

    summary = {
        "evaluated_at": now(),
        "weights": str(weights),
        "data_yaml": str(DATA_YAML),
        "imgsz": imgsz,
        "precision": round(mp, 4), "recall": round(mr, 4), "f1": round(f1, 4),
        "map50": round(map50, 4), "map50_95": round(map5095, 4),
        "best_conf_by_f1": best_conf,
        "speed_ms_per_img": speed,
        "n_parameters": n_params,
        "eval_dir": str(save_dir),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    # final_metrics.csv
    with open(REPORTS / f"final_metrics{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["precision", "recall", "f1", "map50", "map50_95",
                  "best_conf_by_f1", "n_parameters"]:
            w.writerow([k, summary[k]])
        for k, v in (speed or {}).items():
            w.writerow([f"speed_{k}_ms", round(float(v), 3)])
    # per_class_metrics.csv
    with open(REPORTS / f"per_class_metrics{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision",
            "recall", "f1", "ap50", "ap50_95", "test_boxes", "test_images"])
        w.writeheader()
        for pc in per_class:
            w.writerow(pc)

    # copy kurva & confusion matrix ke reports/eval_plots
    plots_out = REPORTS / (f"eval_plots{tag}" if tag else "eval_plots")
    plots_out.mkdir(parents=True, exist_ok=True)
    for png in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
                "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png"]:
        src = save_dir / png
        if src.exists():
            shutil.copy2(src, plots_out / png)

    (REPORTS / f"eval_summary{tag}.json").write_text(
        json.dumps({"overall": summary, "per_class": per_class}, indent=2),
        encoding="utf-8")

    print("\n=== METRIK KESELURUHAN (TEST) ===")
    for k in ["precision", "recall", "f1", "map50", "map50_95", "best_conf_by_f1"]:
        print(f"  {k}: {summary[k]}")
    print("Per-kelas:")
    for pc in per_class:
        print(f"  {pc['class_name']:24s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} mAP50={pc['ap50']:.3f} mAP50-95={pc['ap50_95']:.3f} "
              f"(test_img={pc['test_images']})")
    return summary, per_class, save_dir


# --------------------------------------------------------------------------- #
# Prediksi contoh (TP / FP / FN sederhana)
# --------------------------------------------------------------------------- #

def save_prediction_examples(weights: Path, n=12, conf=0.25, imgsz=IMGSZ, name="pred"):
    from ultralytics import YOLO
    import random
    banner("CONTOH PREDIKSI pada test split")
    out = OUTPUTS / "prediction_examples"
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((PROCESSED / "images" / "test").glob("*"))
    random.Random(SEED).shuffle(imgs)
    sample = imgs[:n]
    model = YOLO(str(weights))
    res = model.predict(source=[str(p) for p in sample], conf=conf, imgsz=imgsz,
                        device=DEVICE, save=True, project=str(out), name=name,
                        exist_ok=True, verbose=False)
    print(f"Disimpan {len(sample)} contoh prediksi ke {out/name}")
    return out / name


# --------------------------------------------------------------------------- #
# Finalisasi artefak
# --------------------------------------------------------------------------- #

def finalize_artifacts(save_dir: Path, copy_best=True):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if best.exists() and copy_best:
        shutil.copy2(best, OUTPUTS / "best_model.pt")
    latest = OUTPUTS / "latest_results"
    latest.mkdir(parents=True, exist_ok=True)
    # pointer file
    (OUTPUTS / "latest_results" / "POINTER.txt").write_text(
        f"best.pt: {best}\nlast.pt: {last}\nrun_dir: {save_dir}\n"
        f"eval: {DETECT_PROJECT/'eval_test'}\n", encoding="utf-8")
    print(f"Artefak difinalisasi: {OUTPUTS/'best_model.pt'}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"run_{ts}.log"
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout
    print(f"Log: {log_path}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--evaluate", nargs="?", const="auto", default=None)
    ap.add_argument("--experiment2", action="store_true",
                    help="Train YOLO11s @960 untuk perbaiki recall tungro, lalu evaluasi.")
    ap.add_argument("--experiment3", action="store_true",
                    help="Train YOLO11s @1280 untuk perbaiki recall tungro, lalu evaluasi.")
    ap.add_argument("--experiment4", action="store_true",
                    help="Train YOLO11s @640 di dataset tungro-merged (region), lalu evaluasi.")
    ap.add_argument("--experiment4b", action="store_true",
                    help="Train YOLO11s @640 di dataset tungro-union (1 box/daun), lalu evaluasi.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.all or args.smoke:
        ok, model_name = run_smoke()
        if args.all and not ok:
            print("Smoke gagal — hentikan sebelum baseline."); sys.exit(1)

    save_dir = DETECT_PROJECT / BASELINE_NAME
    if args.all or args.baseline:
        save_dir = run_baseline(resume=args.resume)
        finalize_artifacts(save_dir)

    if args.all or args.evaluate is not None:
        if args.evaluate and args.evaluate != "auto":
            weights = Path(args.evaluate)
        else:
            weights = save_dir / "weights" / "best.pt"
        run_evaluate(weights, imgsz=IMGSZ, eval_name="eval_test", tag="")
        save_prediction_examples(weights, imgsz=IMGSZ)
        finalize_artifacts(save_dir)

    if args.experiment2:
        exp_dir, exp_imgsz = run_hires_experiment(
            imgsz=960, batch=8, epochs=80, patience=20, name="exp2_yolo11s_hires")
        exp_best = exp_dir / "weights" / "best.pt"
        finalize_artifacts(exp_dir, copy_best=False)
        run_evaluate(exp_best, imgsz=exp_imgsz, eval_name="eval_test_exp2", tag="_exp2")
        print(f"\n[Exp2] Selesai. Bandingkan reports/eval_summary.json (baseline) "
              f"vs reports/eval_summary_exp2.json untuk memilih model final.")

    if args.experiment3:
        exp_dir, exp_imgsz = run_hires_experiment(
            imgsz=1280, batch=4, epochs=60, patience=15, name="exp3_yolo11s_1280")
        exp_best = exp_dir / "weights" / "best.pt"
        finalize_artifacts(exp_dir, copy_best=False)
        run_evaluate(exp_best, imgsz=exp_imgsz, eval_name="eval_test_exp3", tag="_exp3")
        print(f"\n[Exp3] Selesai. Bandingkan reports/eval_summary.json (baseline) "
              f"vs reports/eval_summary_exp3.json untuk memilih model final.")

    if args.experiment4:
        exp_dir = run_experiment4()
        exp_best = exp_dir / "weights" / "best.pt"
        finalize_artifacts(exp_dir, copy_best=False)
        run_evaluate(exp_best, imgsz=IMGSZ, eval_name="eval_test_exp4", tag="_exp4",
                     data_yaml=DATA_YAML_V2,
                     test_labels_dir=PROCESSED_V2 / "labels" / "test")
        save_prediction_examples(exp_best, imgsz=IMGSZ, name="pred_exp4")
        print(f"\n[Exp4] Selesai. Lihat reports/eval_summary_exp4.json "
              f"(tungro = deteksi region, label digabung).")

    if args.experiment4b:
        exp_dir = run_experiment4(data_yaml=DATA_YAML_V3, name="exp4b_tungro_union")
        exp_best = exp_dir / "weights" / "best.pt"
        finalize_artifacts(exp_dir, copy_best=False)
        run_evaluate(exp_best, imgsz=IMGSZ, eval_name="eval_test_exp4b", tag="_exp4b",
                     data_yaml=DATA_YAML_V3,
                     test_labels_dir=PROCESSED_V3 / "labels" / "test")
        save_prediction_examples(exp_best, imgsz=IMGSZ, name="pred_exp4b")
        print(f"\n[Exp4b] Selesai. tungro = deteksi area/daun (1 box/gambar, lebih kasar).")

    print(f"\n[{now()}] SELESAI.")


if __name__ == "__main__":
    main()
