"""
TACYOLO A100 Colab Training Notebook
=====================================

Complete Google Colab script for training on NVIDIA A100 GPU (80GB VRAM).

Pipeline:
  1. Mount Google Drive & clone TACYOLO repo
  2. Download all 6 C-UAS & Armor datasets into one unified folder
  3. GPU-tile every image into 640x640 YOLO-ready tiles on A100
  4. Multi-teacher ensemble distillation (YOLOv8x + YOLO11x + RT-DETR-L)
     -> Weighted Box Fusion consensus labels
  5. Train compact YOLO11s student detector (batch=64 on A100 80GB)
  6. Export to TensorRT FP16 engine + benchmark latency/P99

Usage in Colab:
  Upload this file to Colab, then:
    !python colab_a100_train.py
  Or run cells individually by copying sections.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & GOOGLE DRIVE MOUNT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CELL 1: ENVIRONMENT SETUP & GOOGLE DRIVE MOUNT")
print("=" * 72)

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("✅ Google Drive mounted at /content/drive/MyDrive")
except Exception:
    print("⚠️  Not in Colab or Drive already mounted. Continuing...")

# Persistent project directories on Google Drive
DRIVE_ROOT     = Path("/content/drive/MyDrive/TACYOLO")
DRIVE_DATASETS = DRIVE_ROOT / "datasets" / "tactical_cuas_armor"
DRIVE_WEIGHTS  = DRIVE_ROOT / "trained_weights"
DRIVE_EXPORTS  = DRIVE_ROOT / "exports"

for d in [DRIVE_ROOT, DRIVE_DATASETS, DRIVE_WEIGHTS, DRIVE_EXPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# Clone or pull TACYOLO repo
REPO_DIR = Path("/content/TACYOLO")
REPO_URL = "https://github.com/devansh3108-2/tacyolo.git"  # Update with your repo URL

if REPO_DIR.exists():
    print("🔄 TACYOLO repo exists. Pulling latest...")
    os.system(f"cd {REPO_DIR} && git pull --rebase 2>/dev/null || true")
else:
    print(f"📥 Cloning TACYOLO repo...")
    os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))
os.chdir(str(REPO_DIR))

# Install dependencies
print("\n📦 Installing dependencies...")
os.system("pip install -q ultralytics ensemble-boxes kagglehub huggingface_hub pyyaml opencv-python-headless")

# Verify A100
import torch
print(f"\n🖥️  PyTorch: {torch.__version__}")
print(f"⚡ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"🎯 GPU: {gpu_name} ({gpu_mem:.0f} GB VRAM)")
    IS_A100 = "A100" in gpu_name
    if IS_A100:
        print("🔥 NVIDIA A100 DETECTED! Enabling maximum batch sizes.")
    else:
        print(f"⚠️  GPU is {gpu_name}, not A100. Adjusting batch size accordingly.")
else:
    IS_A100 = False
    print("❌ NO GPU DETECTED! Training will be extremely slow.")

print("\n✅ Cell 1 Complete: Environment Ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: DOWNLOAD ALL 6 DATASETS INTO ONE UNIFIED FOLDER
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CELL 2: DOWNLOADING ALL 6 C-UAS & ARMOR DATASETS")
print("=" * 72)

# Kaggle credentials (set yours here or upload kaggle.json)
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"]      = "your_key"

# Symlink unified dataset folder to Drive for persistence
LOCAL_DATASET = REPO_DIR / "datasets" / "tactical_cuas_armor"
if not LOCAL_DATASET.exists():
    LOCAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(DRIVE_DATASETS), str(LOCAL_DATASET))
    print(f"🔗 Symlinked local dataset -> Drive: {DRIVE_DATASETS}")

t0 = time.time()
os.system(f"python {REPO_DIR}/scripts/download_6_datasets.py --skip-tile --workers 3")
dl_time = time.time() - t0
print(f"\n⏱️  Downloads completed in {dl_time/60:.1f} minutes")

# Count what we have
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
total_src_imgs = sum(1 for p in DRIVE_DATASETS.rglob("*") if p.is_file() and p.suffix.lower() in img_exts)
print(f"📊 Total Source Images Across 6 Datasets: {total_src_imgs}")
print("\n✅ Cell 2 Complete: All 6 Datasets Downloaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: GPU-TILE ALL IMAGES (A100 CUDA TENSOR SLICING)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CELL 3: A100 GPU-ACCELERATED TILING (640x640)")
print("=" * 72)

t0 = time.time()
os.system(f"python {REPO_DIR}/scripts/prepare_cuas_armor_gpu.py --tile-size 640 --overlap 0.2")
tile_time = time.time() - t0

# Count generated tiles
train_tiles = len(list((DRIVE_DATASETS / "images" / "train").glob("*.*"))) if (DRIVE_DATASETS / "images" / "train").exists() else 0
val_tiles = len(list((DRIVE_DATASETS / "images" / "val").glob("*.*"))) if (DRIVE_DATASETS / "images" / "val").exists() else 0
print(f"\n⏱️  Tiling completed in {tile_time/60:.1f} minutes")
print(f"📊 Tiles Generated: {train_tiles} train + {val_tiles} val = {train_tiles + val_tiles} total")
print("\n✅ Cell 3 Complete: GPU Tiling Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4: MULTI-TEACHER ENSEMBLE DISTILLATION (OPTIONAL)
#          YOLOv8x + YOLO11x + RT-DETR-L -> WBF Consensus Labels
# ══════════════════════════════════════════════════════════════════════════════
ENABLE_DISTILLATION = False  # Set to True to run 3-teacher ensemble distillation

if ENABLE_DISTILLATION:
    print("=" * 72)
    print("  CELL 4: MULTI-TEACHER ENSEMBLE DISTILLATION")
    print("=" * 72)

    from tacyolo.distillation import (
        EnsembleDistillationPipeline,
        PipelineConfig,
    )

    distill_config = PipelineConfig.default_tri_teacher(
        input_dir=str(DRIVE_DATASETS / "images" / "train"),
        output_dir=str(DRIVE_ROOT / "distilled_dataset"),
        yolov8_weights="yolov8x.pt",
        yolo11_weights="yolo11x.pt",
        rtdetr_weights="rtdetr-l.pt",
    )
    distill_config.wbf_weights = [1.0, 1.2, 1.1]
    distill_config.iou_thr = 0.55
    distill_config.skip_box_thr = 0.35
    distill_config.epochs = 0  # Don't train inside pipeline, we do it below
    distill_config.device = "0"
    distill_config.batch_size = 32

    pipeline = EnsembleDistillationPipeline(distill_config)
    frame_dets = pipeline.run_stage_1_ensemble()
    fused = pipeline.run_stage_2_wbf(frame_dets)
    partition = pipeline.run_stage_3_partition(frame_dets, fused)

    TRAIN_YAML = partition.dataset_yaml_path
    print(f"\n✅ Distillation Dataset: {TRAIN_YAML}")
    print(f"📊 Train={partition.train_count}, Val={partition.val_count}, Test={partition.test_count}")
else:
    print("⏭️  Skipping multi-teacher distillation (using pre-tiled dataset directly).")
    TRAIN_YAML = DRIVE_DATASETS / "data.yaml"
    print(f"📄 Using existing dataset: {TRAIN_YAML}")

print("\n✅ Cell 4 Complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5: TRAIN YOLO11s STUDENT DETECTOR ON A100
#          Optimized for A100 80GB: batch=64, amp=True, close_mosaic=10
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CELL 5: A100 YOLO11s STUDENT TRAINING")
print("=" * 72)

from ultralytics import YOLO
import yaml

# ─── A100-Optimized Hyperparameters ──────────────────────────────────────────
STUDENT_MODEL  = "yolo11s.pt"
EPOCHS         = 100
IMGSZ          = 640
CLOSE_MOSAIC   = 10       # Disable mosaic in last 10 epochs for refinement
AMP            = True     # Automatic Mixed Precision

# A100 80GB -> batch=64 easily fits; non-A100 -> scale down
if IS_A100:
    BATCH_SIZE = 64
    WORKERS    = 16
else:
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3) if torch.cuda.is_available() else 4
    if gpu_mem_gb >= 24:
        BATCH_SIZE = 32
    elif gpu_mem_gb >= 12:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 8
    WORKERS = 8

# Check for resume from previous training
DRIVE_BEST = DRIVE_WEIGHTS / "best.pt"
DRIVE_LAST = DRIVE_WEIGHTS / "last.pt"
resume_flag = False

if DRIVE_LAST.exists():
    starting_weights = str(DRIVE_LAST)
    resume_flag = True
    print(f"🔄 RESUMING from Drive checkpoint: {DRIVE_LAST}")
elif DRIVE_BEST.exists():
    starting_weights = str(DRIVE_BEST)
    print(f"🌟 Fine-tuning from Drive best.pt: {DRIVE_BEST}")
else:
    starting_weights = STUDENT_MODEL
    print(f"🆕 Fresh training from pretrained: {STUDENT_MODEL}")

print(f"\n🏋️ Training Configuration:")
print(f"   Model: {starting_weights}")
print(f"   Dataset: {TRAIN_YAML}")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE} (A100={IS_A100})")
print(f"   Image Size: {IMGSZ}x{IMGSZ}")
print(f"   Close Mosaic: last {CLOSE_MOSAIC} epochs")
print(f"   AMP: {AMP}")
print(f"   Workers: {WORKERS}")
print(f"   Resume: {resume_flag}\n")

# Load model
model = YOLO(starting_weights)

# Project directory on Drive for persistence
PROJECT_DIR = str(DRIVE_ROOT / "runs")
RUN_NAME = "a100_yolo11s_cuas_armor"

t0 = time.time()
results = model.train(
    data=str(TRAIN_YAML),
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMGSZ,
    device="0",
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    workers=WORKERS,
    resume=resume_flag,
    cache="disk",
    amp=AMP,
    close_mosaic=CLOSE_MOSAIC,
    patience=25,
    cos_lr=True,          # Cosine learning rate decay for smoother convergence
    label_smoothing=0.1,  # Slight label smoothing for better calibration
    overlap_mask=True,    # Better instance separation
    verbose=True,
)
train_time = time.time() - t0

# Save best weights to Drive
run_dir = Path(PROJECT_DIR) / RUN_NAME
best_pt = run_dir / "weights" / "best.pt"
last_pt = run_dir / "weights" / "last.pt"

if best_pt.exists():
    shutil.copy2(best_pt, DRIVE_WEIGHTS / "best.pt")
    shutil.copy2(best_pt, DRIVE_WEIGHTS / "tactical_yolo11s_cuas_armor.pt")
    print(f"\n⭐ Best weights saved to Drive: {DRIVE_WEIGHTS / 'best.pt'}")
if last_pt.exists():
    shutil.copy2(last_pt, DRIVE_WEIGHTS / "last.pt")
    print(f"💾 Last checkpoint saved to Drive: {DRIVE_WEIGHTS / 'last.pt'}")

print(f"\n⏱️  Training completed in {train_time/60:.1f} minutes ({train_time/3600:.1f} hours)")

# Print validation metrics
if hasattr(results, "results_dict"):
    rd = results.results_dict
    print(f"\n📊 Validation Metrics:")
    for k, v in rd.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")

print("\n✅ Cell 5 Complete: Student Training Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6: EXPORT TO TENSORRT FP16 + LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  CELL 6: TENSORRT FP16 EXPORT & LATENCY BENCHMARK")
print("=" * 72)

# Use best weights
export_weights = DRIVE_WEIGHTS / "best.pt"
if not export_weights.exists():
    export_weights = best_pt if best_pt.exists() else Path(starting_weights)

print(f"📦 Exporting: {export_weights}")
export_model = YOLO(str(export_weights))

# Export to TensorRT FP16 engine
try:
    engine_path = export_model.export(
        format="engine",
        imgsz=IMGSZ,
        half=True,
        batch=1,
        device="0",
        workspace=8,
        dynamic=False,
        verbose=True,
    )
    engine_path = Path(engine_path)
    print(f"⚡ TensorRT Engine: {engine_path}")

    # Copy engine to Drive
    drive_engine = DRIVE_EXPORTS / engine_path.name
    shutil.copy2(engine_path, drive_engine)
    print(f"💾 Engine saved to Drive: {drive_engine}")
except Exception as e:
    print(f"⚠️  TensorRT engine export failed: {e}")
    print("🔄 Exporting ONNX as fallback for Jetson trtexec compilation...")
    onnx_path = export_model.export(
        format="onnx",
        imgsz=IMGSZ,
        half=True,
        batch=1,
        dynamic=False,
        simplify=True,
    )
    onnx_path = Path(onnx_path)
    shutil.copy2(onnx_path, DRIVE_EXPORTS / onnx_path.name)
    print(f"📄 ONNX saved: {DRIVE_EXPORTS / onnx_path.name}")
    print(f"💡 On Jetson, run: trtexec --onnx={onnx_path.name} --saveEngine=tactical.engine --fp16")
    engine_path = onnx_path

# ─── Latency Benchmark ──────────────────────────────────────────────────────
print(f"\n⏱️  Benchmarking inference latency...")
import numpy as np

bench_model = YOLO(str(export_weights))
dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)

# Warmup
for _ in range(20):
    _ = bench_model(dummy, device="0", verbose=False)

# Timed iterations with CUDA Events
BENCH_ITERS = 100
latencies = []
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

for _ in range(BENCH_ITERS):
    start_ev.record()
    _ = bench_model(dummy, device="0", verbose=False)
    end_ev.record()
    torch.cuda.synchronize()
    latencies.append(start_ev.elapsed_time(end_ev))

lat = np.array(latencies)
print(f"\n{'='*50}")
print(f"  LATENCY BENCHMARK ({BENCH_ITERS} iterations)")
print(f"{'='*50}")
print(f"  Mean Latency:   {np.mean(lat):.2f} ms")
print(f"  Median (P50):   {np.median(lat):.2f} ms")
print(f"  P90 Latency:    {np.percentile(lat, 90):.2f} ms")
print(f"  P95 Latency:    {np.percentile(lat, 95):.2f} ms")
print(f"  P99 Latency:    {np.percentile(lat, 99):.2f} ms")
print(f"  Mean FPS:       {1000.0/np.mean(lat):.1f}")
print(f"{'='*50}")

print("\n✅ Cell 6 Complete: Export & Benchmark Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  🎉 TACYOLO A100 PIPELINE COMPLETE!")
print("=" * 72)
print(f"""
  📁 Google Drive Artifacts (persistent):
     Dataset:    {DRIVE_DATASETS}
     Best .pt:   {DRIVE_WEIGHTS / 'best.pt'}
     Last .pt:   {DRIVE_WEIGHTS / 'last.pt'}
     TRT Engine: {DRIVE_EXPORTS}
     Train Runs: {DRIVE_ROOT / 'runs'}

  🚀 To deploy on Jetson:
     1. Copy {DRIVE_EXPORTS}/*.engine to Jetson
     2. Run: python tacyolo --weights tactical.engine --source /dev/video0

  🔄 To resume training later:
     Just re-run this notebook — it auto-detects Drive checkpoints!
""")
