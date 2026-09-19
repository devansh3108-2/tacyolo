# DET-YOLO hands-off Colab pipeline (GitHub sync)

One-in / one-out loop. You open the notebook once from GitHub; after that Colab:

1. Pulls latest `state.json` + `weights/DET-YOLO.pt` from this repo
2. Downloads **one** Kaggle dataset from `download_queue.txt`
3. Trains a few epochs
4. **Deletes** that dataset from Colab disk
5. Pushes updated weights + state back to GitHub
6. Takes the next slug — no manual upload between steps

## One-time setup (about 2 minutes)

1. Create a GitHub **fine-grained or classic PAT** with `repo` scope (Settings → Developer settings → Personal access tokens).
2. In Colab: 🔑 icon (Secrets) → add:
   - `GH_TOKEN` = your PAT
   - `KAGGLE_API_TOKEN` = your Kaggle token (or paste `kaggle.json` contents)
3. Open the notebook from GitHub (badge below) → Runtime → GPU → **Run all**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devansh3108-2/tacyolo/blob/main/colab_pipeline/DET_YOLO_GitHub_Pipeline.ipynb)

Do not put tokens in the repo. Colab reads them from Secrets only.

## Knobs

Edit the config cell: `EPOCHS_PER_DS`, `IMGSZ`, `BATCH`, `MAX_IMAGES`.

Set `MAX_DATASETS = 1` for a smoke test; `None` walks the full remaining queue.

## Resume

`state.json` tracks `done` / `failed` / `current`. Re-run after disconnect — it skips finished slugs. To retry a failed ref, remove it from `failed` and push.

## Weights + Git LFS

Checkpoints land in `colab_pipeline/weights/DET-YOLO.pt` (Git LFS). The first cycle seeds `yolov8n.pt` if that file is missing.

```bash
git lfs install
git pull
# colab_pipeline/weights/DET-YOLO.pt
```

## Desktop side-by-side

Desktop can keep archiving to `D:\DET-YOLO_kaggle_datasets`. Colab does **its own** download per cycle (no GB upload from your PC). Pull latest weights anytime:

```bash
git pull
# colab_pipeline/weights/DET-YOLO.pt
```

## Total queue size

`download_queue.txt` is 150 Kaggle `owner/slug` lines (UAV / thermal / drone / person detection and related public sets). ~596 GB measured (140/150) · ~638 GB estimated for all 150 — delete-after-train is required. Replace slugs in the queue file if you want a different mix.
