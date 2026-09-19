# DET-YOLO offline pipeline (desktop download, offline train, GitHub state)

**Colab does not use Kaggle.** There is no `KAGGLE_API_TOKEN` and no `kagglehub` in the notebook.

Split of work:

| Machine | Job | Kaggle? |
|---|---|---|
| **Windows desktop** | Parallel download of `download_queue.txt` to `D:\DET-YOLO_kaggle_datasets`. On fail, immediately download a replacement from `replenish_pool.txt`. | Yes (local `kaggle.json` only) |
| **Windows desktop (preferred train)** | `train_offline_loop.py`: next **ready local folder** → train a few epochs → archive/delete → next. | **No** |
| **Colab (optional)** | Same offline trainer, only if you copied those folders to Google Drive and mounted them. | **No** |

Shared GitHub files: `download_queue.txt`, `replenish_pool.txt`, `state.json`, `weights/DET-YOLO.pt`.

## 1) Desktop: Kaggle credentials (once)

On the Windows PC only:

1. Create `C:\Users\<you>\.kaggle\kaggle.json` from https://www.kaggle.com/settings (API token).
2. Never commit that file. It is gitignored.

```powershell
pip install kagglehub ultralytics opencv-python-headless pyyaml tqdm
cd path\to\tacyolo\colab_pipeline
```

## 2) Desktop: parallel download

```powershell
python download_desktop.py --data-root D:\DET-YOLO_kaggle_datasets --workers 8
```

Each success lands at `D:\DET-YOLO_kaggle_datasets\<owner>--<slug>\`. Failures are recorded in `state.json` (`download_failed`) and a pool slug is queued immediately. Re-run the command anytime; already-local folders are skipped.

Target: ~150 successful folders, not 150 attempts.

## 3) Desktop: offline train (preferred)

```powershell
python train_offline_loop.py --data-root D:\DET-YOLO_kaggle_datasets --epochs 8
```

Loop:

1. Pick the next queue slug whose folder is already on disk (skip ones the downloader has not finished).
2. Build a YOLO train set from images + optional `.txt` labels.
3. If **no labels**: skip, write a replenish note, try the next **local** `replenish_pool.txt` folder.
4. Train a few epochs from `weights/DET-YOLO.pt` (seeds `yolov8n.pt` if missing).
5. Copy `best.pt` → `weights/DET-YOLO.pt`, update `state.json`, `git push`.
6. Move the dataset to `D:\DET-YOLO_kaggle_datasets\_done\` (or `--after-train delete`).
7. Next ready folder.

Smoke test: `python train_offline_loop.py --max-datasets 1 --no-push`

## 4) Optional Colab — still no Kaggle

Only if you copy (a subset of) `D:\DET-YOLO_kaggle_datasets` to Google Drive.

1. Colab secret `GH_TOKEN` (repo PAT) — **not** a Kaggle token.
2. Open the notebook badge → GPU → **Run all**.
3. Mount Drive and set `DATA_ROOT` to that folder.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devansh3108-2/tacyolo/blob/main/colab_pipeline/DET_YOLO_GitHub_Pipeline.ipynb)

The notebook installs `ultralytics` only. It never calls Kaggle.

## Resume

`state.json` tracks `done` / `failed` / `skipped` / `downloaded` / `download_failed` / `replenished` / `current`. Re-run download or train; finished slugs are skipped.

## Weights + Git LFS

```bash
git lfs install
git pull
# colab_pipeline/weights/DET-YOLO.pt
```

## Queue size

`download_queue.txt` is 150 Kaggle `owner/slug` lines (~596 GB measured / ~638 GB estimated). That bulk grab is **desktop-only**. `replenish_pool.txt` is ~80 alternates, used when a download or unlabeled folder fails.
