# DET-YOLO hands-off Colab pipeline (GitHub sync)

One-in / one-out loop. You open the notebook once from GitHub; after that Colab:

1. Pulls latest `state.json` + `weights/DET-YOLO.pt` from this repo
2. Downloads **one** Kaggle dataset from `download_queue.txt`
3. Trains a few epochs
4. **Deletes** that dataset from Colab disk
5. Pushes updated weights + state back to GitHub
6. Takes the next slug — no manual upload between steps

If a download or train **fails**, Colab marks it in `state.json` and immediately tries a replacement from `replenish_pool.txt` until that slot succeeds or the pool is empty. The target is ~150 **successful** trains, not 150 attempts.

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

Set `MAX_DATASETS = 1` for a smoke test (one **successful** train, with replenish on fail); `None` aims for the full ~150 successes.

## Resume

`state.json` tracks `done` / `failed` / `current` / `replenished`. Re-run after disconnect — it skips finished slugs. To retry a failed ref, remove it from `failed` and push.

## Failure replenish

Goal: ~150 successful trains, not 150 attempts.

1. A queue slug fails (Kaggle 404, no images, CUDA OOM, …).
2. It is appended to `state.json` `failed` and pushed.
3. The loop **immediately** pops the next unused slug from `replenish_pool.txt` (UAV / thermal / drone / person sets that are **not** in `download_queue.txt`).
4. Repeat until a train succeeds or the pool is empty, then continue with the next main-queue slug.
5. After the main queue is exhausted, leftover pool slugs backfill any success deficit (e.g. fails from an earlier run).

`replenish_pool.txt` is ~80 alternates. Replace or extend it if you want a different mix.

## Parallel desktop download vs Colab train

These are **two different machines and two different download paths**. Do not mix them.

| | Windows desktop | Colab |
|---|---|---|
| Role | Archive the queue | Train |
| How | **Parallel** download of all ~150 `download_queue.txt` slugs (many workers) into e.g. `D:\DET-YOLO_kaggle_datasets` | **One-at-a-time**: kagglehub download → train a few epochs → **delete** from Colab disk → git push → next |
| Path | Local disk on the PC | Colab VM via `kagglehub.dataset_download` |

Colab never reads the desktop folder and never downloads the full 150 in parallel. Desktop never trains. Pull weights from GitHub on either side:

```bash
git pull
# colab_pipeline/weights/DET-YOLO.pt
```

## Weights + Git LFS

Checkpoints land in `colab_pipeline/weights/DET-YOLO.pt` (Git LFS). The first cycle seeds `yolov8n.pt` if that file is missing.

```bash
git lfs install
git pull
# colab_pipeline/weights/DET-YOLO.pt
```

## Total queue size

`download_queue.txt` is 150 Kaggle `owner/slug` lines (UAV / thermal / drone / person detection and related public sets). ~596 GB measured (140/150) · ~638 GB estimated for all 150 — that bulk parallel grab is the **desktop** job. Colab delete-after-train is required so the VM stays one-dataset-at-a-time.
