# DET-YOLO weights

The offline train loop writes `DET-YOLO.pt` here after each successful dataset and (by default) `git push`es it.

This folder is empty on purpose until the first training run — do **not** commit a large `.pt` blob in a PR. Git LFS is configured for `weights/*.pt`.

```bash
git pull
# colab_pipeline/weights/DET-YOLO.pt
```
