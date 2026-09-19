# DET-YOLO weights

The Colab loop writes `DET-YOLO.pt` here after each successful dataset cycle and `git push`es it.

This folder is empty on purpose until the first training run — do **not** commit a large `.pt` blob in a PR. Git LFS is configured for `weights/*.pt` (see `colab_pipeline/.gitattributes` and the repo-root `.gitattributes`).

After a Colab pass:

```bash
git pull
# colab_pipeline/weights/DET-YOLO.pt
```
