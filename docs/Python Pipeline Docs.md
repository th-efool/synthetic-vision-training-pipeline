# Python Pipeline Documentation

## Overview
This document describes the Python-side pipeline for converting Unity Perception outputs, preparing datasets, training YOLOv8 (synthetic-only and mixed), evaluating results, and exporting plots.  
Paths below assume you run commands from the repo root.

## Prerequisites
- Python 3.10+ (3.14 worked in dev but 3.10–3.12 recommended)
- A working virtual environment (venv or conda)
- Unity output already exported under `1. unity_project/SyntheticOutput/` (or prepackaged zips under `3. datasets/`)
- If you want COCO real data: download `val2017` and `instances_val2017.json` (Kaggle or COCO site)

---

## Minimal `requirements.txt` (place at repo root)
```
ultralytics==8.3.232
torch==2.9.1 # CPU or GPU build matching your machine
torchvision==0.24.1
opencv-python==4.10.0
pandas==2.2.2
plotly==6.5.0
Pillow==10.4.0
tqdm==4.66.5
patool # optional, for .rar extraction in prepare_datasets.py
```

Install:
```bash
python -m venv .venv
```

```
# Windows
.venv\Scripts\activate
```

```
# macOS / Linux
source .venv/bin/activate
```

pip install -r requirements.txt

### Repo paths used by the Python pipeline
```bash
2. python_scripts/src/
  ├─ helperscripts/
  │   └─ UnityToYOLO.py          # converts Perception output -> YOLO dataset
  ├─ prepare_datasets.py         # (optional) extracts archives from 3. datasets/
  ├─ SyntheticTraining.ipynb
  ├─ MixedTraining.ipynb
  ├─ datasets/
  │   ├─ yolo_synthetic_dataset/
  │   └─ yolo_mixed_dataset/
  └─ runs/                       # training outputs: runs/detect/<run_name>/
```
## 1) Prepare datasets
1. Extract the zips you received under `3. datasets/` into:
```
2. python_scripts/src/datasets/yolo_synthetic_dataset/
3. python_scripts/src/datasets/yolo_mixed_dataset/
```
2. Copy corresponding YAMLs (`yolo_synthetic_dataset.yaml` → `dataset.yaml` inside each dataset folder).

## 2) Convert Unity Perception → YOLO (if you exported from Unity)
Place class folders (cup, bottle, chair, laptop, book) next to the converter:
```
  2.python_scripts/src/helperscripts/
  cup/
  bottle/
  chair/
  laptop/
  book/
  UnityToYOLO.py
```
Run converter (from that folder or with correct cwd):
```
cd 2. python_scripts/src/helperscripts
python UnityToYOLO.py --val-ratio 0.1 --mode copy -v
```
Output will be:
```
2. python_scripts/src/helperscripts/yolo_synthetic_dataset/
  images/train, images/val
  labels/train, labels/val
  dataset.yaml
  classes.txt
```
Move or copy that `yolo_synthetic_dataset` to `2. python_scripts/src/datasets/` (not strictly necessary if notebooks point to helperscripts path — but keep consistent).

## 3) Data sanity checks (quick)
Paste this into a notebook cell (or run as script) to visually confirm labels match images:
```python
from pathlib import Path
from PIL import Image, ImageDraw
import random

DATASET_ROOT = Path("2. python_scripts/src/datasets/yolo_synthetic_dataset")
images_dir = DATASET_ROOT / "images" / "train"
labels_dir = DATASET_ROOT / "labels" / "train"

imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
print("images found:", len(imgs))
for img_path in random.sample(imgs, min(4,len(imgs))):
    im = Image.open(img_path).convert("RGB")
    w,h = im.size
    draw = ImageDraw.Draw(im)
    label_file = labels_dir / (img_path.stem + ".txt")
    if label_file.exists():
        for ln in label_file.read_text().splitlines():
            cid,x,y,w_n,h_n = ln.split()
            x,y,w_n,h_n = map(float, (x,y,w_n,h_n))
            left = (x - w_n/2) * w
            top  = (y - h_n/2) * h
            right= (x + w_n/2) * w
            bottom=(y + h_n/2) * h
            draw.rectangle([left,top,right,bottom], outline="red", width=2)
    display(im)
```
---

## 4) Training (YOLOv8) — Notebooks
Two notebooks are provided:
- `SyntheticTraining.ipynb` — trains on synthetic-only dataset
- `MixedTraining.ipynb` — trains on combined real + synthetic dataset
### Jupyter: recommended flow
1. Open notebook in Jupyter or VSCode (working dir: `2. python_scripts/src/`)
2. Verify `DATASET_ROOT` variable at top points to:
```
DATASET_ROOT = Path.cwd() / "datasets" / "yolo_synthetic_dataset"  # or yolo_mixed_dataset
```
3. Run cells in order. Notebooks call Ultralytics `model.train(...)`.

## 5) Evaluation & plotting
- Ultralytics writes `runs/detect/<name>/results.csv` and images.
- A hardcoded plotting notebook `plot_hardcoded_yolo_run.ipynb` is included to reproduce the exact curves from a completed run (creates `runs/detect/<run>/plots_hardcoded/training_curves.html`).
To produce plots from an actual `results.csv`:

```python
import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv("runs/detect/mixed_run/results.csv")
# plot traces...
```
If you only want the saved hardcoded visualization (from logs), run the provided plotting cell — it writes `training_curves.html` and (optionally) PNG (requires `kaleido`).

## 6) Expected outputs (after training)
```
2. python_scripts/src/runs/detect/<run_name>/
  ├─ weights/best.pt
  ├─ weights/last.pt
  ├─ results.csv
  ├─ results.png
  ├─ predictions/   <-- sample predicted images
  └─ plots_hardcoded/training_curves.html
```

## 7) Troubleshooting (quick)
- **No images found / zero URLs:** ensure dataset extraction paths are correct and YAML points to right directories.
- **requests/connect timeouts when scraping:** network / firewall may block. Prefer using COCO official download (Kaggle CLI) instead of web scraping.
- **Ultralytics recursion/plot errors:** set `plots=False` in `model.train(...)` or skip training and run plotting separately.
- **scikit-learn or build errors on Windows:** prefer installing binary wheels (conda) or use Python versions with compatible wheel builds. For heavy installs, use conda environment with prebuilt `scikit-learn`, `numpy`, `torch`.
- **Kaleido missing for PNG export:** install with `!pip install -U kaleido` in notebook.

## 8) Reproducibility notes
- Pin the python packages (use the provided `requirements.txt`) and run in the same Python version used for development.
- Use fixed seeds where available:
    - In `UnityToYOLO.py` use `--val-ratio` consistently.
    - In Unity `PrefabPlacementRandomizer`, set `useSeed = true` and `seed = <int>` to reproduce synthetic renders.
- Document the exact `yolov8` model (e.g., `yolov8n.pt`) and hyperparameters used.
