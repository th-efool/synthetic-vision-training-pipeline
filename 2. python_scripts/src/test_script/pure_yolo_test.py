#!/usr/bin/env python3
# pure_yolo_test.py
"""
Evaluate a *pure* (COCO-pretrained) YOLOv8 model on a real dataset organized as:
  real_dataset/<class_name>/*.jpg

This script DOES NOT train or fine-tune anything. It only runs inference
with the specified model (default: yolov8n.pt from Ultralytics, COCO classes)
and evaluates predictions against folder labels (top-1 detection per image).

Outputs (directory by default: eval_results_pure_yolo/):
 - results.csv            (per-image predictions)
 - per_class_metrics.csv
 - confusion_matrix.html  (interactive Plotly heatmap)
 - summary.json

Usage:
  python pure_yolo_test.py --model yolov8n.pt --data real_dataset --device cpu
  python pure_yolo_test.py --data real_dataset               # uses yolov8n.pt by default

Notes:
 - The script treats the folder name as the ground-truth class for each image.
 - Prediction logic selects the highest-confidence detection per image and maps
   the predicted class name (or numeric id) to folder labels when possible.
 - Uses Plotly for the confusion matrix (avoids matplotlib recursion issues).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# -------- argparse --------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate COCO-pretrained YOLO on real dataset (folder-class format).")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO .pt model (default: yolov8n.pt)")
    p.add_argument("--data", type=str, default="real_dataset", help="Root folder with class subfolders (default: real_dataset)")
    p.add_argument("--out", type=str, default="pure_yolo_eval_results", help="Output folder for evaluation results")
    p.add_argument("--device", type=str, default="cpu", help="Device for inference ('cpu' or '0' for CUDA)")
    p.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--batch", type=int, default=8, help="Batch size for predict()")
    return p.parse_args()

args = parse_args()

MODEL_PATH = Path(args.model)
REAL_DATASET_ROOT = Path(args.data)
OUT_DIR = Path(args.out)
CONF_THRESHOLD = float(args.conf)
DEVICE = args.device
IMG_SIZE = int(args.imgsz)
BATCH_SIZE = int(args.batch)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- basic checks --------
if not MODEL_PATH.exists():
    print(f"Model '{MODEL_PATH}' not found locally. The Ultralytics YOLO loader will attempt to use the model string '{MODEL_PATH.name}' and auto-download if possible.")
if not REAL_DATASET_ROOT.exists():
    raise FileNotFoundError(f"Real dataset root not found: {REAL_DATASET_ROOT.resolve()}")

# -------- imports (install missing libs automatically) --------
try:
    from ultralytics import YOLO
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_fscore_support
    from tqdm import tqdm
except Exception:
    print("Installing required packages (ultralytics, pandas, plotly, scikit-learn, tqdm, numpy)...")
    import os
    os.system(f"{sys.executable} -m pip install ultralytics pandas plotly scikit-learn tqdm numpy --quiet")
    from ultralytics import YOLO
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_fscore_support
    from tqdm import tqdm

# -------- build image list from folder-structure --------
image_label_pairs = []
class_names = []
for class_dir in sorted([p for p in REAL_DATASET_ROOT.iterdir() if p.is_dir()]):
    class_names.append(class_dir.name)
    for img in sorted(class_dir.iterdir()):
        if img.suffix.lower() in IMAGE_EXTS:
            image_label_pairs.append((img, class_dir.name))

if not image_label_pairs:
    raise RuntimeError(f"No images found under {REAL_DATASET_ROOT} (extensions: {IMAGE_EXTS})")

print(f"Found {len(image_label_pairs)} images across {len(class_names)} classes.")
print("Classes:", class_names)

class_to_index = {c: i for i, c in enumerate(class_names)}

# -------- load model --------
print("Loading model:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))  # will accept local path or model string (e.g., 'yolov8n.pt')

# robustly extract names map from model
model_names = {}
# prefer model.names (modern ultralytics)
try:
    names_obj = getattr(model, "names", None)
    if isinstance(names_obj, (list, dict)):
        if isinstance(names_obj, list):
            model_names = {i: n for i, n in enumerate(names_obj)}
        else:
            model_names = {int(k): v for k, v in names_obj.items()}
except Exception:
    model_names = {}

# fallback try model.model.names
if not model_names:
    try:
        mm = getattr(model, "model")
        names_obj = getattr(mm, "names", None)
        if isinstance(names_obj, list):
            model_names = {i: n for i, n in enumerate(names_obj)}
        elif isinstance(names_obj, dict):
            model_names = {int(k): v for k, v in names_obj.items()}
    except Exception:
        model_names = {}

if model_names:
    print("Detected model class names (sample):", {k: model_names[k] for k in sorted(list(model_names.keys())[:8])})
else:
    print("No model class-name mapping detected; predictions will be numeric ids when necessary.")

# -------- inference loop (batching) --------
rows = []
paths = [p for p, _ in image_label_pairs]
gts = [gt for _, gt in image_label_pairs]

print("Running inference on dataset...")
for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Batches"):
    batch_paths = paths[start:start + BATCH_SIZE]
    preds = model.predict(source=[str(p) for p in batch_paths], device=DEVICE, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
    for img_path, res in zip(batch_paths, preds):
        pred_label_name = "none"
        pred_score = 0.0
        pred_class_id = None

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            # no detections
            pass
        else:
            # extract confs and cls robustly across ultralytics versions
            try:
                confs = boxes.conf.cpu().numpy().ravel().tolist()
                cls_ids = boxes.cls.cpu().numpy().ravel().astype(int).tolist()
            except Exception:
                try:
                    confs = [float(b.conf) for b in boxes]
                    cls_ids = [int(b.cls) for b in boxes]
                except Exception:
                    confs, cls_ids = [], []

            if confs:
                best_idx = int(np.argmax(confs))
                pred_score = float(confs[best_idx])
                pred_class_id = int(cls_ids[best_idx])
                pred_label_name = model_names.get(pred_class_id, str(pred_class_id))

        rows.append({
            "image": str(img_path),
            "gt_label": gts[start + batch_paths.index(img_path)],
            "pred_label": pred_label_name,
            "pred_score": float(pred_score),
            "pred_class_id": int(pred_class_id) if pred_class_id is not None else None
        })

# -------- save per-image results --------
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "results.csv", index=False)
print("Saved per-image results to", (OUT_DIR / "results.csv").resolve())

# -------- compute metrics --------
# true labels -> numeric
y_true = [class_to_index[x] for x in df["gt_label"].tolist()]

# predicted -> numeric using name match or numeric fallback, else -1
lower_map = {k.lower(): v for k, v in class_to_index.items()}
y_pred = []
for pl in df["pred_label"].tolist():
    if pl == "none":
        y_pred.append(-1)
        continue
    if pl in class_to_index:
        y_pred.append(class_to_index[pl])
        continue
    if pl.lower() in lower_map:
        y_pred.append(lower_map[pl.lower()])
        continue
    # attempt numeric parse
    try:
        pid = int(pl)
        if 0 <= pid < len(class_names):
            y_pred.append(pid)
            continue
    except Exception:
        pass
    # last fallback: try mapping via model_names->folder names
    mapped = None
    try:
        for idx, name in model_names.items():
            if isinstance(name, str) and name.lower() in lower_map:
                mapped = lower_map[name.lower()]
                break
    except Exception:
        mapped = None
    if mapped is not None:
        y_pred.append(mapped)
        continue
    y_pred.append(-1)

import numpy as np
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
correct = (y_true_np == y_pred_np)
accuracy = correct.sum() / len(correct)
print(f"Overall accuracy: {accuracy:.4f} ({correct.sum()}/{len(correct)})")

# per-class metrics
labels_for_metrics = list(range(len(class_names)))
prec, rec, f1, support = precision_recall_fscore_support(y_true_np, y_pred_np, labels=labels_for_metrics, zero_division=0)
per_class = pd.DataFrame({
    "class": class_names,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "support": support
})
per_class.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
print("Saved per-class metrics to", (OUT_DIR / "per_class_metrics.csv").resolve())
print(per_class)

# confusion matrix (include 'none' as final column)
all_labels = class_names + ["none"]
cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
for gt_idx, pred_idx in zip(y_true_np.tolist(), y_pred_np.tolist()):
    row = int(gt_idx)
    col = int(pred_idx) if pred_idx >= 0 else (len(all_labels) - 1)
    cm[row, col] += 1

# plotly heatmap and save
fig = go.Figure(data=go.Heatmap(z=cm.tolist(), x=all_labels, y=class_names, colorscale="Viridis"))
fig.update_layout(title="Confusion Matrix (gt rows, pred cols). 'none' = no detection", xaxis_title="Predicted", yaxis_title="Ground truth", height=600, width=700)
html_path = OUT_DIR / "confusion_matrix.html"
fig.write_html(str(html_path))
print("Saved confusion matrix to", html_path.resolve())

# summary json
summary = {
    "overall_accuracy": float(accuracy),
    "num_images": int(len(df)),
    "per_class": per_class.to_dict(orient="records")
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary to", (OUT_DIR / "summary.json").resolve())

print("Done.")
