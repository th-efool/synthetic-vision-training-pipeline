# test_real_dataset.py
# %%
"""
Evaluate a YOLOv8 model on a real dataset organized as:
  real_dataset/<class_name>/*.jpg

Outputs (eval_results/):
 - results.csv         (per-image predictions)
 - per_class_metrics.csv
 - confusion_matrix.html
 - summary.json
"""
from pathlib import Path
import os
import sys
import json

# Config
MODEL_PATH = Path("synth_run_best.pt")          # path to your trained model
REAL_DATASET_ROOT = Path("real_dataset") # root with class subfolders
OUT_DIR = Path("synth_eval_results")
CONF_THRESHOLD = 0.25
DEVICE = "cpu"     # "cpu" or "0" for GPU
BATCH_SIZE = 8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Basic checks
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
if not REAL_DATASET_ROOT.exists():
    raise FileNotFoundError(f"Real dataset root not found: {REAL_DATASET_ROOT.resolve()}")

# Optional imports (install if missing)
try:
    from ultralytics import YOLO
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_fscore_support
    from tqdm import tqdm
except Exception:
    print("Installing missing packages...")
    os.system(f"{sys.executable} -m pip install ultralytics pandas plotly scikit-learn tqdm numpy --quiet")
    from ultralytics import YOLO
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_fscore_support
    from tqdm import tqdm

# Build list of images and class folders
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

# Load model
print("Loading model:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))

# Robustly get model class-name mapping
model_names = {}
# Try multiple places to find names dict
if hasattr(model, "names") and isinstance(getattr(model, "names"), (dict, list)):
    names_obj = getattr(model, "names")
    if isinstance(names_obj, list):
        model_names = {i: n for i, n in enumerate(names_obj)}
    else:
        # dict-like (keys may be ints or strings)
        model_names = {int(k): v for k, v in names_obj.items()}
else:
    # try model.model.names (some ultralytics versions)
    try:
        names_obj = getattr(model, "model").names
        if isinstance(names_obj, list):
            model_names = {i: n for i, n in enumerate(names_obj)}
        else:
            model_names = {int(k): v for k, v in names_obj.items()}
    except Exception:
        model_names = {}

if model_names:
    print("Detected model class names:", model_names)
else:
    print("No model class-name mapping detected; predicted classes will be numeric strings.")

# Prepare lists for results
rows = []
paths = [p for p, _ in image_label_pairs]
gts = [gt for _, gt in image_label_pairs]

# Run inference in batches
print("Running inference...")
for start in tqdm(range(0, len(paths), BATCH_SIZE)):
    batch_paths = paths[start:start + BATCH_SIZE]
    preds = model.predict(source=[str(p) for p in batch_paths], device=DEVICE, imgsz=512, conf=CONF_THRESHOLD, verbose=False)
    # preds is a list of Results in same order
    for img_path, res in zip(batch_paths, preds):
        # default when no boxes
        pred_label_name = "none"
        pred_score = 0.0
        pred_class_id = None

        boxes = getattr(res, "boxes", None)
        # boxes can be None, or have .cls/.conf attributes or be sequence
        if boxes is None or len(boxes) == 0:
            pass
        else:
            try:
                # ultralytics >=8: boxes.cls and boxes.conf are tensors
                confs = boxes.conf.cpu().numpy().ravel().tolist()
                cls_ids = boxes.cls.cpu().numpy().ravel().astype(int).tolist()
            except Exception:
                # fallback older versions / unexpected types
                try:
                    confs = [float(b.conf) for b in boxes]
                    cls_ids = [int(b.cls) for b in boxes]
                except Exception:
                    confs = []
                    cls_ids = []

            if confs:
                best_idx = int(np.argmax(confs))
                pred_score = float(confs[best_idx])
                pred_class_id = int(cls_ids[best_idx])
                pred_label_name = model_names.get(pred_class_id, str(pred_class_id))

        rows.append({
            "image": str(img_path),
            "gt_label": gts[start + batch_paths.index(img_path)],
            "pred_label": pred_label_name,
            "pred_score": float(pred_score) if pred_score is not None else 0.0,
            "pred_class_id": int(pred_class_id) if pred_class_id is not None else None
        })

# Save per-image results
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "results.csv", index=False)
print("Saved per-image results to", (OUT_DIR / "results.csv").resolve())

# Map to numeric labels for metrics
y_true = [class_to_index[x] for x in df["gt_label"].tolist()]
# predicted: match predicted label name to class index, else -1
lower_map = {k.lower(): v for k, v in class_to_index.items()}
y_pred = []
for pl in df["pred_label"].tolist():
    if pl == "none":
        y_pred.append(-1)
        continue
    # exact match
    if pl in class_to_index:
        y_pred.append(class_to_index[pl])
        continue
    # case-insensitive
    if pl.lower() in lower_map:
        y_pred.append(lower_map[pl.lower()])
        continue
    # try matching numeric string to index
    try:
        pid = int(pl)
        # if pid in model_names and names match folder labels, attempt to map by comparing names
        mapped_name = model_names.get(pid, None)
        if mapped_name and mapped_name in class_to_index:
            y_pred.append(class_to_index[mapped_name])
            continue
        # otherwise accept numeric id if within range
        if 0 <= pid < len(class_names):
            y_pred.append(pid)
            continue
    except Exception:
        pass
    y_pred.append(-1)

y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
correct = (y_true_np == y_pred_np)
accuracy = correct.sum() / len(correct)
print(f"Overall accuracy: {accuracy:.4f} ({correct.sum()}/{len(correct)})")

# Per-class metrics
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

# Confusion matrix (include "none" as final column)
all_labels = class_names + ["none"]
cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
for gt_idx, pred_idx in zip(y_true_np.tolist(), y_pred_np.tolist()):
    row = int(gt_idx)
    col = int(pred_idx) if pred_idx >= 0 else (len(all_labels) - 1)
    cm[row, col] += 1

# Plotly heatmap
fig = go.Figure(data=go.Heatmap(z=cm.tolist(), x=all_labels, y=class_names, colorscale="Viridis"))
fig.update_layout(title="Confusion Matrix (gt rows, pred cols). 'none' = no detection", xaxis_title="Predicted", yaxis_title="Ground truth", height=600, width=700)
html_path = OUT_DIR / "confusion_matrix.html"
fig.write_html(str(html_path))
print("Saved confusion matrix to", html_path.resolve())

# Summary JSON
summary = {
    "overall_accuracy": float(accuracy),
    "num_images": int(len(df)),
    "per_class": per_class.to_dict(orient="records")
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary to", (OUT_DIR / "summary.json").resolve())

# %%
# End
