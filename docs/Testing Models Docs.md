**Purpose (short):** run inference with a YOLOv8 model on a folder-structured real dataset (`real_dataset/<class_name>/*.jpg`) and produce per-image predictions, per-class metrics, a confusion matrix, and a small JSON summary. These docs cover the three scripts in `src/test_script/`:
- `test_real_dataset.py` — generic evaluator (use your trained model).
- `pure_yolo_test.py` — runs COCO-pretrained (or any `.pt`) model without training.
- `test_real_dataset.py` (alternate copy) — same as first but packaged for mixed-run names.

---
### Prereqs
1. Python 3.8+ (venv recommended).
2. Install packages:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/linux
source .venv/bin/activate
pip install ultralytics pandas plotly scikit-learn tqdm numpy
# optional for PNG exports:
pip install -U kaleido
```
3. Dataset layout (example):
```
real_dataset/
  cup/
    img001.jpg
    img002.jpg
  bottle/
    ...
```
4. Model file (for your trained model): e.g. `runs/detect/mixed_run/weights/best.pt` or `yolov8n.pt`.

### What each script does (one-liners)
- `test_real_dataset.py`  
    Run a specified YOLO model on your folder dataset, saves per-image CSV, per-class metrics, confusion matrix (interactive HTML), and `summary.json`.
- `pure_yolo_test.py`  
    Same evaluation flow but intended for running a COCO-pretrained YOLO (e.g., `yolov8n.pt`) — useful as baseline.
- `mixed / alternate test`  
    Variant that is shipped in the repo with slightly different default paths (check header vars).
### Common config (edit at top of script or pass args)
- `MODEL_PATH` or `--model` — path to `.pt` model (or model string like `yolov8n.pt`).
- `REAL_DATASET_ROOT` or `--data` — root folder with class subfolders.
- `OUT_DIR` or `--out` — output dir for results.
- `CONF_THRESHOLD` / `--conf` — detection confidence threshold (default 0.25).
- `DEVICE` / `--device` — `"cpu"` or GPU id (e.g. `"0"`).
- `IMG_SIZE` / `--imgsz` — inference size (512 recommended).
- `BATCH_SIZE` / `--batch` — batch size for prediction.
### Run examples

From project root, using defaults in scripts:

```bash
# evaluate your trained model
python src/test_script/test_real_dataset.py

# evaluate a COCO-pretrained model (baseline)
python src/test_script/pure_yolo_test.py --model yolov8n.pt --data real_dataset --out pure_yolo_eval_results

# override device and conf threshold
python src/test_script/test_real_dataset.py --model runs/detect/mixed_run/weights/best.pt --device 0 --conf 0.3
```

### Outputs (per run)
Folder: `<OUT_DIR>` (default names shown in scripts)
```
results.csv            # per-image predictions: image, gt_label, pred_label, pred_score, pred_class_id
per_class_metrics.csv  # precision, recall, f1, support per ground-truth class
confusion_matrix.html  # interactive Plotly heatmap (gt rows, pred cols); 'none' column = no detection
summary.json           # overall_accuracy, num_images, per_class metrics (JSON)
```

### Notes about label mapping & common pitfalls
- The scripts treat the **folder name** as the ground-truth class. Ensure folder names match expected class strings (case-insensitive matching is attempted).
- Model class names: the evaluator tries multiple ways to extract `model.names` (ultralytics API changed across versions). If mapping fails, predictions will be numeric ids — the script attempts to map numeric ids back to your folder names when possible.
- If you see many `"none"` predictions (no detection), try lowering `--conf` (e.g., `0.2`) or increasing `imgsz`.
- If `results.csv` is empty, check:
    - `MODEL_PATH` exists (or allow autoload for `yolov8n.pt`),
    - `REAL_DATASET_ROOT` path is correct,
    - image extensions are supported (`.jpg/.png/...`).

### Quick troubleshooting
- **Missing packages**: scripts auto-install required packages, but prefer installing manually in venv.
- **Model aut-download**: passing `yolov8n.pt` string will let Ultralytics auto-download the official weights if not present.
- **Confusion matrix rendering**: uses Plotly; open the produced `confusion_matrix.html` in a browser.
- **Large GPU memory / CPU-only**: set `--device cpu` to avoid CUDA errors (training uses CPU in your environment logs).
---
