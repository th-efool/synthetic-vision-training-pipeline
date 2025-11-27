## Week 3 Progress Log
**Project:** *Training a Large Model Using Unity Perception Images*  
**Dates:** Nov 13, 2025 – Nov 19, 2025

---

### 1. Objectives
- Replace placeholder data with proper **real** and **synthetic** datasets.  
- Implement a stable Unity → YOLO conversion pipeline.  
- Run a first full training on the synthetic-only dataset.  

---

### 2. Work Completed
- Finalized Unity Perception output format: each capture stored as `sequence.X/step0.camera.png` + `step0.frame_data.json`.  
- Implemented `UnityToYOLO.py` to:
  - parse Perception JSON,
  - convert 2D bounding boxes to YOLO `(class x y w h)` format,
  - split into `train/val`,
  - write `dataset.yaml` and `classes.txt`.  
- Generated a first batch of synthetic images for all five classes (cup, bottle, chair, laptop, book).  
- Structured repo datasets as:
  - `yolo_synthetic_dataset/`  
  - `yolo_mixed_dataset/` (placeholder for upcoming mixed data).  
- Ran an initial **synthetic-only** YOLOv8 training run to validate:
  - labels load correctly,
  - loss decreases over time,
  - sample predictions make sense on synthetic images.

---

### 3. Issues / Blockers
- Unity export path and folder naming were inconsistent at first (sequence folders, step filenames), requiring extra checks in the converter.  
- Some frames contained no valid bounding boxes → added logic to skip or handle empty annotations.  
- Class-name inconsistencies (e.g., “books” vs “book”) required canonicalization inside the converter script.

---

### 4. Next Steps
- Clean up Unity scenes and regenerate a more balanced synthetic dataset (similar image counts per class).  
- Finish curation of the real-image dataset and convert to YOLO format.  
- Set up **mixed** training (real + synthetic) and define evaluation splits.  
- Start logging metrics (mAP, per-class AP) in a consistent format for the final results section.

---

### Summary
The core Unity → YOLO conversion pipeline is now working, and synthetic-only training has been verified end-to-end. The project is ready to scale up data volume and move toward mixed-dataset experiments.
