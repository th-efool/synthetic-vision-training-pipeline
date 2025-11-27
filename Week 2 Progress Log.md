## Week 2 Progress Log
**Project:** *Training a Large Model Using Unity Perception Images*  
**Date:** Nov 12, 2025

---

### 1. Objectives
- Convert Unity Perception JSON annotations into YOLO TXT format.  
- Prepare a small COCO 2017 subset (cup, bottle, chair, laptop, book).  
- Generate synthetic image samples to verify the processing pipeline.  
- Implement basic training and evaluation scripts.

---

### 2. Work Completed
- Wrote a synthetic data generation utility to produce ~250 synthetic-style images and ~250 placeholder “real” images for testing.  
- Implemented `download_coco_subset.py` to extract the required COCO classes into YOLO format (requires local COCO files).  
- Added initial training and evaluation scripts (`train.py`, `evaluate.py`) using the Ultralytics YOLO API.

---

### 3. Issues / Blockers
- COCO 2017 must be downloaded manually (Kaggle API), as it cannot be included in the repo.  
- Unity Editor is required for real Perception renders; placeholder synthetic images were used this week for pipeline validation.

---

### 4. Next Steps
- Download COCO `val2017` + `instances_val2017.json` and run:  
  `python src/download_coco_subset.py --coco_annotations <path> --images_dir <path>`  
- Replace placeholder real data with the actual COCO subset and retrain.  
- Export true Unity Perception synthetic dataset and replace placeholder synthetic samples.

---
