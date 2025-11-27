## Week 4 Progress Log
**Project:** *Training a Large Model Using Unity Perception Images*  
**Dates:** Nov 20, 2025 – Nov 27, 2025

---

### 1. Objectives
- Produce the final synthetic dataset using Unity Perception.  
- Prepare the real-image dataset and mixed dataset structure.  
- Train and compare **synthetic-only** vs **mixed** YOLOv8 models.  
- Finish documentation and reproducibility instructions.

---

### 2. Work Completed
- Generated per-class Unity Perception captures:
  - configured `FixedLengthScenario` for each object class,
  - used consistent seeds and randomizers (position/rotation/light) for reproducibility,
  - exported all frames under `SyntheticOutput/solo/sequence.*`.  
- Ran `UnityToYOLO.py` on the final output, producing:

  `yolo_synthetic_dataset/images/{train,val}` and `labels/{train,val}` + `dataset.yaml`.

- Organized final datasets under:

  `2. python_scripts/src/datasets/`  
  - `yolo_synthetic_dataset/`  
  - `yolo_mixed_dataset/` (synthetic + real).

- Implemented and ran two Jupyter notebooks:
  - `SyntheticTraining.ipynb` → trains YOLOv8 on synthetic-only data.  
  - `MixedTraining.ipynb` → trains YOLOv8 on combined real + synthetic data.

- Saved training artifacts:
  - `runs/detect/synth_only_run/` and `runs/detect/mixed_run/`,  
  - metrics plots (loss curves, PR curves),  
  - sample prediction images on real test images.

- Wrote/cleaned:
  - **Reproducibility** section (Python + Unity steps),  
  - dataset setup instructions for `3. datasets/` → `2. python_scripts/src/datasets/`,  
  - Week 1–4 logs and initial README content.

---

### 3. Issues / Blockers
- Environment differences (package versions, PyTorch / CUDA) required pinning key versions in `requirements.txt`.  
- Some Unity sequences produced off-center or missing boxes; mitigated by filtering low-quality frames and adjusting randomizer ranges.  
- Mixed dataset initially had class imbalance; fixed by re-running Unity captures to roughly match class counts.

---

### 4. Next Steps
- Record the **demo video**:
  - brief project overview,
  - walkthrough of Unity scene and converter,
  - run of `SyntheticTraining.ipynb` and `MixedTraining.ipynb`,
  - discussion of metrics and visual examples.  
- Final pass on README, docs, and results figures.  
- Double-check that a fresh clone + following the reproducibility steps successfully recreates training and evaluation.

---

### Summary
By the end of Week 4, the full pipeline is in place: Unity Perception → YOLO dataset → synthetic-only and mixed training → evaluation and visual results. The project is feature-complete and ready for final polishing and video recording.
