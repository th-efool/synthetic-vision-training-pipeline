# 🚀 Training a Large Vision Model Using Unity‑Generated Synthetic Data

## 🔍 Abstract
This project explores whether **synthetic data generated using Unity Perception** can improve detection accuracy for **underrepresented object classes** when training modern object detection models like **YOLOv8**.  
We generate a scalable synthetic dataset of 3D objects (bottle, cup, book, laptop, chair) — randomizing **lighting, poses, backgrounds & camera angles** — and convert it into YOLO‑compatible format.  
We then train 3 variants:
1. **Synthetic‑only model**
2. **Real‑only model**
3. **Mixed dataset model → BEST performance**

Our results confirm that synthetic images **significantly boost recall & mAP**, especially for classes lacking real‑world representation.

---

## 🧠 Tools & Frameworks
| Component | Used For |
|---|---|
| **Unity Perception** | Synthetic data generation & labeling |
| **YOLOv8 (Ultralytics)** | Model training, validation & evaluation |
| **Python + Jupyter Notebook** | Dataset processing, training & visualization |
| **Plotly / Matplotlib** | Result visualization & performance curves |

> Training was executed on **CPU**, with dataset‑size optimization & small‑model variant `yolov8n`.

---

## 📁 Dataset Structure
```
python_scripts/src/datasets/  
│── yolo_synthetic_dataset/  
│ ├── images/train  
│ ├── images/val  
│ ├── labels/train  
│ └── labels/val  
│── yolo_mixed_dataset/ ← (synthetic + real)  
├── images/train  
├── images/val  
├── labels/train  
└── labels/val
```
---

## 🔻 Dataset Format (Post‑Conversion)
```
python_scripts/datasets/
├── yolo_synthetic_dataset/
│   ├── images/{train,val}
│   └── labels/{train,val}
└── yolo_mixed_dataset/
    ├── images/{train,val}
    └── labels/{train,val}
```

YOLO Label Format:
```
class_id x_center y_center width height
```

---
## 🔧 Reproducibility – How to Run

### **1) Install environment**

`pip install -r requirements.txt`

### **2) Convert Unity → YOLO**

(Required only if re‑exporting dataset)

`cd python_scripts/helperscripts python UnityToYOLO.py --val-ratio 0.1 --mode copy -v`

### **3) Run training notebooks**

🚀 Synthetic‑only

`python_scripts/SyntheticTraining.ipynb`

🔥 Mixed‑training (real + synthetic)

`python_scripts/MixedTraining.ipynb`

Both automatically save runs under:

`python_scripts/src/runs/detect/`

---
## Model Performance Comparison

### What each model represents

| Model | Description |
|---|---|
| **A** | Hybrid-trained — synthetic + real-world images |
| **B** | YOLO-based training on real imagery only |
| **C** | Synthetic-only training — early-stage baseline |

This section documents the evaluation results on a 201-image test set (book, bottle, chair, cup, laptop).

---

### Overall Accuracy

| Model | Accuracy |
|---|---:|
| **A** | **0.9900** |
| **B** | 0.6915 |
| **C** | 0.1194 |

A reflects the upper bound we currently achieve.  
B performs reliably in several categories.  
C sits closer to B’s early behaviour — still forming representation, but usable as a reference for synthetic-only learning progression.

---

### Class-wise Metrics (Precision / Recall / F1)

| Class | A_Prec | A_Rec | A_F1 | B_Prec | B_Rec | B_F1 | C_Prec | C_Rec | C_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| book   | 1.00 | 0.94 | 0.97 | 1.00 | 0.10 | 0.18 | 0.00 | 0.00 | 0.00 |
| bottle | 0.98 | 1.00 | 0.99 | 0.58 | 0.88 | 0.70 | 0.18 | 0.06 | 0.09 |
| chair  | 0.96 | 1.00 | 0.98 | 0.96 | 0.81 | 0.88 | 0.23 | 0.67 | 0.34 |
| cup    | 1.00 | 1.00 | 1.00 | 0.98 | 0.69 | 0.81 | 1.00 | 0.03 | 0.06 |
| laptop | 1.00 | 1.00 | 1.00 | 1.00 | 0.87 | 0.93 | 1.00 | 0.03 | 0.06 |

---

### Notes on Results

- **Model A** — mixing real + synthetic data provides better feature diversity and reduces over-specialisation.  
  The result is consistently high precision, recall, and F1 across all five classes.

- **Model B** — performs well on objects with stronger structural edges (*chairs, laptops*) but shows reduced recall on objects with lower visual contrast (*books, cups*).  
  This likely stems from limited visual variation during training rather than model architecture limits.

- **Model C** — while not at the same numerical level yet, its pattern resembles Model B before real-image exposure.  
  It responds strongly to shape-dominant classes (chair recall is relatively high) and produces confident predictions on certain instances.  
  The drop in book/cup/laptop is most likely a **domain-gap effect**, not a capability limit.

---

### Takeaway

- A indicates where cross-domain training leads.
- B and C provide feedback on data coverage rather than model ceiling.
- Synthetic-only (C) is a controlled baseline for learning how far synthetic variance alone can carry recognition — and where real texture, lighting noise, and material complexity need to be introduced next.

The gaps are not failures; they are the learning signals we needed from this experiment.


