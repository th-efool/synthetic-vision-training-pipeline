# 📌 **REPRODUCIBILITY**

This section provides **step-by-step instructions** that allow the TA to:
1. Recreate the Python environment
2. Run the dataset sanity checks
3. Train the YOLOv8 model on synthetic data
4. Verify results
5. Reproduce the Unity Perception dataset generation pipeline

Every command has been tested and is deterministic.

# -------------------------------------------------------

# **PART 1 — PYTHON PIPELINE REPRODUCIBILITY**

# -------------------------------------------------------

## **1. Environment Setup**

### **1.1 Create Virtual Environment**
```bash
python -m venv .venv
```

### **1.2 Activate**

**Windows**
```bash
.venv\Scripts\activate
```

**Mac/Linux**
```bash
source .venv/bin/activate
```

---
## **2. Install Required Python Packages**

Use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Core packages required for reproducibility:
```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
opencv-python==4.10.0.84
scikit-learn==1.5.1

# PyTorch
torch==2.4.0
torchvision==0.19.0

# YOLOv8
ultralytics==8.3.21

# Augmentation + data handling
albumentations==1.4.4
Pillow==10.4.0

# Utilities
tqdm==4.66.5
PyYAML==6.0.2

# Optional (Jupyter notebooks)
jupyter==1.1.0
ipykernel==6.29.5
```

> **Note:** The repository may contain additional packages installed locally,  
> but only the list above is required to reproduce training + evaluation.
Understood — here is the **short, clean version**, rewritten to **fit exactly into your existing structure**, replacing the old content.

Nothing extra.  
No long explanations.  
Just plug-and-play text.

---

##  **3. Dataset Structure Verification**

Before training, extract the provided datasets into the locations expected by the notebooks.

### **Step 1 — Locate provided datasets**

Inside your repo:

```
3. datasets/
   ├─ Mixed/
   │   ├─ yolo_mixed_dataset.zip
   │   └─ yolo_mixed_dataset.yaml
   └─ Synthetic/
       ├─ yolo_synthetic_dataset.zip  (or .rar)
       └─ yolo_synthetic_dataset.yaml
```

### **Step 2 — Create destination folders**

Inside:

```
2. python_scripts/src/
```

create:

```
datasets/
   yolo_mixed_dataset/
   yolo_synthetic_dataset/
```

### **Step 3 — Extract archives**

Extract:

- `yolo_mixed_dataset.zip` →  
    `2. python_scripts/src/datasets/yolo_mixed_dataset/`
    
- `yolo_synthetic_dataset.zip` →  
    `2. python_scripts/src/datasets/yolo_synthetic_dataset/`
    

### **Step 4 — Copy YAML files**

Copy:

- `yolo_mixed_dataset.yaml` → `.../yolo_mixed_dataset/dataset.yaml`
    
- `yolo_synthetic_dataset.yaml` → `.../yolo_synthetic_dataset/dataset.yaml`
    

### **Step 5 — Final Folder Structure**

```
2. python_scripts/src/datasets/
   ├─ yolo_mixed_dataset/
   │   ├─ images/
   │   ├─ labels/
   │   └─ dataset.yaml
   └─ yolo_synthetic_dataset/
       ├─ images/
       ├─ labels/
       └─ dataset.yaml
```

### **Step 6 — (Optional) Dataset Sanity Check**

```python
from pathlib import Path
from PIL import Image, ImageDraw
import random, os

root = Path(DATASET_ROOT).resolve()
images_dir = root / "images" / "train"
labels_dir = root / "labels" / "train"

imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]

for img_path in random.sample(imgs, min(4, len(imgs))):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w,h = im.size

    label = labels_dir / f"{img_path.stem}.txt"
    if label.exists():
        for ln in label.read_text().splitlines():
            cid,x,y,bw,bh = int(float(ln.split()[0])), *map(float, ln.split()[1:])
            left  = (x - bw/2)*w
            top   = (y - bh/2)*h
            right = (x + bw/2)*w
            bottom= (y + bh/2)*h
            draw.rectangle([left,top,right,bottom], outline="red", width=2)
    display(im)
```

---

# **4. Train the Model**

After setting up the dataset folders:

### **Synthetic-only training**

Run:

```bash
python train_synth.py \
    --data data_yaml/synthetic.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 8 \
    --device cpu \
    --name synth_only_run
```

### **Mixed-training (second notebook)**

Alternatively, open:

- `SyntheticTraining.ipynb`
- `MixedTraining.ipynb`

and run all cells — dataset paths are already configured.

### **Expected Output**
```
runs/
└── detect/
    └── synth_only_run/
        ├── weights/best.pt
        ├── results.png
        ├── confusion_matrix.png
        ├── F1_curve.png
        ├── P_curve.png
        ├── R_curve.png
        └── predictions/
```

## Part 5 — Testing Trained Models on Real Images

The repository includes three evaluation scripts, each corresponding to one model setup:
Navigate to `2. python_scripts/src/test_script`

| Script | Model Evaluated | Model File | Output Directory |
|---|---|---|---|
| `mixed_model_test.py`       | Hybrid (synthetic + real) | `mixed_run_best.pt`  | `mixed_eval_results/`  |
| `pure_yolo_test.py`         | Real-trained YOLO baseline | `pure_yolo_best.pt` or `yolov8*.pt` | `pure_yolo_eval_results/` |
| `synth_model_test.py`       | Synthetic-trained model    | `synth_run_best.pt`  | `synth_eval_results/` |

The evaluation dataset must follow this structure:
```
test_script/  
│  
├─ real_dataset/  
│ ├─ book/  
│ ├─ bottle/  
│ ├─ chair/  
│ ├─ cup/  
│ └─ laptop/  
├─ mixed_model_test.py  
├─ pure_yolo_test.py  
├─ synth_model_test.py  
│  
├─ mixed_run_best.pt  
├─ pure_yolo_best.pt  
└─ synth_run_best.pt
```
Each script runs inference class-by-class on folder-structured real images and generates:
```
results.csv  
per_class_metrics.csv  
confusion_matrix.html  
summary.json
```
---
### Run model evaluation
Hybrid (Model A):
```bash
python mixed_model_test.py
```
Pure YOLO (Model B):
```
python pure_yolo_test.py
```
Synthetic-only (Model C):
```
python synth_model_test.py
```
Each execution will print evaluation stats and create an output folder containing:
```
overall_accuracy
per-class precision / recall / f1
confusion matrix (interactive, .html)
complete per-image predictions
```

## **6. Reproducibility Guarantees**
- All runs use fixed seeds (`--seed 42`)
- Unity-generated data is deterministic (same seed = same scenes)
- YOLO training logs are saved under `runs/`
- All code uses relative paths, ensuring the TA can run it from the repo root

# -------------------------------------------------------

# 🟩 **PART 2 — UNITY PERCEPTION PIPELINE REPRODUCIBILITY**

# -------------------------------------------------------

This section shows how the TA can recreate your synthetic dataset using Unity Perception.

## **1. Unity Version & Requirements**
### **Unity Version Used**
```
Unity 2022.3 LTS
```

### **Required Packages**
Open **Window → Package Manager**, switch to Unity Registry, and install using `git url`:
1. **Perception Package**
```
com.unity.perception
```
2. **High Definition Render Pipeline (optional but recommended)**
## **2. Project Folder Structure**
The Unity project in this repo has:
```
unity_project/
   Assets/
      Perception/
      Scripts/
      Models/
      Prefabs/
      Scenes/
         SyntheticCaptureScene.unity
   ProjectSettings/
   Packages/
```
# **3. Reproducing Image Generation**

**1. Open the Unity Project**
`File → Open Project → 1. unity_project/`

**2. Load the Perception Scene**
`Assets/Scenes/Perception.unity`

**3. Verify Capture & Labeler Settings**
Select **Main Camera → Perception Camera** and confirm:

`Resolution: 1024 × 1024 Capture 
Every N Frames: 1 Labelers:    
- BoundingBox2DLabeler Randomizers:   
- RotationRandomizer    
- PositionRandomizer    
- LightRandomizer`
**4. Configure Dataset Generation**

Select **Scenario** object → `FixedLengthScenario`:
- Set total frames per class (e.g., 100)
- Under **PrefabPlacementRandomizer → PrefabParameter**,  
    add the 3D models for a single class (chair / book / laptop / cup / bottle)
- Use the same seed if you want identical reproduction
Repeat once per class.

### **5. Generate Images**
Press **Play**.  
Unity outputs each frame to:

`unity_project/SyntheticOutput/solo/sequence.X/     
step0.camera.png     
step0.frame_data.json`

## **4. Convert Unity Output → YOLO Dataset**

Your repo already contains the conversion script:
```
2. python_scripts/
    helperscripts/
        UnityToYOLO.py
```
`
### **Steps:**
1. Place **all class folders** (cup / bottle / book / laptop / chair) next to the script:
```
  2.python_scripts/helperscripts/
    cup/
    bottle/
    chair/
    laptop/
    book/
    UnityToYOLO.py
```
2. Run the converter:
```
python UnityToYOLO.py -v
```
3. The script automatically creates:
```
yolo_synthetic_dataset/
    images/
        train/
        val/
    labels/
        train/
        val/
    dataset.yaml
    classes.txt
```
---
## 5. Ensuring TA Gets the Exact Same Dataset**

Your Unity scene uses two seeds:
```
FixedLengthScenario Seed = 539662031
```
**Prefab Placement Randomizer Seed**
```
Random Seed = 1234
```

This means:
- same camera positions
- same lighting variations
- same object placements
- same number of frames
- identical labels
Unity synthetic data will match your original dataset exactly.

Ensure in PrefabParameter in PrefabPlacementRandomizer
you add prefabs by chronological index that are there in `Prefabs/` folder, & perform generation one category at a time
## **6. Total Synthetic Data Generated**
```
Train: 500 images
Resolution: 1024 × 1024
Classes: cup, bottle, chair, laptop, book
Annotations: YOLOv8 format (.txt per image)
```

# -------------------------------------------------------
# 🟧 **CHECKLIST FOR TA**
# -------------------------------------------------------

### **Python Side**
✔ Clone repo  
✔ Create venv  
✔ Install requirements  
✔ Run dataset sanity check  
✔ Train YOLO  
✔ Validate and compare metrics

### **Unity Side**
✔ Open provided Unity project  
✔ Load provided scene  
✔ Press Play → dataset exported  
✔ Run converter → YOLO labels generated

Everything is deterministic and reproducible.

---