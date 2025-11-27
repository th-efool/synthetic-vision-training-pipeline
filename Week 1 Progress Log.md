**Project:** _Training a Large Model Using Unity Perception Images_  
**Dates:** _Nov 3, 2025 – Nov 9, 2025_

---

### **1. Weekly Objective**
- Understand dataset imbalance issues in object detection.
- Set up Unity Perception to generate synthetic images.
- Explore real-world datasets suitable for augmentation.
### **2. Work Completed**
- Installed **Unity Hub**, **Unity Editor**, and the **Perception package**.
- Built a sample Perception scene with labeled objects.
- Generated an initial synthetic dataset (~100 labeled images).
- Reviewed external datasets; shortlisted:
    - **COCO 2017** (primary real-world source)
    - **Pascal VOC 2012** (secondary validation source)
- Installed **YOLOv8** and confirmed training pipeline works on a small test set.
### **3. Challenges**
- Unity output consumes significant storage and compute.
- Unity Perception provides **JSON labels**, while YOLOv8 requires **YOLO TXT** labels.
### **4. Next Steps**
- Implement JSON → YOLO label conversion.
- Prepare clean datasets for synthetic-only and mixed training.
- Start fine-tuning YOLOv8 on the combined dataset.
- Record early performance metrics for the baseline.
---
### **Summary**
Environment setup and dataset exploration are complete. The upcoming week will focus on data conversion, dataset structuring, and creating the first reproducible training pipeline.
