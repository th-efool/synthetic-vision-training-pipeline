#!/usr/bin/env python3
# auto_annotate_yolo.py
"""
Auto-generate YOLO-format bounding boxes from images in class folders.

Input structure:
  input_root/
    book/
      img1.jpg
      img2.png
    cup/
      ...
    ...

Output structure (created):
  output_root/
    images/train, images/val
    labels/train, labels/val

Usage:
  python auto_annotate_yolo.py --input input_root --output output_dataset \
      --classes bottle cup book laptop chair --val 0.1 --seed 0

Notes:
 - This uses simple CV heuristics (grayscale + Otsu + morphology + contour area)
   and works best when the object is prominent on a simple/contrasting background.
 - If no contour is found for an image, a fallback full-image bbox is written.
 - You can tune min_area_ratio for stricter filtering, or enable visualization.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import shutil
import sys
from typing import List, Tuple
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Auto-generate YOLO labels from class-folders of images.")
    p.add_argument("--input", "-i", required=True, help="Input root with class subfolders.")
    p.add_argument("--output", "-o", required=True, help="Output dataset root to create images/labels train/val.")
    p.add_argument("--classes", "-c", nargs="+", required=True, help="Ordered list of class names (indices 0..N-1).")
    p.add_argument("--val", type=float, default=0.1, help="Validation fraction (default 0.1).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for train/val split.")
    p.add_argument("--min_area_ratio", type=float, default=0.002, help="Min contour area ratio to accept (default 0.002).")
    p.add_argument("--vis", action="store_true", help="Save a visualization image per input into output/vis (for quick QC).")
    p.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"], help="Allowed extensions.")
    return p.parse_args()

def ensure_dirs(root: Path):
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

def list_images_in_classfolder(folder: Path, exts: List[str]):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

def detect_bboxes_from_image(img_bgr: np.ndarray, min_area_ratio: float=0.002) -> List[Tuple[int,int,int,int]]:
    """
    Return list of bounding boxes (x1,y1,x2,y2) detected in image using threshold+contours.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # blur then Otsu
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If background is bright and object dark, invert so object is foreground
    # Choose polarity by comparing mean of foreground area
    if np.mean(gray[th==255]) < np.mean(gray[th==0]):
        th = cv2.bitwise_not(th)

    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = max(1, int(min_area_ratio * w * h))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        if ww <= 3 or hh <= 3:
            continue
        boxes.append((x, y, x+ww, y+hh))

    # Sort boxes by area descending (largest first)
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes

def write_yolo_label_file(label_path: Path, class_id: int, boxes: List[Tuple[int,int,int,int]], img_w:int, img_h:int):
    """
    Writes YOLO-format lines: class_id cx cy w h
    """
    lines = []
    for (x1,y1,x2,y2) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw/2.0
        cy = y1 + bh/2.0
        lines.append(f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

def copy_image(dst_dir: Path, src_path: Path) -> Path:
    dst = dst_dir / src_path.name
    shutil.copy2(src_path, dst)
    return dst

def main():
    args = parse_args()
    input_root = Path(args.input)
    out_root = Path(args.output)
    class_names = args.classes
    val_frac = float(args.val)
    seed = int(args.seed)
    min_area_ratio = float(args.min_area_ratio)
    vis_flag = bool(args.vis)
    exts = [e.lower() if e.startswith('.') else f".{e.lower()}" for e in args.exts]

    if not input_root.exists():
        print("Input not found:", input_root)
        sys.exit(1)

    ensure_dirs(out_root)
    images_train_dir = out_root / "images" / "train"
    images_val_dir = out_root / "images" / "val"
    labels_train_dir = out_root / "labels" / "train"
    labels_val_dir = out_root / "labels" / "val"
    if vis_flag:
        vis_dir = out_root / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # map class folder -> index using provided classes list
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    # gather images and their class (based on parent folder name)
    image_class_pairs = []
    for class_folder in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        cls_name = class_folder.name
        if cls_name not in class_to_index:
            print(f"Warning: folder '{cls_name}' not in provided classes list -> skipping")
            continue
        img_list = list_images_in_classfolder(class_folder, exts)
        for img in img_list:
            image_class_pairs.append((img, cls_name))

    if not image_class_pairs:
        print("No images found in input folders. Exiting.")
        sys.exit(1)

    # deterministic shuffle + train/val split
    rng = random.Random(seed)
    rng.shuffle(image_class_pairs)
    split_index = int(len(image_class_pairs) * (1.0 - val_frac))
    train_pairs = image_class_pairs[:split_index]
    val_pairs = image_class_pairs[split_index:]

    print(f"Total images: {len(image_class_pairs)} (train={len(train_pairs)}, val={len(val_pairs)})")

    def process_pairs(pairs, split_name: str):
        images_out_dir = images_train_dir if split_name == "train" else images_val_dir
        labels_out_dir = labels_train_dir if split_name == "train" else labels_val_dir
        for src_path, cls_name in pairs:
            img = cv2.imread(str(src_path))
            if img is None:
                print("Failed to read:", src_path)
                continue
            h, w = img.shape[:2]
            boxes = detect_bboxes_from_image(img, min_area_ratio=min_area_ratio)

            # fallback: if nothing detected, create full-image bbox
            if not boxes:
                boxes = [(0,0,w,h)]

            # Optionally, you might want only the largest box (uncomment if desired)
            # boxes = [boxes[0]]

            # copy image to images_out_dir
            dst_img = images_out_dir / src_path.name
            shutil.copy2(src_path, dst_img)

            # write label file
            class_id = class_to_index[cls_name]
            label_file = labels_out_dir / (src_path.stem + ".txt")
            write_yolo_label_file(label_file, class_id, boxes, w, h)

            # optional visualization image
            if vis_flag:
                vis = img.copy()
                for (x1,y1,x2,y2) in boxes:
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
                    text = f"{cls_name} ({class_id})"
                    cv2.putText(vis, text, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.imwrite(str(vis_dir / src_path.name), vis)

    process_pairs(train_pairs, "train")
    process_pairs(val_pairs, "val")

    # write dataset yaml file that references these folders (relative paths)
    yaml_content = f"""names:
{chr(10).join([f'  - {n}' for n in class_names])}
nc: {len(class_names)}
train: images/train
val: images/val
"""
    (out_root / "dataset.yaml").write_text(yaml_content)

    print("Done. Output written to:", out_root)
    print(" - images/train, images/val, labels/train, labels/val")
    if vis_flag:
        print(" - visualizations in:", vis_dir)

if __name__ == "__main__":
    main()
