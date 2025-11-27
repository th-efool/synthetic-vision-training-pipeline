#!/usr/bin/env python3
"""
auto_annotate_yolo.py

Auto-generate YOLO-format bounding boxes from images in class folders.

Changes from original:
 - --input and --output are optional.
 - If --input is omitted, current directory (".") is used and class folders are inferred.
 - If --output is omitted, a folder named "dataset" is created in current directory.
 - If --classes is omitted, class names are inferred from subfolders of input.
 - Keeps previous CV heuristic fallback (full-image bbox if nothing detected).
Usage examples:
  python auto_annotate_yolo.py
  python auto_annotate_yolo.py --input ./real_images --output ./dataset
  python auto_annotate_yolo.py --classes bottle cup book laptop chair
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
    p.add_argument("--input", "-i", required=False, help="Input root with class subfolders. (default: current directory)")
    p.add_argument("--output", "-o", required=False, help="Output dataset root to create images/labels train/val. (default: ./dataset)")
    p.add_argument("--classes", "-c", nargs="+", required=False, help="Ordered list of class names (indices 0..N-1). If omitted, inferred from input folders.")
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
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pick polarity
    try:
        if np.mean(gray[th==255]) < np.mean(gray[th==0]):
            th = cv2.bitwise_not(th)
    except Exception:
        pass
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
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
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes

def write_yolo_label_file(label_path: Path, class_id: int, boxes: List[Tuple[int,int,int,int]], img_w:int, img_h:int):
    lines = []
    for (x1,y1,x2,y2) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw/2.0
        cy = y1 + bh/2.0
        lines.append(f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

def main():
    args = parse_args()
    input_root = Path(args.input) if args.input else Path(".")
    out_root = Path(args.output) if args.output else Path("dataset")
    exts = [e.lower() if e.startswith('.') else f".{e.lower()}" for e in args.exts]

    if not input_root.exists():
        print(f"Input root not found: {input_root.resolve()}")
        sys.exit(1)

    # infer classes from args or from subfolders
    if args.classes and len(args.classes) > 0:
        class_names = args.classes
    else:
        # find subfolders in input_root (exclude hidden)
        class_names = [p.name for p in sorted([p for p in input_root.iterdir() if p.is_dir() and not p.name.startswith('.')])]
        if not class_names:
            print("No class folders found in input directory and --classes not provided.")
            print("Either create class subfolders under the input directory or pass --classes.")
            sys.exit(1)

    ensure_dirs(out_root)
    images_train_dir = out_root / "images" / "train"
    images_val_dir   = out_root / "images" / "val"
    labels_train_dir = out_root / "labels" / "train"
    labels_val_dir   = out_root / "labels" / "val"
    if args.vis:
        vis_dir = out_root / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    # collect images
    image_class_pairs = []
    for class_folder_name in class_names:
        class_folder = input_root / class_folder_name
        if not class_folder.exists() or not class_folder.is_dir():
            print(f"Warning: class folder '{class_folder}' not found under input -> skipping")
            continue
        for img in list_images_in_classfolder(class_folder, exts):
            image_class_pairs.append((img, class_folder_name))

    if not image_class_pairs:
        print("No images found in input class folders. Exiting.")
        sys.exit(1)

    # deterministic shuffle and split
    rng = random.Random(int(args.seed))
    rng.shuffle(image_class_pairs)
    split_index = int(len(image_class_pairs) * (1.0 - float(args.val)))
    train_pairs = image_class_pairs[:split_index]
    val_pairs   = image_class_pairs[split_index:]
    print(f"Total images: {len(image_class_pairs)} (train={len(train_pairs)}, val={len(val_pairs)})")
    min_area_ratio = float(args.min_area_ratio)

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
            if not boxes:
                boxes = [(0,0,w,h)]  # fallback
            # copy image to output (preserve filename; avoid collisions by prefixing if needed)
            dst_img = images_out_dir / src_path.name
            if dst_img.exists():
                # prefix with incremental counter
                i = 1
                while (images_out_dir / f"{dst_img.stem}_{i}{dst_img.suffix}").exists():
                    i += 1
                dst_img = images_out_dir / f"{dst_img.stem}_{i}{dst_img.suffix}"
            shutil.copy2(src_path, dst_img)
            # write label file
            class_id = class_to_index[cls_name]
            label_file = labels_out_dir / (dst_img.stem + ".txt")
            write_yolo_label_file(label_file, class_id, boxes, w, h)
            # optional visualization
            if args.vis:
                vis = img.copy()
                for (x1,y1,x2,y2) in boxes:
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
                text = f"{cls_name} ({class_id})"
                cv2.putText(vis, text, (max(2,x1), max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.imwrite(str(vis_dir / dst_img.name), vis)

    process_pairs(train_pairs, "train")
    process_pairs(val_pairs, "val")

    # write dataset yaml (relative paths)
    yaml_content = "names:\n" + "\n".join([f"  - {n}" for n in class_names]) + f"\n\nnc: {len(class_names)}\n\ntrain: images/train\nval: images/val\n"
    (out_root / "dataset.yaml").write_text(yaml_content)
    print("Done. Output written to:", out_root.resolve())
    print(" - images/train, images/val, labels/train, labels/val")
    if args.vis:
        print(" - visualizations in:", vis_dir.resolve())

if __name__ == "__main__":
    main()
