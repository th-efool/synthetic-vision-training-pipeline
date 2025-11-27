#!/usr/bin/env python3
"""
UnityToYOLO.py

Auto-converts Unity Perception output organized into per-class folders (your current layout)
into a YOLO dataset at ./yolo_synthetic_dataset without requiring you to pass input/output paths.

Expected layout (what you have):

/your-working-dir/
  cup/               <-- class folder
    sequence.0/
      step0.camera.png
      step0.frame_data.json
    sequence.1/
      ...
  laptop/
    sequence.0/
    ...
  UnityToYOLO.py     <-- this script (or placed in the same folder)

What this script does:
- Detects all immediate subfolders in the script's directory and treats them as class names (sorted)
- Walks each class folder for sequence subfolders and looks for an image + frame_data.json pair
- Parses Unity Perception JSON (captures -> annotations -> values -> origin/dimension)
- Converts annotations to YOLO normalized format and writes labels
- Copies images into yolo_synthetic_dataset/images/{train,val}/ and labels into yolo_synthetic_dataset/labels/{train,val}/
- Writes yolo_synthetic_dataset/classes.txt automatically

No manual paths required. Run without args.

Optional CLI flags (only for val split and mode):
  --val-ratio  : fraction to reserve for validation (default 0.1)
  --mode       : copy or symlink images into the dataset (default copy)
  --include-empty : include images without annotations
  -v / --verbose : print progress

"""

from pathlib import Path
import json
import random
import shutil
import os
import argparse
from typing import Any, List, Tuple, Dict, Optional
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "yolo_synthetic_dataset"


def find_class_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')])


def find_sequence_dirs_in_class(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])


def find_pair_in_seq(seq_dir: Path) -> Optional[Tuple[Path, Path]]:
    # Look for *.camera.png (Unity) or any *.png and frame_data.json
    png_candidates = list(seq_dir.glob("*.camera.png")) or list(seq_dir.glob("*.png"))
    json_candidates = list(seq_dir.glob("*frame_data.json")) or list(seq_dir.glob("*.frame_data.json")) or list(seq_dir.glob("*.json"))
    if not png_candidates or not json_candidates:
        return None
    return (png_candidates[0], json_candidates[0])


def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_unity_perception_frame(parsed: Any, image_size: Tuple[int,int]) -> List[Tuple[Any,float,float,float,float]]:
    results: List[Tuple[Any,float,float,float,float]] = []
    if not isinstance(parsed, dict):
        return results
    captures = parsed.get('captures')
    if not isinstance(captures, list):
        return results
    img_w, img_h = image_size
    for capture in captures:
        if not isinstance(capture, dict):
            continue
        annotations = capture.get('annotations')
        if not isinstance(annotations, list):
            continue
        for ann in annotations:
            values = ann.get('values')
            if not isinstance(values, list):
                continue
            for v in values:
                if not isinstance(v, dict):
                    continue
                class_id = v.get('labelId') if 'labelId' in v else v.get('label_id')
                label_name = v.get('labelName') or v.get('label')
                class_key = class_id if class_id is not None else label_name
                origin = v.get('origin')
                dim = v.get('dimension')
                if not (isinstance(origin, list) and isinstance(dim, list) and len(origin) >= 2 and len(dim) >= 2):
                    continue
                left = float(origin[0])
                top = float(origin[1])
                width_px = float(dim[0])
                height_px = float(dim[1])
                x_center = (left + width_px / 2.0) / img_w
                y_center = (top + height_px / 2.0) / img_h
                width = width_px / img_w
                height = height_px / img_h
                # clamp
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                results.append((class_key, x_center, y_center, width, height))
    return results


def prepare_output_dirs(output_root: Path) -> None:
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError('mode must be copy or symlink')


def write_yolo_label(path: Path, annotations: List[Tuple[int,float,float,float,float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for idx, x, y, w, h in annotations:
        lines.append(f"{idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))


def main(val_ratio: float, mode: str, include_empty: bool, verbose: bool) -> None:
    class_dirs = find_class_folders(SCRIPT_DIR)
    if not class_dirs:
        print('No class folders found next to the script. Put your class folders (cup, book, etc.) next to this script and run again.')
        return

    classes = [p.name for p in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    print('Detected classes:', classes)

    prepare_output_dirs(OUTPUT_ROOT)
    # save classes.txt
    with open(OUTPUT_ROOT / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(classes))

    all_items: List[Tuple[Path, Path, str]] = []  # image, json, class_name

    for class_dir in class_dirs:
        seq_dirs = find_sequence_dirs_in_class(class_dir)
        if not seq_dirs and verbose:
            print(f'no sequences in class folder: {class_dir.name}')
        for seq in seq_dirs:
            pair = find_pair_in_seq(seq)
            if pair is None:
                if verbose:
                    print(f'skipping sequence (no pair): {seq}')
                continue
            img_path, json_path = pair
            all_items.append((img_path, json_path, class_dir.name))

    if not all_items:
        print('No image+json pairs found in any sequences.')
        return

    random.shuffle(all_items)
    split_index = int(len(all_items) * (1.0 - val_ratio))

    for i, (img_path, json_path, class_name) in enumerate(all_items):
        split = 'train' if i < split_index else 'val'
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            if verbose:
                print(f'failed to open image {img_path}: {e}')
            continue

        parsed = load_json(json_path)
        annotations_parsed = parse_unity_perception_frame(parsed, (img_w, img_h))

        final_annotations: List[Tuple[int,float,float,float,float]] = []
        # If parser returned annotations with labelName/labelId, map them to indices when possible.
        for class_key, x_c, y_c, w_n, h_n in annotations_parsed:
            # class_key might be numeric id or labelName
            mapped_idx = None
            if isinstance(class_key, int):
                mapped_idx = class_key
            else:
                if class_key is None:
                    mapped_idx = None
                else:
                    mapped_idx = class_to_idx.get(str(class_key))
            if mapped_idx is None:
                # Fallback: assume the folder this sequence lives in is the correct class for all objects
                mapped_idx = class_to_idx[class_name]
            final_annotations.append((mapped_idx, x_c, y_c, w_n, h_n))

        if not final_annotations and not include_empty:
            if verbose:
                print(f'skipping {img_path} (no annotations found)')
            continue

        # Make output filenames unique: use class_seqname_imagename
        seq_label = img_path.parent.name
        # simple numeric filenames instead of long descriptive names
        unique_stem = str(i)  # global index-based name
        dst_image = OUTPUT_ROOT / f"images/{split}/{unique_stem}{img_path.suffix}"
        dst_label = OUTPUT_ROOT / f"labels/{split}/{unique_stem}.txt"

        copy_or_link(img_path, dst_image, mode)
        write_yolo_label(dst_label, final_annotations)

        if verbose:
            print(f'[{split}] {img_path} -> {dst_image} labels={len(final_annotations)}')

    print('Done. Output written to', OUTPUT_ROOT)
    print('Classes file:', OUTPUT_ROOT / 'classes.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Unity Perception -> YOLO dataset (no path args required)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fraction reserved for validation')
    parser.add_argument('--mode', choices=('copy','symlink'), default='copy', help='How to transfer images (default copy)')
    parser.add_argument('--include-empty', action='store_true', help='Keep images without annotations')
    parser.add_argument('-v','--verbose', action='store_true')
    args = parser.parse_args()
    main(args.val_ratio, args.mode, args.include_empty, args.verbose)
