#!/usr/bin/env python3
"""
UnityToYOLO.py

Improved behavior:
- Uses a two-pass approach to read all JSONs first and build a map of labelId->labelName where possible.
- Canonicalizes label names (lowercase, strip, simple plural -> singular) to robustly map messy names like "books" -> "book".
- Enforces your exact requested class order (1:bottle,2:cup,3:book,4:laptop,5:chair) and maps every annotation to that 0-based index.
- If an annotation only has a numeric labelId (no labelName), the script will use the discovered labelId->labelName mapping. If still unknown, falls back to the sequence folder's class.
- Writes proper label files with 0-based indices 0..4.

Run: python UnityToYOLO.py -v
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

# User-requested canonical order (1-based human mapping -> we use 0-based internally)
REQUESTED_ORDER = ["bottle", "cup", "book", "laptop", "chair"]
CANON_TO_IDX = {name: idx for idx, name in enumerate(REQUESTED_ORDER)}


def find_class_folders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')])


def find_sequence_dirs_in_class(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])


def find_pair_in_seq(seq_dir: Path) -> Optional[Tuple[Path, Path]]:
    png_candidates = list(seq_dir.glob("*.camera.png")) or list(seq_dir.glob("*.png"))
    json_candidates = list(seq_dir.glob("*frame_data.json")) or list(seq_dir.glob("*.frame_data.json")) or list(seq_dir.glob("*.json"))
    if not png_candidates or not json_candidates:
        return None
    return (png_candidates[0], json_candidates[0])


def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canonicalize_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    # simple plural removal: books -> book, chairs -> chair
    if s.endswith('ies'):
        s = s[:-3] + 'y'
    elif s.endswith('s') and not s.endswith('ss'):
        s = s[:-1]
    # common typos normalization
    s = s.replace(' ', '_')
    return s


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
                # extract labelId and labelName if present
                class_id = v.get('labelId') if 'labelId' in v else v.get('label_id')
                label_name = v.get('labelName') or v.get('label') or v.get('labelName')
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
                results.append((class_id, label_name, x_center, y_center, width, height))
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
        if lines:
            f.write('\n'.join(lines) + '\n')
        else:
            f.write('')


def main(val_ratio: float, mode: str, include_empty: bool, verbose: bool) -> None:
    detected_class_dirs = find_class_folders(SCRIPT_DIR)
    if not detected_class_dirs:
        print('No class folders found next to the script. Put your class folders (cup, book, etc.) next to this script and run again.')
        return

    detected_names = [p.name for p in detected_class_dirs]
    # Build final classes list: prefer requested order, then append extras
    classes: List[str] = [name for name in REQUESTED_ORDER if name in detected_names]
    classes += [name for name in detected_names if name not in classes]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    print('Detected classes (final order):', classes)

    # collect all image/json pairs first
    all_items: List[Tuple[Path, Path, str]] = []  # img, json, class_folder
    for name in classes:
        class_dir = SCRIPT_DIR / name
        if not class_dir.exists() or not class_dir.is_dir():
            if verbose:
                print(f'skipping missing class folder: {name}')
            continue
        seq_dirs = find_sequence_dirs_in_class(class_dir)
        for seq in seq_dirs:
            pair = find_pair_in_seq(seq)
            if pair is None:
                if verbose:
                    print(f'skipping sequence (no pair): {seq}')
                continue
            img_path, json_path = pair
            all_items.append((img_path, json_path, name))

    if not all_items:
        print('No image+json pairs found in any sequences.')
        return

    # First pass: build labelId -> labelName mapping by scanning all JSONs
    id_to_name: Dict[int, str] = {}
    for img_path, json_path, _ in all_items:
        try:
            parsed = load_json(json_path)
        except Exception:
            continue
        captures = parsed.get('captures') if isinstance(parsed, dict) else None
        if not captures or not isinstance(captures, list):
            continue
        for cap in captures:
            anns = cap.get('annotations') if isinstance(cap, dict) else None
            if not anns or not isinstance(anns, list):
                continue
            for ann in anns:
                vals = ann.get('values') if isinstance(ann, dict) else None
                if not vals or not isinstance(vals, list):
                    continue
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    lid = v.get('labelId') if 'labelId' in v else v.get('label_id')
                    lname = v.get('labelName') or v.get('label') or None
                    if lid is not None and lname is not None:
                        try:
                            lid_int = int(lid)
                            cname = canonicalize_name(lname)
                            if cname:
                                id_to_name[lid_int] = cname
                        except Exception:
                            pass
    if verbose:
        print('Discovered labelId->name mapping (from JSONs):', id_to_name)

    # Prepare output dirs and classes file
    prepare_output_dirs(OUTPUT_ROOT)
    with open(OUTPUT_ROOT / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(classes) + '\n')

    random.shuffle(all_items)
    split_index = int(len(all_items) * (1.0 - val_ratio))

    for i, (img_path, json_path, class_folder_name) in enumerate(all_items):
        split = 'train' if i < split_index else 'val'
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            if verbose:
                print(f'failed to open image {img_path}: {e}')
            continue

        try:
            parsed = load_json(json_path)
        except Exception as e:
            if verbose:
                print(f'failed to load json {json_path}: {e}')
            parsed = {}

        annotations_parsed = parse_unity_perception_frame(parsed, (img_w, img_h))

        final_annotations: List[Tuple[int,float,float,float,float]] = []
        for lid, lname, x_c, y_c, w_n, h_n in annotations_parsed:
            mapped_idx: Optional[int] = None
            # If labelName provided, use canonicalized name -> requested mapping
            c_name = canonicalize_name(lname)
            if c_name and c_name in CANON_TO_IDX:
                mapped_idx = CANON_TO_IDX[c_name]
            # else if numeric label id present, try discovered id->name mapping
            if mapped_idx is None and lid is not None:
                try:
                    lid_int = int(lid)
                    if lid_int in id_to_name:
                        cand = id_to_name[lid_int]
                        if cand in CANON_TO_IDX:
                            mapped_idx = CANON_TO_IDX[cand]
                    else:
                        # assume Unity labelId is 1-based and corresponds to REQUESTED_ORDER index
                        # i.e., labelId 1 -> bottle (index 0)
                        if 1 <= lid_int <= len(REQUESTED_ORDER):
                            mapped_idx = lid_int - 1
                except Exception:
                    pass
            # fallback: assume sequence folder class
            if mapped_idx is None:
                mapped_idx = class_to_idx.get(class_folder_name, 0)

            # final sanity
            if mapped_idx < 0:
                mapped_idx = 0
            if mapped_idx >= len(REQUESTED_ORDER):
                # skip annotations that map outside requested classes
                if verbose:
                    print(f'skipping annotation with mapped_idx={mapped_idx} for {img_path}')
                continue

            final_annotations.append((mapped_idx, x_c, y_c, w_n, h_n))

        if not final_annotations and not include_empty:
            if verbose:
                print(f'skipping {img_path} (no annotations found)')
            continue

        unique_stem = str(i)
        dst_image = OUTPUT_ROOT / f"images/{split}/{unique_stem}{img_path.suffix}"
        dst_label = OUTPUT_ROOT / f"labels/{split}/{unique_stem}.txt"

        copy_or_link(img_path, dst_image, mode)
        write_yolo_label(dst_label, final_annotations)

        if verbose:
            print(f'[{split}] {img_path} -> {dst_image} labels={len(final_annotations)}')

    # Write YAML file automatically (with requested order)
    yaml_path = OUTPUT_ROOT / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write("names:\n")
        for name in REQUESTED_ORDER:
            yf.write(f"  - {name}\n")
        yf.write(f"\nnc: {len(REQUESTED_ORDER)}\n")
        yf.write(f"\ntrain: {OUTPUT_ROOT}/images/train\n")
        yf.write(f"val: {OUTPUT_ROOT}/images/val\n")

    print('Done. Output written to', OUTPUT_ROOT)
    print('Classes file:', OUTPUT_ROOT / 'classes.txt')
    print('YAML file:', yaml_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Unity Perception -> YOLO dataset (no path args required)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fraction reserved for validation')
    parser.add_argument('--mode', choices=('copy','symlink'), default='copy', help='How to transfer images (default copy)')
    parser.add_argument('--include-empty', action='store_true', help='Keep images without annotations')
    parser.add_argument('-v','--verbose', action='store_true')
    args = parser.parse_args()
    main(args.val_ratio, args.mode, args.include_empty, args.verbose)