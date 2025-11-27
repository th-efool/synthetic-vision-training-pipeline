#!/usr/bin/env python3
# fix_labels_1to0.py
from pathlib import Path
import shutil

ROOT = Path("yolo_synthetic_dataset")
LABEL_DIRS = [ROOT / "labels" / "train", ROOT / "labels" / "val"]
BACKUP_DIR = ROOT / "labels_backup_1based"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def process_file(p: Path):
    text = p.read_text(encoding="utf-8").strip()
    if text == "":
        return
    lines = text.splitlines()
    out_lines = []
    changed = False
    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            # non-numeric class id — skip or warn
            out_lines.append(line)
            continue
        new_cid = cid - 1
        if new_cid < 0:
            raise ValueError(f"Negative class id after conversion in file {p}: original {cid}")
        if new_cid != cid:
            changed = True
        out_lines.append(" ".join([str(new_cid)] + parts[1:]))
    if changed:
        # backup original
        target_backup = BACKUP_DIR / p.name
        if not target_backup.exists():
            shutil.copy2(p, target_backup)
        p.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        print(f"[fixed] {p}")
    else:
        print(f"[ok]   {p} (no change)")

def main():
    for d in LABEL_DIRS:
        if not d.exists():
            print("Skipping missing folder:", d)
            continue
        for f in sorted(d.glob("*.txt")):
            process_file(f)
    print("Done. Originals copied to:", BACKUP_DIR)

if __name__ == "__main__":
    main()
