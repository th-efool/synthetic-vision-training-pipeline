#!/usr/bin/env python3
"""
download_images_per_class_ddg.py

Downloads images for target classes using DuckDuckGo image JSON (i.js).
Creates folders under datasets/real_unlabeled/images/<class_name>.

Usage:
    python download_images_per_class_ddg.py --per_class 100 --workers 8
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse
import requests

# Optional tqdm progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}

DEFAULT_CLASSES = ["cup", "bottle", "chair", "laptop", "book"]

DDG_IJS = "https://duckduckgo.com/i.js"

def fetch_ddg_image_urls(query: str, max_urls: int = 100, pause: float = 0.4):
    """
    Use DuckDuckGo i.js to fetch image results for a query.
    Returns list of direct image URLs (strings), up to max_urls.
    """
    urls = []
    params = {"q": query}
    s = 0
    attempts = 0
    while len(urls) < max_urls and attempts < 50:
        params["s"] = s
        try:
            resp = requests.get(DDG_IJS, params=params, headers=HEADERS, timeout=15)
            # DDG sometimes returns 403 if abused; handle gracefully
            if resp.status_code != 200:
                # brief wait and retry
                attempts += 1
                time.sleep(1.0)
                continue
            data = resp.json()
        except Exception:
            attempts += 1
            time.sleep(1.0)
            continue

        results = data.get("results") or data.get("image_results") or data.get("results", [])
        if not results:
            # no more results
            break

        for item in results:
            # DuckDuckGo's JSON uses 'image' or 'thumbnail' or 'url' fields; handle common keys
            url = item.get("image") or item.get("thumbnail") or item.get("url") or item.get("src")
            if not url:
                # some entries contain 'icon' or nested fields
                possible = [v for k, v in item.items() if isinstance(v, str) and v.startswith("http")]
                url = possible[0] if possible else None
            if url and url not in urls:
                urls.append(url)
                if len(urls) >= max_urls:
                    break

        # update for next page (DDG uses 'next' via 'next' in JSON or s offset)
        s += len(results)
        time.sleep(pause)
    return urls[:max_urls]

def guess_extension_from_url(url: str, default=".jpg"):
    p = urlparse(url)
    name = Path(p.path).name
    if "." in name:
        ext = Path(name).suffix
        if ext.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"]:
            return ext
    # try to infer from query params or content-type later
    return default

def download_one(url: str, out_path: Path, timeout: int = 20):
    """
    Download a single url to out_path. Returns (ok:bool, info:str).
    """
    try:
        # skip if present
        if out_path.exists() and out_path.stat().st_size > 1000:
            return True, "exists"
        with requests.get(url, headers=HEADERS, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out_path)
        if out_path.stat().st_size < 500:
            out_path.unlink(missing_ok=True)
            return False, "too-small"
        return True, "ok"
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, str(e)

def download_for_class(class_name: str, per_class: int, out_root: Path, workers: int, pause_between_pages: float):
    safe_name = class_name.replace(" ", "_")
    target_dir = out_root / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{class_name}] Querying DuckDuckGo...")
    urls = fetch_ddg_image_urls(class_name, max_urls=per_class, pause=pause_between_pages)
    print(f"[{class_name}] Found {len(urls)} candidate URLs.")

    seen = set()
    clean_urls = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        clean_urls.append(u)
    urls = clean_urls

    tasks = []
    for i, u in enumerate(urls):
        ext = guess_extension_from_url(u)
        filename = f"{safe_name}_{i:04d}{ext}"
        out_path = target_dir / filename
        tasks.append((u, out_path))

    succ = 0
    fail = 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, u, out): (u, out) for (u, out) in tasks}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Downloading [{class_name}]"):
            ok, info = fut.result()
            url, outp = futures[fut]
            if ok:
                succ += 1
            else:
                fail += 1
            results.append((url, outp, ok, info))
    print(f"[{class_name}] Done. success={succ} failed={fail} saved_to={target_dir}")
    return results

def main():
    p = argparse.ArgumentParser(description="Download images per class into datasets/real_unlabeled/images/<class>/ using DuckDuckGo")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names to download")
    p.add_argument("--per_class", type=int, default=50, help="Images per class")
    p.add_argument("--workers", type=int, default=8, help="Parallel downloads")
    p.add_argument("--out_root", type=str, default="datasets/real_unlabeled/images", help="Root output folder")
    p.add_argument("--pause", type=float, default=0.4, help="Pause between DDG pages (seconds)")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Output root:", out_root.resolve())
    print("Classes:", args.classes)
    print("Per class:", args.per_class)
    print("Workers:", args.workers)
    print("Starting... (press Ctrl-C to stop)")

    for cls in args.classes:
        try:
            download_for_class(cls, args.per_class, out_root, args.workers, args.pause)
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            return
        except Exception as e:
            print(f"Error while processing class '{cls}': {e}")

    print("All done. Check images under:", out_root)

if __name__ == "__main__":
    import argparse
    main()
