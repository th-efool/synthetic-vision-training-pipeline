#!/usr/bin/env python3
"""
ddg_debug.py
Quick diagnostic for DuckDuckGo image JSON endpoint.
Runs a single query and prints status, timing, first few URLs or raw response.
"""

import time
import requests
from urllib.parse import quote_plus

DDG_IJS = "https://duckduckgo.com/i.js"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}

def fetch_once(query="cup", timeout=8):
    params = {"q": query}
    start = time.time()
    try:
        r = requests.get(DDG_IJS, params=params, headers=HEADERS, timeout=timeout)
        elapsed = time.time() - start
        print(f"HTTP {r.status_code} - elapsed {elapsed:.2f}s")
        # small safety: print content-length header if present
        print("Content-Length header:", r.headers.get("Content-Length"))
        # try parse json
        try:
            data = r.json()
            # print keys and count
            keys = list(data.keys())
            print("JSON keys:", keys)
            results = data.get("results") or data.get("image_results") or data.get("results", [])
            print("Num results found:", len(results))
            # print first 8 result entries (compact)
            for i, item in enumerate(results[:8]):
                # pretty-safe extraction of url
                url = item.get("image") or item.get("thumbnail") or item.get("url") or item.get("src")
                print(f"{i+1:2d}. {url}")
            if len(results) == 0:
                print("Results present but empty. Raw response head:")
                print(r.text[:2000])
        except ValueError:
            print("Response not JSON. Raw head:")
            print(r.text[:2000])
    except requests.exceptions.RequestException as e:
        print("Request failed:", type(e).__name__, e)

if __name__ == "__main__":
    print("Testing DuckDuckGo image endpoint (one request)...")
    fetch_once("cup", timeout=8)
