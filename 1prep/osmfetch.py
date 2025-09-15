#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_osm_graphs.py

Batch-download OSM road networks with OSMnx and save as GraphML.
This script accepts list files whose lines are either:
  - "District, City, Country"  (district-level)
  - "City, Country"            (city-level)
Edit the two user-config paths below; other settings remain unchanged.
"""

import os
import sys
import argparse
import osmnx as ox

# === CONFIG ================================================================
LIST_ROOT = r"0data/meta/analysis"  # Folder containing the .txt lists or a parent (relative to repo root).

# === OTHER SETTINGS (UNCHANGED) =============================================
NETWORK_TYPE      = "drive"   # 'drive' | 'walk' | 'bike' | 'all'
USE_CACHE         = True
LOG_CONSOLE       = False
OVERPASS_ENDPOINT = "https://overpass.openstreetmap.fr/api/interpreter"
TIMEOUT           = 240
RETRY_COUNT       = 3
# ============================================================================

FAILED_CITIES = []  # record failed items (kept for compatibility)

def ensure_dir(path):
    """Ensure directory exists; create if missing."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def download_place_graph(place_name, out_dir):
    """
    Download network by administrative boundary for `place_name`.
    Saves to {out_dir}/{slug}.graphml; on failure, log and continue.
    """
    ensure_dir(out_dir)
    slug = (
        place_name.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("/", "-")
    )
    out_fp = os.path.join(out_dir, f"{slug}.graphml")
    if os.path.isfile(out_fp):
        print(f"[~] exists, skip: {slug}.graphml")
        return
    try:
        print(f"[→] downloading: {place_name}")
        G = ox.graph_from_place(
            place_name,
            network_type=NETWORK_TYPE,
            simplify=False
        )
        ox.save_graphml(G, out_fp)
        print(f"[✔] saved: {out_fp}")
    except Exception as e:
        print(f"[!] failed {place_name}: {e}")
        # do not raise; continue to next

def main():
    parser = argparse.ArgumentParser(
        description="Find .txt lists under a folder; for each parent folder, create 'graphml/group_XX' and save GraphML there."
    )
    parser.add_argument(
        "list_root",
        nargs="?",
        default="0data/meta/analysis",
        help="Folder containing .txt lists (relative to repo root)."
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.list_root)
    if not os.path.isdir(root_dir):
        print(f"Error: folder not found: {root_dir}")
        sys.exit(1)

    ox.settings.use_cache = USE_CACHE
    ox.settings.log_console = LOG_CONSOLE
    ox.settings.overpass_endpoint = OVERPASS_ENDPOINT
    ox.settings.timeout = TIMEOUT
    ox.settings.retry_count = RETRY_COUNT

    dir_to_txts = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        txts = [os.path.join(dirpath, fn) for fn in filenames if fn.lower().endswith(".txt")]
        if txts:
            dir_to_txts[dirpath] = sorted(txts)

    if not dir_to_txts:
        print(f"Error: no .txt lists found under: {root_dir}")
        sys.exit(1)

    for d, txts in sorted(dir_to_txts.items()):
        out_base = os.path.join(d, "graphml")
        ensure_dir(out_base)
        for idx, txt in enumerate(txts, 1):
            group_dir = os.path.join(out_base, f"group_{idx:02d}")
            ensure_dir(group_dir)

            print("\n" + "=" * 80)
            print(f"[LIST ] {os.path.relpath(txt, start=root_dir)}")
            print(f"[GROUP] group_{idx:02d}")
            print(f"[OUT  ] {os.path.relpath(group_dir, start=root_dir)}")

            with open(txt, "r", encoding="utf-8") as f:
                raw_lines = [ln.strip() for ln in f if ln.strip()]

            seen = set()
            places = []
            for ln in raw_lines:
                if ln.lstrip().startswith("#"):
                    continue
                if ln not in seen:
                    places.append(ln)
                    seen.add(ln)

            if not places:
                print("[WARN] empty list, skip.")
                continue

            for place in places:
                download_place_graph(place, group_dir)

    print("\n[✔] Done.")

if __name__ == "__main__":
    main()
