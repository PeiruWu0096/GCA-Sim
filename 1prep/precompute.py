#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphML → precomputed .pt

Output structure:
{
  'node_states': Tensor[N,1]        # node degree (no normalization)
  'neighbors'  : List[LongTensor]   # each [deg_u]
  'space_info' : List[FloatTensor]  # each [deg_u,2] -> [dist01, angle_gap01]
}
"""

import os, math, random
from pathlib import Path
from functools import lru_cache
import argparse
import networkx as nx
import torch
import numpy as np

# ----------------- PATH & SEED -----------------
ROOT_DIR = r"0data/meta/analysis"   # Path to the parent folder; all GraphML files under it will be processed recursively.
RANDOM_SEED = 3704

random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

# ----------------- Geodesy utils -----------------
R_EARTH = 6_371_000.0  # meter

@lru_cache(maxsize=None)
def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0

def haversine_m(lon1, lat1, lon2, lat2) -> float:
    """Haversine distance in meters (fast)."""
    lon1, lat1, lon2, lat2 = map(_deg2rad, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R_EARTH * math.asin(math.sqrt(a))

def angle_deg(lon1, lat1, lon2, lat2) -> float:
    """Bearing: 0° = east, clockwise, return in [0,360)."""
    dx = (lon2 - lon1) * math.cos(_deg2rad((lat1 + lat2)/2))
    dy =  lat2 - lat1
    ang = math.degrees(math.atan2(dy, dx))
    return (ang + 360.0) % 360.0

# ----------------- Main pipeline -----------------
def process_graph(path_in: Path, path_out: Path):
    G = nx.read_graphml(path_in)
    # --- node index map ---
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # --- node initial states: node degree (no normalization) ---
    node_states = torch.tensor([float(G.degree(n)) for n in nodes], dtype=torch.float32).unsqueeze(1)

    # --- neighbors & spatial info ---
    neighbors   = []
    space_infos = []
    lons = nx.get_node_attributes(G, 'x')
    lats = nx.get_node_attributes(G, 'y')

    # try fallback 'lon'/'lat' if 'x'/'y' missing; otherwise raise
    if not lons or not lats:
        lons = nx.get_node_attributes(G, 'lon')
        lats = nx.get_node_attributes(G, 'lat')
    if not lons or not lats:
        raise ValueError(f"{path_in} missing node coordinates: need 'x/y' or 'lon/lat'")

    for u in nodes:
        idx_u = idx_map[u]
        # one pass neighbor list to drive nb & feats consistently
        neigh_nodes = list(G.neighbors(u))
        nb = [idx_map[v] for v in neigh_nodes]
        neighbors.append(torch.tensor(nb, dtype=torch.long))
        if nb:
            lon_u, lat_u = float(lons[u]), float(lats[u])

            # build distances/angles with the same neigh_nodes order
            dists = []
            angs  = []
            for v in neigh_nodes:
                lon_v, lat_v = float(lons[v]), float(lats[v])
                dists.append(haversine_m(lon_u, lat_u, lon_v, lat_v))
                angs.append(angle_deg(lon_u, lat_u, lon_v, lat_v))  # [0,360)
            deg_u = len(neigh_nodes)
            # sort by absolute angle and keep mapping to original indices
            order = sorted(range(deg_u), key=lambda i: angs[i])
            angs_sorted = [angs[i] for i in order]

            # circular gaps between successive sorted angles
            gaps = []
            for i in range(deg_u):
                a = angs_sorted[i]
                b = angs_sorted[(i + 1) % deg_u]
                d = (b - a) % 360.0
                gaps.append(d if d > 0.0 else 360.0)

            # for each edge (by its sorted position k), sum of adjacent gaps = gaps[k-1] + gaps[k]
            sum_two_sides = [0.0] * deg_u  # write back to original order
            if deg_u == 1:
                sum_two_sides[0] = 360.0
            else:
                for k, idx_orig in enumerate(order):
                    prev_k = (k - 1) % deg_u
                    next_k = k
                    sum_deg = gaps[prev_k] + gaps[next_k]  # (0,360], =360 when deg=2
                    sum_two_sides[idx_orig] = sum_deg

            # assemble features in the same order as nb/neigh_nodes
            # distance normalized by per-node max distance
            max_d = max(dists) if deg_u > 0 else 0.0
            denom = max(max_d, 1e-6)
            dist01 = [float(d) / denom for d in dists]

            # angle-gap normalized to [0,1]
            ang01 = [min(float(a), 360.0) / 360.0 for a in sum_two_sides]

            feats = [[dist01[i], ang01[i]] for i in range(deg_u)]
            space_infos.append(torch.tensor(feats, dtype=torch.float32))
        else:
            space_infos.append(torch.zeros((0, 2), dtype=torch.float32))
    # --- pack & save ---
    # consistency check: neighbors vs. space_info row counts must match
    mismatch = []
    for u, (nb_u, si_u) in enumerate(zip(neighbors, space_infos)):
        deg_nb = int(nb_u.numel())
        deg_si = int(si_u.size(0)) if isinstance(si_u, torch.Tensor) and si_u.ndim >= 1 else 0
        if deg_nb != deg_si:
            mismatch.append((u, deg_nb, deg_si))
    if mismatch:
        print(f"[WARN] {path_in} has {len(mismatch)} nodes with neighbor/space_info size mismatch; first 3: {mismatch[:3]}")

    data = {
        'node_states': node_states,
        'neighbors':   neighbors,
        'space_info':  space_infos
    }
    path_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path_out)

def main(root_dir: str):
    files = list(Path(root_dir).rglob("*.graphml"))
    total = len(files)
    print(f"Found {total} GraphML files; generating .pt ...")
    for i, f in enumerate(sorted(files), 1):
        # Determine base and group to mirror 'graphml/group_XX' → 'pre/group_XX'
        p = f.parent
        if p.name.startswith("group_") and p.parent.name == "graphml":
            group = p.name
            base  = p.parent.parent
        elif p.name == "graphml":
            group = "group_01"
            base  = p.parent
        else:
            group = "group_01"
            base  = p

        pre_dir = Path(base) / "pre" / group
        pre_dir.mkdir(parents=True, exist_ok=True)
        outf = pre_dir / (f.stem + ".pt")

        rel_show = f.relative_to(root_dir)
        print(f"[{i}/{total}] {rel_show}  →  {outf.relative_to(base)}", flush=True)
        try:
            process_graph(f, outf)
        except Exception as e:
            print(f"  [Error] {f}: {e}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Precompute tensors from GraphML files")
    ap.add_argument("root", nargs="?", default=ROOT_DIR, help="Parent folder to scan recursively")
    args = ap.parse_args()
    main(args.root)