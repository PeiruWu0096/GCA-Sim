#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPD (Network Portrait Divergence) evaluation script.

Changes in this revision:
- All user-editable I/O paths and seed are placed at the very top.
- Comments are translated to concise English; redundant/legacy explanations removed.
- Keeps outputs and filenames consistent with the previous script:
  - analysis_metrics.json / analysis_metrics.csv
  - cluster_assignments.csv
  - pairwise_distance_matrix.csv
  - similarity_heatmap_ordered.png (+ leaf_order/matrix/linkage/style sidecars)
- No training; forward-only evaluation over *.graphml files.
"""

# =========================
# 0) Paths & seed
# =========================
# Default: district set; replace "district" with "city" to test the city set.
GRAPH_DIR = r"0data/meta/analysis/district/graphml"
# Results root; a subfolder named after this script will be created.
SAVE_DIR_BASE = r"0data/output/analysissave"
SEED = 3407
USE_ARGPARSE = False

# (Optional) If you prefer CLI override, set USE_ARGPARSE=True and run:
#   python eval_npd.py --seed 3415 --save_dir /path/to/save --graph_dir /path/to/graphs
USE_ARGPARSE = False

# =========================
# Imports & basic setup
# =========================
import os
import math
import json
import csv
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import networkx as nx

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# 1) Core hyper-parameters
# =========================
NUM_ITER = 5       # Iterations here are not used by NPD (kept for compatibility)
MAX_K    = 25      # Upper bound for #clusters during agglomerative cut
MIN_K    = 3       # Lower bound (>=2); matches prior evaluation convention

# =========================
# 2) Similarity & clustering
# =========================
class SimilarityAndClustering:
    """
    Utilities to convert a per-graph "portrait" distance matrix into
    clustering assignments and validation metrics.
    """

    def __init__(self, bins: int = 128):
        self.bins = bins
        self.eps  = 1e-8

    def cluster_and_metrics(self, sims: torch.Tensor):
        """
        sims: torch.Tensor [K,K], a *distance* matrix (smaller = more similar).
        Returns: labels (np.ndarray[K]), metrics (dict), Z (linkage matrix)
        """
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE")
        sims_np = sims.detach().cpu().numpy()
        Kcur = sims_np.shape[0]

        # Symmetrize & clean
        D = sims_np.copy()
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)

        # Single linkage tree; later we cut for different k
        dvec = squareform(D, checks=False)
        Z = linkage(dvec, method='average')

        # Classical MDS embedding for CH/DB (deterministic)
        dd = D ** 2
        n  = dd.shape[0]
        J  = np.eye(n) - np.ones((n, n)) / n
        B  = -0.5 * J.dot(dd).dot(J)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]; V = V[:, idx]
        k_emb = min(8, int((w > 1e-12).sum()) or 1)
        X_emb = V[:, :k_emb] * np.sqrt(np.maximum(w[:k_emb], 0.0))
        X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)

        # Enumerate k in [MIN_K, min(MAX_K, Kcur)], choose by rank sum of Sil↑ + DB↓
        records = []
        k_max = int(min(MAX_K, Kcur))
        for k in range(max(MIN_K, 2), max(MIN_K + 1, k_max + 1)):
            labels_k = fcluster(Z, t=k, criterion='maxclust') - 1
            if np.unique(labels_k).size < 2:
                continue
            try:
                sil_k = silhouette_score(D, labels_k, metric='precomputed')
            except Exception:
                sil_k = -1.0
            try:
                ch_k = calinski_harabasz_score(X_emb, labels_k)
            except Exception:
                ch_k = -1.0
            try:
                db_k = davies_bouldin_score(X_emb, labels_k)
            except Exception:
                db_k = np.inf
            records.append((k, labels_k, sil_k, ch_k, db_k))

        if not records:
            i, j = np.unravel_index(np.argmax(D), D.shape)
            labels = np.zeros(Kcur, dtype=int)
            labels[j] = 1
        else:
            ks, lbls, sils, chs, dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs  = np.asarray(dbs,  dtype=float)
            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)   # Sil: larger is better
            db_rk  = ( np.argsort(np.argsort(dbs ))).astype(float)   # DB : smaller is better
            best_idx = int(np.argmin(sil_rk + db_rk))
            labels = lbls[best_idx]

        # Basic intra/inter distances
        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(D[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(D[mask_diff].mean()) if mask_diff.any() else float('nan')

        # Soft silhouette (scaled rows + median tau)
        with torch.no_grad():
            SOFTSIL_TAU = 0.20
            d_t = torch.from_numpy(D).to(device=sims.device, dtype=sims.dtype)
            K = d_t.size(0)
            eye = torch.eye(K, device=d_t.device, dtype=torch.bool)

            BIG = 1e6
            d_for_sort = d_t + eye.to(d_t.dtype) * BIG
            vals, _ = torch.sort(d_for_sort, dim=1)
            n_off = max(1, K - 1)
            q_idx = max(0, min(n_off - 1, int(round(0.75 * (n_off - 1)))))
            row_scale = vals[:, q_idx:q_idx+1].clamp_min(1e-6)

            d_scaled = (d_t / row_scale).masked_fill(eye, 0.0)
            off_vals = d_t[~eye]
            tau = (off_vals.median().clamp_min(1e-6)
                   if off_vals.numel() > 0
                   else torch.tensor(SOFTSIL_TAU, device=d_t.device, dtype=d_t.dtype))

            NEG_BIG = 1e9
            logits_intra = (-d_scaled / tau).masked_fill(eye, -NEG_BIG)
            logits_inter = ( d_scaled / tau).masked_fill(eye, -NEG_BIG)

            p_intra = torch.softmax(logits_intra, dim=1)
            a_i = (p_intra * d_scaled).sum(dim=1) / (p_intra.sum(dim=1) + 1e-6)

            p_inter = torch.softmax(logits_inter, dim=1)
            b_i = (p_inter * d_scaled).sum(dim=1) / (p_inter.sum(dim=1) + 1e-6)

            sil_i = (b_i - a_i) / (b_i + a_i + 1e-6)
            soft_sil_val = float(sil_i.mean().item())

        try:
            sil_pre = silhouette_score(D, labels, metric='precomputed')
        except Exception:
            sil_pre = -1.0
        try:
            ch = calinski_harabasz_score(X_emb, labels)
        except Exception:
            ch = -1.0
        try:
            db = davies_bouldin_score(X_emb, labels)
        except Exception:
            db = np.inf

        return labels, {
            'Cluster_Count': int(np.unique(labels).size),
            'Intra_Dist': float(intra),
            'Inter_Dist': float(inter),
            'Silhouette': float(sil_pre),
            'Soft_Silhouette': float(soft_sil_val),
            'Calinski_Harabasz': float(ch),
            'Davies_Bouldin': float(db)
        }, Z

# =========================
# 3) NPD: portraits & distance
# =========================
class NPDMethod:
    """
    Network Portrait Divergence (Bagrow & Bollt): approximate network portraits
    via truncated BFS histograms and compute pairwise JSD-based distances.
    """

    def __init__(self, similarity: SimilarityAndClustering, seed: int):
        self.sim = similarity
        self.seed = seed
        self.L_MAX = 10             # Max BFS layer (row count = L_MAX+1)
        self.SAMPLE_CAP = 1500      # Max BFS sources for large graphs
        self.EXACT_NODE_CAP = 8000  # If N <= this threshold, use all nodes
        self.RNG = np.random.default_rng(self.seed)

    def _approximate_portrait(self, G: nx.Graph):
        """
        Returns:
          row_counters: list[Counter] of length L_MAX+1
          S: number of BFS sources used
        """
        N = G.number_of_nodes()
        if N == 0:
            return [Counter() for _ in range(self.L_MAX + 1)], 0
        nodes = list(G.nodes())
        if N <= self.EXACT_NODE_CAP:
            sources = nodes
        else:
            S = min(self.SAMPLE_CAP, N)
            sources = self.RNG.choice(nodes, size=S, replace=False).tolist()

        row_counters = [Counter() for _ in range(self.L_MAX + 1)]
        for s in sources:
            dist = nx.single_source_shortest_path_length(G, s, cutoff=self.L_MAX)
            counts = np.bincount(list(dist.values()), minlength=self.L_MAX + 1)[:self.L_MAX + 1]
            for l in range(self.L_MAX + 1):
                k = int(counts[l]) if l < counts.size else 0
                row_counters[l][k] += 1
        return row_counters, len(sources)

    @torch.no_grad()
    def run(self, graph_paths):
        # 1) Read graphs and normalize to simple undirected graphs without self-loops
        graphs = []
        for i, gp in enumerate(graph_paths, 1):
            print(f"\rForward {i}/{len(graph_paths)}", end="")
            G = nx.read_graphml(gp)
            if G.is_directed():
                G = G.to_undirected()
            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            graphs.append(G)
        print()

        # 2) Build approximate portraits
        portraits = []
        samples_per_graph = []
        for G in graphs:
            rc, S = self._approximate_portrait(G)
            portraits.append(rc)
            samples_per_graph.append(S)

        # 3) JSD over aligned rows (concatenated) → sqrt(JS) per pair
        def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
            M = 0.5 * (P + Q)
            eps = 1e-12
            kl1 = np.sum(np.where(P > 0, P * (np.log(P + eps) - np.log(M + eps)), 0.0))
            kl2 = np.sum(np.where(Q > 0, Q * (np.log(Q + eps) - np.log(M + eps)), 0.0))
            return 0.5 * (kl1 + kl2)

        def npd_distance(p1, s1, p2, s2) -> float:
            L = max(len(p1), len(p2))
            rows_P, rows_Q = [], []
            for l in range(L):
                c1 = p1[l] if l < len(p1) else Counter()
                c2 = p2[l] if l < len(p2) else Counter()
                kmax = 0
                if c1: kmax = max(kmax, max(c1.keys()))
                if c2: kmax = max(kmax, max(c2.keys()))
                rowP = np.zeros(kmax + 1, dtype=np.float64)
                rowQ = np.zeros(kmax + 1, dtype=np.float64)
                for k, v in c1.items():
                    if k <= kmax: rowP[k] = v
                for k, v in c2.items():
                    if k <= kmax: rowQ[k] = v
                rows_P.append(rowP)
                rows_Q.append(rowQ)
            P = np.concatenate(rows_P) if rows_P else np.zeros(1, dtype=np.float64)
            Q = np.concatenate(rows_Q) if rows_Q else np.zeros(1, dtype=np.float64)
            P_sum = P.sum(); Q_sum = Q.sum()
            if P_sum > 0: P = P / P_sum
            if Q_sum > 0: Q = Q / Q_sum
            return float(np.sqrt(js_divergence(P, Q)))

        K = len(graphs)
        D = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            D[i, i] = 0.0
            for j in range(i + 1, K):
                d = npd_distance(portraits[i], samples_per_graph[i], portraits[j], samples_per_graph[j])
                D[i, j] = D[j, i] = d

        sims = torch.tensor(D, dtype=torch.float32, device=DEVICE)
        labels, metrics, Z = self.sim.cluster_and_metrics(sims)
        return sims, labels, metrics, Z

# =========================
# 4) Visualization
# =========================
def plot_similarity_blocked(sim_matrix: np.ndarray, city_names, labels, Z, save_path: str, style: dict = None):
    """
    Draw an ordered (by dendrogram leaves) distance heatmap with cluster blocks highlighted.
    """
    if style is None:
        style = {
            "cmap_colors": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
            "border_color": "#2b2b2b",
            "dendro_color": "#606060",
            "label_color":  "#2b2b2b",
            "gridline_color": "#d9d9d9",
            "cbar_label": "Network Portrait Divergence"
        }
    cmap = LinearSegmentedColormap.from_list("simseq", style["cmap_colors"])

    D = np.array(sim_matrix, dtype=float).copy()
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    D = 0.5 * (D + D.T)
    K = D.shape[0]

    leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = np.asarray(leaves, dtype=int)

    D_ord = D[order][:, order]
    cities_ord = [city_names[i] for i in order]
    labels_ord = np.asarray(labels)[order]

    # Make the central heatmap square (each cell is a square)
    cell = 0.28
    S = max(6.0, K * cell)
    wd = max(1.2, 0.25 * S)
    hd = max(1.0, 0.22 * S)
    cbar_w = 0.45
    margin_w, margin_h = 0.6, 0.6
    fig_w = wd + S + cbar_w + margin_w
    fig_h = S + hd + margin_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    gs = gridspec.GridSpec(2, 3, width_ratios=[wd, S, cbar_w], height_ratios=[S, hd], wspace=0.02, hspace=0.02)
    ax_dy   = fig.add_subplot(gs[0, 0])
    ax_hm   = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_dx   = fig.add_subplot(gs[1, 1])

    dendrogram(Z, ax=ax_dy, orientation='left',   color_threshold=None, no_labels=True)
    ax_dy.set_ylim(-0.5, K - 0.5)
    ax_dy.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for sp in ax_dy.spines.values():
        sp.set_visible(False)

    dendrogram(Z, ax=ax_dx, orientation='bottom', color_threshold=None, no_labels=True)
    ax_dx.set_xlim(-0.5, K - 0.5)
    ax_dx.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for sp in ax_dx.spines.values():
        sp.set_visible(False)

    vmin, vmax = float(np.nanmin(D_ord)), float(np.nanmax(D_ord) if np.nanmax(D_ord) > 0 else 1.0)
    im = ax_hm.imshow(
        D_ord, cmap=cmap, interpolation='nearest',
        origin='lower', aspect='equal',
        extent=[-0.5, K - 0.5, -0.5, K - 0.5],
        vmin=vmin, vmax=vmax
    )
    ax_hm.set_xticks(np.arange(K))
    ax_hm.set_yticks(np.arange(K))
    ax_hm.set_xticklabels(cities_ord, rotation=90, fontsize=8, color=style["label_color"])
    ax_hm.set_yticklabels(cities_ord, fontsize=8, color=style["label_color"])
    ax_hm.tick_params(axis='both', which='both', length=0, pad=1.5)
    ax_hm.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax_hm.grid(which='minor', color=style["gridline_color"], linewidth=0.5)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(style["cbar_label"], fontsize=9, color=style["label_color"])

    # Draw cluster blocks
    boundaries = []
    start = 0
    for i in range(1, K):
        if labels_ord[i] != labels_ord[i - 1]:
            boundaries.append((start, i - 1))
            start = i
    boundaries.append((start, K - 1))
    for (s, e) in boundaries:
        w = e - s + 1
        rect = Rectangle((s - 0.5, s - 0.5), w, w, fill=False, linewidth=2.0, edgecolor=style["border_color"])
        ax_hm.add_patch(rect)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    # Sidecar files for replotting/debug
    base = os.path.splitext(save_path)[0]
    with open(base + "_leaf_order.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index_in_order", "City", "Cluster"])
        for i, (c, lab) in enumerate(zip(cities_ord, labels_ord.tolist())):
            w.writerow([i, c, int(lab)])
    with open(base + "_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cities_ord)
        for i in range(K):
            w.writerow([cities_ord[i]] + [f"{D_ord[i, j]:.6g}" for j in range(K)])
    np.save(base + "_linkage.npy", Z)
    try:
        with open(base + "_style.json", "w", encoding="utf-8") as f:
            json.dump(style, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =========================
# 5) Main
# =========================
def main():
    global GRAPH_DIR, SAVE_DIR, SEED
    if USE_ARGPARSE:
        import argparse
        p = argparse.ArgumentParser(description="NPD Graph Similarity Analysis")
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--save_dir', type=str, required=True)
        p.add_argument('--graph_dir', type=str, required=True)
        a = p.parse_args()
        SEED, GRAPH_DIR = a.seed, a.graph_dir
        SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
        SAVE_DIR = os.path.join(a.save_dir, SCRIPT_NAME)
    else:
        SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
        SAVE_DIR = os.path.join(SAVE_DIR_BASE, SCRIPT_NAME)

    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # List *.graphml
    all_graphs = []
    for root, _, files in os.walk(GRAPH_DIR):
        for f in files:
            if f.lower().endswith('.graphml'):
                all_graphs.append(os.path.join(root, f))
    all_graphs.sort()

    if len(all_graphs) == 0:
        raise RuntimeError(f"No .graphml files found under: {GRAPH_DIR}")
    if len(all_graphs) != 50:
        print(f"[WARN] Found {len(all_graphs)} graphs; expected 50 (continuing with all found).")

    city_names = [os.path.splitext(os.path.basename(p))[0] for p in all_graphs]

    # Run NPD → distances → clustering → metrics
    sim = SimilarityAndClustering(bins=128)
    method = NPDMethod(similarity=sim, seed=SEED)

    sims_t, labels, metrics, Z = method.run(all_graphs)

    # Save metrics
    metrics_path_json = os.path.join(SAVE_DIR, 'analysis_metrics.json')
    metrics_path_csv  = os.path.join(SAVE_DIR, 'analysis_metrics.csv')
    with open(metrics_path_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Save assignments
    assign_csv = os.path.join(SAVE_DIR, 'cluster_assignments.csv')
    with open(assign_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['City', 'Cluster'])
        for c, lab in zip(city_names, labels.tolist()):
            w.writerow([c, int(lab)])

    # Save distance matrix
    sims_np = sims_t.detach().cpu().numpy()
    def save_square_csv(mat, labels, path):
        K = mat.shape[0]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([''] + labels)
            for i in range(K):
                w.writerow([labels[i]] + [f"{x:.6g}" for x in mat[i].tolist()])
    save_square_csv(sims_np, city_names, os.path.join(SAVE_DIR, 'pairwise_distance_matrix.csv'))

    # Heatmap
    heatmap_path = os.path.join(SAVE_DIR, 'similarity_heatmap_ordered.png')
    plot_similarity_blocked(sims_np, city_names, labels, Z, heatmap_path)

    # Console summary
    print("\n===== Analysis Summary =====")
    print(f"Seed: {SEED}")
    print(f"Clusters: {metrics['Cluster_Count']}")
    def _fmt4(v: float) -> str:
        try:
            av = abs(float(v))
        except Exception:
            return str(v)
        return f"{v:.4e}" if (av != 0.0 and av < 1e-3) else f"{v:.4f}"
    print(f"Intra_Dist: {_fmt4(metrics['Intra_Dist'])} | Inter_Dist: {_fmt4(metrics['Inter_Dist'])}")
    print(f"Silhouette(precomputed dist): {_fmt4(metrics['Silhouette'])} | Soft_Silhouette: {_fmt4(metrics['Soft_Silhouette'])}")
    print(f"Calinski-Harabasz: {_fmt4(metrics['Calinski_Harabasz'])} | Davies-Bouldin: {_fmt4(metrics['Davies_Bouldin'])}")
    print(f"Saved to: {SAVE_DIR}")
    print(f"- cluster_assignments.csv")
    print(f"- pairwise_distance_matrix.csv")
    print(f"- {os.path.basename(heatmap_path)}")
    print(f"- analysis_metrics.json / analysis_metrics.csv")

if __name__ == "__main__":
    main()
