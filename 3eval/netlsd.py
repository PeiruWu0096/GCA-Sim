#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NetLSD evaluation (forward-only):
- Edit the two path constants below if you run without CLI flags.
- No training/annealing/schedulers; forward inference only.
- Outputs (names unchanged): analysis_metrics.json/.csv, cluster_assignments.csv,
  pairwise_distance_matrix.csv, similarity_heatmap_ordered.png
"""

import os
import math
import json
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import scipy.sparse as sp
import networkx as nx

# ========= 0) PATHS =========
# Default: district set; replace "district" with "city" to test the city set.
GRAPH_DIR = r'0data/meta/analysis/district/graphml'
# Results root; a subfolder named after this script will be created.
SAVE_DIR_DEFAULT = r'0data/output/analysissave'

# ========= 1) Seed & device =========
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= 2) Core constants (kept consistent with test/main) =========
NUM_ITER = 5          # number of state-update iterations (conceptual for alignment)
TEMP_INIT = 2.0       # placeholder (not important after hardening)
TEMP_MIN  = 0.08
MAX_K = 25
MIN_K = 3

# ========= 3) Similarity & clustering (NetLSD downstream stays identical) =========
class SimilarityAndClustering:
    def __init__(self, bins=128):
        self.bins = bins
        self.eps  = 1e-8

    def soft_hist(self, x_norm: torch.Tensor):
        # x_norm in [0,1]
        x = x_norm.view(-1).float()
        centers = torch.linspace(0.0, 1.0, self.bins, device=x.device, dtype=x.dtype)
        scale = 0.05 + 0.10 * x.std().clamp_min(1e-3)
        diffs = x.unsqueeze(1) - centers.unsqueeze(0)
        weights = torch.exp(-0.5 * (diffs / scale) ** 2)
        hist = weights.sum(dim=0) + self.eps
        return hist / hist.sum()

    def compute_js_distance(self, all_states):
        """
        all_states: List[List[Tensor]] (T states per graph).
        For each time step: min-max to [0,1] -> soft histogram -> JS distance;
        then average over time steps.
        """
        hists_per_graph = []
        for state_list in all_states:
            hist_ts = []
            for st in state_list:
                flat = st.view(-1).float()
                xmin, xmax = flat.min(), flat.max()
                if xmax <= xmin:
                    xmax = xmin + 1e-4
                x_norm = (flat - xmin) / (xmax - xmin)
                hist_ts.append(self.soft_hist(x_norm))
            hists_per_graph.append(torch.stack(hist_ts, dim=0))  # [T, bins]
        hists = torch.stack(hists_per_graph, dim=0)              # [K, T, bins]

        def js_div(p, q):
            m = 0.5 * (p + q)
            term1 = p * torch.log((p + 1e-12) / (m + 1e-12))
            term2 = q * torch.log((q + 1e-12) / (m + 1e-12))
            return 0.5 * (term1.sum(-1) + term2.sum(-1))

        sims = None
        T = hists.size(1)
        for t in range(T):
            a = hists[:, t, :]
            js = js_div(a.unsqueeze(1), a.unsqueeze(0))  # [K,K]
            sims = js if sims is None else sims + js
        sims = sims / T
        return sims  # This is a distance (smaller = more similar)

    def cluster_and_metrics(self, sims: torch.Tensor):
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE")
        sims_np = sims.detach().cpu().numpy()
        Kcur = sims_np.shape[0]

        # Precomputed distance matrix D
        D = sims_np.copy()
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)

        # Single linkage build; cut for different k
        dvec = squareform(D, checks=False)
        Z = linkage(dvec, method='average')

        # Classical MDS for CH/DB scores
        dd = D ** 2
        n  = dd.shape[0]
        J  = np.eye(n) - np.ones((n, n)) / n
        B  = -0.5 * J.dot(dd).dot(J)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        k_emb = min(8, int((w > 1e-12).sum()) or 1)
        X_emb = V[:, :k_emb] * np.sqrt(np.maximum(w[:k_emb], 0.0))
        X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)

        # Pick k via rank-sum: Sil ↑, DB ↓
        records = []
        k_max = int(min(MAX_K, Kcur))
        for k in range(MIN_K, max(MIN_K + 1, k_max + 1)):
            labels_k = fcluster(Z, t=k, criterion='maxclust') - 1
            if np.unique(labels_k).size < 2:
                continue
            try:
                sil_k = silhouette_score(D, labels_k, metric='precomputed')
            except Exception:
                sil_k = -1.0
            try:
                db_k = davies_bouldin_score(X_emb, labels_k)
            except Exception:
                db_k = np.inf
            records.append((k, labels_k, sil_k, db_k))

        if not records:
            i, j = np.unravel_index(np.argmax(D), D.shape)
            labels = np.zeros(Kcur, dtype=int)
            labels[j] = 1
        else:
            ks, lbls, sils, _dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs  = np.asarray(_dbs, dtype=float)
            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)
            db_rk  = ( np.argsort(np.argsort(dbs ))).astype(float)
            rank_sum = sil_rk + db_rk
            best_idx = int(np.argmin(rank_sum))
            labels = lbls[best_idx]

        # Intra/Inter & Soft-Sil (evaluation version)
        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(D[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(D[mask_diff].mean()) if mask_diff.any() else float('nan')

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
            tau = off_vals.median().clamp_min(1e-6) if off_vals.numel() > 0 else torch.tensor(SOFTSIL_TAU, device=d_t.device, dtype=d_t.dtype)
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

# ========= 4) Method wrapper (pure NetLSD) =========
class AnalysisMethod:
    """
    Encapsulate the baseline method to be easily swapped if needed.
    """
    def __init__(self, similarity: SimilarityAndClustering, seed: int):
        self.sim = similarity
        self.seed = seed

    @torch.no_grad()
    def run_once(self, graph_paths):
        # 1) Read graphs and sanitize (simple undirected, no self-loops)
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

        # 2) NetLSD time grid (log-spaced)
        T = 64
        t_min, t_max = 1e-3, 1e+3
        times = np.logspace(np.log10(t_min), np.log10(t_max), T).astype(np.float64)

        # 3) SLQ params (probes & Lanczos steps)
        R = 16
        m = 50

        def normalized_laplacian_sparse(G: nx.Graph) -> sp.csr_matrix:
            n = G.number_of_nodes()
            if n == 0:
                return sp.csr_matrix((0, 0), dtype=np.float64)
            idx = {u: i for i, u in enumerate(G.nodes())}
            rows, cols = [], []
            for u, v in G.edges():
                i, j = idx[u], idx[v]
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)
            data = np.ones(len(rows), dtype=np.float64)
            A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
            deg = np.asarray(A.sum(axis=1)).ravel()
            with np.errstate(divide='ignore'):
                dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            Dinv = sp.diags(dinv, offsets=0, dtype=np.float64, format='csr')
            I = sp.eye(n, dtype=np.float64, format='csr')
            L = I - (Dinv @ A @ Dinv)
            return L

        def lanczos_tridiag(A: sp.csr_matrix, v0: np.ndarray, m: int):
            # Symmetric Lanczos to build T_m (tridiagonal), sparse matvec only.
            n = A.shape[0]
            v_prev = np.zeros(n, dtype=np.float64)
            v = v0 / max(1e-12, np.linalg.norm(v0))
            alphas = np.zeros(m, dtype=np.float64)
            betas  = np.zeros(m - 1, dtype=np.float64)
            for k in range(m):
                w = A.dot(v)
                if k > 0:
                    w = w - betas[k - 1] * v_prev
                alpha = float(np.dot(v, w))
                w = w - alpha * v
                if k > 0:
                    gamma = float(np.dot(v_prev, w))
                    w = w - gamma * v_prev
                beta = float(np.linalg.norm(w))
                alphas[k] = alpha
                if k < m - 1:
                    betas[k] = beta
                v_prev, v = v, (w / max(1e-12, beta)) if beta > 0 else (v, np.zeros_like(v))
                if beta < 1e-14 and k < m - 1:
                    alphas = alphas[:k + 1]
                    betas  = betas[:k]
                    break
            Tm = np.diag(alphas)
            if betas.size > 0:
                Tm += np.diag(betas, k=1) + np.diag(betas, k=-1)
            return Tm

        RNG = np.random.default_rng(self.seed)

        def slq_heat_trace(L: sp.csr_matrix, times: np.ndarray, R: int, m: int, rng) -> np.ndarray:
            # Hutch++/SLQ-style trace(exp(-tL)) estimate via tridiagonalization.
            n = L.shape[0]
            if n == 0:
                return np.zeros_like(times, dtype=np.float64)
            acc = np.zeros_like(times, dtype=np.float64)
            for r in range(R):
                z = rng.choice([-1.0, 1.0], size=n).astype(np.float64)   # Rademacher probe
                Tm = lanczos_tridiag(L, z, m)
                evals, evecs = np.linalg.eigh(Tm)
                q1_sq = (evecs[0, :] ** 2).astype(np.float64)
                for i, t in enumerate(times):
                    acc[i] += float(np.sum(q1_sq * np.exp(-t * evals)))
            return (n / float(R)) * acc

        # 4) Per-graph NetLSD signatures h(t)
        signatures = []
        for G in graphs:
            L = normalized_laplacian_sparse(G)
            sig = slq_heat_trace(L, times, R=R, m=m, rng=RNG)
            signatures.append(sig)
        S = np.vstack(signatures)  # [K, T]

        # Scale: normalize by h(0)=n; cosine distance downstream
        denom = np.clip(S[:, 0:1], 1e-12, None)
        S_norm = S / denom
        S_norm = S_norm / np.clip(np.linalg.norm(S_norm, axis=1, keepdims=True), 1e-12, None)

        Kc = S_norm @ S_norm.T
        Kc = np.clip(Kc, -1.0, 1.0)
        D = np.sqrt(np.maximum(2.0 - 2.0 * Kc, 0.0))
        sims = torch.tensor(D, dtype=torch.float32, device=DEVICE)  # keep name "sims" (meaning: distances)

        labels, metrics, Z = self.sim.cluster_and_metrics(sims)
        return sims, labels, metrics, Z

# ========= 5) Visualization =========
def plot_similarity_blocked(sim_matrix: np.ndarray, city_names, labels, Z, save_path: str, style: dict = None):
    """
    Plot the similarity (distance) matrix with dendrograms and cluster blocks.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    if style is None:
        style = {
            "cmap_colors": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
            "border_color": "#2b2b2b",
            "dendro_color": "#606060",
            "label_color":  "#2b2b2b",
            "gridline_color": "#d9d9d9",
            "cbar_label": "NetLSD Distance"
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

    ax_dy = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_dx = fig.add_subplot(gs[1, 1])

    dendrogram(Z, ax=ax_dy, orientation='left', color_threshold=None, no_labels=True)
    ax_dy.set_ylim(-0.5, K - 0.5)
    ax_dy.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax_dy.spines.values():
        spine.set_visible(False)

    dendrogram(Z, ax=ax_dx, orientation='bottom', color_threshold=None, no_labels=True)
    ax_dx.set_xlim(-0.5, K - 0.5)
    ax_dx.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax_dx.spines.values():
        spine.set_visible(False)

    vmin, vmax = float(np.nanmin(D_ord)), float(np.nanmax(D_ord) if np.nanmax(D_ord) > 0 else 1.0)
    im = ax_hm.imshow(
        D_ord, cmap=cmap, interpolation='nearest', origin='lower', aspect='equal',
        extent=[-0.5, K - 0.5, -0.5, K - 0.5], vmin=vmin, vmax=vmax
    )
    ax_hm.set_xticks(np.arange(K)); ax_hm.set_yticks(np.arange(K))
    ax_hm.set_xticklabels(cities_ord, rotation=90, fontsize=8, color=style["label_color"])
    ax_hm.set_yticklabels(cities_ord, fontsize=8, color=style["label_color"])
    ax_hm.tick_params(axis='both', which='both', length=0, pad=1.5)
    ax_hm.set_xticks(np.arange(-0.5, K, 1), minor=True); ax_hm.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax_hm.grid(which='minor', color=style["gridline_color"], linewidth=0.5)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(style["cbar_label"], fontsize=9, color=style["label_color"])

    # draw cluster blocks
    boundaries = []
    start = 0
    for i in range(1, K):
        if labels_ord[i] != labels_ord[i - 1]:
            boundaries.append((start, i - 1)); start = i
    boundaries.append((start, K - 1))
    for (s, e) in boundaries:
        rect = Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1, fill=False, linewidth=2.0, edgecolor=style["border_color"])
        ax_hm.add_patch(rect)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    # Export table-form leaf order/matrix for replotting
    base = os.path.splitext(save_path)[0]
    with open(base + "_leaf_order.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["Index_in_order", "City", "Cluster"])
        for i, (c, lab) in enumerate(zip(cities_ord, labels_ord.tolist())):
            w.writerow([i, c, int(lab)])
    with open(base + "_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow([""] + cities_ord)
        for i in range(K):
            w.writerow([cities_ord[i]] + [f"{D_ord[i, j]:.6g}" for j in range(K)])

# ========= 6) Main =========
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NetLSD Graph Similarity Analysis")
    parser.add_argument('--seed', type=int, default=3415, help='Random seed.')
    parser.add_argument('--graph_dir', type=str, default=GRAPH_DIR, help='Root directory containing *.graphml.')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR_DEFAULT, help='Directory to save analysis results.')
    args = parser.parse_args()

    SEED = args.seed
    root_graph = args.graph_dir
    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    SAVE_DIR = os.path.join(args.save_dir, SCRIPT_NAME)

    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Discover graphs
    all_graphs = []
    for root, _, files in os.walk(root_graph):
        for f in files:
            if f.lower().endswith('.graphml'):
                all_graphs.append(os.path.join(root, f))
    all_graphs.sort()
    if not all_graphs:
        raise RuntimeError(f"No .graphml files found under: {root_graph}")
    if len(all_graphs) != 50:
        print(f"[WARN] Found {len(all_graphs)} graphs; expected 50. Will evaluate all discovered files.")

    city_names = [os.path.splitext(os.path.basename(p))[0] for p in all_graphs]

    sim = SimilarityAndClustering(bins=128)
    method = AnalysisMethod(similarity=sim, seed=SEED)
    sims_t, labels, metrics, Z = method.run_once(all_graphs)

    # ---- Persist outputs (names unchanged) ----
    metrics_path_json = os.path.join(SAVE_DIR, 'analysis_metrics.json')
    metrics_path_csv  = os.path.join(SAVE_DIR, 'analysis_metrics.csv')
    with open(metrics_path_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    assign_csv = os.path.join(SAVE_DIR, 'cluster_assignments.csv')
    with open(assign_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['City', 'Cluster'])
        for c, lab in zip(city_names, labels.tolist()):
            w.writerow([c, int(lab)])

    sims_np = sims_t.detach().cpu().numpy()
    def save_square_csv(mat, labels, path):
        K = mat.shape[0]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([''] + labels)
            for i in range(K):
                w.writerow([labels[i]] + [f"{x:.6g}" for x in mat[i].tolist()])
    save_square_csv(sims_np, city_names, os.path.join(SAVE_DIR, 'pairwise_distance_matrix.csv'))

    heatmap_path = os.path.join(SAVE_DIR, 'similarity_heatmap_ordered.png')
    plot_similarity_blocked(sims_np, city_names, labels, Z, heatmap_path)

    # Console summary
    print("\n===== Analysis Summary =====")
    print(f"Seed: {SEED}")
    print(f"Graphs: {len(all_graphs)} | Clusters: {metrics['Cluster_Count']}")
    def _fmt4(v: float) -> str:
        try:
            av = abs(float(v))
        except Exception:
            return str(v)
        return f"{v:.4e}" if (av != 0.0 and av < 1e-3) else f"{v:.4f}"
    print(f"Intra_Dist: {_fmt4(metrics['Intra_Dist'])} | Inter_Dist: {_fmt4(metrics['Inter_Dist'])}")
    print(f"Silhouette(precomputed): {_fmt4(metrics['Silhouette'])} | Soft_Silhouette: {_fmt4(metrics['Soft_Silhouette'])}")
    print(f"Calinski-Harabasz: {_fmt4(metrics['Calinski_Harabasz'])} | Davies-Bouldin: {_fmt4(metrics['Davies_Bouldin'])}")
    print(f"Saved to: {SAVE_DIR}")
    print(f"- cluster_assignments.csv")
    print(f"- pairwise_distance_matrix.csv")
    print(f"- similarity_heatmap_ordered.png")
    print(f"- analysis_metrics.json / analysis_metrics.csv")

if __name__ == "__main__":
    main()
