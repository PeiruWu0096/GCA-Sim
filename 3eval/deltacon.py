#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeltaCon-based evaluation (forward-only).
- Edit input/output paths at the top.
- No training, annealing, or schedulers; this is a pure analysis script.
- Uses hierarchical clustering on a precomputed distance matrix and reports:
  Cluster_Count, Intra/Inter distances, Silhouette (precomputed), Soft_Silhouette,
  Calinski-Harabasz, Davies-Bouldin.
- Output file names remain unchanged for downstream compatibility.
"""

import os
import json
import csv
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# =========================
# 0) EDITABLE PATHS & PARAMS
# Default: district set; replace "district" with "city" to test the city set.
INPUT_GRAPH_DIR = r'0data/meta/analysis/district/graphml'

# Results root; a subfolder named after this script will be created.
OUTPUT_SAVE_DIR_DEFAULT = r'0data/output/analysissave'

# Default random seed (can be overridden by --seed)
SEED_DEFAULT = 3407

# Clustering search range (kept consistent with test pipeline)
MAX_K = 25
MIN_K = 3

# Soft-silhouette temperature (consistent with test)
SOFTSIL_TAU_DEFAULT = 0.20

# DeltaCon signatures: groups (g), per-group quantiles (Q), repeats (R)
DELTACON_G = 64
DELTACON_Q = 64
DELTACON_R = 3

# Torch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# Utilities
# =========================
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
# 1) Clustering & Metrics
# =========================
class SimilarityAndClustering:
    """
    Provides hierarchical clustering from a precomputed distance matrix and
    computes standard metrics, including a differentiable-style soft silhouette.
    """

    def __init__(self, softsil_tau: float = SOFTSIL_TAU_DEFAULT):
        self.softsil_tau = float(softsil_tau)

    def cluster_and_metrics(self, sims: torch.Tensor):
        """
        Args:
            sims: torch.Tensor [K,K], pairwise distance matrix (smaller = more similar)
        Returns:
            labels: np.ndarray [K], cluster assignment
            metrics: dict of clustering quality metrics
            Z: linkage matrix for dendrogram plotting
        """
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE")
        sims_np = sims.detach().cpu().numpy()
        Kcur = sims_np.shape[0]

        # Sanitize: zero diagonal, clip negatives, symmetrize
        D = sims_np.copy()
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)

        # Single linkage tree; evaluate k in [MIN_K..MAX_K]
        dvec = squareform(D, checks=False)
        Z = linkage(dvec, method='average')

        # Classical MDS (Torgerson) for CH/DB (deterministic)
        dd = D ** 2
        n = dd.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(dd).dot(J)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        k_emb = min(8, int((w > 1e-12).sum()) or 1)
        X_emb = V[:, :k_emb] * np.sqrt(np.maximum(w[:k_emb], 0.0))
        X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)

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
                ch_k = calinski_harabasz_score(X_emb, labels_k)
            except Exception:
                ch_k = -1.0
            try:
                db_k = davies_bouldin_score(X_emb, labels_k)
            except Exception:
                db_k = np.inf
            records.append((k, labels_k, sil_k, ch_k, db_k))

        if not records:
            # Fallback: split by farthest pair
            i, j = np.unravel_index(np.argmax(D), D.shape)
            labels = np.zeros(Kcur, dtype=int)
            labels[j] = 1
        else:
            ks, lbls, sils, chs, _dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs = np.asarray(_dbs, dtype=float)
            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)  # larger sil → better
            db_rk = (np.argsort(np.argsort(dbs))).astype(float)     # smaller db → better
            rank_sum = sil_rk + db_rk
            best_idx = int(np.argmin(rank_sum))
            labels = lbls[best_idx]

        # Intra/Inter distances
        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(D[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(D[mask_diff].mean()) if mask_diff.any() else float('nan')

        # Soft silhouette on distances (row-wise scaled by 0.75 quantile)
        with torch.no_grad():
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
            tau = off_vals.median().clamp_min(1e-6) if off_vals.numel() > 0 else torch.tensor(self.softsil_tau, device=d_t.device, dtype=d_t.dtype)
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
# 2) DeltaCon (grouped, multiscale)
# =========================
class DeltaConAnalysis:
    """
    DeltaCon_0 signatures with grouped multi-probe & two-scale diffusion.
    Produces a pairwise distance matrix D (smaller = more similar).
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _build_M(G: nx.Graph, eps: float):
        """Construct sparse operator M(eps) = I + eps^2 D - eps A."""
        n = G.number_of_nodes()
        if n == 0:
            return sp.csr_matrix((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
        idx = {u: i for i, u in enumerate(G.nodes())}
        rows, cols = [], []
        for u, v in G.edges():
            i, j = idx[u], idx[v]
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
        data = np.ones(len(rows), dtype=np.float64)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        deg = np.asarray(A.sum(axis=1)).ravel()
        D = sp.diags(deg, offsets=0, dtype=np.float64, format='csr')
        I = sp.eye(n, dtype=np.float64, format='csr')
        M = I + (eps * eps) * D - eps * A
        return M, deg

    @staticmethod
    def _cg_solve(A, rhs, rtol1=1e-5, maxiter1=2000, rtol2=1e-4, maxiter2=4000):
        """Conjugate gradients with a fallback tolerance for robustness across SciPy versions."""
        try:
            x, info = cg(A, rhs, rtol=rtol1, atol=0.0, maxiter=maxiter1)
        except TypeError:
            x, info = cg(A, rhs, tol=rtol1, maxiter=maxiter1)
        if info != 0:
            try:
                x, info = cg(A, rhs, rtol=rtol2, atol=0.0, maxiter=maxiter2)
            except TypeError:
                x, info = cg(A, rhs, tol=rtol2, maxiter=maxiter2)
        return x

    def _signature(self, G: nx.Graph) -> np.ndarray:
        """
        Grouped multi-probe + two-scale diffusion signature in the RootED domain.
        """
        n = G.number_of_nodes()
        if n == 0:
            return np.zeros(DELTACON_G * DELTACON_Q * 2, dtype=np.float64)

        # Adaptive eps per graph
        _, deg0 = self._build_M(G, 0.0)
        dmax = float(deg0.max()) if deg0.size > 0 else 0.0
        eps1 = 1.0 / (1.0 + max(1.0, dmax))
        eps2 = min(2.0 * eps1, 0.25)
        eps_list = [eps1, eps2]

        sig_runs = []
        for _ in range(DELTACON_R):
            idx = self.rng.permutation(n)
            splits = np.array_split(idx, DELTACON_G)
            sig_all = []
            for eps in eps_list:
                M, _ = self._build_M(G, eps)
                for part in splits:
                    if part.size == 0:
                        sig_all.append(np.zeros(DELTACON_Q, dtype=np.float64))
                        continue
                    b = np.zeros(n, dtype=np.float64)
                    b[part] = 1.0
                    x = self._cg_solve(M, b)
                    x = np.clip(x, 0.0, None)
                    x = np.sqrt(x)  # RootED
                    qs = np.linspace(0.0, 1.0, DELTACON_Q)
                    try:
                        sig = np.quantile(x, qs, method='linear').astype(np.float64)
                    except TypeError:
                        sig = np.quantile(x, qs, interpolation='linear').astype(np.float64)
                    sig_all.append(sig)
            sig_runs.append(np.concatenate(sig_all, axis=0))
        return np.mean(np.vstack(sig_runs), axis=0)

    def run(self, graph_paths):
        """Compute DeltaCon signatures and return a [K,K] distance matrix."""
        graphs = []
        for i, gp in enumerate(graph_paths, 1):
            print(f"\rForward {i}/{len(graph_paths)}", end="")
            G = nx.read_graphml(gp)
            if G.is_directed():
                G = G.to_undirected()
            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                G = nx.Graph(G)  # merge parallel edges
            G.remove_edges_from(nx.selfloop_edges(G))
            graphs.append(G)
        print()

        signatures = [self._signature(G) for G in graphs]
        S = np.vstack(signatures)  # [K, Dsig]
        S_norm = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
        U = S / S_norm
        C = np.clip(U @ U.T, -1.0, 1.0)
        D = np.sqrt(np.maximum(2.0 - 2.0 * C, 0.0))  # cosine distance mapped to [0, 2]→[0, sqrt(4)]
        sims = torch.tensor(D, dtype=torch.float32, device=DEVICE)
        return sims


# =========================
# 3) Visualization
# =========================
def plot_similarity_blocked(sim_matrix: np.ndarray, city_names, labels, Z, save_path: str, style: dict = None):
    """
    Draw a dendrogram-aligned heatmap with block outlines for clusters.
    """
    from matplotlib import gridspec
    from matplotlib.colors import LinearSegmentedColormap

    if style is None:
        style = {
            "cmap_colors": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
            "border_color": "#2b2b2b",
            "dendro_color": "#606060",
            "label_color":  "#2b2b2b",
            "gridline_color": "#d9d9d9",
            "cbar_label": "DeltaCon Distance"
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

    # Export tabular artifacts for reproducibility
    base = os.path.splitext(save_path)[0]
    with open(base + "_leaf_order.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["Index_in_order", "City", "Cluster"])
        for i, (c, lab) in enumerate(zip(cities_ord, labels_ord.tolist())):
            w.writerow([i, c, int(lab)])
    with open(base + "_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow([""] + cities_ord)
        for i in range(K):
            w.writerow([cities_ord[i]] + [f"{D_ord[i, j]:.6g}" for j in range(K)])
    np.save(base + "_linkage.npy", Z)
    try:
        import json as _json
        with open(base + "_style.json", "w", encoding="utf-8") as f:
            _json.dump(style, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =========================
# 4) Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeltaCon Graph Similarity Analysis")
    parser.add_argument('--seed', type=int, default=SEED_DEFAULT, help='Random seed.')
    parser.add_argument('--graph_dir', type=str, default=INPUT_GRAPH_DIR, help='Root directory of .graphml files.')
    parser.add_argument('--save_dir', type=str, default=OUTPUT_SAVE_DIR_DEFAULT, help='Directory to save outputs.')
    args = parser.parse_args()

    SEED = int(args.seed)
    GRAPH_DIR = args.graph_dir
    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    SAVE_DIR = os.path.join(args.save_dir, SCRIPT_NAME)


    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n[INFO] Seed: {SEED}")
    print(f"[INFO] Graph root: {GRAPH_DIR}")
    print(f"[INFO] Saving to: {SAVE_DIR}")

    # Collect .graphml files (recursive)
    all_graphs = []
    for root, _, files in os.walk(GRAPH_DIR):
        for f in files:
            if f.lower().endswith('.graphml'):
                all_graphs.append(os.path.join(root, f))
    all_graphs.sort()
    if not all_graphs:
        raise RuntimeError(f"No .graphml files found under: {GRAPH_DIR}")
    city_names = [os.path.splitext(os.path.basename(p))[0] for p in all_graphs]

    # Run DeltaCon and cluster
    delta = DeltaConAnalysis(seed=SEED)
    sims_t = delta.run(all_graphs)

    sim = SimilarityAndClustering(softsil_tau=SOFTSIL_TAU_DEFAULT)
    labels, metrics, Z = sim.cluster_and_metrics(sims_t)

    # Save metrics (JSON + CSV)
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

    # Plot ordered heatmap
    heatmap_path = os.path.join(SAVE_DIR, 'similarity_heatmap_ordered.png')
    plot_similarity_blocked(sims_np, city_names, labels, Z, heatmap_path)

    # Console summary
    def _fmt4(v: float) -> str:
        try:
            av = abs(float(v))
        except Exception:
            return str(v)
        return f"{v:.4e}" if (av != 0.0 and av < 1e-3) else f"{v:.4f}"

    print("\n===== Analysis Summary =====")
    print(f"Clusters: {metrics['Cluster_Count']}")
    print(f"Intra_Dist: {_fmt4(metrics['Intra_Dist'])} | Inter_Dist: {_fmt4(metrics['Inter_Dist'])}")
    print(f"Silhouette(precomputed): {_fmt4(metrics['Silhouette'])} | Soft_Silhouette: {_fmt4(metrics['Soft_Silhouette'])}")
    print(f"Calinski-Harabasz: {_fmt4(metrics['Calinski_Harabasz'])} | Davies-Bouldin: {_fmt4(metrics['Davies_Bouldin'])}")
    print(f"Saved to: {SAVE_DIR}")
    print(f"- {os.path.basename(assign_csv)}")
    print(f"- pairwise_distance_matrix.csv")
    print(f"- {os.path.basename(heatmap_path)}")
    print(f"- analysis_metrics.json / analysis_metrics.csv")


if __name__ == "__main__":
    main()
