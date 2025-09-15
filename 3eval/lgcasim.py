#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LGCA-Sim evaluation (forward-only).

- Edit the four paths in the "USER-EDITABLE PATHS" block below.
- This script:
    1) Loads precomputed per-graph tensors (mirroring GRAPH_DIR structure).
    2) Runs the exported hardened/pruned LGCA-Sim model forward for NUM_ITER steps.
    3) Builds a Jensen–Shannon distance matrix over per-graph trajectories.
    4) Hierarchically clusters graphs and reports standard clustering metrics.
    5) Saves metrics, assignments, pairwise matrix CSV, and an ordered heatmap.

All comments are in English; extraneous methods were removed for clarity.
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

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ========================= PATHS =========================
# Default: district set; replace "district" with "city" to test the city set.
GRAPH_DIR = r'0data/meta/analysis/district/graphml'
PRE_DIR   = r'0data/meta/analysis/district/pre'
BASE_SAVE_DIR = r'0data/output/analysissave'

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = os.path.join(BASE_SAVE_DIR, SCRIPT_NAME)
# ========================================================

# Reproducibility & device
def set_seed(seed: int = 3407):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Core hyperparameters (aligned with test)
NUM_ITER = 5
TEMP_INIT = 0.9
TEMP_MIN  = 0.08
GATE_ANNEAL_ENABLE = False
MAX_K = 25
MIN_K = 3

# Histogram & soft-silhouette settings (aligned with test)
HIST_BW = 0.07
SOFTSIL_TAU = 0.20
SIMS_GLOBAL_SCALE = 1.0


# ========================= Precompute loader =========================
class PrecomputeModuleAnalysis:
    """
    Loads precomputed tensors from PRE_DIR, mirroring GRAPH_DIR.
    Produces simplified fields:
      - node_states: Tensor[N] or Tensor[N,1]
      - neighbors  : as stored (unused here)
      - space_info : per-source edge attributes (unused here)
      - edge_src, edge_dst: LongTensor[E]
      - edge_info  : FloatTensor[E, 2]
    """
    CACHE = {}

    @staticmethod
    def _rel_path(graph_path: str):
        return os.path.relpath(graph_path, GRAPH_DIR)

    @staticmethod
    def _build_simplified(data):
        neighbors = data['neighbors']
        edge_src, edge_dst, edge_info = [], [], []
        for u, idxs in enumerate(neighbors):
            idx_list = idxs.tolist() if torch.is_tensor(idxs) else list(idxs)
            if not idx_list:
                continue
            for j, v in enumerate(idx_list):
                edge_src.append(u)
                edge_dst.append(v)
                edge_info.append(data['space_info'][u][j].unsqueeze(0))
        if edge_src:
            edge_src  = torch.tensor(edge_src, dtype=torch.long)
            edge_dst  = torch.tensor(edge_dst, dtype=torch.long)
            edge_info = torch.cat(edge_info, dim=0)
        else:
            edge_src  = torch.zeros(0, dtype=torch.long)
            edge_dst  = torch.zeros(0, dtype=torch.long)
            edge_info = torch.zeros((0, 2), dtype=torch.float32)
        return {
            'node_states': data['node_states'],
            'neighbors':   data['neighbors'],
            'space_info':  data['space_info'],
            'edge_src':    edge_src,
            'edge_dst':    edge_dst,
            'edge_info':   edge_info
        }

    def process_graph(self, graph_path: str):
        if graph_path in self.CACHE:
            return self.CACHE[graph_path]

        rel = self._rel_path(graph_path)
        pre_file = os.path.join(PRE_DIR, os.path.splitext(rel)[0] + '.pt')
        if not os.path.isfile(pre_file):
            raise FileNotFoundError(f"Precomputed file not found: {pre_file}")
        raw = torch.load(pre_file, map_location='cpu')

        need_build = not {'edge_src', 'edge_dst', 'edge_info'}.issubset(set(raw.keys()))
        data = self._build_simplified(raw) if need_build else raw

        self.CACHE[graph_path] = data
        return data


# =================== Diff Logic Gate & Networks (eval) ===================
class DiffLogicGate(nn.Module):
    """
    Logic gate with a fixed set of boolean-like functions; weights select among them.
    The exact size of function_logits must match the exported checkpoint.
    """
    _funcs = [
        lambda x, y: x * y,                         # AND
        lambda x, y: x * (1 - y),                   # x AND ~y
        lambda x, y: x,                             # x
        lambda x, y: (1 - x) * y,                   # ~x AND y
        lambda x, y: y,                             # y
        lambda x, y: x + y - 2 * x * y,             # XOR
        lambda x, y: x + y - x * y,                 # OR
        lambda x, y: (1 - x) * (1 - y),             # NOR
        lambda x, y: 1 - (x + y - 2 * x * y),       # XNOR
        lambda x, y: 1 - y,                         # ~y
        lambda x, y: 1 - x * y,                     # NAND
        lambda x, y: 1 - x,                         # ~x
        lambda x, y: (1 - x) + y - (1 - x) * y,     # (~x) OR y
        lambda x, y: x + (1 - y) - x * (1 - y),     # x OR (~y)
    ]
    def __init__(self, temp=TEMP_INIT):
        super().__init__()
        self.num_functions = len(self._funcs)
        self.register_buffer('idx_a', torch.tensor(0))
        self.register_buffer('idx_b', torch.tensor(1))
        self.function_logits = nn.Parameter(torch.zeros(self.num_functions))
        self.register_buffer('temp', torch.tensor(float(temp)))

    def forward(self, x):
        ia = int(self.idx_a.item()) % x.size(1)
        ib = int(self.idx_b.item()) % x.size(1)
        a, b = x[:, ia], x[:, ib]

        def to_prob(u):
            s = u / (1.0 + u.abs())
            return 0.5 * (s + 1.0)

        pa, pb = to_prob(a), to_prob(b)
        outs = torch.stack([f(pa, pb) for f in self._funcs], dim=1)  # [B, F] in probability domain
        m = float(self.num_functions)
        temp = max(1e-5, 1.0 / math.sqrt(m))
        w = F.softmax(self.function_logits / temp, dim=-1)           # [F]
        y = (outs * w.unsqueeze(0)).sum(dim=1, keepdim=True)         # [B,1]
        return torch.clamp(y, 1e-6, 1 - 1e-6)


class DiffLogicGateNetwork(nn.Module):
    def __init__(self, layer_sizes, temp_init=TEMP_INIT):
        super().__init__()
        self.layers = nn.ModuleList()
        for size in layer_sizes:
            gates = nn.ModuleList([DiffLogicGate(temp=temp_init) for _ in range(size)])
            self.layers.append(gates)

    def forward(self, x):
        out = x
        for gates in self.layers:
            z = torch.cat([g(out) for g in gates], dim=1)
            if z.shape != out.shape:
                out = z
            else:
                C = float(out.size(1))
                beta = 1.0 / (1.0 + math.sqrt(C))
                out = beta * out + (1.0 - beta) * z
        return out


class PerceptionUpdateModule(nn.Module):
    """
    Three-branch parallel module (fusion/attn/update) used at evaluation time.
    """
    def __init__(self, layer_cfg):
        super().__init__()
        self.fusion_net  = DiffLogicGateNetwork(layer_cfg['fusion'], temp_init=TEMP_INIT).to(DEVICE)
        self.attn_net    = DiffLogicGateNetwork(layer_cfg['attn'],   temp_init=TEMP_INIT).to(DEVICE)
        self.update_net  = DiffLogicGateNetwork(layer_cfg['update'], temp_init=TEMP_INIT).to(DEVICE)

    def forward(self, pre):
        ns = pre['node_states']               # [N,1]
        N  = ns.size(0)
        edge_src  = pre['edge_src']
        edge_dst  = pre['edge_dst']
        edge_info = pre['edge_info']

        if edge_src.numel() > 0:
            du  = ns[edge_src].view(-1, 1)         # [E,1]
            dv  = ns[edge_dst].view(-1, 1)         # [E,1]
            das = edge_info.view(-1, 2)            # [E,2]

            fusion_in = torch.cat([du, dv], dim=1) # [E,2]
            fused = self.fusion_net(fusion_in)     # [E,Cf]

            # Attention: symmetric max-abs scaling to [-1,1] for stability
            amax = ns.abs().max().clamp_min(1e-6)
            ns_attn = (ns / amax).clamp(-1.0, 1.0)

            du = ns_attn[edge_src].view(-1, 1)
            dv = ns_attn[edge_dst].view(-1, 1)
            attn_in   = torch.cat([du, dv, das], dim=1)   # [E,4]
            attn_raw  = self.attn_net(attn_in)            # [E,1]
            weights_r = attn_raw.squeeze(1)               # [E]

            # Signed per-source L1 normalization: sum(|w|) per source = 1
            abs_sum_per_src = torch.zeros(N, dtype=weights_r.dtype, device=ns.device)
            abs_sum_per_src.index_add_(0, edge_src, weights_r.abs())
            abs_sum_per_src = abs_sum_per_src.clamp_min(1e-6)
            soft_w = weights_r / abs_sum_per_src[edge_src]       # [E]

            weighted = fused * soft_w.unsqueeze(1)               # [E,Cf]
            agg = torch.zeros((N, fused.size(1)), device=ns.device)  # [N,Cf]
            agg.index_add_(0, edge_src, weighted)
        else:
            last_width = len(self.fusion_net.layers[-1])
            agg = torch.zeros((N, last_width), device=ns.device)

        x = torch.cat([ns, agg], dim=1)            # [N, 1+Cf]
        new_states = self.update_net(x)            # [N, Cu]
        return new_states


# =================== Similarity & clustering (eval) ===================
class SimilarityAndClustering(nn.Module):
    def __init__(self, bins=128):
        super().__init__()
        self.bins = bins
        self.eps  = 1e-8

    def soft_hist(self, x_norm: torch.Tensor):
        x = x_norm.view(-1).to(torch.float64)
        centers = torch.linspace(0.0, 1.0, self.bins, device=x.device, dtype=x.dtype)
        scale = torch.tensor(HIST_BW, device=x.device, dtype=x.dtype)
        diffs = x.unsqueeze(1) - centers.unsqueeze(0)
        weights = torch.exp(-0.5 * (diffs / scale) ** 2)
        hist = (weights.sum(dim=0) + self.eps).to(x.dtype)
        hist = hist / hist.sum().clamp_min(self.eps)
        return hist.to(x_norm.dtype)

    def compute_similarity_matrix(self, all_states):
        # Robust linear rescaling per state to [0,1], then JS distance averaged over t
        hists_per_graph = []
        for state_list in all_states:
            hist_ts = []
            for st in state_list:
                flat = st.view(-1).float()
                med  = flat.median()
                mad  = (flat - med).abs().median()
                lo   = med - 3.0 * mad
                hi   = med + 3.0 * mad
                denom = (hi - lo).abs() + 1e-6
                x_norm = ((flat - lo) / denom).clamp(1e-6, 1 - 1e-6)
                hist_ts.append(self.soft_hist(x_norm))
            hists_per_graph.append(torch.stack(hist_ts, dim=0))  # [T, bins]
        hists = torch.stack(hists_per_graph, dim=0)              # [K, T, bins]

        def js_div(p, q, eps=1e-12):
            m = 0.5 * (p + q)
            term1 = torch.special.xlogy(p, (p + eps)) - torch.special.xlogy(p, (m + eps))
            term2 = torch.special.xlogy(q, (q + eps)) - torch.special.xlogy(q, (m + eps))
            return 0.5 * (term1.sum(-1) + term2.sum(-1))

        sims = None
        T = hists.size(1)
        for t in range(T):
            a = hists[:, t, :]
            js = js_div(a.unsqueeze(1), a.unsqueeze(0))  # [K,K]
            sims = js if sims is None else sims + js
        sims = (sims / T) * SIMS_GLOBAL_SCALE
        return sims

    def cluster_and_metrics(self, sims: torch.Tensor):
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE")
        D = sims.detach().cpu().numpy()
        Kcur = D.shape[0]

        # Preprocess distance: zero diag, clip negatives, symmetrize
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)

        dvec = squareform(D, checks=False)
        Z = linkage(dvec, method='average')

        # Classical MDS embedding for CH/DB
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

        # Rank by (Silhouette ↑ + DB ↓)
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
            labels = np.zeros(Kcur, dtype=int)
            if Kcur > 1:
                i, j = np.unravel_index(np.argmax(D), D.shape)
                labels[j] = 1
        else:
            ks, lbls, sils, dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs  = np.asarray(dbs,  dtype=float)
            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)
            db_rk  = ( np.argsort(np.argsort(dbs ))).astype(float)
            rank_sum = sil_rk + db_rk
            best_idx = int(np.argmin(rank_sum))
            labels = lbls[best_idx]

        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(D[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(D[mask_diff].mean()) if mask_diff.any() else float('nan')

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


# ======================= Checkpoint loader (strict) =======================
def load_perception_module(path: str) -> PerceptionUpdateModule:
    obj = torch.load(path, map_location=DEVICE)

    # Direct nn.Module export
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[INFO] Loaded nn.Module from checkpoint.")
        return model

    # Extract state_dict from known keys or treat whole dict as state_dict
    if isinstance(obj, dict):
        state_dict_raw = None
        for k in ['state_dict', 'model_state_dict', 'perception_state_dict', 'perception_update', 'per']:
            if k in obj and isinstance(obj[k], dict):
                state_dict_raw = obj[k]
                break
        if state_dict_raw is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict_raw = obj
        if state_dict_raw is None:
            raise RuntimeError("No usable state_dict found in checkpoint.")
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

    # Parse layer widths from keys
    import re
    pat = re.compile(r'^(fusion_net|attn_net|update_net)\.layers\.(\d+)\.(\d+)\.')
    sizes_tmp = {'fusion_net': {}, 'attn_net': {}, 'update_net': {}}
    for key in state_dict_raw.keys():
        m = pat.match(key)
        if m:
            branch, layer_idx, gate_idx = m.group(1), int(m.group(2)), int(m.group(3))
            prev = sizes_tmp[branch].get(layer_idx, -1)
            sizes_tmp[branch][layer_idx] = max(prev, gate_idx)
    def to_sizes(d):
        if not d:
            return []
        L = max(d.keys()) + 1
        return [d[i] + 1 for i in range(L)]

    layer_cfg = {
        'fusion': to_sizes(sizes_tmp['fusion_net']),
        'attn':   to_sizes(sizes_tmp['attn_net']),
        'update': to_sizes(sizes_tmp['update_net']),
    }
    if not layer_cfg['fusion'] or not layer_cfg['attn'] or not layer_cfg['update']:
        raise RuntimeError("Failed to parse a complete layer configuration from state_dict.")

    print("[INFO] Parsed layer widths:", layer_cfg)

    # Build model skeleton
    model = PerceptionUpdateModule(layer_cfg).to(DEVICE)

    # Keep only keys needed by the model
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict_raw.keys())
    state_dict = {k: v for k, v in state_dict_raw.items() if k in model_keys}

    # Strictly check required gate buffers/params
    required_missing = []
    for branch, sizes in [('fusion_net', layer_cfg['fusion']),
                          ('attn_net',   layer_cfg['attn']),
                          ('update_net', layer_cfg['update'])]:
        for li, width in enumerate(sizes):
            for gi in range(width):
                base = f"{branch}.layers.{li}.{gi}."
                for suf in ["function_logits", "idx_a", "idx_b", "temp"]:
                    k = base + suf
                    if k not in state_dict:
                        required_missing.append(k)
    if required_missing:
        raise RuntimeError(f"Missing required gate keys in checkpoint (first 10 shown): {required_missing[:10]}")

    # Strict load
    model.load_state_dict(state_dict, strict=True)

    # Attention head must output 1 channel
    if len(model.attn_net.layers[-1]) != 1:
        raise RuntimeError("Attention head width must be 1 in the exported model.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ============================== Plotting ==============================
def plot_similarity_blocked(sim_matrix: np.ndarray, city_names, labels, Z, save_path: str, style: dict = None):
    """
    Heatmap with hierarchical clustering blocks (ordered by linkage leaves).
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
            "cbar_label": "Jensen–Shannon distance"
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
    ax_dy   = fig.add_subplot(gs[0, 0])
    ax_hm   = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_dx   = fig.add_subplot(gs[1, 1])

    dendrogram(Z, ax=ax_dy, orientation='left',   color_threshold=None, no_labels=True)
    ax_dy.set_ylim(-0.5, K - 0.5)
    ax_dy.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for s in ax_dy.spines.values():
        s.set_visible(False)

    dendrogram(Z, ax=ax_dx, orientation='bottom', color_threshold=None, no_labels=True)
    ax_dx.set_xlim(-0.5, K - 0.5)
    ax_dx.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for s in ax_dx.spines.values():
        s.set_visible(False)

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
    for s in ax_hm.spines.values():
        s.set_visible(False)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(style["cbar_label"], fontsize=9, color=style["label_color"])

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    # Extra exports for reproducibility
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


# ================================ Main ================================
def main():
    # Ensure output dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Enumerate *.graphml under GRAPH_DIR
    all_graphs = []
    for root, _, files in os.walk(GRAPH_DIR):
        for f in files:
            if f.lower().endswith('.graphml'):
                all_graphs.append(os.path.join(root, f))
    all_graphs.sort()
    if len(all_graphs) == 0:
        raise RuntimeError(f"No .graphml files found under: {GRAPH_DIR}")

    # City names from base filenames
    city_names = [os.path.splitext(os.path.basename(p))[0] for p in all_graphs]

    # Build modules
    pre = PrecomputeModuleAnalysis()
    sim = SimilarityAndClustering(bins=128)

    # Forward all graphs for NUM_ITER steps (random-walk Laplacian; initial state from .pt)
    all_states = []
    for i, gp in enumerate(all_graphs, 1):
        print(f"\rForward {i}/{len(all_graphs)}", end="")
        pre_cpu = pre.process_graph(gp)

        edge_src  = pre_cpu['edge_src'].to(DEVICE, non_blocking=True)
        edge_dst  = pre_cpu['edge_dst'].to(DEVICE, non_blocking=True)

        ns = pre_cpu['node_states']
        ns = ns if ns.dim() == 2 else ns.view(-1, 1)
        cur = ns.to(DEVICE, dtype=torch.float32)

        N = cur.size(0)
        deg = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        if edge_src.numel() > 0:
            deg.index_add_(0, edge_src, torch.ones(edge_src.numel(), dtype=torch.float32, device=DEVICE))
            deg = deg.clamp_min(1e-6)

        states = []
        with torch.no_grad():
            for _ in range(NUM_ITER):
                if edge_src.numel() == 0:
                    nxt = cur
                else:
                    w = (1.0 / deg[edge_src]).view(-1, 1)                # D^{-1}
                    msg = cur[edge_src] * w                               # A row-normalized
                    nxt = torch.zeros_like(cur)
                    nxt.index_add_(0, edge_dst, msg)                      # x_{t+1} = D^{-1}A x_t
                cur = nxt
                states.append(cur.detach())
        all_states.append(states)
    print()

    # Similarity, clustering, metrics
    sims_t = sim.compute_similarity_matrix(all_states)
    labels, metrics, Z = sim.cluster_and_metrics(sims_t)

    # Save metrics JSON/CSV
    metrics_path_json = os.path.join(SAVE_DIR, 'analysis_metrics.json')
    metrics_path_csv  = os.path.join(SAVE_DIR, 'analysis_metrics.csv')
    with open(metrics_path_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Save cluster assignments
    assign_csv = os.path.join(SAVE_DIR, 'cluster_assignments.csv')
    with open(assign_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['City', 'Cluster'])
        for c, lab in zip(city_names, labels.tolist()):
            w.writerow([c, int(lab)])

    # Save pairwise matrix in original order
    sims_np = sims_t.detach().cpu().numpy()
    def save_square_csv(mat, labels, path):
        K = mat.shape[0]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([''] + labels)
            for i in range(K):
                w.writerow([labels[i]] + [f"{x:.6g}" for x in mat[i].tolist()])
    save_square_csv(sims_np, city_names, os.path.join(SAVE_DIR, 'pairwise_js.csv'))

    # Ordered heatmap (leaves from hierarchical tree)
    heatmap_path = os.path.join(SAVE_DIR, 'similarity_heatmap_ordered.png')
    plot_similarity_blocked(sims_np, city_names, labels, Z, heatmap_path)

    # Console summary
    print("\n===== LGCA-Sim Evaluation Summary =====")
    print(f"Clusters: {metrics['Cluster_Count']}")
    def _fmt4(v: float) -> str:
        try:
            av = abs(float(v))
        except Exception:
            return str(v)
        return f"{v:.4e}" if (av != 0.0 and av < 1e-3) else f"{v:.4f}"
    print(f"Intra(JS): {_fmt4(metrics['Intra_Dist'])} | Inter(JS): {_fmt4(metrics['Inter_Dist'])}")
    print(f"Silhouette (precomputed): {_fmt4(metrics['Silhouette'])} | Soft-Silhouette: {_fmt4(metrics['Soft_Silhouette'])}")
    print(f"Calinski-Harabasz: {_fmt4(metrics['Calinski_Harabasz'])} | Davies-Bouldin: {_fmt4(metrics['Davies_Bouldin'])}")
    print(f"Saved to: {SAVE_DIR}")
    print(f"- cluster_assignments.csv")
    print(f"- pairwise_js.csv")
    print(f"- similarity_heatmap_ordered.png")
    print(f"- analysis_metrics.json / analysis_metrics.csv")


if __name__ == "__main__":
    main()
