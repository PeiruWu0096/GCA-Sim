#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估脚本（仅前向推理）：
- 路径、超参数与方法均写死；无需传参，直接 python 本文件即可运行
- 完全复用 test 中的数据初始化、5 次信息迭代、相似性评价与聚类方法/参数
- 不进行任何训练/退火/学习率调度/早停；只用已导出的“硬化+裁剪”模型做前向
- 产出与 test 一致的指标，并生成“按簇排序且框出同簇块”的相似性热力图
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

from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering

import matplotlib
# 允许交互：仅当显式要求时才强制无界面后端
# 若需要无界面渲染，可在运行前设置环境变量 GNCA_FORCE_AGG=1
if os.environ.get("GNCA_FORCE_AGG", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# ========== 0) 固定随机种子与设备 ==========
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

# ========== 1) Paths & run config ==========
# Default: district set; replace "district" with "city" to test the city set.
GRAPH_DIR = r'0data/meta/analysis/district/graphml'
PRE_DIR   = r'0data/meta/analysis/district/pre'
# Results root; a subfolder named after this script will be created.
BASE_SAVE_ROOT = r'0data/output/analysissave'

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = os.path.join(BASE_SAVE_ROOT, SCRIPT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# Default checkpoint: the best model from previous training; replace with any intermediate or final checkpoint if needed.
MODEL_PT = r'0data/output/artifacts/best/3409.pt'

# —— 可选：伴随权重文件（来自 test 的导出），用于覆盖 checkpoint 中的 mix_raw —— 
# 若为 None：自动在 MODEL_PT 同目录下优先查找 'final_mix_raw.pt'，找不到再查 'final_hard_mix_raw.pt'
# 若显式给出路径：将按该路径加载（结构应为 {'fusion':[Tensor...],'attn':[...],'update':[...]}，逐层 1D tensor）
COMPANION_MIX_PATH = None       # 统一 ckpt 已包含 mix_raw；除非兼容历史产物，否则无需指定
APPLY_COMPANION_MIX = False     # 关闭默认覆盖，避免意外篡改 ckpt 内置 mix_raw



# —— 与 test 一致的核心超参/常量 ——
NUM_ITER = 5
TEMP_INIT = 0.9
TEMP_MIN  = 0.08
GATE_ANNEAL_ENABLE = False
MAX_K = 25                  # 与 test 一致
MIN_K = 3                  # 与 test 的“从3开始”一致

# —— 直方图与 Soft-Silhouette 超参数（与 test 保持一致）——
HIST_BW           = 0.07
SOFTSIL_TAU       = 0.20     # 与 test 相同，避免过冷/饱和
SIMS_GLOBAL_SCALE = 1.0      # 与 test 相同：不做额外放大
# ========== 2) 预计算模块（analysis 版本） ==========
class PrecomputeModuleAnalysis:
    """
    加载 analysispre 下的预计算数据，并生成简化版字段：
    - node_states: Tensor[N]
    - neighbors:   List[List[int]] 或 Tensor(list)
    - space_info:  List[Tensor[deg(u), 2]]
    - edge_src, edge_dst: LongTensor[E]
    - edge_info:   FloatTensor[E, 2]
    """
    CACHE = {}

    @staticmethod
    def _rel_path(graph_path: str):
        # 相对 GRAPH_DIR 的相对路径（包含子目录的话也会保留）
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
                # data['space_info'][u][j] → Tensor[2]
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
            raise FileNotFoundError(f"预计算文件未找到: {pre_file}")
        raw = torch.load(pre_file, map_location='cpu')

        # 若没有 edge_* 字段，则构建简化版
        keys = set(raw.keys())
        need_build = not {'edge_src','edge_dst','edge_info'}.issubset(keys)
        data = self._build_simplified(raw) if need_build else raw

        self.CACHE[graph_path] = data
        return data

# ========== 3) 可微逻辑门网络（与 test 一致的结构/前向） ==========
class DiffLogicGate(nn.Module):
    """
    与 ckpt 对齐的九算子门（兼容旧 7 算子：ckpt 若为 9，本实现可无缝加载）
    """
    _OPS = [
        lambda slf, a, b: a + b,    # 0: a+b
        lambda slf, a, b: a - b,    # 1: a-b
        lambda slf, a, b: b - a,    # 2: b-a
        lambda slf, a, b: a,        # 3: pass A
        lambda slf, a, b: b,        # 4: pass B
        lambda slf, a, b: -a,       # 5: -A
        lambda slf, a, b: -b,       # 6: -B
        lambda slf, a, b: torch.maximum(a, b),  # 7: max(A,B)
        lambda slf, a, b: torch.minimum(a, b),  # 8: min(A,B)
    ]

    def __init__(self, temp=TEMP_INIT):
        super().__init__()
        self.num_functions = len(self._OPS)
        self.register_buffer('idx_a', torch.tensor(0))
        self.register_buffer('idx_b', torch.tensor(1))
        # 初始化一个正确尺寸（7）的参数，其数值会被 load_state_dict 覆盖
        self.function_logits = nn.Parameter(torch.zeros(self.num_functions))
        self.register_buffer('temp', torch.tensor(float(temp)))

    def forward(self, x):
        # 修正：直接使用从模型加载的精确索引，不进行取模操作
        ia = int(self.idx_a.item())
        ib = int(self.idx_b.item())
        a = x[:, ia:ia+1]
        b = x[:, ib:ib+1]

        # 评估时，使用确定性的硬门逻辑（取 one-hot 最大概率的操作）
        with torch.no_grad():
            best_op_idx = torch.argmax(self.function_logits).item()
        
        # 直接调用选定的算子
        y = self._OPS[best_op_idx](self, a, b)
        
        if y.dim() == 1:
            return y.unsqueeze(1)
        return y


class DiffLogicGateNetwork(nn.Module):
    """
    由层尺寸列表动态构建，不假设固定层数/节点数；具体尺寸由 checkpoint 解析得到
    """
    def __init__(self, layer_sizes, temp_init=TEMP_INIT):
        super().__init__()
        self.layers = nn.ModuleList()
        for size in layer_sizes:
            gates = nn.ModuleList([DiffLogicGate(temp=temp_init) for _ in range(size)])
            self.layers.append(gates)

    def forward(self, x):
        out = x
        for gates in self.layers:
            # 每个 g(out) 输出 [B, 1], 使用 cat 拼接成 [B, num_gates] 作为下一层输入
            residual = torch.cat([g(out) for g in gates], dim=1)
            out = residual
        return out

class PerceptionUpdateModule(nn.Module):
    """
    三分支并行：fusion / attn / update；层配置由外部动态传入
    仅在首次前向的第1张图上打印一次形状/通道核对信息（debug-once）
    """
    def __init__(self, layer_cfg):
        super().__init__()
        # layer_cfg: {'fusion': [..], 'attn': [..], 'update': [..]}
        self.fusion_net  = DiffLogicGateNetwork(layer_cfg['fusion'], temp_init=TEMP_INIT).to(DEVICE)
        self.attn_net    = DiffLogicGateNetwork(layer_cfg['attn'],   temp_init=TEMP_INIT).to(DEVICE)
        self.update_net  = DiffLogicGateNetwork(layer_cfg['update'], temp_init=TEMP_INIT).to(DEVICE)

        # 调试一次的开关
        self.debug_shapes = True
        self._debug_done  = False

    def forward(self, pre):
        ns = pre['node_states']
        N  = ns.size(0)
        edge_src  = pre['edge_src']
        edge_dst  = pre['edge_dst']
        edge_info = pre['edge_info']

        if edge_src.numel() > 0:
            # —— fusion 和 attention 统一使用原始度/状态 —— 
            du_raw = ns[edge_src].view(-1, 1)
            dv_raw = ns[edge_dst].view(-1, 1)
            das    = edge_info.view(-1, 2)

            fusion_in = torch.cat([du_raw, dv_raw], dim=1)
            fused = self.fusion_net(fusion_in)

            # —— 移除 attention 输入的归一化，使其与 lmain.py 行为一致 ——
            attn_in   = torch.cat([du_raw, dv_raw, das], dim=1)
            weights_r = self.attn_net(attn_in).squeeze(1)          # [E]

            # —— 与 test 一致：按“源点的绝对值和”做归一，保留符号（不 clamp 到正数）
            abs_sum_per_src = torch.zeros(N, dtype=weights_r.dtype, device=ns.device)
            abs_sum_per_src.index_add_(0, edge_src, weights_r.abs())
            abs_sum_per_src = abs_sum_per_src.clamp_min(1e-6)
            soft_w = weights_r / abs_sum_per_src[edge_src]


            # 加权聚合（避免度数带来的能量膨胀）
            weighted = fused * soft_w.unsqueeze(1)                # [E,Cf]
            agg = torch.zeros((N, fused.size(1)), device=ns.device)  # [N,Cf]
            agg.index_add_(0, edge_src, weighted)

            # —— 与 test 一致：直接拼接未夹紧的 agg
            x = torch.cat([ns, agg], dim=1)
            update_delta = self.update_net(x)
        else:
            # 新增：处理图中无边的情况，与 test.py 保持一致
            last_fusion_width = len(self.fusion_net.layers[-1])
            agg = torch.zeros((N, last_fusion_width), device=ns.device)
            x = torch.cat([ns, agg], dim=1)
            update_delta = self.update_net(x)

        # 核心修正：从“替换式”更新改为“增量式”更新，与 test.py 保持一致
        out = ns + update_delta

        # —— 仅打印一次关键形状与校验（增强版 + 边级/源点级 attention 诊断） —— 
        if self.debug_shapes and (not self._debug_done):
            try:
                Cf = fused.size(1)
                Ca = 1   # attn_net 的输出为单通道标量权重
                Cu = out.size(1)
                E  = edge_src.numel()

                # —— 采样若干“源点”检查 |soft_w| 的逐源点绝对和（应≈1）与源点内方差 —— 
                if E > 0:
                    # 统计每个源点的出度
                    deg_per_src = torch.zeros(N, dtype=torch.long, device=ns.device)
                    deg_per_src.index_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.long))
                    uniq_src = torch.nonzero(deg_per_src > 0, as_tuple=False).view(-1)

                    # 优先选择出度≥3，其次≥2；若都没有再退回任意源点
                    cand3 = uniq_src[(deg_per_src[uniq_src] >= 3)]
                    cand2 = uniq_src[(deg_per_src[uniq_src] == 2)]
                    if cand3.numel() > 0:
                        sample_src = cand3[:min(5, cand3.numel())]
                    elif cand2.numel() > 0:
                        sample_src = cand2[:min(5, cand2.numel())]
                    else:
                        sample_src = uniq_src[:min(5, uniq_src.numel())]

                    sums_abs, std_w, std_softw, deg_list = [], [], [], []
                    for s in sample_src.tolist():
                        mask = (edge_src == s)
                        deg  = int(mask.sum().item())
                        deg_list.append(deg)
                        sums_abs.append(float(torch.abs(soft_w[mask]).sum().item()))
                        # 安全 std：当 deg==1 时定义为 0.0，避免警告/NaN
                        if deg >= 2:
                            std_w.append(float(weights_r[mask].std(unbiased=False).item()))
                            std_softw.append(float(soft_w[mask].std(unbiased=False).item()))
                        else:
                            std_w.append(0.0)
                            std_softw.append(0.0)
                else:
                    sample_src = torch.tensor([], dtype=torch.long, device=ns.device)
                    sums_abs, std_w, std_softw, deg_list = [], [], [], []

                # —— 边级样本：优先从“同一源点”取多条边 —— 
                if E > 0:
                    # 选择一个出度最大的源点，打印它的前 k 条边
                    top_src = int(torch.argmax(deg_per_src).item())
                    mask_top = (edge_src == top_src)
                    idx_top = torch.nonzero(mask_top, as_tuple=False).view(-1)
                    k = int(min(5, idx_top.numel()))
                    if k == 0:
                        # 退回到前 k 条边
                        idx_top = torch.arange(min(5, E), device=ns.device)
                        k = idx_top.numel()
                    else:
                        idx_top = idx_top[:k]

                    samp_idx = idx_top
                    samp_src = edge_src[samp_idx].tolist()
                    samp_dst = edge_dst[samp_idx].tolist()
                    samp_in  = attn_in[samp_idx].detach().cpu().numpy()
                    samp_wr  = weights_r[samp_idx].detach().cpu().numpy()
                    samp_w   = weights_r[samp_idx].detach().cpu().numpy()
                    samp_sw  = soft_w[samp_idx].detach().cpu().numpy()
                else:
                    k, samp_idx, samp_src, samp_dst = 0, None, [], []
                    samp_in, samp_wr, samp_w, samp_sw = [], [], [], []

                # —— 注意力原始输出统计 —— 
                wr = weights_r.detach()
                wr_min, wr_med, wr_mean, wr_max = [float(v) for v in (
                    wr.min().item(), wr.median().item(), wr.mean().item(), wr.max().item()
                )]

                # —— 输入尺度统计 (修正：使用 du_raw, dv_raw) —— 
                du_stats  = (float(du_raw.min().item()),  float(du_raw.mean().item()),  float(du_raw.max().item()),  float(du_raw.std().item()))
                dv_stats  = (float(dv_raw.min().item()),  float(dv_raw.mean().item()),  float(dv_raw.max().item()),  float(dv_raw.std().item()))
                das_stats = (float(das.min().item()), float(das.mean().item()), float(das.max().item()), float(das.std().item()))

                print("\n[DEBUG] ===== PerceptionUpdateModule I/O Check (print-once) =====")
                print(f"Nodes: N={N} | Edges: E={E}")
                print(f"Fusion  in shape: {fusion_in.shape}  (expect [E, 2])")
                print(f"Fusion out shape: {fused.shape}      → Cf={Cf}")
                print(f"Attn    in shape: {attn_in.shape}    (expect [E, 4])")
                print(f"Attn   out shape: {weights_r.shape}  → Ca={Ca} (expect 1)")
                print(f"[ATTN] weights_r stats: min/med/mean/max = {wr_min:.4g}/{wr_med:.4g}/{wr_mean:.4g}/{wr_max:.4g}")
                print(f"[IN ] du min/mean/max/std = {du_stats}")
                print(f"[IN ] dv min/mean/max/std = {dv_stats}")
                print(f"[IN ] das min/mean/max/std = {das_stats}")

                # —— 打印逐源点 |soft_w| 和、以及源点内的方差（诊断“是否几乎相等”） —— 
                if sample_src.numel() > 0:
                    sums_str   = ['{:.3f}'.format(v) for v in sums_abs]
                    stdw_str   = ['{:.4f}'.format(v) for v in std_w]
                    stdsw_str  = ['{:.4f}'.format(v) for v in std_softw]
                    print(f"[ATTN] per-src sample deg          ≈ {deg_list}")
                    print(f"[ATTN] per-src sample |sum(soft_w)| ≈ {sums_str}  (expect ≈1)")
                    print(f"[ATTN] per-src sample std(w)       ≈ {stdw_str}")
                    print(f"[ATTN] per-src sample std(soft_w)  ≈ {stdsw_str}")
                else:
                    print("[ATTN] no edges to sample per-src stats.")

                # —— 打印若干边的前后对比（du,dv,das | weights_r (clamp) → soft_w） —— 
                if k > 0:
                    print("[ATTN] edge-level samples (src,dst): [du,dv,das0,das1] | weights_r (clamp) → soft_w")
                    for i in range(k):
                        du_, dv_, a0, a1 = samp_in[i].tolist()
                        print(f"  ({samp_src[i]},{samp_dst[i]}): "
                              f"[{du_:.4f},{dv_:.4f},{a0:.4f},{a1:.4f}] | "
                              f"{samp_wr[i]:+.4f} → {samp_w[i]:+.4f} → {samp_sw[i]:+.4f}")
                else:
                    print("[ATTN] no edges to sample edge-level stats.")

                print(f"Aggregated shape: {agg.shape}         (expect [N, Cf])")
                print(f"Update  in shape: {x.shape}           (expect [N, 1+Cf])")
                print(f"Update out shape: {out.shape}         → Cu={Cu}")
                print("Attention-weighted fusion aggregation is ACTIVE "
                      "(fused * soft_w, then index_add_ by edge_src).")
                print("=================================================================\n")
            finally:
                self._debug_done = True

        return out

# ========== 4) 相似性&聚类模块（与 test 一致） ==========
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

    def compute_similarity_matrix(self, states_list):
        hists_per_graph = []
        for state_list in states_list:
            hist_ts = []
            for st in state_list:
                flat = st.view(-1).float()
                med  = flat.median()
                mad  = (flat - med).abs().median()
                # 稳健线性归一化：把 [med-3*mad, med+3*mad] → [0,1]，再 clamp 到 (0,1)
                lo   = med - 3.0 * mad
                hi   = med + 3.0 * mad
                denom = (hi - lo).abs() + 1e-6
                x_norm = ((flat - lo) / denom).clamp(1e-6, 1 - 1e-6)
                hist_ts.append(self.soft_hist(x_norm))
            hists_per_graph.append(torch.stack(hist_ts, dim=0))  # [T, bins]

        hists = torch.stack(hists_per_graph, dim=0)      # [K, T, bins]

        # —— Jensen–Shannon 距离的平均（跨 T 平均）——
        def js_div(p, q, eps=1e-12):
            m = 0.5 * (p + q)
            term1 = torch.special.xlogy(p, (p + eps)) - torch.special.xlogy(p, (m + eps))
            term2 = torch.special.xlogy(q, (q + eps)) - torch.special.xlogy(q, (m + eps))
            return 0.5 * (term1.sum(-1) + term2.sum(-1))

        sims = None
        T = hists.size(1)
        for t in range(T):
            a = hists[:, t, :]
            js = js_div(a.unsqueeze(1), a.unsqueeze(0))        # [K,K]
            sims = js if sims is None else sims + js
        sims = (sims / T) * SIMS_GLOBAL_SCALE                  # SIMS_GLOBAL_SCALE=1.0
        return sims

    def cluster_and_metrics(self, sims: torch.Tensor):
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE")
        sims_np = sims.detach().cpu().numpy()
        Kcur = sims_np.shape[0]

        # —— 对角置零、负值截断，并对称化 —— 
        D = sims_np.copy()
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)

        # —— linkage 在对称 D 上运行 —— 
        dvec = squareform(D, checks=False)
        Z = linkage(dvec, method='average')

        # —— Classical MDS 供 CH/DB 使用（与 test 一致）——
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

        # —— 枚举 k（从 3 开始），按 Sil↑ + DB↓ 的秩和选最优 —— 
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
            i, j = np.unravel_index(np.argmax(D), D.shape)
            labels = np.zeros(Kcur, dtype=int)
            labels[j] = 1
        else:
            ks, lbls, sils, chs, _dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs  = np.asarray(_dbs, dtype=float)
            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)   # Sil 越大名次越小
            db_rk  = ( np.argsort(np.argsort(dbs ))).astype(float)   # DB 越小名次越小
            rank_sum = sil_rk + db_rk
            best_idx = int(np.argmin(rank_sum))
            labels = lbls[best_idx]

        # —— Intra / Inter（排除对角；inter 仅不同簇对）——
        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(D[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(D[mask_diff].mean()) if mask_diff.any() else float('nan')

        # —— Soft-Sil（评估版，与训练/验证同式）：行 0.75 分位缩放 + 全局 τ=off-diag 中位数 —— 
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
            if off_vals.numel() == 0:
                tau = torch.tensor(SOFTSIL_TAU, device=d_t.device, dtype=d_t.dtype)
            else:
                tau = off_vals.median().clamp_min(1e-6)

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

# ========== 5) 模型加载（兼容多种导出形态） ==========
def load_perception_module(path: str) -> PerceptionUpdateModule:
    obj = torch.load(path, map_location='cpu')

    # —— 1) 统一格式优先 —— 
    if isinstance(obj, dict) and obj.get("kind") == "gnca_model_ckpt" and "state_dict" in obj:
        sd = obj["state_dict"]
        layer_cfg = obj.get("layer_cfg", {})
        # 若 main 的网络按 layer_cfg 构建（建议），否则也可按 state_dict 解析推断
        def _cfg_from_statedict(sd_dict):
            from collections import defaultdict
            sizes = defaultdict(int)
            for k in sd_dict.keys():
                # 形如：fusion_net.layers.0.3.function_logits
                if ".layers." in k:
                    segs = k.split(".")
                    try:
                        branch = segs[0]              # fusion_net / attn_net / update_net
                        li = int(segs[2])             # 层号
                        sizes[(branch, li)] = max(sizes[(branch, li)], 1 + int(segs[3]))
                    except Exception:
                        pass
            out = {"fusion": [], "attn": [], "update": []}
            for b in ("fusion_net","attn_net","update_net"):
                name = "fusion" if b=="fusion_net" else ("attn" if b=="attn_net" else "update")
                li = 0
                while (b, li) in sizes:
                    out[name].append(sizes[(b, li)])
                    li += 1
            return out

        if not layer_cfg or not any(layer_cfg.get(k) for k in ("fusion","attn","update")):
            layer_cfg = _cfg_from_statedict(sd)

        # 注意：PerceptionUpdateModule 现在要求显式传入 layer_cfg
        model = PerceptionUpdateModule(layer_cfg)
        try:
            model.load_state_dict(sd, strict=True)
        except Exception as ex:
            raise RuntimeError(f"Load unified ckpt failed: {ex}")

        model = model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[INFO] Loaded unified checkpoint: epoch={obj.get('epoch','?')} seed={obj.get('seed','?')} "
              f"scale={obj.get('scale','')} group={obj.get('group','')}")
        return model

    # —— 2) 老式：纯 state_dict —— 
    if isinstance(obj, dict) and all(k not in obj for k in ("state","param_groups")):
        # 直接当 state_dict：需先从 key 里推断 layer_cfg
        def _cfg_from_statedict(sd_dict):
            from collections import defaultdict
            sizes = defaultdict(int)
            for k in sd_dict.keys():
                if ".layers." in k:
                    segs = k.split(".")
                    try:
                        branch = segs[0]          # fusion_net / attn_net / update_net
                        li = int(segs[2])         # 层号
                        sizes[(branch, li)] = max(sizes[(branch, li)], 1 + int(segs[3]))
                    except Exception:
                        pass
            out = {"fusion": [], "attn": [], "update": []}
            for b in ("fusion_net","attn_net","update_net"):
                name = "fusion" if b=="fusion_net" else ("attn" if b=="attn_net" else "update")
                li = 0
                while (b, li) in sizes:
                    out[name].append(sizes[(b, li)])
                    li += 1
            return out

        layer_cfg = _cfg_from_statedict(obj)
        if not any(layer_cfg.get(k) for k in ("fusion","attn","update")):
            raise RuntimeError("无法从 legacy state_dict 推断 layer_cfg（fusion/attn/update）")

        model = PerceptionUpdateModule(layer_cfg)
        model.load_state_dict(obj, strict=True)
        model = model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[INFO] Loaded legacy state_dict checkpoint.")
        return model

    # —— 3) 整模块 —— 
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[INFO] Loaded whole nn.Module from checkpoint.")
        return model

    # —— 4) 仍可能是“优化器快照”等 —— 
    raise RuntimeError("Unsupported checkpoint format. Expect a unified 'gnca_model_ckpt' or a model state_dict.")

    # —— 解析 state_dict（更健壮）：支持多层包裹、递归扫描、前缀清理 —— 
    if not isinstance(obj, (dict, nn.Module)):
        raise RuntimeError(f"无法识别的模型文件结构: {type(obj)}")

    # 若直接保存了整个 nn.Module，直接返回（最可靠）
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print("[INFO] Loaded whole nn.Module from checkpoint (skipped state_dict parsing).")
        return model

    import re
    from collections import deque

    def looks_like_statedict(d: dict) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        # 至少有若干 Tensor；或 key 里出现门网络的典型模式
        tcnt = sum(isinstance(v, torch.Tensor) for v in d.values())
        if tcnt >= 10:
            return True
        pat = re.compile(r'(fusion_net|attn_net|update_net)\.layers\.\d+\.\d+\.')
        return any(isinstance(v, torch.Tensor) and pat.search(k) for k, v in d.items())

    def try_strip_prefix(sd: dict):
        """尝试剥离常见前缀，使 key 与我们模型的命名对齐。"""
        prefixes = [
            "module.", "model.", "perception_update.", "perception_module.",
            "per.", "net.", ""
        ]
        pat = re.compile(r'^(fusion_net|attn_net|update_net)\.layers\.\d+\.\d+\.')
        for pref in prefixes:
            if pref == "":
                cand = sd
            else:
                pref_len = len(pref)
                cand = {k[pref_len:]: v for k, v in sd.items() if k.startswith(pref)}
                # 如果没有任何键被剥离，跳过这个前缀
                if len(cand) == 0:
                    continue
            # 有任何 key 匹配我们的层命名，就认为成功
            if any(pat.match(k) for k in cand.keys()):
                return cand
        return sd  # 保底返回原字典

    # 1) 直接就是 tensor 映射
    state_dict_raw = None
    if all(isinstance(v, torch.Tensor) for v in obj.values()):
        state_dict_raw = obj
    else:
        # 2) 常见键尝试
        for k in ['state_dict', 'model_state_dict', 'perception_state_dict', 'perception_update',
                  'per', 'model', 'module', 'ema', 'per_hard', 'per_soft']:
            v = obj.get(k, None)
            if isinstance(v, dict):
                if looks_like_statedict(v):
                    state_dict_raw = v
                    break
                # 二级常见嵌套：有时里面还包了一层 state_dict
                if 'state_dict' in v and isinstance(v['state_dict'], dict) and looks_like_statedict(v['state_dict']):
                    state_dict_raw = v['state_dict']
                    break
        # 3) 递归 BFS 扫描任意子字典
        if state_dict_raw is None:
            dq = deque([obj])
            pat_gate = re.compile(r'(fusion_net|attn_net|update_net)\.layers\.\d+\.\d+\.')
            while dq:
                cur = dq.popleft()
                if not isinstance(cur, dict):
                    continue
                if looks_like_statedict(cur):
                    state_dict_raw = cur
                    break
                for vv in cur.values():
                    if isinstance(vv, dict):
                        dq.append(vv)

    if state_dict_raw is None:
        tops = list(obj.keys())[:12] if isinstance(obj, dict) else []
        raise RuntimeError(f"未找到可用的 state_dict；可用顶层键示例: {tops}")

    # —— 去前缀以匹配我们的命名 —— 
    state_dict_raw = try_strip_prefix(state_dict_raw)

    # —— 解析层尺寸（仅依据 state_dict 中真正出现过的门）——
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
        raise RuntimeError("无法从 state_dict 解析出完整的层配置（fusion/attn/update）")

    print("[INFO] Parsed layer widths from checkpoint:")
    print(f"  fusion: {layer_cfg['fusion']}")
    print(f"  attn  : {layer_cfg['attn']}")
    print(f"  update: {layer_cfg['update']}")

    # —— 构建模型（按解析到的真实结构）——
    model = PerceptionUpdateModule(layer_cfg).to(DEVICE)

    # —— 构造“期望键集合”，并从 state_dict 过滤无关键，仅保留模型真正需要的键 —— 
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict_raw.keys())
    state_dict = {k: v for k, v in state_dict_raw.items() if k in model_keys}

    # 先检查“多余键”（无伤大雅，打印前若干条便于定位导出问题）
    unexpected = sorted(list(state_keys - model_keys))
    if unexpected:
        head = "\n  - " + "\n  - ".join(unexpected[:10])
        more = f"\n  ...(+{len(unexpected)-10} more)" if len(unexpected) > 10 else ""
        print(f"[INFO] {len(unexpected)} unexpected keys are ignored (kept only model-relevant keys):{head}{more}")

    # —— 严格校验：所有“必需键”都必须存在，否则直接报错 —— 
    # 对每个 gate 要有：function_logits（参数）、idx_a（buffer）、idx_b（buffer）、temp（buffer）
    required_missing = []
    # —— 校验并补齐 state_dict 键（支持可选的 mix_raw）——
    required_missing = []
    # 注意：此处 state_dict 已按 model_keys 过滤，只含模型真正需要的键
    for branch, sizes in [('fusion_net', layer_cfg['fusion']),
                          ('attn_net',   layer_cfg['attn']),
                          ('update_net', layer_cfg['update'])]:
        for li, width in enumerate(sizes):
            for gi in range(width):
                base = f"{branch}.layers.{li}.{gi}."
                # 严格必需的键（与旧版一致）
                for suf in ["function_logits", "idx_a", "idx_b", "temp"]:
                    k = base + suf
                    if k not in state_dict:
                        required_missing.append(k)
    if required_missing:
        head = "\n  - " + "\n  - ".join(required_missing[:20])
        more = f"\n  .(+{len(required_missing)-20} more)" if len(required_missing) > 20 else ""
        raise RuntimeError(f"[FATAL] Missing required gate keys in checkpoint (model cannot be trusted):{head}{more}")

    # —— 可选键：mix_raw 若缺失则自动补齐为模型默认值（≈σ(12.0)≈1.0）——
    defaulted_mix = 0
    for k in list(model.state_dict().keys()):
        if k.endswith(".mix_raw") and (k not in state_dict):
            state_dict[k] = model.state_dict()[k].clone()
            defaulted_mix += 1

    # —— 严格加载（strict=True）——
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        # 进一步给出“缺谁”的可读列表
        need = set(model.state_dict().keys())
        have = set(state_dict.keys())
        missing = sorted(list(need - have))
        head = "\n  - " + "\n  - ".join(missing[:20])
        more = f"\n  .(+{len(missing)-20} more)" if len(missing) > 20 else ""
        raise RuntimeError(f"[FATAL] strict load_state_dict failed; missing keys:{head}{more}") from e

    # —— 统一 ckpt 已包含 mix_raw；默认不再进行伴随覆盖 —— 
    applied_source = None
    # 如需兼容历史（早期仅导出 logits/temp、缺失 mix_raw）产物，可临时将 APPLY_COMPANION_MIX 设为 True 并恢复本段逻辑。

    # —— 兼容性提示 —— 
    if defaulted_mix > 0:
        print(f"[WARN] {defaulted_mix} gate(s) missing mix_raw in checkpoint; "
              f"defaulted to 12.0 (σ≈1.0) for backward compatibility.")

    # —— 打印整体权重统计（注明来源）——
    try:
        strengths = []
        for net in (model.fusion_net, model.attn_net, model.update_net):
            for layer in net.layers:
                for g in layer:
                    if hasattr(g, "mix_raw"):
                        strengths.append(float(torch.sigmoid(g.mix_raw).detach().cpu().item()))
        if strengths:
            s_min = min(strengths); s_max = max(strengths); s_mean = sum(strengths) / len(strengths)
            src = f" (from {applied_source})" if applied_source else " (from checkpoint)"
            print(f"[INFO] Gate strength σ(mix_raw){src}: mean={s_mean:.3f}, min={s_min:.3f}, max={s_max:.3f}, count={len(strengths)}")
    except Exception:
        pass

    # 语义校验：注意力头的最后一层宽度必须=1
    if len(model.attn_net.layers[-1]) != 1:
        raise RuntimeError(f"[FATAL] Attention head width must be 1, but got {len(model.attn_net.layers[-1])}. "
                           f"Please check export pipeline.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

# ========== 6) 可替换的“方法”模块（便于对比） ==========
class AnalysisMethod:
    """
    将“方法”收拢成一个类，便于未来替换其它方法做对比：
     - 输入来源、聚类策略、相似性参数均从外层可替换
    """
    def __init__(self, model: PerceptionUpdateModule, precompute: PrecomputeModuleAnalysis,
                 similarity: SimilarityAndClustering):
        self.model = model
        self.pre   = precompute
        self.sim   = similarity

    @torch.no_grad()
    def run_once(self, graph_paths):
        # 逐图前向 5 次迭代，收集各时刻状态
        all_states = []
        for i, gp in enumerate(graph_paths, 1):
            print(f"\rForward {i}/{len(graph_paths)}", end="")
            pre_cpu = self.pre.process_graph(gp)

            # 构造 GPU 常量输入
            # 用“度（原值，不归一化）”作为初始节点数值；attention 内部再按策略归一化
            N = pre_cpu['node_states'].numel() if pre_cpu['node_states'].dim() == 1 else pre_cpu['node_states'].size(0)
            edge_src_cpu = pre_cpu['edge_src']
            ones = torch.ones(edge_src_cpu.numel(), dtype=torch.float32) if edge_src_cpu.numel() > 0 else None
            deg  = torch.zeros((N, 1), dtype=torch.float32)
            if edge_src_cpu.numel() > 0:
                deg.index_add_(0, edge_src_cpu, ones.unsqueeze(1))

            pre_gpu = {
                'node_states': deg.to(DEVICE, non_blocking=True),
                'neighbors':   pre_cpu['neighbors'],
                'space_info':  [t.to(DEVICE, non_blocking=True) for t in pre_cpu['space_info']],
                'edge_src':    pre_cpu['edge_src'].to(DEVICE, non_blocking=True),
                'edge_dst':    pre_cpu['edge_dst'].to(DEVICE, non_blocking=True),
                'edge_info':   pre_cpu['edge_info'].to(DEVICE, non_blocking=True)
            }

            states = []
            current = pre_gpu['node_states']
            for t in range(NUM_ITER):
                iter_in = {
                    'node_states': current,
                    'neighbors':   pre_gpu['neighbors'],
                    'space_info':  pre_gpu['space_info'],
                    'edge_src':    pre_gpu['edge_src'],
                    'edge_dst':    pre_gpu['edge_dst'],
                    'edge_info':   pre_gpu['edge_info']
                }
                current = self.model(iter_in)
                states.append(current.detach())
            all_states.append(states)
        print()
        sims = self.sim.compute_similarity_matrix(all_states)
        labels, metrics, Z = self.sim.cluster_and_metrics(sims)
        return sims, labels, metrics, Z

# ========== 7) 可视化：同簇分块热力图 ==========
def plot_similarity_blocked(sim_matrix: np.ndarray, city_names, labels, Z, save_path: str, style: dict = None):
    """
    进一步改进：
      - 左侧树：叶端固定在最右，非线性距离映射（CDF+γ），整体右移仅留细缝；叶端严格对齐到每个方格中心；
      - 标签：绘图阶段把下划线“_”替换为空格后再渲染；
      - 色带：与主图等高，顶/底端添加数值文本并保留端点刻度，预留右侧空间避免裁切。
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import json, os

    def _rgb255_to_hex(rgb):
        r, g, b = [int(x) for x in rgb]
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _shorten(s: str, n: int) -> str:
        if n is None or n <= 0 or len(s) <= n:
            return s
        return s[: max(1, n - 1)] + "…"

    def _clean(s: str) -> str:
        return str(s).replace("_", " ")

    # —— 默认风格（可覆盖）——
    if style is None:
        style = {}
    style.setdefault("color_low_rgb",  [0,   90, 181])
    style.setdefault("color_high_rgb", [240, 88,  64 ])
    style.setdefault("mid_rgb",        [248, 250, 252])
    style.setdefault("mid_pos",        0.50)
    style.setdefault("border_color",   "#0f0f0f")
    style.setdefault("border_lw",      2.0)
    style.setdefault("label_color",    "#2b2b2b")
    style.setdefault("gridline_color", "#d9d9d9")
    style.setdefault("cbar_label",     "Jensen–Shannon distance")
    style.setdefault("cbar_tick_fmt",  ".2f")
    style.setdefault("show_mid_tick",  False)
    style.setdefault("cbar_label_gap_pt", 4.0)  # 标题左边框与色带右边框的间隔（单位：pt）
    style.setdefault("wspace",         0.18)
    style.setdefault("max_name_chars", 20)
    style.setdefault("x_fontsize",     7)
    style.setdefault("y_fontsize",     8)
    style.setdefault("x_rotation",     78)
    # 树的外观与压缩
    style.setdefault("dendro_color",     "#2c2c2c")
    style.setdefault("dendro_lw",        1.6)
    style.setdefault("dendro_map",       "levels")      # 'levels' 等距；或 'cdf'|'sqrt'|'linear'|'log'
    style.setdefault("dendro_gamma",     1.00)
    style.setdefault("dendro_minfrac",   0.06)          # 左端最小占比略增，进一步缓解拥挤
    style.setdefault("dendro_w_ratio",   0.24)          # 树轴宽度占比（相对 S），原来≈0.16
    style.setdefault("dendro_level_beta",1.8)           # >1 展开左端；=1 等距；<1 向右侧压缩

    # —— 调色：两端色 + 更浅中点 —— 
    low_hex  = _rgb255_to_hex(style["color_low_rgb"])
    mid_hex  = _rgb255_to_hex(style["mid_rgb"])
    high_hex = _rgb255_to_hex(style["color_high_rgb"])
    try:
        mp = float(style.get("mid_pos", 0.5))
        if not (0.0 < mp < 1.0): mp = 0.5
    except Exception:
        mp = 0.5
    cmap = LinearSegmentedColormap.from_list("two_color_mid", [(0.0, low_hex), (mp, mid_hex), (1.0, high_hex)])

    # —— 数据与顺序 —— 
    D = np.array(sim_matrix, dtype=float).copy()
    np.fill_diagonal(D, 0.0); D[D < 0] = 0.0; D = 0.5 * (D + D.T)
    K = D.shape[0]
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = np.asarray(leaves, dtype=int)
    D_ord = D[order][:, order]
    # 标签：先清洗 _ → 空格
    cities_ord = [_clean(city_names[i]) for i in order]
    labels_ord = np.asarray(labels)[order]

    # —— 画布尺寸 —— 
    cell = 0.28
    S = max(6.0, K * cell)
    wd = max(0.9, float(style.get("dendro_w_ratio", 0.24)) * S)   # 可调宽度，默认更宽
    cbar_stub = 0.05 * S
    # 底边留白略收紧（标签仍完整），减小大面积空白
    bottom_pad = max(0.9, 0.16 * S)
    margin_w, margin_h = 0.6, 0.6
    fig_w = wd + S + cbar_stub + margin_w
    fig_h = S + bottom_pad + margin_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    gs = gridspec.GridSpec(2, 3,
        width_ratios=[wd, S, cbar_stub],
        height_ratios=[S, bottom_pad],
        wspace=style["wspace"], hspace=0.02
    )
    ax_dy   = fig.add_subplot(gs[0, 0])
    ax_hm   = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_pad  = fig.add_subplot(gs[1, 1]); ax_pad.axis("off")

    # —— 热力图（严格 1:1；格心=1..K）——
    vmin = float(np.nanmin(D_ord)); vmax = float(np.nanmax(D_ord) if np.nanmax(D_ord) > 0 else 1.0)
    im = ax_hm.imshow(D_ord, cmap=cmap, interpolation='nearest',
                      origin='lower', aspect='equal',
                      extent=[0.5, K + 0.5, 0.5, K + 0.5], vmin=vmin, vmax=vmax)

    # —— 轴刻度与标签（右侧=全名；下方=截断名；均已去下划线）——
    ax_hm.set_xticks(np.arange(1, K + 1)); ax_hm.set_yticks(np.arange(1, K + 1))
    ax_hm.set_xticklabels(cities_ord, rotation=style["x_rotation"], fontsize=style["x_fontsize"],
                          color=style["label_color"], ha="right", va="top")
    ax_hm.set_yticklabels(cities_ord, fontsize=style["y_fontsize"], color=style["label_color"])
    ax_hm.tick_params(axis='both', which='both', length=0, pad=1.5)
    ax_hm.yaxis.tick_right(); ax_hm.yaxis.set_label_position("right")

    # 网格在低 zorder
    ax_hm.set_xticks(np.arange(0.5, K + 0.5, 1.0), minor=True)
    ax_hm.set_yticks(np.arange(0.5, K + 0.5, 1.0), minor=True)
    ax_hm.grid(which='minor', color=style["gridline_color"], linewidth=0.5, zorder=1)
    for spine in ax_hm.spines.values(): spine.set_visible(False)

    # —— 左侧树（叶端固定在最右 + “层级等距”可加权展开）——
    def _draw_left_dendrogram(ax, Z, order_idx, color="#333", lw=1.4,
                              right_gap=0.002, map_kind="levels", gamma=1.0, minfrac=0.04, level_beta=1.8):
        """
        map_kind="levels"：忽略实际合并距离，按合并“层级序号”映射到 [minfrac,1]。
        level_beta>1：对靠左（根部、后期合并）做“ease-in”展开，缓解最左端拥挤；
        level_beta=1 等距；level_beta<1 则更靠近右侧。
        """
        n = len(order_idx)
        y_pos = np.empty(n, dtype=float)
        for i, leaf in enumerate(order_idx):
            # 叶节点严格对齐主图的格心：1..n
            y_pos[leaf] = i + 1

        h = Z[:, 2].astype(float)
        if h.size == 0:
            ax.set_ylim(0.5, n + 0.5)
            ax.set_xlim(0.0, 1.0)
            ax.axis('off')
            ax.set_anchor('E')
            return

        # —— 预处理：为不同映射准备统计量 —— 
        h_sorted = np.sort(h)
        h_min, h_max = float(h_sorted[0]), float(h_sorted[-1])
        span = max(h_max - h_min, 1e-12)

        # 层级数
        m = Z.shape[0]

        def _map_value(idx, val, kind):
            if kind == "levels":
                # u ∈ (0,1]，越靠后合并（越接近根）越大；对 u 施加指数展开
                u = (idx + 1) / m
                u = float(u ** max(1.0, level_beta))  # level_beta>1 扩展靠左空间
                t = minfrac + (1.0 - minfrac) * u
            elif kind == "cdf":
                rank = np.searchsorted(h_sorted, val, side="right") / len(h_sorted)
                t = max(min(rank, 1.0), 1e-6) ** gamma
                t = minfrac + (1.0 - minfrac) * t
            elif kind == "log":
                t = np.log((val - h_min) + 1e-9) / np.log(span + 1e-9)
                t = minfrac + (1.0 - minfrac) * max(min(t, 1.0), 0.0) ** gamma
            elif kind == "sqrt":
                t = ((val - h_min) / span) ** 0.5
                t = minfrac + (1.0 - minfrac) * max(min(t, 1.0), 0.0) ** gamma
            else:  # "linear"
                t = (val - h_min) / span
                t = minfrac + (1.0 - minfrac) * max(min(t, 1.0), 0.0) ** gamma
            return t

        x_leaf = 1.0 - right_gap                # 叶端紧贴右侧，仅留细缝
        x_left_min = minfrac                    # 最左边界
        # 保存每个“节点”（叶子 + 合并后的内部节点）的坐标
        x = np.zeros(n + m, dtype=float)
        y = np.zeros(n + m, dtype=float)
        for i in range(n):
            x[i] = x_leaf
            y[i] = y_pos[i]

        # 逐次合并，按所选映射把竖干放在等距/展开后的位置
        for i in range(m):
            a = int(Z[i, 0]); b = int(Z[i, 1])
            xa, ya = x[a], y[a]
            xb, yb = x[b], y[b]
            # 目标竖干位置
            d = _map_value(i, h[i], map_kind)
            xp = x_leaf - (x_leaf - x_left_min) * d
            yp = (ya + yb) / 2.0
            # 三段折线
            ax.plot([xa, xp], [ya, ya], color=color, linewidth=lw,
                    solid_capstyle='round', solid_joinstyle='round', antialiased=True, zorder=3)
            ax.plot([xb, xp], [yb, yb], color=color, linewidth=lw,
                    solid_capstyle='round', solid_joinstyle='round', antialiased=True, zorder=3)
            ax.plot([xp, xp], [ya, yb], color=color, linewidth=lw,
                    solid_capstyle='round', solid_joinstyle='round', antialiased=True, zorder=3)
            x[n + i] = xp
            y[n + i] = yp

        # 纵向边界与对齐到主图格心
        ax.set_ylim(0.5, n + 0.5)
        ax.set_xlim(0.0, 1.0)
        ax.axis('off')
        ax.set_anchor('E')                      # 轴图元锚定到右侧，避免任何“偏离”
        ax.margins(x=0, y=0)

    # 绘制等距层级树；随后强制把树轴的上下界与主图完全一致，并把树轴右缘贴到主图左缘仅留一条缝
    _draw_left_dendrogram(
        ax=ax_dy, Z=Z, order_idx=order, color=style["dendro_color"], lw=style["dendro_lw"],
        right_gap=0.002, map_kind=style.get("dendro_map", "levels"), gamma=style.get("dendro_gamma", 1.0),
        minfrac=style.get("dendro_minfrac", 0.06), level_beta=style.get("dendro_level_beta", 1.8)
    )
    # 垂直严格对齐：与热图同一 y-limits（每个分支正对每个格心）
    ax_dy.set_ylim(ax_hm.get_ylim())

    # 水平再贴近：把树轴的右边界拉到热图左边界附近，只留极窄缝隙
    try:
        hm_pos = ax_hm.get_position()
        dy_pos = ax_dy.get_position()
        seam = 0.004                                      # figure 坐标下的极细缝
        new_w = dy_pos.width
        new_x1 = hm_pos.x0 - seam
        new_x0 = max(0.02, new_x1 - new_w)               # 保留少量左边距
        ax_dy.set_position([new_x0, hm_pos.y0, new_w, hm_pos.height])
    except Exception:
        pass

    # —— 分簇边框与外框（置顶）——
    boundaries = []; start = 1
    for i in range(2, K + 1):
        if labels_ord[i - 1] != labels_ord[i - 2]: boundaries.append((start, i - 1)); start = i
    boundaries.append((start, K))
    for (s, e) in boundaries:
        w = e - s + 1
        ax_hm.add_patch(Rectangle((s - 0.5, s - 0.5), w, w, fill=False,
                                  linewidth=2.0, edgecolor=style["border_color"], zorder=5))
    ax_hm.add_patch(Rectangle((0.5, 0.5), K, K, fill=False,
                              linewidth=style["border_lw"] * 2.0, edgecolor=style["border_color"], zorder=6))

    # —— 色带：等高 + 端点数值（预留右侧空间）——
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        y_exts = [t.get_window_extent(renderer=rend) for t in ax_hm.get_yticklabels()]
        max_w_in = (max(e.width for e in y_exts) / fig.dpi) if y_exts else 0.0
        pad_in = max_w_in + 0.10
        cbar_w_in = 0.45
        hm_pos = ax_hm.get_position()
        new_x0 = hm_pos.x1 + pad_in / fig_w
        new_w  = cbar_w_in / fig_w
        if new_x0 + new_w > 0.98:
            new_w = max(0.02, 0.98 - new_x0)
        ax_cbar.set_position([new_x0, hm_pos.y0, new_w, hm_pos.height])

        cbar = plt.colorbar(im, cax=ax_cbar)
        # 竖排标签：以“底边中点”为锚点，并保证其外接框中心与色带中心严格重合；再向右脱开固定点值 Δ
        cbar.ax.set_ylabel(style["cbar_label"], fontsize=9, color=style["label_color"],
                           rotation=270, labelpad=0)
        cbar.ax.yaxis.label.set_rotation_mode('anchor')  # 旋转围绕锚点进行

        fig.canvas.draw()
        _rend = fig.canvas.get_renderer()
        _bbox = ax_cbar.get_window_extent(renderer=_rend)

        # 基本间距（pt）+ 额外脱开量（pt，默认 4pt，比之前略大一点）
        _gap_pt_base = float(style.get("cbar_label_gap_pt", 4.0))
        _gap_pt_extra = float(style.get("cbar_label_dx_pt", 4.0))
        _gap_px = (_gap_pt_base + _gap_pt_extra) * fig.dpi / 72.0
        _gap_ax = (_gap_px / _bbox.width) if _bbox.width > 0 else 0.03

        # 初次放置：以色带右边框为基准 + Δ；y=0.5 目标居中
        cbar.ax.yaxis.set_label_coords(1.0 + _gap_ax, 0.5)
        cbar.ax.yaxis.label.set_ha("left")
        cbar.ax.yaxis.label.set_va("center")

        # 二次微调（关键）：用真实文本框中心对齐到色带中心
        fig.canvas.draw()
        _tb = cbar.ax.yaxis.label.get_window_extent(renderer=_rend)
        _cy_ax = 0.5 * (_bbox.y0 + _bbox.y1)
        _cy_txt = 0.5 * (_tb.y0 + _tb.y1)
        _dy_ax = ((_cy_ax - _cy_txt) / _bbox.height) if _bbox.height > 0 else 0.0
        cbar.ax.yaxis.set_label_coords(1.0 + _gap_ax, 0.5 + _dy_ax)

        cbar.outline.set_edgecolor(style["border_color"])
        cbar.outline.set_linewidth(style["border_lw"])
        cbar.ax.tick_params(labelcolor=style["label_color"], color=style["label_color"],
                            length=0, pad=2)
        fmt = style.get("cbar_tick_fmt", ".2f")
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:{fmt}}", f"{vmax:{fmt}}"])
    except Exception:
        # 回退方案：保持同样的“底边中点锚点 + 像素级二次对齐 + 右侧脱开”
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.ax.set_ylabel(style["cbar_label"], fontsize=9, color=style["label_color"],
                           rotation=270, labelpad=0)
        cbar.ax.yaxis.label.set_rotation_mode('anchor')
        try:
            fig.canvas.draw()
            _rend = fig.canvas.get_renderer()
            _bbox = ax_cbar.get_window_extent(renderer=_rend)
            _gap_pt_base = float(style.get("cbar_label_gap_pt", 4.0))
            _gap_pt_extra = float(style.get("cbar_label_dx_pt", 4.0))
            _gap_px = (_gap_pt_base + _gap_pt_extra) * fig.dpi / 72.0
            _gap_ax = (_gap_px / _bbox.width) if _bbox.width > 0 else 0.03
        except Exception:
            _gap_ax = 0.03
        cbar.ax.yaxis.set_label_coords(1.0 + _gap_ax, 0.5)
        cbar.ax.yaxis.label.set_ha("left")
        cbar.ax.yaxis.label.set_va("center")
        try:
            fig.canvas.draw()
            _tb = cbar.ax.yaxis.label.get_window_extent(renderer=_rend)
            _cy_ax = 0.5 * (_bbox.y0 + _bbox.y1)
            _cy_txt = 0.5 * (_tb.y0 + _tb.y1)
            _dy_ax = ((_cy_ax - _cy_txt) / _bbox.height) if _bbox.height > 0 else 0.0
            cbar.ax.yaxis.set_label_coords(1.0 + _gap_ax, 0.5 + _dy_ax)

            cbar.outline.set_edgecolor(style["border_color"])
            cbar.outline.set_linewidth(style["border_lw"])
            fmt = style.get("cbar_tick_fmt", ".2f")
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f"{vmin:{fmt}}", f"{vmax:{fmt}}"])
            cbar.ax.tick_params(labelcolor=style["label_color"], color=style["label_color"],
                                length=0, pad=2)
        except Exception:
            pass

    # —— 保存 & 弹窗（避免裁切右侧与底部，并弹出交互界面）——
    fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight", pad_inches=0.02, format="pdf")
    plt.close(fig)

    # —— 导出辅助文件（保持不变；注意此处仍用原始名字，不替换下划线）——
    base = os.path.splitext(save_path)[0]
    with open(base + "_leaf_order.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f); w.writerow(["Index_in_order", "City", "Cluster"])
        for i in range(K):
            w.writerow([i, cities_ord[i - 1] if i > 0 else cities_ord[0], int(labels_ord[i - 1] if i > 0 else labels_ord[0])])
    with open(base + "_matrix.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f); w.writerow([""] + cities_ord)
        for i in range(K):
            w.writerow([cities_ord[i - 1] if i > 0 else cities_ord[0]] + [f"{D_ord[i - 1, j - 1]:.6g}" for j in range(1, K + 1)])
    np.save(base + "_linkage.npy", Z)
    try:
        with open(base + "_style.json", "w", encoding="utf-8") as f:
            json.dump(style, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ========== 8) 主流程 ==========
def main():
    # 1) 列出 analysis 下的 *.graphml
    all_graphs = []
    for root, _, files in os.walk(GRAPH_DIR):
        for f in files:
            if f.lower().endswith('.graphml'):
                all_graphs.append(os.path.join(root, f))
    all_graphs.sort()

    if len(all_graphs) == 0:
        raise RuntimeError(f"在 {GRAPH_DIR} 下未找到任何 .graphml 文件")
    if len(all_graphs) != 50:
        print(f"[WARN] 发现 {len(all_graphs)} 个图；预期 50。将按发现的全部进行评估。")

    # 城市名 = 文件名（去扩展名）
    city_names = [os.path.splitext(os.path.basename(p))[0] for p in all_graphs]

    # 2) 构建模块并载入已训练模型
    pre = PrecomputeModuleAnalysis()
    sim = SimilarityAndClustering(bins=128)
    model = load_perception_module(MODEL_PT)

    # —— 严格使用 checkpoint 中的门：关闭一切随机 one-hot 调试 —— 
    ENABLE_RANDOM_ONEHOT = False
    # 保留占位，但不再进行任何随机重置或打印，避免覆盖从 simplify.py 硬化后的结果

    method = AnalysisMethod(model, pre, sim)

    # 3) 前向 + 相似性 + 聚类 + 指标
    sims_t, labels, metrics, Z = method.run_once(all_graphs)

    # 4) 保存指标
    metrics_path_json = os.path.join(SAVE_DIR, 'analysis_metrics.json')
    metrics_path_csv  = os.path.join(SAVE_DIR, 'analysis_metrics.csv')
    with open(metrics_path_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_path_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # 5) 保存聚类结果（城市-簇号）
    assign_csv = os.path.join(SAVE_DIR, 'cluster_assignments.csv')
    with open(assign_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['City', 'Cluster'])
        for c, lab in zip(city_names, labels.tolist()):
            w.writerow([c, int(lab)])

    # 6) 保存矩阵（仅 JS 距离）
    sims_np = sims_t.detach().cpu().numpy()

    # 7) 导出“原顺序”的矩阵 CSV（与作图函数内部的单次排序保持一致）
    def save_square_csv(mat, labels, path):
        K = mat.shape[0]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([''] + labels)
            for i in range(K):
                row = [labels[i]] + [f"{x:.6g}" for x in mat[i].tolist()]
                w.writerow(row)

    save_square_csv(sims_np, city_names, os.path.join(SAVE_DIR, 'pairwise_js.csv'))

    # 8) 画热力图（仅在作图函数内部做一次性排序：按层次树叶序）
    heatmap_path = os.path.join(SAVE_DIR, 'similarity_heatmap_ordered.png')
    plot_similarity_blocked(sims_np, city_names, labels, Z, heatmap_path)

    # 10) 控制台简报
    print("\n===== Analysis Summary =====")
    print(f"Clusters: {metrics['Cluster_Count']}")
    def _fmt4(v: float) -> str:
        try:
            av = abs(float(v))
        except Exception:
            return str(v)
        # 对极小值用科学计数法，避免被 .4f 四舍五入成 0.0000
        return f"{v:.4e}" if (av != 0.0 and av < 1e-3) else f"{v:.4f}"

    print(f"Intra_Dist(JS): {_fmt4(metrics['Intra_Dist'])} | Inter_Dist(JS): {_fmt4(metrics['Inter_Dist'])}")
    print(f"Silhouette(precomputed dist): {_fmt4(metrics['Silhouette'])} | Soft_Silhouette: {_fmt4(metrics['Soft_Silhouette'])}")
    print(f"Calinski-Harabasz: {_fmt4(metrics['Calinski_Harabasz'])} | Davies-Bouldin: {_fmt4(metrics['Davies_Bouldin'])}")
    print(f"Saved to: {SAVE_DIR}")
    print(f"- {os.path.basename(assign_csv)}")
    print(f"- pairwise_js.csv")
    print(f"- {os.path.basename(heatmap_path)}")
    print(f"- analysis_metrics.json / analysis_metrics.csv")

if __name__ == "__main__":
    main()
