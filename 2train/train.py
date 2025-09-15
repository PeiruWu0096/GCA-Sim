#!/usr/bin/env python3
"""
Full training script organized into modules 0–4.
Differentiable logic-gate network in PyTorch with early stop, LR scheduling, rehearsal, and full checkpoint saving.
"""
import os
import sys
import math
import argparse
import json
import pickle
from datetime import datetime
import csv
from collections import defaultdict
import random
import copy
import gc

# ========= Runtime hardening (run before numpy/scipy/sklearn/torch imports) =========
# Goal: avoid PATH/BLAS/threads pollution within this process.
def _harden_runtime_env():
    # Threads/BLAS → single-thread to avoid MKL/OpenBLAS/LLVM-OpenMP conflicts
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Disable colored output/hooks that may inject runtime behavior
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("PY_COLORS", "0")
    os.environ.setdefault("RICH_NO_COLOR", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Disable Torch dynamo/compile (can interfere with eval-frame cache)
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    # Reorder PATH: put current Python and its Scripts first; defer risky entries
    danger_keys = ("msys64", "mingw", "graphviz\\bin", "anaconda", "conda", "git\\usr\\bin", "llvm\\bin")
    cur = os.environ.get("PATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    py_root = os.path.dirname(sys.executable)
    py_scripts = os.path.join(py_root, "Scripts")
    safe, later = [], []
    for p in parts:
        if p in (py_root, py_scripts):
            continue
        (later if any(k in p.lower() for k in danger_keys) else safe).append(p)
    new_path = os.pathsep.join([py_root, py_scripts] + safe + later)
    os.environ["PATH"] = new_path
    # Mask sitecustomize/usercustomize hooks (no site-packages path changes here)
    for m in ("sitecustomize", "usercustomize"):
        if m in sys.modules:
            try:
                del sys.modules[m]
            except Exception:
                pass

def _ensure_import_paths():
    """Ensure site-packages are visible when needed (process-local only)."""
    import importlib.util, site
    needed = ("numpy", "scipy", "sklearn", "torch")
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        paths = []
        try:
            sp = site.getsitepackages()
            if isinstance(sp, (list, tuple)):
                paths += sp
        except Exception:
            pass
        try:
            up = site.getusersitepackages()
            if isinstance(up, str):
                paths.append(up)
        except Exception:
            pass
        for p in paths:
            if p and (p not in sys.path):
                sys.path.append(p)

_harden_runtime_env()
_ensure_import_paths()
# ========= end hardening =========

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Limit PyTorch threads (after importing torch)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 0. Setup & Configuration ====================
# ---- Data paths ----
PATHS = {
    'train_graphml': r'0data/meta/train/graphml',
    'train_pre':     r'0data/meta/train/pre',
    'val_graphml':   r'0data/meta/validation/graphml',
    'val_pre':       r'0data/meta/validation/pre',
    'save_root':     r'0data/output/trainsave',
    'tmp_cache':     r'0data/output/trainsave/tmp_cache',
}

TRAIN_PATHS = {'mixed': PATHS['train_graphml']}
VAL_PATHS   = {'mixed': PATHS['val_graphml']}
SAVE_ROOT   = PATHS['save_root']

class PrecomputeModule:
    """Load precomputed data + optional on-disk cache; supports delete-after-use."""
    pre_train_paths = {'mixed': PATHS['train_pre']}
    pre_val_paths   = {'mixed': PATHS['val_pre']}
    CACHE = {}

# ---- Seeds (global) ----
BASE_SEED      = 3407
NUM_SEEDS      = 10

NUM_ITER            = 5
LR_INITIAL          = 5e-1
EPOCHS_PER_GROUP    = 10
EARLY_STOP_PATIENCE = 0
REHEARSAL_INTERVAL  = 10
REHEARSAL_EPOCHS    = 5
REHEARSAL_DROP1     = 0.10
REHEARSAL_DROP2     = 0.02
NO_ANNEAL_EPOCHS    = 0
BATCH_SIZE          = 32
MAX_K               = 20
accum_steps         = 1
ENABLE_GRAD_PROBE   = False

# ---- Histogram & soft-silhouette hyperparameters ----
HIST_BW             = 0.07
SOFTSIL_TAU         = 0.20
SIMS_GLOBAL_SCALE   = 1.0

# ---- Early-stop conditions ----
# Conditions (any required by your trainer): min epochs, soft-sil, hard-sil, min clusters
STOP_MIN_EPOCHS       = 40
STOP_SOFTSIL          = 0.84
STOP_HARDSIL          = 0.91
STOP_MIN_CLUSTERS     = 4
EARLY_STOP_PATIENCE   = 0
STOP_SIL              = STOP_HARDSIL

# ---- Switch from random to finetune ----
RANDOM_SWITCH_HARDSIL = 0.91
RANDOM_SWITCH_SOFTSIL = 0.80

# ---- Finetune targets ----
FINETUNE_TARGET_HARD  = 0.91
FINETUNE_TARGET_SOFT  = 0.84
FINETUNE_PATIENCE     = 10

# ---- Group consistency thresholds (GCI in [0,1]) ----
SWITCH_MIN_GCI        = 0.80
STOP_MIN_GCI          = 0.80

# ---- Random-phase Gaussian perturbation ----
RANDOM_SIGMA          = 0.05

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gate temperature / annealing
TEMP_INIT          = 15.0
TEMP_MIN           = 2
GATE_ANNEAL_ENABLE = False
GATE_ANNEAL_GAMMA  = 0.9  # kept for backward-compatibility

# ---- Adaptive annealing (exp * cosine weight) ----
ANNEAL_BASE_GAMMA     = 0.98
ANNEAL_START_EPOCH    = 6
ANNEAL_TARGET_EPOCH   = 160
ANNEAL_W0             = 0.35
ANNEAL_W1             = 1.10
ANNEAL_BETA           = 1.10
ANNEAL_GAMMA_MIN      = 0.94
ANNEAL_GAMMA_MAX      = 0.995
ANNEAL_SOFTSIL_THRESH = 0.90

# Gate initialization
ONE_HOT_INIT_LOGITS = False
GATE_SKEW_INIT      = True
INIT_BOOST          = 1.2
INIT_SPREAD         = 0.15
INIT_MARGIN_CAP     = 1.2

# ---- Global prior: gate group probabilities (random phase) ----
GATE_PROB_PASS   = 0.56
GATE_PROB_ADDSUB = 0.42
GATE_PROB_NONLIN = 0.02

# Temporary on-disk cache for simplified tensors
TMP_GRAPH_DIR        = PATHS['tmp_cache']
DELETE_TMP_AFTER_USE = True
os.makedirs(TMP_GRAPH_DIR, exist_ok=True)

def print_progress(stage, current, total):
    pct = 100.0 * current / max(1, total)
    sys.stdout.write(f"\r[{stage}] {pct:5.1f}%")
    sys.stdout.flush()

# ---- Unified checkpoint packer ----
def _pack_gnca_ckpt(model: nn.Module, epoch: int, seed: int, scale: str, group: int, layer_cfg: dict, metrics: dict):
    return {
        "kind": "gnca_model_ckpt",
        "version": 1,
        "epoch": int(epoch),
        "seed": int(seed),
        "scale": str(scale) if scale is not None else "",
        "group": int(group) if group is not None else -1,
        "layer_cfg": {
            "fusion": list(layer_cfg.get("fusion", [])),
            "attn":   list(layer_cfg.get("attn",   [])),
            "update": list(layer_cfg.get("update", [])),
        },
        "state_dict": model.state_dict(),
        "metrics": dict(metrics or {}),
    }

# ---- Atomic save ----
def _atomic_save(obj, path: str):
    tmp = path + ".writing"
    torch.save(obj, tmp)
    os.replace(tmp, path)

# ===== Startup self-sanity check (detect undefined refs early) =====
def _self_sanity_check_undeclared_refs():
    import io, re
    try:
        src_path = __file__ if '__file__' in globals() else sys.argv[0]
        with io.open(src_path, 'r', encoding='utf-8') as f:
            src = f.read()
        # Build a clean source for scanning: drop this function body + call, strip string literals
        def_start = re.search(r'^[ \t]*def[ \t]+_self_sanity_check_undeclared_refs\\s*\\(\\)\\s*:', src, flags=re.M|re.S)
        call_after = re.search(r'^[ \t]*_self_sanity_check_undeclared_refs\\s*\\(\\)\\s*$', src, flags=re.M)
        src_scan = src
        if def_start and call_after and call_after.start() > def_start.start():
            src_scan = src[:def_start.start()] + src[call_after.end():]
        str_pat = re.compile(r"('''.*?'''|\"\"\".*?\"\"\"|'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'|\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\")", flags=re.S)
        src_scan = str_pat.sub('""', src_scan)
        # Rule1: forbid unqualified last_epoch_metrics
        pat_unqualified_last = re.compile(r'(?<!\\.)\\blast_epoch_metrics\\b')
        bad_last = [m.group(0) for m in pat_unqualified_last.finditer(src_scan)]
        # Rule2: forbid undefined stop_onehot references (self./Trainer./bare)
        pat_stop_onehot_attr = re.compile(r'\\bself\\.stop_onehot\\b|\\bTrainer\\.stop_onehot\\b|\\.\\s*stop_onehot\\b')
        has_stop_onehot_attr = bool(pat_stop_onehot_attr.search(src_scan))
        pat_unqualified_stop = re.compile(r'(?<!\\.)\\bstop_onehot\\b')
        bad_stop = [m.group(0) for m in pat_unqualified_stop.finditer(src_scan)]
        errs = []
        if bad_last:
            errs.append("Found unqualified 'last_epoch_metrics'. Use 'self.last_epoch_metrics'.")
        if has_stop_onehot_attr or bad_stop:
            errs.append("Found references to 'stop_onehot' which is undefined. Remove or use the four-condition early-stop.")
        if errs:
            raise RuntimeError("Self-sanity check failed:\n  - " + "\n  - ".join(errs))
        else:
            print("[Self-Check] Source self-check passed: no invalid references.")
    except Exception:
        raise

_self_sanity_check_undeclared_refs()

# ==================== 1. Precompute module ====================
    @staticmethod
    def _path_map(graph_path):
        for s, root in TRAIN_PATHS.items():
            if graph_path.startswith(root):
                return PrecomputeModule.pre_train_paths[s], root, os.path.relpath(graph_path, root)
        for s, root in VAL_PATHS.items():
            if graph_path.startswith(root):
                return PrecomputeModule.pre_val_paths[s], root, os.path.relpath(graph_path, root)
        raise ValueError(f"Unknown graph path: {graph_path}")

    @staticmethod
    def _tmp_file(rel_path):
        safe = rel_path.replace("\\", "__").replace("/", "__")
        return os.path.join(TMP_GRAPH_DIR, os.path.splitext(safe)[0] + ".pt")

    @staticmethod
    def _build_simplified(data):
        neighbors = data['neighbors']
        edge_src, edge_dst, edge_info = [], [], []
        for u, idxs in enumerate(neighbors):
            idx_list = idxs.tolist() if torch.is_tensor(idxs) else idxs
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
            edge_info = torch.zeros((0, 2))
        return {
            'node_states': data['node_states'],
            'neighbors':   data['neighbors'],
            'space_info':  data['space_info'],
            'edge_src':    edge_src,
            'edge_dst':    edge_dst,
            'edge_info':   edge_info
        }

    def process_graph(self, graph_path):
        if graph_path in PrecomputeModule.CACHE:
            return PrecomputeModule.CACHE[graph_path]

        pre_root, base_root, rel_path = self._path_map(graph_path)
        tmp_file = self._tmp_file(rel_path)

        def _looks_valid_pt(path):
            try:
                size = os.path.getsize(path)
                if size < 256:
                    return False
                with open(path, 'rb') as f:
                    head = f.read(4)
                return (head[:2] == b'\x80\x04') or (head == b'PK\x03\x04')
            except Exception:
                return False

        def _safe_load_pt(path):
            return torch.load(path, map_location='cpu', weights_only=False)

        # 1) Prefer cache; if invalid/corrupt, rebuild
        if os.path.isfile(tmp_file):
            if not _looks_valid_pt(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
            else:
                try:
                    def _to_tensors(obj):
                        if isinstance(obj, np.ndarray):
                            return torch.from_numpy(obj)
                        if isinstance(obj, list):
                            return [_to_tensors(x) for x in obj]
                        if isinstance(obj, dict):
                            return {k: _to_tensors(v) for k, v in obj.items()}
                        return obj
                    with open(tmp_file, 'rb') as f:
                        loaded = pickle.load(f)
                    simplified = _to_tensors(loaded)
                    PrecomputeModule.CACHE[graph_path] = simplified
                    return simplified
                except Exception as e:
                    print(f"[WARN] Read cache failed {tmp_file}: {e} — will regenerate", flush=True)
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass

        # 2) Rebuild from precomputed .pt
        pre_file = os.path.join(pre_root, os.path.splitext(rel_path)[0] + '.pt')
        if not os.path.isfile(pre_file):
            raise FileNotFoundError(f"Precomputed file not found: {pre_file}")
        if not _looks_valid_pt(pre_file):
            raise RuntimeError(f"Invalid precomputed file header/size: {pre_file}")

        raw = _safe_load_pt(pre_file)
        simplified = self._build_simplified(raw)

        # 3) Atomic write cache: .writing → os.replace
        tmp_sw = tmp_file + ".writing"
        try:
            def _to_plain(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy()
                if isinstance(obj, list):
                    return [_to_plain(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: _to_plain(v) for k, v in obj.items()}
                return obj
            with open(tmp_sw, 'wb') as f:
                pickle.dump(_to_plain(simplified), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_sw, tmp_file)
        except Exception:
            try:
                if os.path.exists(tmp_sw):
                    os.remove(tmp_sw)
            except OSError:
                pass

        PrecomputeModule.CACHE[graph_path] = simplified
        return simplified

    @staticmethod
    def release(graph_path, delete_file=False):
        """
        Release in-memory cache; optionally delete the temp file on disk.
        If delete_file=True, remove the corresponding .pt in TMP_GRAPH_DIR.
        """
        if graph_path in PrecomputeModule.CACHE:
            del PrecomputeModule.CACHE[graph_path]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if delete_file:
            try:
                pre_root, base_root, rel_path = PrecomputeModule._path_map(graph_path)
                tmp_file = PrecomputeModule._tmp_file(rel_path)
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass

    @classmethod
    def clear_cache(cls):
        """Clear class-level in-memory cache only (keep temp files)."""
        try:
            cls.CACHE.clear()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def reset_all(cls, delete_tmp=True):
        """
        Hard reset across seeds/rounds:
          1) clear in-memory cache
          2) optionally delete temp .pt files under TMP_GRAPH_DIR
          3) clear CUDA cache and force GC
        """
        try:
            cls.CACHE.clear()
        except Exception:
            pass
        if delete_tmp:
            try:
                if os.path.isdir(TMP_GRAPH_DIR):
                    for name in os.listdir(TMP_GRAPH_DIR):
                        p = os.path.join(TMP_GRAPH_DIR, name)
                        if os.path.isfile(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

# ==================== 2. Differentiable gates & perception update ====================
class DiffLogicGate(nn.Module):
    """
    Simplified gate set: add, subtract, passthrough, and simple min/max.
    """
    # Operator list (kept as in original order)
    _OPS = [
        lambda slf, a, b: a + b,                  # 0: a+b
        lambda slf, a, b: a - b,                  # 1: a-b
        lambda slf, a, b: b - a,                  # 2: b-a
        lambda slf, a, b: a,                      # 3: pass A
        lambda slf, a, b: b,                      # 4: pass B
        lambda slf, a, b: -a,                     # 5: -A
        lambda slf, a, b: -b,                     # 6: -B
        lambda slf, a, b: torch.maximum(a, b),    # 7: max(a,b)
        lambda slf, a, b: torch.minimum(a, b),    # 8: min(a,b)
    ]
    def __init__(self, in_dim, temp=TEMP_INIT, idx_a=None, idx_b=None):
        super().__init__()
        self.num_functions = len(self._OPS)

        if idx_a is None or idx_b is None:
            ia, ib = np.random.choice(in_dim, 2, replace=False)
        else:
            ia, ib = idx_a, idx_b
        self.register_buffer('idx_a', torch.tensor(int(ia)))
        self.register_buffer('idx_b', torch.tensor(int(ib)))

        # Lightly biased initialization with group probabilities:
        # passthrough (incl. neg) > add/sub >> nonlinears (max/min)
        logits = torch.randn(self.num_functions, device=DEVICE) * float(INIT_SPREAD)
        if GATE_SKEW_INIT:
            # Group probabilities (can be overridden by global GATE_PROB_*; else defaults below)
            prob_pass   = float(globals().get('GATE_PROB_PASS',   0.56))  # passthrough incl. sign: 3,4,5,6
            prob_addsub = float(globals().get('GATE_PROB_ADDSUB', 0.42))  # add/sub: 0,1,2
            prob_nonlin = float(globals().get('GATE_PROB_NONLIN', 0.02))  # nonlinears (max/min): 7,8
            total = max(1e-6, prob_pass + prob_addsub + prob_nonlin)
            prob_pass, prob_addsub, prob_nonlin = prob_pass/total, prob_addsub/total, prob_nonlin/total

            pass_ids   = [3, 4, 5, 6]  # PASS_A, PASS_B, -A, -B
            addsub_ids = [0, 1, 2]     # ADD, A-B, B-A
            nonlin_ids = [7, 8]        # MAX, MIN

            import numpy as _np
            p = _np.zeros(self.num_functions, dtype=float)
            # Bias within passthrough group: favor PASS_A/PASS_B slightly more than -A/-B
            pass_w = {3: 0.32, 4: 0.32, 5: 0.18, 6: 0.18}
            sw = sum(pass_w.values())
            for i in pass_ids:   p[i] = prob_pass   * (pass_w[i] / sw)
            for i in addsub_ids: p[i] = prob_addsub / max(1, len(addsub_ids))
            for i in nonlin_ids: p[i] = prob_nonlin / max(1, len(nonlin_ids))
            p = p / p.sum()

            # Sample a dominant function k with weight p, then apply a mild positive bias
            k = int(_np.random.choice(self.num_functions, p=p))
            logits[k] += float(INIT_BOOST)

            # Cap top1-top2 margin to avoid early lock-in
            vals, idxs = torch.topk(logits, k=2)
            if (vals[0] - vals[1]) > float(INIT_MARGIN_CAP):
                logits[idxs[0]] = vals[1] + float(INIT_MARGIN_CAP)

            # Cap top1-top2 margin again (kept intentionally as original)
            vals, idxs = torch.topk(logits, k=2)
            if (vals[0] - vals[1]) > float(INIT_MARGIN_CAP):
                logits[idxs[0]] = vals[1] + float(INIT_MARGIN_CAP)

        self.function_logits = nn.Parameter(logits)
        self.register_buffer('temp', torch.tensor(float(temp)))

    def forward(self, x):
        ia = int(self.idx_a.item())
        ib = int(self.idx_b.item())
        a = x[:, ia:ia+1]
        b = x[:, ib:ib+1]

        if self.training:
            outs = torch.stack([op(self, a, b) for op in self._OPS], dim=1).squeeze(-1)
            w = torch.softmax(self.function_logits / self.temp, dim=0)
            y = (outs * w.unsqueeze(0)).sum(dim=1)
            return y.unsqueeze(1)
        else:
            with torch.no_grad():
                best_op_idx = torch.argmax(self.function_logits).item()
            y = self._OPS[best_op_idx](self, a, b)
            if y.dim() == 1:
                return y.unsqueeze(1)
            return y

class DiffLogicGateNetwork(nn.Module):
    """
    Multi-layer gate network. Each layer selects two inputs per gate and concatenates outputs.
    Input channels = previous layer's output channels (first layer uses original input dimension).
    """
    # Add master_rng to control internal randomness (without altering global RNG)
    def __init__(self, input_dim, layer_sizes, temp_init=TEMP_INIT, master_rng=None):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim

        # Create a local RNG if not provided (recommended to pass one from caller)
        if master_rng is None:
            master_rng = random.Random()

        for layer_idx, size in enumerate(layer_sizes):
            pairs = []
            remaining = list(range(in_dim))
            rng = master_rng

            while remaining and len(pairs) < size:
                ia = remaining.pop(0)
                cand = list(range(in_dim))
                cand.remove(ia)
                ib = int(rng.choice(cand))
                pairs.append((int(ia), int(ib)))

            while len(pairs) < size:
                # Use rng.sample for without-replacement sampling (Python's random semantics)
                ia, ib = rng.sample(range(in_dim), 2)
                pairs.append((int(ia), int(ib)))

            rng.shuffle(pairs)

            gates = nn.ModuleList([
                DiffLogicGate(in_dim, temp=temp_init, idx_a=ia, idx_b=ib)
                for ia, ib in pairs
            ])
            self.layers.append(gates)
            in_dim = size

    def forward(self, x):
        out = x
        for gates in self.layers:
            # Each g(out) returns [B,1]; concatenate to [B, num_gates]
            residual = torch.cat([g(out) for g in gates], dim=1)
            if residual.shape != out.shape:
                out = residual
            else:
                beta = 1.0 / (1.0 + math.sqrt(residual.size(1)))  # kept for parity
                out = residual
        return out

    def set_temp(self, new_temp: float):
        for layer in self.layers:
            for g in layer:
                g.temp.fill_(new_temp)

    def anneal_temp(self, gamma: float):
        for layer in self.layers:
            for g in layer:
                g.temp.mul_(gamma).clamp_(min=TEMP_MIN)


class PerceptionUpdateModule(nn.Module):
    """
    Parallel neighbor-aggregation with three subnetworks sharing the same gate logic.
    Initialization follows random-walk Laplacian (RW-Lap) behavior:
      - fusion: use dv (neighbor state)
      - attn  : carry du and apply per-source L1 normalization → sum(|w|) per source = 1
      - update: h = 0.5 * (1 + x - mean_N) realized via gates (kept as incremental update)
    """
    # Add master_rng and pass it to subnets
    def __init__(self, master_rng=None):
        super().__init__()
        if master_rng is None:
            master_rng = random.Random()

        self.fusion_net  = DiffLogicGateNetwork(2,  [6, 4, 3], temp_init=TEMP_INIT, master_rng=master_rng).to(DEVICE)
        self.attn_net    = DiffLogicGateNetwork(4,  [8, 8, 4, 2, 1], temp_init=TEMP_INIT, master_rng=master_rng).to(DEVICE)
        self.update_net  = DiffLogicGateNetwork(4,  [12, 12, 8, 8, 4, 2, 1], temp_init=TEMP_INIT, master_rng=master_rng).to(DEVICE)

    def _hard_select_op(self, gate: DiffLogicGate, op_id: int, ia: int, ib: int,
                        logit_pos: float = 8.0, logit_neg: float = -8.0):
        with torch.no_grad():
            gate.idx_a.fill_(int(ia))
            gate.idx_b.fill_(int(ib))
            logits = torch.full_like(gate.function_logits, float(logit_neg))
            logits[int(op_id)] = float(logit_pos)
            gate.function_logits.copy_(logits)

    def _uniformize_logits(self, gate: DiffLogicGate):
        """Set all function logits to zero (uniform softmax) to ease later learning."""
        with torch.no_grad():
            gate.function_logits.zero_()

    def _bias_passthrough(self, gate: DiffLogicGate, ia: int, ib: int = None):
        """
        Set input connections only; keep global initialization bias (passthrough > linear > others).
        """
        with torch.no_grad():
            gate.idx_a.fill_(int(ia))
            if ib is None:
                ib = 0 if ia != 0 else 1
            gate.idx_b.fill_(int(ib))
            # Do not modify function_logits here.

    def _init_random_walk_laplacian_start(self):
        """
        Strict RW-Lap start:
          - fusion : first layer hard-set to pass B (dv); later layers pass A@ch0 to replicate across channels
          - attn   : pass A@ch0 (du); per-source L1 normalization yields 1/deg(u) when uniform
          - update : first layer uses a-b (x - mean_N); later layers pass A@ch0
        """
        # fusion: dv then identity propagation
        for g in self.fusion_net.layers[0]:
            self._hard_select_op(g, op_id=4, ia=0, ib=1)   # pass B: dv (op_id=4)
        for li in range(1, len(self.fusion_net.layers)):
            for g in self.fusion_net.layers[li]:
                self._hard_select_op(g, op_id=3, ia=0, ib=0)  # pass A@ch0 (op_id=3)

        # attn: use du then per-source L1 normalization downstream
        for layer in self.attn_net.layers:
            for g in layer:
                self._hard_select_op(g, op_id=3, ia=0, ib=1)  # pass A: du (op_id=3)

        # update: first layer a-b (x - mean_N), later layers identity
        for g in self.update_net.layers[0]:
            self._hard_select_op(g, op_id=1, ia=0, ib=1)  # a-b (op_id=1)

        for li in range(1, len(self.update_net.layers)):
            for g in self.update_net.layers[li]:
                self._hard_select_op(g, op_id=3, ia=0, ib=1)  # pass A@ch0 (op_id=3)

    def forward(self, pre):
        ns = pre['node_states']
        N  = ns.size(0)
        edge_src  = pre['edge_src'].to(DEVICE)
        edge_dst  = pre['edge_dst'].to(DEVICE)
        edge_info = pre['edge_info'].to(DEVICE)

        if edge_src.numel() > 0:
            # Use raw states for fusion and attention
            du_raw = ns[edge_src].view(-1, 1)
            dv_raw = ns[edge_dst].view(-1, 1)
            das    = edge_info.view(-1, 2)

            fused = self.fusion_net(torch.cat([du_raw, dv_raw], dim=1))

            # Keep attention input unnormalized (match lmain.py behavior)
            attn_in   = torch.cat([du_raw, dv_raw, das], dim=1)
            weights_r = self.attn_net(attn_in).squeeze(1)  # [E]
            # Per-source signed normalization: sum(|w|) per source = 1
            w = weights_r
            sum_abs_per_src = torch.zeros(N, dtype=w.dtype, device=DEVICE).index_add_(0, edge_src, w.abs())
            sum_abs_per_src = sum_abs_per_src.clamp_min(1e-6)
            soft_w = w / sum_abs_per_src[edge_src]

            weighted = fused * soft_w.unsqueeze(1)
            agg = torch.zeros((N, fused.size(1)), device=DEVICE)
            agg.index_add_(0, edge_src, weighted)
        else:
            agg = torch.zeros((N, self.fusion_net.layers[-1].__len__()), device=DEVICE)

        x = torch.cat([ns, agg], dim=1)
        update_delta = self.update_net(x)
        # Incremental update: ns ← ns + Δ
        new_states = ns + update_delta
        return new_states

    def set_global_temp(self, new_temp: float):
        self.fusion_net.set_temp(new_temp)
        self.attn_net.set_temp(new_temp)
        self.update_net.set_temp(new_temp)

    def anneal_all(self, gamma: float = GATE_ANNEAL_GAMMA):
        if not GATE_ANNEAL_ENABLE:
            return
        self.fusion_net.anneal_temp(gamma)
        self.attn_net.anneal_temp(gamma)
        self.update_net.anneal_temp(gamma)

    def set_eval_hard(self, flag: bool):
        """
        Eval-only switch: when True, gates use one-hot top-1 path in forward().
        Training keeps soft mixture.
        """
        for net in (self.fusion_net, self.attn_net, self.update_net):
            for layer in net.layers:
                for g in layer:
                    setattr(g, 'force_hard_eval', bool(flag))

# ==================== 3. Similarity evaluation module ====================
class SimilarityModule(nn.Module):
    def __init__(self, bins=128):
        super().__init__()
        self.bins = bins
        self.eps = 1e-8
        self.num_iters = NUM_ITER

    def soft_hist(self, x_norm):
        """
        x_norm is scaled to [0, 1].
        Use fixed centers = linspace(0,1,bins) with a Gaussian soft-assignment (differentiable).
        """
        # Use float64 for numerical stability; cast back at the end
        x = x_norm.view(-1).to(torch.float64)
        centers = torch.linspace(0.0, 1.0, self.bins, device=x.device, dtype=x.dtype)
        # Fixed bandwidth to avoid cross-batch drift
        scale = torch.tensor(HIST_BW, device=x.device, dtype=x.dtype)
        diffs = x.unsqueeze(1) - centers.unsqueeze(0)
        weights = torch.exp(-0.5 * (diffs / scale) ** 2)
        hist = (weights.sum(dim=0) + self.eps).to(x.dtype)
        hist = hist / hist.sum()
        return hist.to(x_norm.dtype)

    def compute_similarity_matrix(self, all_states):
        """
        all_states: List[List[Tensor]] with T states per graph.
        Robustly clamp each state to (1e-6, 1-1e-6) via MAD-based scaling,
        build histograms with fixed bandwidth, compute JS divergence at aligned timesteps,
        then average across t and multiply by SIMS_GLOBAL_SCALE.
        """
        hists_per_graph = []
        for state_list in all_states:
            hist_ts = []
            for st in state_list:
                flat = st.view(-1).float()
                med  = flat.median()
                mad  = (flat - med).abs().median()
                # Robust linear scaling: map [med-3*mad, med+3*mad] → [0, 1], then clamp
                lo   = med - 3.0 * mad
                hi   = med + 3.0 * mad
                denom = (hi - lo).abs() + 1e-6
                x_norm = ((flat - lo) / denom).clamp(1e-6, 1.0 - 1e-6)
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
        sims = sims / T
        sims = sims * SIMS_GLOBAL_SCALE
        return sims

    def cluster_and_metrics(self, sims):
        """
        Agglomerative clustering (average linkage, precomputed distances) + automatic k selection.
        Evaluate k in [2, MAX_K] via silhouette (precomputed distances) and also report soft-silhouette.
        """
        if not torch.isfinite(sims).all():
            raise RuntimeError("SIMS_NONFINITE in Validation")
        sims_np = sims.detach().cpu().numpy()
        Kcur = sims_np.shape[0]

        # Preprocess distance matrix for silhouette (consistent with original behavior)
        sims_fill = sims_np.copy()
        np.fill_diagonal(sims_fill, 0.0)
        sims_fill[sims_fill < 0] = 0.0

        # One-time linkage tree + rank by Sil↑ and DB↓ on the same tree
        # 1) Build symmetric, non-negative D and condensed vector dvec
        D = sims_np.copy()
        np.fill_diagonal(D, 0.0)
        D[D < 0] = 0.0
        D = 0.5 * (D + D.T)
        dvec = squareform(D, checks=False)

        # 2) Single linkage call; cut for k=2..MAX_K
        Z = linkage(dvec, method='average')
        # Classical MDS for CH/DB features (reused across k)
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
        for k in range(3, max(4, k_max + 1)):
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

        # Fallback + rank-sum selection (Sil high, DB low)
        if not records:
            i, j = np.unravel_index(np.argmax(D), D.shape)
            labels = np.zeros(Kcur, dtype=int)
            labels[j] = 1
        else:
            ks, lbls, sils, _dbs = zip(*records)
            sils = np.asarray(sils, dtype=float)
            dbs  = np.asarray(_dbs, dtype=float)

            sil_rk = (-np.argsort(np.argsort(sils))).astype(float)   # larger Sil → smaller rank
            db_rk  = ( np.argsort(np.argsort(dbs ))).astype(float)   # smaller DB  → smaller rank

            rank_sum = sil_rk + db_rk
            best_idx = int(np.argmin(rank_sum))
            labels = lbls[best_idx]
        n_active = int(np.unique(labels).size)

        # Intra/Inter means (exclude diagonal; inter uses different-cluster pairs only)
        mask_same = (labels[:, None] == labels[None, :])
        mask_same &= ~np.eye(labels.size, dtype=bool)
        mask_diff = (labels[:, None] != labels[None, :])
        intra = float(sims_np[mask_same].mean()) if mask_same.any() else float('nan')
        inter = float(sims_np[mask_diff].mean()) if mask_diff.any() else float('nan')

        # Soft-silhouette (eval version): row 0.75-quantile scaling + global tau = off-diagonal median
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

        # Hard-label metrics (same definitions as original)
        sil_pre = silhouette_score(D, labels, metric='precomputed')
        try:
            ch = calinski_harabasz_score(X_emb, labels)
        except Exception:
            ch = -1.0
        try:
            db = davies_bouldin_score(X_emb, labels)
        except Exception:
            db = np.inf

        return {
            'Cluster_Count': int(n_active),
            'Intra_Dist': float(intra),
            'Inter_Dist': float(inter),
            'Silhouette': float(sil_pre),
            'Soft_Silhouette': float(soft_sil_val),
            'Calinski_Harabasz': float(ch),
            'Davies_Bouldin': float(db)
        }

# ==================== 4. Main training/control module ====================
class Trainer:
    def __init__(self, args):
        # ---- constants ----
        self.args = args
        self.EPS      = 1e-8
        self.EPS_DIV  = 1e-6

        # ---- modules ----
        self.pre = PrecomputeModule()
        
        # Create an independent RNG from the main seed to build the model (reproducible wiring per seed)
        self.seed = getattr(self.args, 'seed', None)
        master_rng = random.Random(self.seed)
        self.per = PerceptionUpdateModule(master_rng=master_rng).to(DEVICE)
        
        self.sim = SimilarityModule(bins=128).to(DEVICE)
        # Snapshot of the very first initialization (structure/indices/temperature/weights)
        import copy
        self._first_init_state = copy.deepcopy(self.per.state_dict())
        # Optionally record RNG state for exact reproduction (without touching global RNG)
        try:
            self._rng_state0 = {
                'py':   random.getstate(),
                'np':   np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        except Exception:
            self._rng_state0 = None

        # ---- logging/saving ----
        self.scale_best = {}
        self.seed = getattr(self.args, 'seed', None)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # If multi-seed mode, an outer caller passes run_root; otherwise fall back to SAVE_ROOT/timestamp
        run_root = getattr(self.args, 'run_root', None)
        if run_root:
            self.save_dir = os.path.join(run_root, f'seed_{self.seed}')
        else:
            self.save_dir = os.path.join(SAVE_ROOT, ts)

        os.makedirs(self.save_dir, exist_ok=True)
        cfg_to_save = dict(vars(args))
        cfg_to_save['seed'] = self.seed
        torch.save(cfg_to_save, os.path.join(self.save_dir, 'config.pt'))

        # ---- global best snapshot ----
        # Keep *best_global* under save_dir always pointing to a global-best candidate
        self.global_metric_name = 'Silhouette'   # change here if Soft_Silhouette is preferred
        self.global_best = {
            'metric': -float('inf'),
            'ckpt':   None   # {'model': state_dict, 'opt': state_dict, ...}
        }

        # Per-epoch rows (for final CSV writing)
        self.epoch_rows = []
        # ---- optimizer (stable finetuning): param groups + sane LR cap + mild weight decay ----
        gate_logits = []
        for module in self.per.modules():
            if hasattr(module, "function_logits"):
                gate_logits.append(module.function_logits)
        gate_ids = {id(p) for p in gate_logits}
        other_params = [p for p in self.per.parameters() if id(p) not in gate_ids]

        # Slightly higher base LR cap and relatively larger LR for non-gate params (still conservative)
        base_lr = min(float(getattr(self.args, 'lr', 1.5e-2)), 1.5e-2)
        self.current_optimizer = torch.optim.AdamW(
            [
                {'params': gate_logits,  'lr': base_lr},          # gate logits
                {'params': other_params, 'lr': base_lr * 0.75},   # other parameters
            ],
            betas=(0.9, 0.97),
            eps=self.EPS,
            weight_decay=1e-4
        )

        # ---- phase-aware cosine scheduler for finetune stage ----
        # Only effective when self.phase == 'finetune'; random phase untouched (e.g., LR may be set to zero)
        class _PhaseAwareCosineScheduler:
            def __init__(self, opt, owner, min_factor=0.85, max_factor=1.15,
                         warmup=3, period=EPOCHS_PER_GROUP):
                self.opt = opt
                self.owner = owner
                self.min_factor = float(min_factor)
                self.max_factor = float(max_factor)
                self.warmup = int(max(0, warmup))
                self.period = max(1, int(period))
                self.base_lrs = [g['lr'] for g in opt.param_groups]
                self.t = 0

            def step(self):
                # Adjust LR only in finetune phase; keep zeroed LRs untouched
                if getattr(self.owner, 'phase', 'random') != 'finetune':
                    self.t = 0
                    return
                self.t += 1
                if self.warmup > 0 and self.t <= self.warmup:
                    fac = self.min_factor + (1.0 - self.min_factor) * (self.t / float(self.warmup))
                else:
                    import math
                    x = min(1.0, self.t / float(self.period))
                    fac = self.min_factor + (self.max_factor - self.min_factor) * (0.5 - 0.5 * math.cos(math.pi * x))
                for g, base in zip(self.opt.param_groups, self.base_lrs):
                    if g.get('lr', 0.0) <= 0.0:
                        continue
                    g['lr'] = base * fac

        self.scheduler = _PhaseAwareCosineScheduler(self.current_optimizer, self,
                                                    min_factor=0.85, max_factor=1.15,
                                                    warmup=3, period=EPOCHS_PER_GROUP)


        # thresholds & phase parameters (centralized from hyperparameter section; CLI may override)
        self.phase = 'random'  # start in random/search phase

        # switching thresholds to enter finetune (args override if provided)
        self.random_switch_sil_hard = float(getattr(self.args, 'random_switch_hardsil', RANDOM_SWITCH_HARDSIL))
        self.random_switch_sil_soft = float(getattr(self.args, 'random_switch_softsil', RANDOM_SWITCH_SOFTSIL))

        # finetune targets (one of the early-stop criteria)
        self.finetune_target_hard   = float(getattr(self.args, 'finetune_target_hard', FINETUNE_TARGET_HARD))
        self.finetune_target_soft   = float(getattr(self.args, 'finetune_target_soft', FINETUNE_TARGET_SOFT))

        # group consistency (GCI) thresholds
        self.switch_min_gci         = float(getattr(self.args, 'switch_min_gci', SWITCH_MIN_GCI))
        self.stop_min_gci           = float(getattr(self.args, 'stop_min_gci',   STOP_MIN_GCI))

        # patience for falling back from finetune (consecutive underperformance, not degradation)
        self.finetune_patience      = int(getattr(self.args, 'finetune_patience', FINETUNE_PATIENCE))

        # noise strength in random phase
        self.random_sigma           = float(getattr(self.args, 'random_sigma', RANDOM_SIGMA))

        self._finetune_epochs  = 0

        print(
            "[Trainer] Two-phase mode: RANDOM(param perturb) → FINETUNE(slow lr). "
            f"switch@(hard_sil>{self.random_switch_sil_hard} & soft_sil>{self.random_switch_sil_soft} & GCI>={self.switch_min_gci}), "
            f"target@(hard_sil>{self.finetune_target_hard} & soft_sil>{self.finetune_target_soft} & GCI>={self.stop_min_gci}), "
            f"patience={self.finetune_patience}, random_sigma={self.random_sigma}."
        )

        # ---- gradient statistics (probe every epoch) ----
        self.grad_ema = None
        self.grad_ema_beta = 0.05
        self.grad_scale_factor = 3.0
        self.big_grad_thresh = 100.0  # optional: threshold for large-gradient warnings

        # ---- annealing control (adaptive) ----
        self.anneal_started = False
        self.anneal_start_epoch = ANNEAL_START_EPOCH
        self.anneal_metric_thresh = ANNEAL_SOFTSIL_THRESH
        self.prev_soft_sil = None            # previous epoch Soft-Sil (for deltas)
        self.anneal_exp = 1.0                # exponent on base_gamma: >1 faster, <1 slower
        # plateau detection + "micro-thaw" cooldown to avoid frequent toggling
        self.plateau_count = 0               # count consecutive tiny deltas
        self.thaw_cooldown = 0               # when >0, skip micro-thaw this epoch

        # ---- global epoch counter + stop thresholds ----
        self.total_epochs_run = 0
        self.stop_min_epochs = int(getattr(self.args, 'min_epochs', STOP_MIN_EPOCHS))
        # backward compatibility: if args.stop_sil exists but is None, keep defaults; derive 4-condition thresholds
        _legacy_stop_sil = getattr(self.args, 'stop_sil', None)
        self.stop_softsil     = float(getattr(self.args, 'stop_softsil', STOP_SOFTSIL) if _legacy_stop_sil is None else STOP_SOFTSIL)
        self.stop_hardsil     = float(_legacy_stop_sil if _legacy_stop_sil is not None else getattr(self.args, 'stop_hardsil', STOP_HARDSIL))
        self.stop_min_clusters = int(getattr(self.args, 'min_clusters', STOP_MIN_CLUSTERS))
        # early stop is independent of temperature/one-hot
        self._seed_should_stop = False
        self.last_epoch_metrics = None

        self._last_rehearsal_epoch = 0
        self.finetune_monitor = {'best_sil': -1.0, 'last_check_epoch': 0, 'degradation_threshold': 0.05}

        # === diagnostics: ensure all gate function_logits are in the optimizer param list ===
        gate_params = []
        for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
            for layer in net.layers:
                for g in layer:
                    gate_params.append(g.function_logits)

        opt_param_ids = {id(p) for group in self.current_optimizer.param_groups for p in group['params']}
        missed = [i for i, p in enumerate(gate_params) if (id(p) not in opt_param_ids) or (p.requires_grad is False)]
        if missed:
            print(f"[WARN] {len(missed)} gate logits NOT in optimizer or requires_grad=False. Indices={missed[:20]}...")

    def _log_gradient_stats(self, model, epoch):
        """Gradient probe: report average gradient norms per layer of each gate network."""
        print("\n" + "="*20 + f" [GRADIENT PROBE @ Epoch {epoch}] " + "="*20)
        all_grads = []
        for name, net in [('fusion_net', model.fusion_net), 
                          ('attn_net', model.attn_net), 
                          ('update_net', model.update_net)]:
            layer_norms = []
            for i, layer in enumerate(net.layers):
                grads = [g.function_logits.grad.norm().item() 
                         for g in layer 
                         if g.function_logits.grad is not None]
                if grads:
                    avg_norm = sum(grads) / len(grads)
                    layer_norms.append(f"L{i}={avg_norm:.2e}")
                    all_grads.extend(grads)
            print(f"- {name:10s} : {', '.join(layer_norms)}")

        if all_grads:
            total_avg = sum(all_grads) / len(all_grads)
            print(f"--- Total Avg Grad Norm: {total_avg:.2e} ---\n")
        else:
            print("--- No gradients found. ---\n")

    def _restore_first_init(self):
        """Restore self.per to the first-initialization snapshot (isomorphic and parameter-equal)."""
        import copy
        if getattr(self, "_first_init_state", None) is None:
            self._first_init_state = copy.deepcopy(self.per.state_dict())
        self.per.load_state_dict(self._first_init_state, strict=True)

    def _perturb_from_first_init(self, epoch_idx: int, sigma: float):
        """
        Add reproducible Gaussian noise on top of the first-initialization
        (perturb function_logits only; keep indices/structure/temp).
        - Independence across rounds via a derived sub-seed from (BASE_SEED, current seed, epoch_idx)
        - Do not pollute global RNG: save/restore RNG states
        """
        # 1) save global RNG
        try:
            py_st  = random.getstate()
            np_st  = np.random.get_state()
            tc_st  = torch.get_rng_state()
            tcu_st = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        except Exception:
            py_st = np_st = tc_st = tcu_st = None

        # 2) derived deterministic sub-seed
        sid  = int(getattr(self, "seed", 0) or 0)
        base = int(BASE_SEED)
        ep   = int(epoch_idx)
        sub_seed = (base * 1000003 + sid * 7919 + ep * 271) & 0x7fffffff
        random.seed(sub_seed)
        np.random.seed(sub_seed % (2**32 - 1))
        torch.manual_seed(sub_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sub_seed)

        # 3) restore first init then perturb (optionally resample dominant gate class in random phase)
        self._restore_first_init()
        resample = bool(getattr(self.args, 'random_resample_gate_bias', True))
        for mod in self.per.modules():
            if hasattr(mod, "function_logits") and isinstance(mod.function_logits, torch.nn.Parameter):
                with torch.no_grad():
                    if resample and hasattr(mod, 'num_functions'):
                        # class probabilities (globally overridable by GATE_PROB_*), consistent with init
                        prob_pass   = float(globals().get('GATE_PROB_PASS',   0.56))
                        prob_addsub = float(globals().get('GATE_PROB_ADDSUB', 0.42))
                        prob_nonlin = float(globals().get('GATE_PROB_NONLIN', 0.02))
                        total = max(1e-6, prob_pass + prob_addsub + prob_nonlin)
                        prob_pass, prob_addsub, prob_nonlin = prob_pass/total, prob_addsub/total, prob_nonlin/total

                        pass_ids   = [3, 4, 5, 6]
                        addsub_ids = [0, 1, 2]
                        nonlin_ids = [7, 8]

                        import numpy as _np
                        p = _np.zeros(int(mod.num_functions), dtype=float)
                        pass_w = {3: 0.32, 4: 0.32, 5: 0.18, 6: 0.18}
                        sw = sum(pass_w.values())
                        for i in pass_ids:   p[i] = prob_pass   * (pass_w[i] / sw)
                        for i in addsub_ids: p[i] = prob_addsub / max(1, len(addsub_ids))
                        for i in nonlin_ids: p[i] = prob_nonlin / max(1, len(nonlin_ids))
                        p = p / p.sum()

                        k = int(_np.random.choice(int(mod.num_functions), p=p))
                        # reset logits with mild positive bias for the sampled dominant op
                        logits_new = torch.randn_like(mod.function_logits) * float(INIT_SPREAD)
                        logits_new[k] += float(INIT_BOOST)

                        vals, idxs = torch.topk(logits_new, k=2)
                        if (vals[0] - vals[1]) > float(INIT_MARGIN_CAP):
                            logits_new[idxs[0]] = vals[1] + float(INIT_MARGIN_CAP)
                        mod.function_logits.copy_(logits_new)

                        vals, idxs = torch.topk(logits_new, k=2)
                        if (vals[0] - vals[1]) > float(INIT_MARGIN_CAP):
                            logits_new[idxs[0]] = vals[1] + float(INIT_MARGIN_CAP)
                        mod.function_logits.copy_(logits_new)

                    # add small Gaussian noise to maintain exploration
                    noise = torch.randn_like(mod.function_logits) * float(sigma)
                    mod.function_logits.add_(noise)

        # 4) restore global RNG
        try:
            if py_st  is not None: random.setstate(py_st)
            if np_st  is not None: np.random.set_state(np_st)
            if tc_st  is not None: torch.set_rng_state(tc_st)
            if tcu_st is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(tcu_st)
        except Exception:
            pass

    def _freeze_optimizer_lr_once(self):
        """Temporarily set optimizer LR to 0 for the current round (random phase only); return a backup."""
        opt = self.current_optimizer
        backup = [g['lr'] for g in opt.param_groups]
        for g in opt.param_groups:
            g['lr'] = 0.0
        return backup

    def _restore_optimizer_lr(self, backup):
        """Restore LRs that were zeroed by _freeze_optimizer_lr_once."""
        if backup is None:
            return
        opt = self.current_optimizer
        for g, lr in zip(opt.param_groups, backup):
            g['lr'] = lr

    def _save_epoch_checkpoint(self, epoch, current_metrics):
        """Helper: safely save a hardened checkpoint for this epoch without mutating the main model."""
        try:
            # Create a hardened deep copy internally; do not touch self.per
            import copy
            per_hard = copy.deepcopy(self.per)

            def _hardify_module(mod):
                for layer in getattr(mod, 'layers', []):
                    for g in layer:
                        with torch.no_grad():
                            soft = torch.softmax(g.function_logits / max(float(g.temp), 1e-6), dim=0)
                            k = int(torch.argmax(soft).item())
                            hard_logits = torch.full_like(g.function_logits, -1e4)
                            hard_logits[k] = 1e4
                            g.function_logits.copy_(hard_logits)
                            g.temp.copy_(torch.tensor(TEMP_MIN))
            
            for sub in (per_hard.fusion_net, per_hard.attn_net, per_hard.update_net):
                _hardify_module(sub)

            # collect layer_cfg
            def _infer_layer_cfg(mod):
                return [len(layer) for layer in mod.layers]
            layer_cfg = {
                "fusion": _infer_layer_cfg(self.per.fusion_net),
                "attn":   _infer_layer_cfg(self.per.attn_net),
                "update": _infer_layer_cfg(self.per.update_net),
            }
            
            ckpt_metrics = dict(current_metrics or {})
            _scale = getattr(self, "cur_scale", "")
            _group = int(getattr(self, "cur_group_idx", -1))

            ckpt = _pack_gnca_ckpt(per_hard, epoch=epoch, seed=int(self.seed),
                                   scale=_scale, group=_group, layer_cfg=layer_cfg, metrics=ckpt_metrics)

            ckpt_name = f"ckpt_epoch{int(epoch):04d}.pt"
            ckpt_path = os.path.join(self.save_dir, ckpt_name)
            _atomic_save(ckpt, ckpt_path)
            print(f"[Checkpoint] Saved unified HARD checkpoint: {ckpt_path}")

        except Exception as _ex:
            print(f"[Checkpoint][WARN] Save failed: {_ex}")

    def _log_gradient_stats(self, model, epoch):
        """Gradient probe: report average gradient norms per layer of each gate network."""
        print("\n" + "="*20 + f" [GRADIENT PROBE @ Epoch {epoch}] " + "="*20)
        all_grads = []
        for name, net in [('fusion_net', model.fusion_net), 
                          ('attn_net', model.attn_net), 
                          ('update_net', model.update_net)]:
            layer_norms = []
            for i, layer in enumerate(net.layers):
                grads = [g.function_logits.grad.norm().item() 
                         for g in layer 
                         if g.function_logits.grad is not None]
                if grads:
                    avg_norm = sum(grads) / len(grads)
                    layer_norms.append(f"L{i}={avg_norm:.2e}")
                    all_grads.extend(grads)
            print(f"- {name:10s} : {', '.join(layer_norms)}")

        if all_grads:
            total_avg = sum(all_grads) / len(all_grads)
            print(f"--- Total Avg Grad Norm: {total_avg:.2e} ---\n")
        else:
            print("--- No gradients found. ---\n")


    def run_epoch(self, graphs, mode):
        """Stream graphs and compute similarities (for validation/evaluation only)."""
        states_list = []
        try:
            # Always eval with a hardened deep copy during validation
            import copy
            per_eval = copy.deepcopy(self.per)
            per_eval.eval()
            forward_model = per_eval

            for i, g in enumerate(graphs):
                print(f"\rProcessing {mode} graph {i+1}/{len(graphs)}", end="")
                
                if isinstance(g, dict):
                    pre_cpu = g
                else:
                    pre_cpu = self.pre.process_graph(g)

                _ns0 = pre_cpu['node_states'].to(DEVICE, non_blocking=True)
                if _ns0.dim() == 1:
                    _ns0 = _ns0.view(-1, 1)
                _N = _ns0.size(0)
                _e_src = pre_cpu['edge_src'].to(DEVICE, non_blocking=True)
                _ones = torch.ones(_e_src.numel(), dtype=_ns0.dtype, device=_ns0.device)
                _deg = torch.zeros((_N, 1), dtype=_ns0.dtype, device=_ns0.device)
                if _e_src.numel() > 0:
                    _deg.index_add_(0, _e_src, _ones.unsqueeze(1))

                pre_gpu = {
                    'node_states': _deg,
                    'neighbors':   pre_cpu['neighbors'],
                    'space_info':  [t.to(DEVICE, non_blocking=True) for t in pre_cpu['space_info']],
                    'edge_src':    _e_src,
                    'edge_dst':    pre_cpu['edge_dst'].to(DEVICE, non_blocking=True),
                    'edge_info':   pre_cpu['edge_info'].to(DEVICE, non_blocking=True)
                }

                states = []
                with torch.no_grad():
                    current_state = pre_gpu['node_states']
                    for _ in range(NUM_ITER):
                        iter_input = {
                            'node_states': current_state,
                            'neighbors':   pre_cpu['neighbors'],
                            'space_info':  pre_cpu['space_info'],
                            'edge_src':    pre_gpu['edge_src'],
                            'edge_dst':    pre_gpu['edge_dst'],
                            'edge_info':   pre_gpu['edge_info'],
                        }
                        current_state = forward_model(iter_input)
                        states.append(current_state.detach())
                states_list.append(states)

                del pre_gpu
                if not isinstance(g, dict):
                    self.pre.release(g, delete_file=DELETE_TMP_AFTER_USE)
        finally:
            pass
        
        print()
        sims = self.sim.compute_similarity_matrix(states_list)
        return self.sim.cluster_and_metrics(sims)

    def train_group(self, train_g, val_group_lists, scale,
                    epochs=EPOCHS_PER_GROUP, early_stop=False, do_anneal=True,
                    group_idx=0, is_search_phase=False):        # allow cross-group continuity without forced resets
        # Reuse the externally created optimizer; do not rebuild here
        opt = self.current_optimizer
        history = []

        best, no_imp = -float('inf'), 0
        # local early-stop trackers (best value and no-improvement count)
        best_val_softsil = -float('inf')
        no_improve_cnt = 0

        # group-scoped logging: print once when thresholds are met; do not break group loop
        early_stop_logged = False

        lr_fixed = opt.param_groups[0]['lr']        # LR at group entry


        # ---- rolling safety snapshots for rollback ---- 
        from collections import deque
        snapshots = deque(maxlen=3)
        def take_snapshot():
            import copy
            return {
                'model': copy.deepcopy(self.per.state_dict()),
                'opt':   copy.deepcopy(opt.state_dict()),
                'anneal_started': bool(self.anneal_started),
                'anneal_exp': float(self.anneal_exp),
                'prev_soft_sil': None if self.prev_soft_sil is None else float(self.prev_soft_sil)
            }
        snapshots.append(take_snapshot())
        collapse_retries = 0

        for e in range(1, epochs + 1):
            
            # Respect caller's explicit phase selection; avoid cross-group phase leakage
            is_random_epoch_this_turn = bool(is_search_phase)

            if is_random_epoch_this_turn:
                # Randomization routine:
                # 1) restore first-initialization structure and weights
                self._restore_first_init()
                
                # 2) create a sub-seed
                _sub = random.randint(0, 2**31 - 1)
                
                # 3) temporary RNGs controlled by the sub-seed
                temp_rng = random.Random(_sub)
                g_cpu = torch.Generator().manual_seed(_sub)

                # 4) reinitialize weights under the temporary RNGs (no global RNG pollution)
                with torch.no_grad():
                    for _net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
                        for _layer in _net.layers:
                            for g in _layer:
                                _logits = torch.randn(g.function_logits.shape, generator=g_cpu, device=g.function_logits.device) * float(INIT_SPREAD)
                                if GATE_SKEW_INIT:
                                    _k = temp_rng.randint(0, int(g.num_functions) - 1)
                                    _logits[_k] += float(INIT_BOOST)
                                    _vals, _idxs = torch.topk(_logits, k=2)
                                    if (_vals[0] - _vals[1]) > float(INIT_MARGIN_CAP):
                                        _logits[_idxs[0]] = _vals[1] + float(INIT_MARGIN_CAP)
                                g.function_logits.copy_(_logits)
                                g.temp.fill_(float(TEMP_INIT))

            # save a hardened checkpoint at the start of each epoch
            self._save_epoch_checkpoint(self.total_epochs_run, self.last_epoch_metrics)

            # ---- periodic/high-water memory guard ----
            try:
                need_hard_clean = (e % 40 == 0)   # tune per machine if needed
                try:
                    import psutil
                    proc = psutil.Process(os.getpid())
                    rss_gb = proc.memory_info().rss / (1024**3)
                    if (rss_gb > 20.0) or (psutil.virtual_memory().percent > 85):
                        need_hard_clean = True
                        print(f"[WARN] High RSS {rss_gb:.2f} GB or mem>85% — force cleanup.")
                except Exception:
                    pass
                if need_hard_clean:
                    try:
                        PrecomputeModule.clear_cache()
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    try:
                        import gc as _gc
                        _gc.collect()
                    except Exception:
                        pass
            except Exception:
                pass

            # =========== Training phase: deferred computation pattern ===========
            # Initialize accumulators outside the loop to avoid NameError
            batch_losses = []
            batch_soft_sil_values = []
            # Step 1: forward all graphs in the group and collect states
            all_states_for_sim = []
            for i, g_path in enumerate(train_g):
                pre_cpu = self.pre.process_graph(g_path)
                _ns0 = pre_cpu['node_states'].to(DEVICE)
                _N = _ns0.size(0)
                _e_src = pre_cpu['edge_src'].to(DEVICE)
                _ones = torch.ones(_e_src.numel(), dtype=_ns0.dtype, device=_ns0.device)
                _deg = torch.zeros((_N, 1), dtype=_ns0.dtype, device=_ns0.device)
                if _e_src.numel() > 0:
                    _deg.index_add_(0, _e_src, _ones.unsqueeze(1))
                pre_gpu = {
                    'node_states': _deg.requires_grad_(True),
                    'edge_src': _e_src,
                    'edge_dst': pre_cpu['edge_dst'].to(DEVICE),
                    'edge_info': pre_cpu['edge_info'].to(DEVICE)
                }
                self.pre.release(g_path, delete_file=DELETE_TMP_AFTER_USE)
                states_one_graph = []
                current_state = pre_gpu['node_states']
                for _ in range(NUM_ITER):
                    iter_input = {
                        'node_states': current_state,
                        'edge_src': pre_gpu['edge_src'],
                        'edge_dst': pre_gpu['edge_dst'],
                        'edge_info': pre_gpu['edge_info'],
                    }
                    current_state = self.per(iter_input)
                    states_one_graph.append(current_state)
                all_states_for_sim.append(states_one_graph)

            # Step 2: compute total loss once using all collected states
            opt.zero_grad(set_to_none=True)
            sims = self.sim.compute_similarity_matrix(all_states_for_sim)
            # If non-finite appears during training, rollback to last safe snapshot and skip this group
            if not torch.isfinite(sims).all():
                print("[WARN] Non-finite similarity in training — rollback to last safe snapshot and skip this group.")
                snap = snapshots[-1]
                self.per.load_state_dict(snap['model'])
                opt.load_state_dict(snap['opt'])
                self.anneal_started = snap['anneal_started']
                self.anneal_exp     = snap['anneal_exp']
                self.prev_soft_sil  = snap['prev_soft_sil']
                torch.cuda.empty_cache()
                return None

            # ======== Validate first, then form the full loss ========
            # A) run validation to get metrics (e.g., hard silhouette)
            metrics_each = []
            for gi, vg in enumerate(val_group_lists, 1):
                m = self.run_epoch(vg, f'Val-G{gi}')
                metrics_each.append(m)
            val_metrics = {
                k: float(np.mean([m[k] for m in metrics_each]))
                for k in metrics_each[0]
            }

            # B) compute soft silhouette and auxiliaries on training sims
            eye = torch.eye(sims.size(0), device=sims.device, dtype=torch.bool)
            d = sims.clone()
            BIG = 1e6
            d_for_sort = d + eye.to(d.dtype) * BIG
            vals, _ = torch.sort(d_for_sort, dim=1)
            n_off = max(1, d.size(0) - 1)
            q_idx = max(0, min(n_off - 1, int(round(0.75 * (n_off - 1)))))
            row_scale = vals[:, q_idx:q_idx+1].clamp_min(1e-6)
            d_scaled = (d / row_scale).masked_fill(eye, 0.0)
            off_vals = d[~eye]
            tau = off_vals.median().clamp_min(1e-6) if off_vals.numel() > 0 else torch.tensor(SOFTSIL_TAU, device=d.device, dtype=d.dtype)
            NEG_BIG = 1e9
            logits_intra = (-d_scaled / tau).masked_fill(eye, -NEG_BIG)
            logits_inter = (d_scaled / tau).masked_fill(eye, -NEG_BIG)
            p_intra = torch.softmax(logits_intra, dim=1)
            a_i = (p_intra * d_scaled).sum(dim=1) / (p_intra.sum(dim=1) + self.EPS_DIV)
            p_inter = torch.softmax(logits_inter, dim=1)
            b_i = (p_inter * d_scaled).sum(dim=1) / (p_inter.sum(dim=1) + self.EPS_DIV)
            sil_i = (b_i - a_i) / (b_i + a_i + self.EPS_DIV)
            soft_sil = sil_i.mean()
            sil_loss = (1.0 - soft_sil)
            margin_val = b_i.mean() - a_i.mean()
            margin_loss = F.softplus(0.10 - margin_val)
            size_p = (p_intra.sum(dim=0) / (p_intra.sum() + self.EPS))
            ent_cur = -(size_p * torch.log(size_p + self.EPS)).sum()
            ent_floor = 0.50 * math.log(MAX_K)
            ent_loss = F.relu(ent_floor - ent_cur)

            # C) reward term uses validation hard-sil (detached)
            hard_sil_reward = torch.tensor(val_metrics['Silhouette'] + 1.0, device=DEVICE)
            log_probs = []
            for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
                for layer in net.layers:
                    for g in layer:
                        probs = torch.softmax(g.function_logits / g.temp, dim=0)
                        log_p = torch.log(probs + self.EPS).mean()
                        log_probs.append(log_p)
            total_log_prob = torch.stack(log_probs).mean()
            # Encourage sharper distributions (smaller log_prob → larger -log_prob)
            reward_loss = -total_log_prob * hard_sil_reward.detach()

            # D) final loss
            loss = 3.5 * sil_loss + 0.1 * margin_loss + 0.05 * ent_loss + 1.0 * reward_loss

            # ================= Backprop & update =================
            loss.backward()
            from torch.nn.utils import clip_grad_norm_
            total_norm = float(clip_grad_norm_(self.per.parameters(), max_norm=1.0))
            # Layer-wise grad scaling (front layers > back layers)
            if not is_random_epoch_this_turn:
                print("  Applying layered gradient scaling to stabilize critical gates.")
                for net_name, net in [('fusion', self.per.fusion_net), ('attn', self.per.attn_net), ('update', self.per.update_net)]:
                    num_layers = len(net.layers)
                    for i, layer in enumerate(net.layers):
                        scale_factor = 1.5 - (i / max(1, num_layers - 1))
                        for g in layer:
                            if g.function_logits.grad is not None:
                                g.function_logits.grad.mul_(scale_factor)
            if ENABLE_GRAD_PROBE:
                self._log_gradient_stats(self.per, e)

            # Update params only in finetune phase
            if not is_random_epoch_this_turn:
                opt.step()
            else:
                print("  Skipping optimizer step for random exploration epoch.")
            
            batch_losses.append(loss.item())
            batch_soft_sil_values.append(soft_sil.detach().item())

            if hasattr(self, 'scheduler') and (self.scheduler is not None):
                self.scheduler.step()

            avg_loss = sum(batch_losses) / max(1, len(batch_losses))
            train_soft_sil = float(sum(batch_soft_sil_values) / max(1, len(batch_soft_sil_values)))

            # LR scheduling & state
            self.per.train()
            best = max(best, val_metrics['Silhouette'])

            # ===== consistency & phase/stop logic =====
            try:
                sil_each  = [float(m.get('Silhouette', -1.0))      for m in metrics_each]
                ssoft_each= [float(m.get('Soft_Silhouette', -1.0)) for m in metrics_each]
            except Exception:
                sil_each, ssoft_each = [], []

            def _gci(xs):
                if not xs:
                    return 0.0
                rng = max(xs) - min(xs)
                return max(0.0, 1.0 - (rng / 2.0))

            gci_hard = _gci(sil_each)
            gci_soft = _gci(ssoft_each)
            gci = min(gci_hard, gci_soft)
            val_metrics['Group_Consistency'] = float(gci)

            switch_min_gci = float(getattr(self, 'switch_min_gci', 0.80))
            stop_min_gci   = float(getattr(self, 'stop_min_gci',   0.80))

            try:
                _clus_each = [float(m.get('Cluster_Count', 0.0)) for m in (metrics_each or [])]
                _avg_clusters = sum(_clus_each) / max(1, len(_clus_each))
            except Exception:
                _avg_clusters = float(val_metrics.get('Cluster_Count', 0.0))
            val_metrics['Avg_Cluster_Count'] = float(_avg_clusters)

            if self.phase == 'random':
                cond_switch = (
                    (val_metrics['Silhouette'] >= self.random_switch_sil_hard) and
                    (train_soft_sil            >= self.random_switch_sil_soft) and
                    (gci                        >= switch_min_gci)              and
                    (_avg_clusters              >= float(self.stop_min_clusters))
                )
                if cond_switch:
                    print(
                        f"\n[Phase Switch] Conditions met → RANDOM → FINETUNE | "
                        f"Val Hard-Sil {val_metrics['Silhouette']:.4f} ≥ {self.random_switch_sil_hard}, "
                        f"Train Soft-Sil {train_soft_sil:.4f} ≥ {self.random_switch_sil_soft}, "
                        f"GCI {gci:.3f} ≥ {switch_min_gci:.2f}, "
                        f"Avg-k {_avg_clusters:.2f} ≥ {self.stop_min_clusters}"
                    )
                    self._finetune_epochs = 0
                    self.phase = 'finetune'

            elif self.phase == 'finetune':
                cur_sil = val_metrics.get('Silhouette', -1.0)
                if cur_sil >= self.random_switch_sil_hard:
                    self._finetune_epochs = 0
                else:
                    self._finetune_epochs += 1
                if self._finetune_epochs >= self.finetune_patience:
                    print(f"\n[Phase Switch] Finetune patience reached ({self.finetune_patience}). Reverting to RANDOM.")
                    self.phase = 'random'

            cond_hardsil   = (val_metrics.get('Silhouette', -1.0) >= self.finetune_target_hard)
            cond_softsil   = (train_soft_sil                      >= self.finetune_target_soft)
            cond_gci       = (gci                                 >= stop_min_gci)
            cond_clusters  = (_avg_clusters                        >= float(self.stop_min_clusters))
            if cond_hardsil and cond_softsil and cond_gci and cond_clusters:
                print(
                    f"[Early-Stop] Hit at global_epoch={self.total_epochs_run}: "
                    f"Val Hard-Sil={val_metrics.get('Silhouette', -1.0):.4f} (≥{self.finetune_target_hard}), "
                    f"Train Soft-Sil={train_soft_sil:.4f} (≥{self.finetune_target_soft}), "
                    f"GCI={gci:.3f} (≥{stop_min_gci:.2f}), "
                    f"Avg-k={val_metrics['Avg_Cluster_Count']:.2f} (≥{self.stop_min_clusters})."
                )
                self._seed_should_stop = True

            # group-best snapshot
            if e == 1:
                self._group_best_metric = -float('inf')
                self._group_best_ckpt   = None
            cur_group_metric = float(val_metrics['Silhouette'])
            if cur_group_metric > self._group_best_metric + 1e-12:
                self._group_best_metric = cur_group_metric
                self._group_best_ckpt   = take_snapshot()

            # global-best snapshot
            gname = getattr(self, 'global_metric_name', 'Silhouette')
            cur_global_metric = float(val_metrics.get(gname, -float('inf')))
            if not hasattr(self, 'global_best'):
                self.global_best = {'metric': -float('inf'), 'ckpt': None}
            if cur_global_metric > float(self.global_best['metric']) + 1e-12:
                self.global_best['metric'] = cur_global_metric
                self.global_best['ckpt']   = take_snapshot()
                torch.save(self.per.state_dict(), os.path.join(self.save_dir, 'model_best_global.pt'))
                torch.save(opt.state_dict(),   os.path.join(self.save_dir, 'opt_best_global.pt'))

            # =========== logging & persistence ===========
            history.append({
                'stage': scale,
                'group': train_g,
                'epoch': e,
                'train_loss': float(avg_loss),
                'val': {k: float(v) for k, v in val_metrics.items()},
                'rehearsal': False
            })
            import copy
            _per_hard = copy.deepcopy(self.per)
            def _hardify_module(mod):
                for layer in getattr(mod, 'layers', []):
                    for g in layer:
                        with torch.no_grad():
                            soft = torch.softmax(g.function_logits / max(float(g.temp), 1e-6), dim=0)
                            k = int(torch.argmax(soft).item())
                            hard_logits = torch.full_like(g.function_logits, -1e4)
                            hard_logits[k] = 1e4
                            g.function_logits.copy_(hard_logits)
                            g.temp.copy_(torch.tensor(TEMP_MIN))
            for _sub in (_per_hard.fusion_net, _per_hard.attn_net, _per_hard.update_net):
                _hardify_module(_sub)
            torch.save(_per_hard.state_dict(), os.path.join(self.save_dir, f'model_{scale}_epoch{e}.pt'))
            torch.save(opt.state_dict(),      os.path.join(self.save_dir, f'opt_{scale}_epoch{e}.pt'))
            torch.save(_per_hard.state_dict(), os.path.join(self.save_dir, 'model_last_run.pt'))
            torch.save(opt.state_dict(),       os.path.join(self.save_dir, 'opt_last_run.pt'))

            print(f"  Train Loss: {avg_loss:.4f} | Train Soft-Sil: {train_soft_sil:.4f}")
            print(f"  Val   → Clusters: {val_metrics['Cluster_Count']}, "
                  f"Intra_Dist: {val_metrics['Intra_Dist']:.4f}, "
                  f"Inter_Dist: {val_metrics['Inter_Dist']:.4f}")
            print(f"        Sil: {val_metrics['Silhouette']:.4f}, "
                  f"Soft_Sil: {val_metrics['Soft_Silhouette']:.4f}, "
                  f"CH: {val_metrics['Calinski_Harabasz']:.1f}, "
                  f"DB: {val_metrics['Davies_Bouldin']:.4f}")
            if self.scheduler:
                self.scheduler.step()

            # Early stop (4 conditions AND): soft/hard thresholds, min clusters, min epochs
            if not hasattr(self, '_global_epoch'):
                self._global_epoch = 0
            self._global_epoch += 1
            try:
                _clus_each = [float(m.get('Cluster_Count', 0.0)) for m in metrics_each]
                _avg_clusters = sum(_clus_each) / max(1, len(_clus_each))
            except Exception:
                _avg_clusters = float(val_metrics.get('Cluster_Count', 0.0))
            val_metrics['Avg_Cluster_Count'] = float(_avg_clusters)
            if (val_metrics.get('Soft_Silhouette', -1.0) >= STOP_SOFTSIL and
                val_metrics.get('Silhouette', -1.0)      >  STOP_HARDSIL  and
                _avg_clusters                             >= STOP_MIN_CLUSTERS and
                self._global_epoch >= STOP_MIN_EPOCHS):
                print(f"[Early-Stop] Hit at global_epoch={self._global_epoch}: "
                      f"Soft-Sil={val_metrics['Soft_Silhouette']:.4f} (≥{STOP_SOFTSIL}), "
                      f"Sil={val_metrics['Silhouette']:.4f} (>{STOP_HARDSIL}), "
                      f"Avg-k={val_metrics['Avg_Cluster_Count']:.2f} (≥{STOP_MIN_CLUSTERS}).")
                try:
                    with open(os.path.join(self.save_dir, 'EARLY_STOPPED'), 'w', encoding='utf-8') as _f:
                        _f.write(str(self._global_epoch))
                except Exception:
                    pass
                e = epochs

            # No group-end rollback; keep global-best export mechanism
            pass

            # ===== gate stats & new stability-based stop =====
            from collections import deque
            gate_probs, gate_temps, margins = [], [], []
            uncommitted = []
            for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
                for layer in net.layers:
                    for g in layer:
                        g.committed = False
                        g.committed_idx = None
                        if hasattr(g.function_logits, 'requires_grad'):
                            g.function_logits.requires_grad_(True)
                        if not hasattr(g, 'stable_count'):
                            g.stable_count = 0
                        with torch.no_grad():
                            soft = torch.softmax(g.function_logits / g.temp, dim=0)
                            vals, idxs = torch.topk(soft, k=2, dim=0)
                            win = int(idxs[0].item())
                            z_top2 = torch.topk(g.function_logits, k=2, dim=0).values
                            margin = float((z_top2[0] - z_top2[1]).item())
                            margins.append(margin)
                            gate_probs.append(float(vals[0].item()))
                            gate_temps.append(float(g.temp))
                            prev = getattr(g, 'last_winner', -1)
                            if prev == win:
                                g.stable_count = int(g.stable_count) + 1
                            else:
                                g.stable_count = 1
                            g.last_winner = win
                            if not g.committed:
                                uncommitted.append((g, margin))

            avg_gate_p     = sum(gate_probs) / max(1, len(gate_probs)) if gate_probs else 0.0
            min_gate_p     = min(gate_probs) if gate_probs else 0.0
            avg_gate_temp  = sum(gate_temps) / max(1, len(gate_temps)) if gate_temps else 0.0
            min_gate_temp  = min(gate_temps) if gate_temps else TEMP_INIT

            print(f"        Avg_Gate_MaxP: {avg_gate_p:.3f} | "
                  f"Min_Gate_MaxP: {min_gate_p:.3f} | "
                  f"Avg_Gate_Temp: {avg_gate_temp:.3f} | "
                  f"Min_Gate_Temp: {min_gate_temp:.3f}")

            if margins:
                m = np.array(margins, dtype=float)
                print(f"[Diag] Gate top2-margin: min={m.min():.4f}, p25={np.percentile(m,25):.4f}, "
                      f"median={np.median(m):.4f}, p75={np.percentile(m,75):.4f}, p95={np.percentile(m,95):.4f}")
                cnt_small_1e3 = int((m < 1e-3).sum())
                cnt_small_1e2 = int((m < 1e-2).sum())
                print(f"[Diag] Gate margins <1e-3: {cnt_small_1e3}/{len(m)}, <1e-2: {cnt_small_1e2}/{len(m)}")

            val_metrics['Avg_Gate_MaxP']  = float(avg_gate_p)
            val_metrics['Min_Gate_MaxP']  = float(min_gate_p)
            val_metrics['Avg_Gate_Temp']  = float(avg_gate_temp)
            val_metrics['Min_Gate_Temp']  = float(min_gate_temp)

            if not hasattr(self, '_avgp_hist'):
                self._avgp_hist = deque(maxlen=6)
            if not hasattr(self, '_avgt_hist'):
                self._avgt_hist = deque(maxlen=6)
            self._avgp_hist.append(avg_gate_p)
            self._avgt_hist.append(avg_gate_temp)

            def _stable(hist, tol=1e-3):
                return (len(hist) == hist.maxlen) and (max(hist) - min(hist) <= tol)

            A_ok = (avg_gate_p >= 0.985) and _stable(self._avgp_hist, 1e-3)
            B_ok = (avg_gate_temp <= 0.11) and _stable(self._avgt_hist, 1e-3)
            if (self.total_epochs_run >= self.stop_min_epochs) and (A_ok or B_ok):
                print(f"[STOP] New criterion met at epoch={self.total_epochs_run} | "
                      f"AvgP={avg_gate_p:.3f} (stable={_stable(self._avgp_hist)}), "
                      f"AvgT={avg_gate_temp:.3f} (stable={_stable(self._avgt_hist)}).")
                return history

            # Row for CSV/aggregation
            cur_lr = self.current_optimizer.param_groups[0]['lr']
            group_name = os.path.basename(os.path.dirname(train_g[0])) if (isinstance(train_g, list) and len(train_g) > 0) else 'unknown'
            row = {
                'seed': self.seed if self.seed is not None else -1,
                'scale': scale,
                'group': group_name,
                'epoch': int(e),
                'train_loss': float(avg_loss),
                'Cluster_Count': float(val_metrics['Cluster_Count']),
                'Intra_Dist': float(val_metrics['Intra_Dist']),
                'Inter_Dist': float(val_metrics['Inter_Dist']),
                'Silhouette': float(val_metrics['Silhouette']),
                'Soft_Silhouette': float(val_metrics['Soft_Silhouette']),
                'Calinski_Harabasz': float(val_metrics['Calinski_Harabasz']),
                'Davies_Bouldin': float(val_metrics['Davies_Bouldin']),
                'Avg_Gate_MaxP': float(avg_gate_p),
                'Min_Gate_MaxP': float(min_gate_p),
                'lr': float(cur_lr),
            }
            self.epoch_rows.append(row)
            self.last_epoch_metrics = row

            # Global epoch counter (for stop conditions)
            self.total_epochs_run += 1

            # Refresh safety snapshot
            snapshots.append(take_snapshot())

            # Adaptive annealing after validation (optional)
            if do_anneal and GATE_ANNEAL_ENABLE:
                if not self.anneal_started:
                    by_epoch  = (self.total_epochs_run >= max(ANNEAL_START_EPOCH, self.anneal_start_epoch))
                    by_metric = (val_metrics.get('Silhouette', -1.0) >= self.anneal_metric_thresh)
                    if by_epoch or by_metric:
                        self.anneal_started = True
                        self.prev_soft_sil = val_metrics.get('Silhouette', None)
                        self.anneal_exp = 1.
                if self.anneal_started:
                    t  = self.total_epochs_run
                    t0 = max(ANNEAL_START_EPOCH, self.anneal_start_epoch)
                    T  = max(1, ANNEAL_TARGET_EPOCH - t0)
                    p  = min(1.0, max(0.0, (t - t0) / T))
                    cos_w = 0.5 * (1.0 - math.cos(math.pi * p))
                    gamma_epoch = ANNEAL_GAMMA_MAX - (ANNEAL_GAMMA_MAX - ANNEAL_GAMMA_MIN) * cos_w
                    gamma_epoch = max(ANNEAL_GAMMA_MIN, min(ANNEAL_GAMMA_MAX, gamma_epoch))
                    if not hasattr(self, 'prev_min_p'):
                        self.prev_min_p = None
                    if not hasattr(self, 'prev_sil_for_plateau'):
                        self.prev_sil_for_plateau = None
                    if not hasattr(self, 'reanneal_left'):
                        self.reanneal_left = 0
                        self.reanneal_total = 0
                    if not hasattr(self, 'thaw_cooldown'):
                        self.thaw_cooldown = 0
                    sil_cur = val_metrics.get('Silhouette', -1.0)
                    d_sil  = float('inf') if self.prev_sil_for_plateau is None else abs(sil_cur   - self.prev_sil_for_plateau)
                    d_minp = float('inf') if self.prev_min_p is None            else abs(min_gate_p - self.prev_min_p)
                    unmet_sil  = (sil_cur    < STOP_HARDSIL)
                    HARDEN_MINP_THRESH = 0.90
                    unmet_minp = (min_gate_p < HARDEN_MINP_THRESH)
                    plateau_sil  = (d_sil  <= 2e-3)
                    plateau_minp = (d_minp <= 1e-3)
                    if (unmet_sil and plateau_sil) or (unmet_minp and plateau_minp):
                        self.plateau_count = getattr(self, 'plateau_count', 0) + 1
                    else:
                        self.plateau_count = 0
                    ADD_HEAT       = 0.30
                    REANNEAL_SPAN  = 12
                    GAMMA_RE_MAX   = 0.985
                    GAMMA_RE_MIN   = 0.945
                    trigger_heat = (self.total_epochs_run >= self.stop_min_epochs) and (self.plateau_count >= 3) and (self.thaw_cooldown == 0)
                    if trigger_heat:
                        with torch.no_grad():
                            for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
                                for layer in net.layers:
                                    for g in layer:
                                        g.temp.add_(ADD_HEAT)
                                        g.temp.clamp_(min=TEMP_MIN, max=TEMP_INIT)
                        self.reanneal_total = REANNEAL_SPAN
                        self.reanneal_left  = REANNEAL_SPAN
                        self.thaw_cooldown  = 2
                        self.plateau_count  = 0
                        print(f"[HEAT] epoch={self.total_epochs_run} | add={ADD_HEAT} | reanneal={REANNEAL_SPAN}ep")
                    elif self.thaw_cooldown > 0:
                        self.thaw_cooldown -= 1
                    if self.reanneal_left > 0:
                        q = 1.0 - (self.reanneal_left / max(1, self.reanneal_total))
                        w = 0.5 * (1.0 - math.cos(math.pi * q))
                        gamma_re = GAMMA_RE_MAX - (GAMMA_RE_MAX - GAMMA_RE_MIN) * w
                        self.per.anneal_all(gamma_re)
                        self.reanneal_left -= 1
                    else:
                        if (self.total_epochs_run >= t0) or (sil_cur >= ANNEAL_SOFTSIL_THRESH):
                            if p < 1.0:
                                self.per.anneal_all(gamma_epoch)
                    self.prev_sil_for_plateau = sil_cur
                    self.prev_min_p = min_gate_p
                    if p >= 1.0 and self.reanneal_left <= 0:
                        self.per.set_global_temp(TEMP_MIN)

            # Soft-sil based patience (best-on-plateau tracking)
            stop_metric = val_metrics['Soft_Silhouette']
            if stop_metric > best_val_softsil + 1e-6:
                best_val_softsil = stop_metric
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            # Gate hardening diagnostics (unchanged thresholds)
            gate_maxps = []
            for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
                for layer in net.layers:
                    for g in layer:
                        with torch.no_grad():
                            gate_maxps.append(float(torch.softmax(g.function_logits / g.temp, dim=0).max().item()))
            avg_maxp = float(np.mean(gate_maxps)) if gate_maxps else 0.0
            frac_hardened = float(np.mean([p >= 0.9 for p in gate_maxps])) if gate_maxps else 0.0
            if not hasattr(self, 'avg_maxp_hist'):
                self.avg_maxp_hist = []
            self.avg_maxp_hist.append(avg_maxp)
            hist_win = 5
            if len(self.avg_maxp_hist) >= 2:
                tail = self.avg_maxp_hist[-hist_win:]
                delta_avg_maxp = float(max(tail) - min(tail))
            else:
                delta_avg_maxp = float('inf')
            cond_hardened = (frac_hardened >= 0.6) and (delta_avg_maxp < 0.005)


        # -------- end of group: clear EWC accumulators --------
        self.ewc_importance = {}
        self.ewc_prev_params = {}

        torch.cuda.empty_cache()
        group_name = os.path.basename(os.path.dirname(train_g[0])) if (isinstance(train_g, list) and len(train_g) > 0) else 'unknown'
        with open(os.path.join(self.save_dir, f'metrics_{scale}_{group_name}.json'), 'w') as f:
            json.dump(history, f)
        return best

    def train(self):
        scales = ['mixed']
        # 1) pre-scan all groups
        groups_per_scale = {
            s: sorted([
                os.path.join(TRAIN_PATHS[s], d)
                for d in os.listdir(TRAIN_PATHS[s])
                if d.startswith('group')
            ])
            for s in scales
        }
        # environment cleanup
        PrecomputeModule.reset_all(delete_tmp=True)
        gc.collect()

        # Stage 1: random search on the first group
        print("\n===== Stage 1: Random Search on First Group =====")
        scale = 'mixed'
        first_group_path = groups_per_scale[scale][0]
        first_group_files = sorted([os.path.join(first_group_path, f) for f in os.listdir(first_group_path)])
        val_root = VAL_PATHS[scale]
        val_groups = sorted([os.path.join(val_root, d) for d in os.listdir(val_root) if d.startswith('group')])
        val_group_lists = [sorted([os.path.join(gdir, f) for f in os.listdir(gdir) if f.endswith('.graphml')]) for gdir in val_groups]
        self.phase = 'random'
        search_epoch = 0
        while self.phase == 'random':
            search_epoch += 1
            print(f"\n--- Random Search Epoch {search_epoch} on group {os.path.basename(first_group_path)} ---")
            # train_group handles randomized weights internally
            self.train_group(first_group_files, val_group_lists, scale, epochs=1, do_anneal=False, group_idx=0, is_search_phase=True)
            if self._seed_should_stop:
                break

        if self._seed_should_stop:
            print("\n[Early-Stop] Target met during random search phase. Training finished.")
        else:
            # Stage 2: finetune across all groups
            print("\n===== Stage 2: Finetuning on All Groups =====")
            self.phase = 'finetune'
            for gi, grp in enumerate(groups_per_scale[scale]):
                if self._seed_should_stop:
                    break
                tg = sorted([os.path.join(grp, f) for f in os.listdir(grp)])
                group_name = os.path.basename(grp)
                print(f"\n===== Finetuning on {scale} / {group_name} =====")
                done = 0
                while (done < EPOCHS_PER_GROUP) and (not self._seed_should_stop):
                    # If reverted to random during this group's finetune, resume random exploration until target
                    while (self.phase == 'random') and (not self._seed_should_stop):
                        print(f"[Re-Random @ {group_name}] resume random exploration until target, then continue finetune…")
                        self.train_group(tg, val_group_lists, scale, epochs=1, do_anneal=False, group_idx=gi, is_search_phase=True)
                    if self._seed_should_stop:
                        break
                    print(f"Finetuning {group_name}: epoch {done+1}/{EPOCHS_PER_GROUP}")
                    self.train_group(tg, val_group_lists, scale, epochs=1, do_anneal=False, group_idx=gi, is_search_phase=False)
                    done += 1


        # ---- Unified export (soft) ----
        try:
            def _infer_layer_cfg(mod):
                return [len(layer) for layer in mod.layers]
            layer_cfg = {
                "fusion": _infer_layer_cfg(self.per.fusion_net),
                "attn":   _infer_layer_cfg(self.per.attn_net),
                "update": _infer_layer_cfg(self.per.update_net),
            }
            final_metrics = dict(self.last_epoch_metrics or {})
            final_ckpt = _pack_gnca_ckpt(
                model=self.per,
                epoch=-1,
                seed=int(self.cur_seed if hasattr(self, "cur_seed") else 0),
                scale=getattr(self, "cur_scale", ""),
                group=int(getattr(self, "cur_group_idx", -1)),
                layer_cfg=layer_cfg,
                metrics=final_metrics
            )
            _atomic_save(final_ckpt, os.path.join(self.save_dir, 'final_model.pt'))
            print("[Export] Saved unified final checkpoint → final_model.pt")
        except Exception as ex:
            print(f"[Export][WARN] Save unified final failed: {ex}")

        # ---- Export one-hot (hard) in the same format ----
        try:
            per_hard = copy.deepcopy(self.per)
            onehot_indices = []

            def _hardify_module(mod):
                for layer in getattr(mod, 'layers', []):
                    for g in layer:
                        with torch.no_grad():
                            soft = F.softmax(g.function_logits / max(float(g.temp), 1e-6), dim=0)
                            k = int(torch.argmax(soft).item())
                            onehot_indices.append(k)
                            hard_logits = torch.full_like(g.function_logits, -1e4)
                            hard_logits[k] = 1e4
                            g.function_logits.copy_(hard_logits)
                            g.temp.copy_(torch.tensor(TEMP_MIN))

            for sub in (per_hard.fusion_net, per_hard.attn_net, per_hard.update_net):
                _hardify_module(sub)

            layer_cfg_h = {
                "fusion": [len(l) for l in per_hard.fusion_net.layers],
                "attn":   [len(l) for l in per_hard.attn_net.layers],
                "update": [len(l) for l in per_hard.update_net.layers],
            }
            hard_ckpt = {
                "kind": "gnca_model_ckpt",
                "version": 1,
                "epoch": -1,
                "seed": int(self.cur_seed if hasattr(self, "cur_seed") else 0),
                "scale": getattr(self, "cur_scale", ""),
                "group": int(getattr(self, "cur_group_idx", -1)),
                "layer_cfg": layer_cfg_h,
                "state_dict": per_hard.state_dict(),  # one-hot + TEMP_MIN
                "metrics": dict(self.last_epoch_metrics or {}),
                "onehot_indices": onehot_indices,
                "mode": "hard"
            }
            _atomic_save(hard_ckpt, os.path.join(self.save_dir, 'final_model_hard.pt'))
            with open(os.path.join(self.save_dir, 'gates_onehot_indices.json'), 'w', encoding='utf-8') as f:
                json.dump({'indices': onehot_indices}, f, ensure_ascii=False, indent=2)
            print("[Export] Saved unified hard checkpoint → final_model_hard.pt (one-hot)")
        except Exception as ex:
            print(f"[Export][WARN] Save unified hard failed: {ex}")

        # Write per-epoch table for this seed
        csv_path = os.path.join(self.save_dir, 'epoch_metrics.csv')
        if self.epoch_rows:
            base_headers = [
                'seed','scale','group','epoch','train_loss',
                'Cluster_Count','Intra_Dist','Inter_Dist',
                'Silhouette','Soft_Silhouette','Calinski_Harabasz','Davies_Bouldin',
                'Avg_Gate_MaxP','Min_Gate_MaxP','lr'
            ]
            seen = set(base_headers)
            extra = []
            for r in self.epoch_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        extra.append(k)
            headers = base_headers + extra

            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in self.epoch_rows:
                    writer.writerow(r)

        print("Training completed!")

        # Summarize this seed: final one-hot sharpness and Soft-Sil
        all_probs = []
        for net in (self.per.fusion_net, self.per.attn_net, self.per.update_net):
            for layer in net.layers:
                for g in layer:
                    all_probs.append(torch.softmax(g.function_logits / g.temp, dim=0).max().item())
        onehot_min = float(min(all_probs)) if all_probs else float('nan')

        if self.last_epoch_metrics is not None:
            final_softsil = float(self.last_epoch_metrics.get('Soft_Silhouette', float('nan')))
            final_scale   = self.last_epoch_metrics.get('scale')
            final_group   = self.last_epoch_metrics.get('group')
            final_epoch   = int(self.last_epoch_metrics.get('epoch', -1))
        else:
            final_softsil = float('nan')
            final_scale   = None
            final_group   = None
            final_epoch   = -1

        return {
            'seed': self.seed,
            'save_dir': self.save_dir,
            'epoch_csv': csv_path if self.epoch_rows else None,
            'final_softsil': final_softsil,
            'final_scale': final_scale,
            'final_group': final_group,
            'final_epoch': final_epoch,
            'onehot_min_p': onehot_min,
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=LR_INITIAL)
    parser.add_argument('--device', type=str, default=str(DEVICE))
    parser.add_argument('--seed', type=int, default=BASE_SEED)
    parser.add_argument('--num_seeds', type=int, default=NUM_SEEDS)
    parser.add_argument('--min_epochs', type=int, default=STOP_MIN_EPOCHS, help='Minimum global training epochs before any stop check')
    parser.add_argument('--stop_softsil', type=float, default=STOP_SOFTSIL, help='Stop threshold: validation Soft-Silhouette (current epoch)')
    parser.add_argument('--stop_hardsil', type=float, default=STOP_HARDSIL, help='Stop threshold: validation hard Silhouette (current epoch)')
    parser.add_argument('--min_clusters', type=int, default=STOP_MIN_CLUSTERS, help='Stop condition: minimum number of clusters (>= value)')
    # Backward-compat alias for hard Silhouette; hidden from CLI
    parser.add_argument('--stop_sil', dest='stop_hardsil', type=float, help=argparse.SUPPRESS)

    args = parser.parse_args([])

    # Fixed run config (no CLI needed)
    args.seed          = BASE_SEED
    args.num_seeds     = NUM_SEEDS
    args.min_epochs    = STOP_MIN_EPOCHS
    args.stop_softsil  = STOP_SOFTSIL
    args.stop_hardsil  = STOP_HARDSIL
    args.min_clusters  = STOP_MIN_CLUSTERS

    # Single seed
    if args.num_seeds == 1:
        set_seed(args.seed)
        Trainer(args).train()
    else:
        # Multi-seed: sequential run + summary
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_root = os.path.join(SAVE_ROOT, f'multiseed_{run_ts}')
        os.makedirs(run_root, exist_ok=True)

        seeds = [args.seed + i for i in range(args.num_seeds)]
        print(f"Multi-seed run → seeds = {seeds}")
        summaries = []

        for s in seeds:
            # Per-seed pre-run: hard cleanup + crash tracing
            try:
                import faulthandler
                faulthandler.enable()
            except Exception:
                pass
            try:
                PrecomputeModule.reset_all(delete_tmp=True)  # Clear class-level cache and temp files
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            args_i = copy.deepcopy(args)
            args_i.seed = s
            args_i.run_root = run_root
            set_seed(s)
            print(f"\n===== Seed {s} =====")
            summ = Trainer(args_i).train()
            summaries.append(summ)

        # Write summary JSON
        with open(os.path.join(run_root, 'multiseed_summary.json'), 'w') as f:
            json.dump(summaries, f, indent=2)

        # Read per-seed epoch CSVs and compute cross-seed means grouped by (scale, group, epoch)
        rows_all = []
        for summ in summaries:
            csv_path = summ.get('epoch_csv')
            if csv_path and os.path.isfile(csv_path):
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows_all.append(r)

        if rows_all:
            key_cols = ('scale', 'group', 'epoch')
            num_cols = [
                'train_loss','Cluster_Count','Intra_Dist','Inter_Dist',
                'Silhouette','Soft_Silhouette','Calinski_Harabasz','Davies_Bouldin',
                'Avg_Gate_MaxP','Min_Gate_MaxP','lr'
            ]
            buckets = defaultdict(list)
            for r in rows_all:
                key = (r['scale'], r['group'], int(r['epoch']))
                buckets[key].append(r)

            out_rows = []
            for key, items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
                out = {'scale': key[0], 'group': key[1], 'epoch': int(key[2]), 'num_seeds': len(items)}
                for c in num_cols:
                    vals = []
                    for it in items:
                        try:
                            vals.append(float(it[c]))
                        except Exception:
                            pass
                    out[c] = float(np.mean(vals)) if vals else float('nan')
                out_rows.append(out)

            avg_csv = os.path.join(run_root, 'epoch_metrics_averaged.csv')
            headers = ['scale','group','epoch','num_seeds'] + num_cols
            with open(avg_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in out_rows:
                    writer.writerow(r)

            # Print key averages: cross-seed Soft_Silhouette at last epoch per scale
            last_softsil = defaultdict(list)
            for r in out_rows:
                last_softsil[r['scale']].append((r['epoch'], r['Soft_Silhouette']))
            print("\n==== Cross-seed averaged (by scale, last epoch) ====")
            for sc, pairs in last_softsil.items():
                if pairs:
                    pairs.sort(key=lambda x: x[0])
                    print(f"{sc}: Soft_Sil(avg) at last epoch = {pairs[-1][1]:.4f}")

        # Overall averages (top level)
        valid_soft = [s.get('final_softsil') for s in summaries if s.get('final_softsil') is not None]
        if valid_soft:
            vals = [float(v) for v in valid_soft if (isinstance(v, (int, float)) and not math.isnan(float(v)))]
            print("\n==== Multi-seed summary ====")
            if vals:
                print(f"Avg final Soft_Sil (last group at stop): {float(np.mean(vals)):.4f}")
            else:
                print("Avg final Soft_Sil (last group at stop): N/A")
            print(f"Min onehot prob across seeds: {min(s.get('onehot_min_p', 0.0) for s in summaries):.3f}")
            print(f"All per-seed results saved under: {run_root}")
