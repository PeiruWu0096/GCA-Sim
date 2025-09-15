# -*- coding: utf-8 -*-
"""
Offline export & visualization utilities:
- Harden (strict one-hot) -> Prune to reachable gates only
- Gate-type mapping aligned with the 9 differentiable gates used in test/train
- Export Graphviz .dot (labeled & clean) and render to .svg if `dot` exists
- Emit SystemVerilog (SV) based on the pruned topology (no orphan gates)
- Optionally, generate DigitalJS-style SVG if yosys + netlistsvg are available
- SymPy algebraic expressions are intentionally skipped per user request
"""

import os, json, copy, subprocess, sys, shutil, re
from collections import defaultdict, OrderedDict
import torch
import torch.nn.functional as F

# =========================
# PATHS
# =========================
# MODEL_DIR: a run folder that contains checkpoints.
# MODEL_IN : path to the checkpoint to export (intermediate or final).
# OUT_DIR  : where all artifacts will be written under the run folder.
# Default: expects "final_model_hard.pt" inside MODEL_DIR.
MODEL_DIR = r"0data/output/trainsave/your_run"
MODEL_IN  = os.path.join(MODEL_DIR, "final_model_hard.pt")  # set this to a specific checkpoint if needed
if os.path.dirname(MODEL_IN):
    MODEL_DIR = os.path.dirname(MODEL_IN)
OUT_DIR   = os.path.join(MODEL_DIR, "difflogic_export")

# =========================
# Hardening options (strict one-hot)
# =========================
ONEHOT_ON  = 120.0     # exp(-120) under float32 ~ 0 → softmax becomes strict one-hot
ONEHOT_OFF = -120.0
TEMP_FIXED = 1.0

# =========================
# Gate set (must match the 9 functions used in your training/test code)
# =========================
OPS9 = [
    ("ADD"   , lambda a,b: a + b),                    # 0
    ("SUB_AB", lambda a,b: a - b),                    # 1
    ("SUB_BA", lambda a,b: b - a),                    # 2
    ("PASS_A", lambda a,b: a),                        # 3
    ("PASS_B", lambda a,b: b),                        # 4
    ("NEG_A" , lambda a,b: -a),                       # 5
    ("NEG_B" , lambda a,b: -b),                       # 6
    ("MAX"   , lambda a,b: torch.maximum(a,b)),       # 7
    ("MIN"   , lambda a,b: torch.minimum(a,b)),       # 8
]
def func14_idx_to_name(idx:int)->str:
    return OPS9[idx][0]

# Kept as an empty mapping for compatibility; not used in this script.
LUT2_NAME = {}

# =========================
# Output paths (DO NOT rename)
# =========================
HARDENED_PT = os.path.join(OUT_DIR, "final_model_hardened.pt")
PRUNED_PT   = os.path.join(OUT_DIR, "final_model_hardened_pruned.pt")
GRAPH_JSON  = os.path.join(OUT_DIR, "logic_net_pruned.json")
LABELED_DOT = os.path.join(OUT_DIR, "logic_net_pruned_labeled.dot")
LABELED_SVG = os.path.join(OUT_DIR, "logic_net_pruned_labeled.svg")
CLEAN_DOT   = os.path.join(OUT_DIR, "logic_net_pruned_clean.dot")
CLEAN_SVG   = os.path.join(OUT_DIR, "logic_net_pruned_clean.svg")
DIGITAL_SV  = os.path.join(OUT_DIR, "difflogic_top.sv")
DIGITAL_JSON= os.path.join(OUT_DIR, "difflogic.json")
DIGITAL_SVG = os.path.join(OUT_DIR, "digitaljs.svg")
DIGITAL_SVG_CLEAN = os.path.join(OUT_DIR, "digitaljs_clean.svg")
ALG_TXT     = os.path.join(OUT_DIR, "algebraic_expressions_sympy.txt")

# =========================
# Helpers
# =========================
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_state_dict(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, (dict, OrderedDict)) and "state_dict" in data:
        sd = data["state_dict"]
    else:
        sd = data
    if not isinstance(sd, (dict, OrderedDict)):
        raise ValueError("The loaded file does not contain a valid state_dict.")
    return sd

def stat_hardness(sd):
    probs=[]
    for k,v in sd.items():
        if k.endswith("function_logits"):
            base=k[:-len("function_logits")]
            t = sd.get(base+"temp", torch.tensor(1.0))
            p = F.softmax(v/t, dim=0).max().item()
            probs.append(p)
    if not probs:
        return None
    return {"num_gates":len(probs), "min":min(probs), "avg":sum(probs)/len(probs)}

def harden_same_shape(sd):
    out=copy.deepcopy(sd)
    for k,v in list(sd.items()):
        if k.endswith("function_logits"):
            base=k[:-len("function_logits")]
            t = sd.get(base+"temp", torch.tensor(TEMP_FIXED))
            idx = int(torch.argmax(v/t).item())
            new = torch.full_like(v, ONEHOT_OFF); new[idx]=ONEHOT_ON
            out[k]=new
            if base+"temp" in out:
                out[base+"temp"]=torch.tensor(float(TEMP_FIXED))
    return out

def list_networks(sd):
    nets={}
    for name in ("fusion_net","attn_net","update_net"):
        prefix=name+".layers."
        ks=[k for k in sd if k.startswith(prefix) and k.endswith(".idx_a")]
        if not ks:
            continue
        layers=defaultdict(list)
        for k in ks:
            parts=k.split("."); L=int(parts[2]); G=int(parts[3]); layers[L].append(G)
        nets[name]={"layers":[{"gates":sorted(layers[L])} for L in range(max(layers)+1)]}
    return nets

def read_gate(sd, net, L, G):
    base=f"{net}.layers.{L}.{G}."
    ia=int(sd[base+"idx_a"]); ib=int(sd[base+"idx_b"])
    idx=int(torch.argmax(sd[base+"function_logits"]).item())
    return ia,ib,idx

def prune_reachable(sd):
    topo=list_networks(sd); pruned={}
    for name,info in topo.items():
        Ls=info["layers"]
        if not Ls:
            pruned[name]={"layers":[]}
            continue
        Lmax=len(Ls)-1
        keep=[set() for _ in range(Lmax+1)]
        for g in Ls[Lmax]["gates"]:
            keep[Lmax].add(g)
        for L in range(Lmax-1,-1,-1):
            used=set()
            for gi in keep[L+1]:
                ia,ib,_=read_gate(sd,name,L+1,gi)
                used.update([int(ia),int(ib)])
            for g in Ls[L]["gates"]:
                if g in used:
                    keep[L].add(g)
        info_out={"layers":[]}
        for L in range(Lmax+1):
            order=sorted(keep[L]); idmap={old:i for i,old in enumerate(order)}
            info_out["layers"].append({"gates":order,"idmap":idmap})
        pruned[name]=info_out
    return pruned

def build_compact_pruned_state_dict(sd_hardened, pruned):
    out=OrderedDict()
    def copy_gate(net,L,oldg,newg,has_prev,prev_map):
        bo=f"{net}.layers.{L}.{oldg}."
        bn=f"{net}.layers.{L}.{newg}."
        out[bn+"function_logits"]=sd_hardened[bo+"function_logits"].clone()
        if bo+"temp" in sd_hardened:
            out[bn+"temp"]=torch.tensor(float(TEMP_FIXED))
        ia=int(sd_hardened[bo+"idx_a"]); ib=int(sd_hardened[bo+"idx_b"])
        if has_prev:
            ia=prev_map[ia]; ib=prev_map[ib]
        out[bn+"idx_a"]=torch.tensor(int(ia))
        out[bn+"idx_b"]=torch.tensor(int(ib))
    for net,info in pruned.items():
        if not info["layers"]:
            continue
        prev=None
        for L,meta in enumerate(info["layers"]):
            order=meta["gates"]; idmap=meta["idmap"]
            has_prev=(L>0 and len(info["layers"][L-1]["gates"])>0)
            for oldg in order:
                copy_gate(net,L,oldg,idmap[oldg],has_prev,prev)
            prev=idmap
    return out

# ---------- DOT export (with L0 inputs, labeled & clean) ----------
def export_graph_with_inputs(sd, pruned, dot_label, dot_clean, json_path):
    nodes, edges=[],[]
    for name,info in pruned.items():
        Ls=info["layers"]
        if not Ls:
            continue
        used=set()
        for g in Ls[0]["gates"]:
            ia,ib,_=read_gate(sd,name,0,g); used.update([int(ia),int(ib)])
        used=sorted(used)
        for i in used:
            nid=f"{name}_IN_{i}"
            nodes.append({"id":nid,"net":name,"layer":-1,"label":f"in[{i}]", "is_input":True})
        for L,meta in enumerate(Ls):
            for g in meta["gates"]:
                _,_,idx=read_gate(sd,name,L,g)
                label=func14_idx_to_name(idx)
                nodes.append({"id":f"{name}_L{L}_G{g}","net":name,"layer":L,"label":label,"func_idx":idx})
        for g in Ls[0]["gates"]:
            ia,ib,_=read_gate(sd,name,0,g)
            if ia in used:
                edges.append({"src":f"{name}_IN_{ia}","dst":f"{name}_L0_G{g}", "label": "a"})
            if ib in used:
                edges.append({"src":f"{name}_IN_{ib}","dst":f"{name}_L0_G{g}", "label": "b"})
        for L in range(1,len(Ls)):
            prev=set(Ls[L-1]["gates"])
            for g in Ls[L]["gates"]:
                ia,ib,_=read_gate(sd,name,L,g)
                if ia in prev:
                    edges.append({"src":f"{name}_L{L-1}_G{ia}","dst":f"{name}_L{L}_G{g}", "label": "a"})
                if ib in prev:
                    edges.append({"src":f"{name}_L{L-1}_G{ib}","dst":f"{name}_L{L}_G{g}", "label": "b"})
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump({"nodes":nodes,"edges":edges},f,ensure_ascii=False,indent=2)

    def _dot(show_label=True):
        lines=[]
        lines.append("digraph G {")
        lines.append("  rankdir=LR;")
        lines.append("  edge [fontsize=8];")
        if show_label:
            lines.append('  node [shape=box, style=rounded, fontsize=10];')
        else:
            lines.append('  node [shape=box, style=rounded, fontsize=10, label=""];')
        by_net=defaultdict(list)
        for n in nodes:
            by_net[n["net"]].append(n)
        for net,nds in by_net.items():
            lines.append(f'  subgraph cluster_{net} {{')
            lines.append(f'    label="{net}";')
            by_layer=defaultdict(list)
            for n in nds:
                by_layer[n["layer"]].append(n)
            for L in sorted(by_layer.keys()):
                items=" ".join([f'"{x["id"]}"' for x in sorted(by_layer[L], key=lambda z:z["id"])])
                lines.append(f'    {{ rank=same; {items} }}')
            for n in nds:
                lab=n["label"] if show_label else ""
                lines.append(f'    "{n["id"]}" [label="{lab}"];')
            lines.append("  }")
        for e in edges:
            edge_label = e.get("label", "")
            label_attr = f' [label="{edge_label}"]' if edge_label else ""
            lines.append(f'  "{e["src"]}" -> "{e["dst"]}"{label_attr};')
        lines.append("}")
        return "\n".join(lines)

    with open(dot_label,"w",encoding="utf-8") as f:
        f.write(_dot(True))
    with open(dot_clean,"w",encoding="utf-8") as f:
        f.write(_dot(False))

def render_svg(dot_path, svg_path):
    try:
        subprocess.run(["dot","-Tsvg",dot_path,"-o",svg_path], check=True)
        print(f"[OK] Graphviz render: {svg_path}")
    except Exception as e:
        print(f"[WARN] Graphviz render failed: {e}")

# ---------- SV emission based on pruned topology ----------
def list_layers(sd):
    nets={}
    for name in ("fusion_net","attn_net","update_net"):
        prefix=name+".layers."
        ks=[k for k in sd if k.startswith(prefix) and k.endswith(".idx_a")]
        if not ks:
            continue
        layers=defaultdict(list)
        for k in ks:
            parts=k.split("."); L=int(parts[2]); G=int(parts[3]); layers[L].append(G)
        nets[name]={"layers":[{"gates":sorted(layers[L])} for L in range(max(layers)+1)]}
    return nets

def collect_L0_inputs(sd, nets):
    inputs={}
    for net,info in nets.items():
        used=set()
        if info["layers"]:
            for g in info["layers"][0]["gates"]:
                ia=int(sd[f"{net}.layers.0.{g}.idx_a"]); ib=int(sd[f"{net}.layers.0.{g}.idx_b"])
                used.update([ia,ib])
        inputs[net]=sorted(used) if used else [0]
    return inputs

def emit_sv_from_pruned_compact(sd_compact):
    """
    Generate hierarchical SV:
    - For each net and each layer, define a stage module <net>_stage_L<L>
    - Top 'difflogic_top' wires stages in sequence; no multi-driver aliases
    - Only 2-input primitives via behavioral assigns; width inference is delegated
    """
    nets = list_layers(sd_compact)
    prim_in = collect_L0_inputs(sd_compact, nets)
    in_maps = {net: {idx: i for i, idx in enumerate(in_list)} for net, in_list in prim_in.items()}

    def inst_for_idx(y, a, b, idx, net, L, g, lines):
        name = func14_idx_to_name(idx)
        if name == "ADD":
            lines.append(f"    assign {y} = {a} + {b};")
        elif name == "SUB_AB":
            lines.append(f"    assign {y} = {a} - {b};")
        elif name == "SUB_BA":
            lines.append(f"    assign {y} = {b} - {a};")
        elif name == "PASS_A":
            lines.append(f"    assign {y} = {a};")
        elif name == "PASS_B":
            lines.append(f"    assign {y} = {b};")
        elif name == "NEG_A":
            lines.append(f"    assign {y} = -{a};")
        elif name == "NEG_B":
            lines.append(f"    assign {y} = -{b};")
        else:
            lines.append(f"    assign {y} = 1'b0;")

    stage_defs = []
    for net, info in nets.items():
        Lmax = len(info['layers']) - 1 if info['layers'] else -1
        for L, meta in enumerate(info['layers']):
            gates = meta['gates']
            Nin = (len(prim_in[net]) if L == 0 else len(info['layers'][L-1]['gates']))
            Nout = len(gates)
            m = []
            m.append(f"(* keep_hierarchy = 1 *) module {net}_stage_L{L} (input [{Nin-1}:0] in_v, output [{Nout-1}:0] out_v);")
            for gi in gates:
                m.append(f"  (* keep = 1 *) wire {net}_L{L}_G{gi};")
            for gi in gates:
                ia = int(sd_compact[f"{net}.layers.{L}.{gi}.idx_a"])
                ib = int(sd_compact[f"{net}.layers.{L}.{gi}.idx_b"])
                idx = int(torch.argmax(sd_compact[f"{net}.layers.{L}.{gi}.function_logits"]).item())
                if L == 0:
                    amap = in_maps[net]
                    assert (ia in amap) and (ib in amap), f"L0 index not found for {net} L{L} G{gi}: ia={ia}, ib={ib}, map={amap}"
                    a = f"in_v[{amap[ia]}]"
                    b = f"in_v[{amap[ib]}]"
                else:
                    a = f"in_v[{ia}]"
                    b = f"in_v[{ib}]"
                y = f"{net}_L{L}_G{gi}"
                inst_for_idx(y, a, b, idx, net, L, gi, m)
            for oi, gi in enumerate(gates):
                m.append(f"  assign out_v[{oi}] = {net}_L{L}_G{gi};")
            m.append("endmodule")
            m.append("")
            stage_defs.append("\n".join(m))

    port_names = []
    for net in nets.keys():
        port_names += [f"in_{net}", f"out_{net}"]

    lines = []
    lines.append("// Hierarchical gate-level netlist with per-layer modules; no alias assigns.")
    lines.append("module difflogic_top (" + ", ".join(port_names) + ");")

    for net, info in nets.items():
        Nin  = max(1, len(prim_in[net]))
        last = info['layers'][-1]['gates'] if info['layers'] else []
        Nout = max(1, len(last))
        lines.append(f"  input  [{Nin-1}:0]  in_{net};")
        lines.append(f"  output [{Nout-1}:0] out_{net};")

    for net, info in nets.items():
        if not info['layers']:
            lines.append(f"  assign out_{net}[0] = 1'b0;")
            continue
        lines.append(f"  wire [{len(info['layers'][0]['gates'])-1}:0] {net}_L0;")
        lines.append(f"  {net}_stage_L0 u_{net}_L0 (.in_v(in_{net}), .out_v({net}_L0));")
        for L in range(1, len(info['layers'])):
            Ncur = len(info['layers'][L]['gates'])
            lines.append(f"  wire [{Ncur-1}:0] {net}_L{L};")
            lines.append(f"  {net}_stage_L{L} u_{net}_L{L} (.in_v({net}_L{L-1}), .out_v({net}_L{L}));")
        Lmax = len(info['layers']) - 1
        lines.append(f"  assign out_{net} = {net}_L{Lmax};")

    lines.append("endmodule")
    lines.append("")
    lines.extend(stage_defs)
    return "\n".join(lines)

# ---------- SymPy expressions (intentionally skipped) ----------
def has_sympy():
    try:
        import sympy as sp  # noqa
        return True
    except Exception:
        return False

def algebraic_expressions_sympy(sd_compact, out_txt):
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("Algebraic expression generation is skipped by design.\n")
        print(f"[INFO] Skipped SymPy algebraic expressions (placeholder written): {out_txt}")
    except Exception as e:
        print(f"[WARN] Failed to write placeholder for algebraic expressions: {e}")

# ---------- Optional: offline netlistsvg pipeline ----------
def has_cmd(cmd):
    try:
        subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def make_gate_svg_offline():
    yosys_ok = has_cmd("yosys")
    nls_ok   = has_cmd("netlistsvg")
    if not yosys_ok or not nls_ok:
        print("[INFO] yosys or netlistsvg not found; skipping DigitalJS-style SVG.\n"
              f"      After installation, run in OUT_DIR:\n"
              f"      yosys -p \"read_verilog {os.path.basename(DIGITAL_SV)}; prep -top difflogic_top; write_json {os.path.basename(DIGITAL_JSON)}\"\n"
              f"      netlistsvg {os.path.basename(DIGITAL_JSON)} -o {os.path.basename(DIGITAL_SVG)}")
        return
    try:
        subprocess.run(["yosys","-p",f"read_verilog {DIGITAL_SV}; prep -top difflogic_top; write_json {DIGITAL_JSON}"], check=True)
        subprocess.run(["netlistsvg",DIGITAL_JSON,"-o",DIGITAL_SVG], check=True)
        with open(DIGITAL_SVG,"r",encoding="utf-8") as f:
            svg=f.read()
        svg=re.sub(r"<text[^>]*>.*?</text>","",svg,flags=re.DOTALL)
        with open(DIGITAL_SVG_CLEAN,"w",encoding="utf-8") as f:
            f.write(svg)
        print(f"[OK] DigitalJS SVG: {DIGITAL_SVG}")
        print(f"[OK] DigitalJS SVG (no text): {DIGITAL_SVG_CLEAN}")
    except Exception as e:
        print("[WARN] DigitalJS SVG pipeline failed: ", e)

# =========================
# Main
# =========================
def main():
    if not os.path.isfile(MODEL_IN):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_IN}")
    os.makedirs(OUT_DIR, exist_ok=True)

    sd = load_state_dict(MODEL_IN)
    s = stat_hardness(sd)
    if s:
        print(f"[INFO] gates={s['num_gates']}  minP={s['min']:.4f}  avgP={s['avg']:.4f}")

    # 1) harden
    sd_hardened = harden_same_shape(sd)
    torch.save(sd_hardened, HARDENED_PT)
    print(f"[OK] Saved hardened checkpoint: {HARDENED_PT}")

    # 2) prune reachable
    pruned = prune_reachable(sd_hardened)

    # 3) compact pruned state_dict (as single source for all exports)
    sd_compact = build_compact_pruned_state_dict(sd_hardened, pruned)
    torch.save({"state_dict": sd_compact, "meta":{"sizes":{k:[len(L['gates']) for L in v['layers']] for k,v in pruned.items()}}}, PRUNED_PT)
    print(f"[OK] Saved compact pruned checkpoint: {PRUNED_PT}")

    # 4) DOT (+SVG)
    export_graph_with_inputs(sd_hardened, pruned, LABELED_DOT, CLEAN_DOT, GRAPH_JSON)
    render_svg(LABELED_DOT, LABELED_SVG)
    render_svg(CLEAN_DOT,   CLEAN_SVG)

    # 5) SV (no orphan gates)
    sv = emit_sv_from_pruned_compact(sd_compact)
    with open(DIGITAL_SV,"w",encoding="utf-8") as f:
        f.write(sv)
    print(f"[OK] Wrote SV: {DIGITAL_SV}")

    # 6) optional DigitalJS-style SVG
    make_gate_svg_offline()

    # 7) skip SymPy expressions by design
    algebraic_expressions_sympy(sd_compact, ALG_TXT)
    print("\n=== DONE ===")
    print("Output dir:", OUT_DIR)
    print("If you need gate-symbol SVG without text, check digitaljs_clean.svg (if generated).")

if __name__=="__main__":
    main()
