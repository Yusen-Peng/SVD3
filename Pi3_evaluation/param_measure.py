import torch
import argparse
import time
import csv
import os
import pandas as pd
import plotly.express as px
from fvcore.nn import FlopCountAnalysis

import torch.profiler as prof
import rootutils    
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from vggt.models.vggt import VGGT
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profiler_efficiency(model, imgs, dtype, csv_path="profile.csv"):
    with prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as p:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])

    # Collect key averages
    events = p.key_averages()

    # Write to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        writer.writerow([
            "Name",
            "CPU time total (us)",
            "CUDA time total (us)",
            "Calls",
            "Input shapes",
            "Self CPU Mem (KB)",
            "Self CUDA Mem (KB)"
        ])
        # rows
        for evt in events:
            writer.writerow([
                evt.key,
                evt.cpu_time_total,
                evt.cuda_time_total,
                evt.count,
                evt.input_shapes,
                evt.self_cpu_memory_usage / 1024,
                evt.self_cuda_memory_usage / 1024
            ])

    print(f"✅Profiler results written to {csv_path}")
    return res

def build_profiler_plots(
    csv_path: str = "profile.csv",
    png_path: str = "topk_cuda_ops.png",
    top_k: int = 10,
    width: int = 520,          # ✅ smaller canvas
    row_h: int = 28,           # ✅ controls “shorter”
    bar_gap: float = 0.25,     # ✅ thinner spacing between bars
    font_size: int = 11,       # ✅ smaller font
    left_margin: int = 170,    # ✅ smaller left gutter
):
    import pandas as pd
    import plotly.express as px

    df = pd.read_csv(csv_path)

    col_map = {
        "Name": "Name",
        "CPU time total (us)": "CPU_us",
        "CUDA time total (us)": "CUDA_us",
        "Calls": "Calls",
        "Input shapes": "Input shapes",
        "Self CPU Mem (KB)": "Self CPU Mem (KB)",
        "Self CUDA Mem (KB)": "Self CUDA Mem (KB)",
    }
    df = df.rename(columns=col_map)

    for c in ["CPU_us", "CUDA_us", "Calls", "Self CPU Mem (KB)", "Self CUDA Mem (KB)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = (
        df.groupby("Name", as_index=False)
          .agg({
              "CPU_us": "sum",
              "CUDA_us": "sum",
              "Calls": "sum",
          })
    )

    agg["CUDA_ms"] = agg["CUDA_us"] / 1000.0

    top = (
        agg.sort_values("CUDA_ms", ascending=False)
           .head(max(1, int(top_k)))
           .copy()
    )

    total_cuda_ms = top["CUDA_ms"].sum()  # (still “top-k total”, compact plot)
    title = f"Top {len(top)} Ops by CUDA Time (topK total={total_cuda_ms:.1f} ms)"


    top["Label"] = top["Name"].map(lambda x: x[6:])

    # ✅ smaller height proportional to K
    height = max(180, row_h * len(top) + 70)

    fig = px.bar(
        top.sort_values("CUDA_ms", ascending=True),
        x="CUDA_ms",
        y="Label",
        orientation="h",
        title=title,
        labels={"CUDA_ms": "CUDA time (ms)", "Label": ""},
    )

    fig.update_layout(
        width=width,
        height=height,
        bargap=bar_gap,
        margin=dict(l=left_margin, r=20, t=45, b=35),
        xaxis=dict(title="CUDA time (ms)", ticks="outside"),
        yaxis=dict(automargin=False),
        font=dict(size=font_size, family="Times New Roman", color="black"),
        showlegend=False,
    )

    # ✅ thinner bars
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="%{y}<br>CUDA: %{x:.2f} ms<extra></extra>",
    )

    fig.write_image(png_path, scale=2)
    print(f"[build_profiler_plots] wrote: {png_path}")
    return top


import torch
import torch.nn as nn
import csv
from collections import defaultdict

def module_param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters(recurse=False))

def breakdown_params_by_type(model: nn.Module):
    type2params = defaultdict(int)
    name2params = {}

    for name, m in model.named_modules():
        # count only params directly owned by this module (avoid double counting)
        n = module_param_count(m)
        if n > 0:
            t = type(m).__name__
            type2params[t] += n
            name2params[name] = n

    total = sum(p.numel() for p in model.parameters())
    return total, dict(type2params), name2params



def leaf_modules_with_params(model: nn.Module):
    """
    Returns (name, module) pairs where:
      - module has params directly (recurse=False)
      - AND no child module has params directly
    This avoids container modules like DinoVisionTransformer / Pi3 showing up.
    """
    # first mark which modules have direct params
    direct_params = {}
    for name, m in model.named_modules():
        n = sum(p.numel() for p in m.parameters(recurse=False))
        direct_params[name] = n

    # then keep only those with params and no param-owning children
    leaf = []
    for name, m in model.named_modules():
        if direct_params[name] == 0:
            continue
        has_child_params = False
        prefix = name + "." if name else ""
        for child_name, child_n in direct_params.items():
            if child_n == 0:
                continue
            if child_name.startswith(prefix) and child_name != name:
                has_child_params = True
                break
        if not has_child_params:
            leaf.append((name, m))
    return leaf

def params_by_type_leaf_only(model: nn.Module):
    type2params = defaultdict(int)
    total = sum(p.numel() for p in model.parameters())

    leaf = leaf_modules_with_params(model)
    for _, m in leaf:
        n = sum(p.numel() for p in m.parameters(recurse=False))
        type2params[type(m).__name__] += n

    rows = sorted(type2params.items(), key=lambda x: x[1], reverse=True)
    rows = [(t, n, 100.0*n/total) for t, n in rows]
    return total, rows

def write_param_reports(model: nn.Module, out_dir=".", top_k_modules=30):
    os.makedirs(out_dir, exist_ok=True)
    type_csv = os.path.join(out_dir, "params_by_type.csv")
    total, type_rows = params_by_type_leaf_only(model)
    with open(type_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ModuleType", "Params", "Percent"])
        for t, n, p in type_rows:
            w.writerow([t, n, p])
    print(f"✅ wrote: {type_csv}")

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='/data/wanghaoxuan/yusen_stuff/examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--ckpt", type=str, default='/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/pi3_model.safetensors',
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    args = parser.parse_args()


    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)


    if "pi3_model" in args.ckpt.lower():
        if args.ckpt is not None:
            model = Pi3().to(device).eval()
            if args.ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(args.ckpt)
            else:
                weight = torch.load(args.ckpt, map_location=device, weights_only=False)
            
            model.load_state_dict(weight)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        write_param_reports(model, out_dir="profile_reports_pi3", top_k_modules=40)
    elif "vggt_model" in args.ckpt.lower():
        model = VGGT().to(device).eval()
        from safetensors.torch import load_file
        weight = load_file(args.ckpt)
        model.load_state_dict(weight, strict=True)
        write_param_reports(model, out_dir="profile_reports_vggt", top_k_modules=40)
    else:
        raise ValueError(f"Unknown model type in checkpoint path: {args.ckpt}")