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


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='/data/wanghaoxuan/yusen_stuff/examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default='/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/pi3_model.safetensors',
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--efficiency_measure", type=str, default='simple', choices=['simple', 'profiler'],
                        help="Type of efficiency measurement to perform. Default: 'simple'")

    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
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
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    res = profiler_efficiency(model, imgs, dtype)
    build_profiler_plots(
        csv_path="profile.csv",
        top_k=5
    )

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # # 5. Save points
    # print(f"Saving point cloud to: {args.save_path}")
    # write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    # print("Done.")
