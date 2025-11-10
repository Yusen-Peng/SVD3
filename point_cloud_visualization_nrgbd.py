import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def render_pcd(ax, ply_path, point_size=0.3, elev=20, azim=45):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
    if cols is not None and cols.shape[0] == pts.shape[0]:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=cols)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, color="gray")

    # zoom to bounding box
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2.0
    max_range = (maxs - mins).max() / 2.0
    ax.set_xlim(center[0]-max_range, center[0]+max_range)
    ax.set_ylim(center[1]-max_range, center[1]+max_range)
    ax.set_zlim(center[2]-max_range, center[2]+max_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

if __name__ == "__main__":
    MODE = 'sparse'  # 'dense' or 'sparse'

    pred_path = f"outputs/mv_recon/NRGBD-{MODE}/kitchen-pred.ply"
    gt_path = f"outputs/mv_recon/NRGBD-{MODE}/kitchen-gt.ply"

    fig = plt.figure(figsize=(6, 3))   # one row, two columns
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    render_pcd(ax1, gt_path)
    ax1.set_title("Ground Truth", fontsize=10)
    render_pcd(ax2, pred_path)
    ax2.set_title("Prediction", fontsize=10)

    plt.tight_layout()
    out_path = "kitchen-comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved side-by-side visualization → {out_path}")