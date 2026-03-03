import open3d as o3d
import numpy as np

def robust_aabb_crop(
    pcd: o3d.geometry.PointCloud,
    q_low=2.0, q_high=98.0,      # robust bounds on x/y
    z_low=1.0, z_high=99.5,      # robust bounds on z (often want to keep more)
    pad_ratio=0.08,              # enlarge box a bit
    max_points_for_stats=2_000_000,  # speed guard
):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    # subsample for stats if cloud is huge (keeps it fast)
    if len(pts) > max_points_for_stats:
        idx = np.random.choice(len(pts), size=max_points_for_stats, replace=False)
        pts_stat = pts[idx]
    else:
        pts_stat = pts

    # percentile bounds (robust to outliers / flying junk)
    x0, x1 = np.percentile(pts_stat[:, 0], [q_low, q_high])
    y0, y1 = np.percentile(pts_stat[:, 1], [q_low, q_high])
    z0, z1 = np.percentile(pts_stat[:, 2], [z_low, z_high])

    # pad the box a bit
    dx = (x1 - x0) * pad_ratio
    dy = (y1 - y0) * pad_ratio
    dz = (z1 - z0) * pad_ratio

    x0, x1 = x0 - dx, x1 + dx
    y0, y1 = y0 - dy, y1 + dy
    z0, z1 = z0 - dz, z1 + dz

    mask = (
        (pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
        (pts[:, 1] >= y0) & (pts[:, 1] <= y1) &
        (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
    )

    return pcd.select_by_index(np.where(mask)[0])

if __name__ == "__main__":
    path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/outputs_ada/mv_recon/NRGBD-dense/breakfast_room-pred.ply"
    pcd = o3d.io.read_point_cloud(path)

    # Optional: voxel downsample a tiny bit to stabilize stats (doesn't change geometry much)
    # pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # pcd_crop = robust_aabb_crop(
    #     pcd,
    #     q_low=1.0, q_high=99.0,
    #     z_low=0.5, z_high=99.7,
    #     pad_ratio=0.10
    # )
    
    pcd_crop = robust_aabb_crop(
        pcd,
        q_low=3.0,  q_high=97.0,
        z_low=1.0,  z_high=99.3,
        pad_ratio=0.05
    )

    o3d.io.write_point_cloud("cropped.ply", pcd_crop)
    print(f"saved -> cropped.ply  (kept {len(pcd_crop.points)}/{len(pcd.points)} points)")