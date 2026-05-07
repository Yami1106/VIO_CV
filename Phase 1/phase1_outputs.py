import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix.
def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R

# Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w].
def rot_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)

# Extract all numeric values from a text block containing a bracketed vector.
def parse_vector_block(text):
    text = text.replace("\n", " ")
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    return np.array([float(x) for x in nums], dtype=float)

# Parse the MSCKF log file to extract estimated poses (timestamp, position, orientation).
def parse_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    est = []
    i = 0
    while i < len(lines):
        if "+++publish:" in lines[i]:
            t = None
            q = None
            p = None
            j = i + 1
            while j < len(lines) and "+++publish:" not in lines[j]:
                line = lines[j].strip()
                if line.startswith("timestamp:"):
                    t = float(line.split("timestamp:")[1].strip())
                elif line.startswith("orientation:"):
                    block = line.split("orientation:")[1]
                    while "]" not in block and j + 1 < len(lines):
                        j += 1
                        block += " " + lines[j].strip()
                    q = parse_vector_block(block)
                elif line.startswith("position:"):
                    block = line.split("position:")[1]
                    while "]" not in block and j + 1 < len(lines):
                        j += 1
                        block += " " + lines[j].strip()
                    p = parse_vector_block(block)
                j += 1

            if t is not None and q is not None and p is not None and len(q) == 4 and len(p) == 3:
                est.append([t, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
            i = j
        else:
            i += 1

    est = np.array(est, dtype=float)
    if len(est) == 0:
        raise RuntimeError("No estimates were parsed from the log.")
    return est

# Load EuRoC ground truth CSV and return poses as [timestamp, px, py, pz, qx, qy, qz, qw].
def load_euroc_gt(dataset_path):
    gt_path = f"{dataset_path}/mav0/state_groundtruth_estimate0/data.csv"
    df = pd.read_csv(gt_path)
    ts = df.iloc[:, 0].to_numpy(dtype=float) * 1e-9
    px = df.iloc[:, 1].to_numpy(dtype=float)
    py = df.iloc[:, 2].to_numpy(dtype=float)
    pz = df.iloc[:, 3].to_numpy(dtype=float)
    qw = df.iloc[:, 4].to_numpy(dtype=float)
    qx = df.iloc[:, 5].to_numpy(dtype=float)
    qy = df.iloc[:, 6].to_numpy(dtype=float)
    qz = df.iloc[:, 7].to_numpy(dtype=float)
    gt = np.column_stack([ts, px, py, pz, qx, qy, qz, qw])
    return gt

# Match estimated and ground truth poses by closest timestamp within max_dt tolerance.
def time_match(est, gt, max_dt=0.01):
    gt_ts = gt[:, 0]
    matched_est = []
    matched_gt = []

    for e in est:
        t = e[0]
        idx = np.argmin(np.abs(gt_ts - t))
        if abs(gt_ts[idx] - t) <= max_dt:
            matched_est.append(e)
            matched_gt.append(gt[idx])

    matched_est = np.array(matched_est, dtype=float)
    matched_gt = np.array(matched_gt, dtype=float)

    if len(matched_est) < 5:
        raise RuntimeError("Too few matched timestamps between estimate and ground truth.")
    return matched_est, matched_gt

# Compute SE(3) alignment (Umeyama) between source and destination 3D point sets.
def se3_align(src_pts, dst_pts):
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    return R, t

# Apply a rigid SE(3) transform (R, t) to a set of 3D points.
def apply_se3(R, t, pts):
    return (R @ pts.T).T + t

# Compute ATE metrics (RMSE, MAE, median, max, final drift) between aligned estimate and ground truth.
def compute_metrics(est_aligned, gt):
    errors = np.linalg.norm(est_aligned - gt, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    med = np.median(errors)
    mx = np.max(errors)
    final_drift = np.linalg.norm(est_aligned[-1] - gt[-1])
    return {
        "rmse_ate": rmse,
        "mae": mae,
        "median": med,
        "max": mx,
        "final_drift": final_drift,
        "all_errors": errors
    }

# Save pose data to a TUM-format text file (timestamp tx ty tz qx qy qz qw).
def save_tum(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(f"{row[0]:.9f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                    f"{row[4]:.8f} {row[5]:.8f} {row[6]:.8f} {row[7]:.8f}\n")

# Plot XY and XZ trajectory comparisons between aligned estimate and ground truth.
def plot_trajectories(est_xyz, gt_xyz, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(est_xyz[:, 0], est_xyz[:, 1], color="blue", linewidth=2, label="Estimate")
    axes[0].plot(gt_xyz[:, 0], gt_xyz[:, 1], color="deeppink", linewidth=2, label="Ground truth")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("XY trajectory")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(est_xyz[:, 0], est_xyz[:, 2], color="blue", linewidth=2, label="Estimate")
    axes[1].plot(gt_xyz[:, 0], gt_xyz[:, 2], color="deeppink", linewidth=2, label="Ground truth")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("XZ trajectory")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# Create an animated MP4 showing the estimate and ground truth trajectories being drawn over time.
def make_animation(est_xyz, gt_xyz, out_mp4):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Phase 1 trajectory visualization")
    ax.grid(True)

    all_x = np.concatenate([est_xyz[:, 0], gt_xyz[:, 0]])
    all_y = np.concatenate([est_xyz[:, 1], gt_xyz[:, 1]])
    pad_x = 0.5 * max(1.0, np.max(all_x) - np.min(all_x))
    pad_y = 0.5 * max(1.0, np.max(all_y) - np.min(all_y))
    ax.set_xlim(np.min(all_x) - 0.1 * pad_x, np.max(all_x) + 0.1 * pad_x)
    ax.set_ylim(np.min(all_y) - 0.1 * pad_y, np.max(all_y) + 0.1 * pad_y)

    est_line, = ax.plot([], [], color="blue", linewidth=2, label="Estimate")
    gt_line, = ax.plot([], [], color="deeppink", linewidth=2, label="Ground truth")
    ax.legend()

    n = min(len(est_xyz), len(gt_xyz))

    def update(frame):
        est_line.set_data(est_xyz[:frame, 0], est_xyz[:frame, 1])
        gt_line.set_data(gt_xyz[:frame, 0], gt_xyz[:frame, 1])
        return est_line, gt_line

    ani = FuncAnimation(fig, update, frames=n, interval=30, blit=True)
    ani.save(out_mp4, writer="ffmpeg", fps=30)
    plt.close()

# Main entry point: parse log, load ground truth, align, compute metrics, and save all outputs.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to phase1_run.log")
    parser.add_argument("--dataset", required=True, help="Path to EuRoC MH_01_easy")
    parser.add_argument("--outdir", default="phase1_outputs", help="Output directory")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    est = parse_log(args.log)
    gt = load_euroc_gt(args.dataset)

    est_m, gt_m = time_match(est, gt, max_dt=0.01)

    est_xyz = est_m[:, 1:4]
    gt_xyz = gt_m[:, 1:4]

    R_align, t_align = se3_align(est_xyz, gt_xyz)
    est_aligned_xyz = apply_se3(R_align, t_align, est_xyz)

    est_aligned = est_m.copy()
    est_aligned[:, 1:4] = est_aligned_xyz

    metrics = compute_metrics(est_aligned_xyz, gt_xyz)

    save_tum(f"{args.outdir}/estimate_raw_tum.txt", est_m)
    save_tum(f"{args.outdir}/groundtruth_matched_tum.txt", gt_m)
    save_tum(f"{args.outdir}/estimate_aligned_tum.txt", est_aligned)

    plot_trajectories(est_aligned_xyz, gt_xyz, f"{args.outdir}/phase1_trajectory_overlay.png")
    make_animation(est_aligned_xyz, gt_xyz, f"{args.outdir}/Output.mp4")

    with open(f"{args.outdir}/phase1_metrics.txt", "w") as f:
        f.write(f"RMSE_ATE: {metrics['rmse_ate']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"Median: {metrics['median']:.6f}\n")
        f.write(f"Max: {metrics['max']:.6f}\n")
        f.write(f"Final_Drift: {metrics['final_drift']:.6f}\n")

    print("Saved outputs to:", args.outdir)
    print(f"RMSE ATE      : {metrics['rmse_ate']:.6f} m")
    print(f"MAE           : {metrics['mae']:.6f} m")
    print(f"Median ATE    : {metrics['median']:.6f} m")
    print(f"Max ATE       : {metrics['max']:.6f} m")
    print(f"Final drift   : {metrics['final_drift']:.6f} m")

if __name__ == "__main__":
    main()