"""
test_full_stack_vio.py
-----------------------
Post-processing optimization on top of dead-reckoned trajectories.

  1. Relative translation constraints from network predictions
  2. Loop closure constraint (trajectory is closed → first ≈ last pose)
  3. Smoothness prior (penalizes sudden jumps)

Usage:
    python test_full_stack_vio.py --all --splits test
    python test_full_stack_vio.py --all --splits train val test
"""

import os
import csv
import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.optimize import least_squares

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import (VisionOnlyDataset, IMUOnlyDataset, VisualInertialDataset,
                         load_split_csv)

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "evaluation")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")


# ============================================================
# Quaternion / rotation utilities
# ============================================================

def normalize_quaternion(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else np.array([0.0, 0.0, 0.0, 1.0])


def canonical_quaternion(q):
    q = normalize_quaternion(q)
    if q[3] < 0:
        q = -q
    return q


def quat_to_rotmat(q):
    qx, qy, qz, qw = canonical_quaternion(q)
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ])


# ============================================================
# Dead-reckoning (baseline, same as test.py)
# ============================================================

def dead_reckon(pred_translations, rotations, gt_poses_csv):
    gt_all = []
    with open(gt_poses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_all.append({
                "position": np.array([float(row["px"]), float(row["py"]), float(row["pz"])]),
                "quaternion": np.array([float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])]),
            })

    frame_step = 10
    num_frames = len(pred_translations) + 1
    gt_positions = []
    for k in range(num_frames):
        idx = k * frame_step
        if idx < len(gt_all):
            gt_positions.append(gt_all[idx]["position"])
    gt_positions = np.array(gt_positions)

    pred_positions = [gt_positions[0].copy()]
    current_pos = gt_positions[0].copy()
    current_R = quat_to_rotmat(gt_all[0]["quaternion"])

    rotations_world = [current_R.copy()]

    for i in range(len(pred_translations)):
        t_rel = pred_translations[i]
        q_rel = rotations[i]
        R_rel = quat_to_rotmat(q_rel)

        t_world = current_R @ t_rel
        current_pos = current_pos + t_world
        current_R = current_R @ R_rel

        pred_positions.append(current_pos.copy())
        rotations_world.append(current_R.copy())

    return np.array(pred_positions), gt_positions, rotations_world


# ============================================================
# Pose Graph Optimization
# ============================================================

def pose_graph_optimize(pred_translations, rotations, gt_poses_csv,
                        loop_closure_weight=1.0,
                        smoothness_weight=0.1,
                        relative_weight=1.0):
    """
    Optimize all N+1 positions jointly using least-squares.

    Variables: positions[1..N] as a flat vector (position[0] is fixed to GT start).

    Constraints:
    1. Relative: pos[i+1] - pos[i] ≈ R[i] @ t_rel[i]  (from network)
    2. Loop closure: pos[N] ≈ pos[0]  (closed trajectory)
    3. Smoothness: pos[i+1] - 2*pos[i] + pos[i-1] ≈ 0  (penalize jerk)
    """

    # First do normal dead-reckoning to get initial positions and rotations
    dr_positions, gt_positions, rotations_world = dead_reckon(
        pred_translations, rotations, gt_poses_csv
    )

    N = len(pred_translations)  # number of edges
    start_pos = dr_positions[0].copy()

    # Compute world-frame translations from predictions
    t_world_predictions = []
    for i in range(N):
        t_world = rotations_world[i] @ pred_translations[i]
        t_world_predictions.append(t_world)
    t_world_predictions = np.array(t_world_predictions)

    # Initial guess: dead-reckoned positions (excluding fixed start)
    x0 = dr_positions[1:].flatten()  # (N*3,)

    def residuals(x):
        positions = x.reshape(-1, 3)  # (N, 3) — positions[0] is pos[1], etc.
        res = []

        # --- Relative constraints ---
        # pos[1] - pos[0_fixed] ≈ t_world[0]
        r = relative_weight * (positions[0] - start_pos - t_world_predictions[0])
        res.extend(r)

        # pos[i+1] - pos[i] ≈ t_world[i]
        for i in range(1, N):
            r = relative_weight * (positions[i] - positions[i-1] - t_world_predictions[i])
            res.extend(r)

        # --- Loop closure ---
        # Last position should return to start
        r_loop = loop_closure_weight * (positions[-1] - start_pos)
        res.extend(r_loop)

        # --- Smoothness (second derivative penalty) ---
        # For i=1: pos[1] - 2*pos[0] + start
        r_smooth = smoothness_weight * (positions[1] - 2*positions[0] + start_pos)
        res.extend(r_smooth)

        for i in range(1, N-1):
            r_smooth = smoothness_weight * (positions[i+1] - 2*positions[i] + positions[i-1])
            res.extend(r_smooth)

        return np.array(res)

    # Solve
    result = least_squares(residuals, x0, method='trf', max_nfev=500, verbose=0)

    optimized = np.vstack([start_pos, result.x.reshape(-1, 3)])

    return optimized, dr_positions, gt_positions


def compute_ate_rmse(pred_pos, gt_pos):
    n = min(len(pred_pos), len(gt_pos))
    errors = np.linalg.norm(pred_pos[:n] - gt_pos[:n], axis=1)
    return np.sqrt(np.mean(errors ** 2)), errors


# ============================================================
# Predictions
# ============================================================

@torch.no_grad()
def get_predictions(model, loader, device, mode):
    model.eval()
    all_preds, all_labels = [], []
    all_gyro_qs, all_gt_qs = [], []

    for batch in loader:
        if mode == "vision":
            x, label = batch
            pred = model(x.to(device))
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.numpy())
        elif mode == "imu":
            x, label, gyro_q, gt_q = batch
            pred = model(x.to(device))
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.numpy())
            all_gyro_qs.append(gyro_q.numpy())
            all_gt_qs.append(gt_q.numpy())
        elif mode == "visual_inertial":
            img, imu, label, gyro_q, gt_q = batch
            pred = model(img.to(device), imu.to(device), gyro_q.to(device))
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.numpy())
            all_gyro_qs.append(gyro_q.numpy())
            all_gt_qs.append(gt_q.numpy())

    result = {"preds": np.concatenate(all_preds), "labels": np.concatenate(all_labels)}
    if all_gyro_qs:
        result["gyro_qs"] = np.concatenate(all_gyro_qs)
    return result


# ============================================================
# Plotting
# ============================================================

GT_COLOR = "#2563EB"
DR_COLOR = "#F59E0B"
OPT_COLOR = "#10B981"


def plot_trajectory_comparison(dr_pos, opt_pos, gt_pos, title, save_path):
    fig = plt.figure(figsize=(22, 7), facecolor='white')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], color=GT_COLOR, linewidth=2.0, label='Ground Truth')
    ax1.plot(dr_pos[:, 0], dr_pos[:, 1], color=DR_COLOR, linewidth=1.2, linestyle='--', label='Dead-Reckoned', alpha=0.7)
    ax1.plot(opt_pos[:, 0], opt_pos[:, 1], color=OPT_COLOR, linewidth=1.5, linestyle='-', label='Optimized')
    ax1.plot(gt_pos[0, 0], gt_pos[0, 1], 'ko', markersize=8, label='Start')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top view (X-Y)'); ax1.legend(fontsize=8); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 2], color=GT_COLOR, linewidth=2.0, label='GT')
    ax2.plot(dr_pos[:, 0], dr_pos[:, 2], color=DR_COLOR, linewidth=1.2, linestyle='--', label='DR', alpha=0.7)
    ax2.plot(opt_pos[:, 0], opt_pos[:, 2], color=OPT_COLOR, linewidth=1.5, label='Opt')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side view (X-Z)'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], color=GT_COLOR, linewidth=2.0, label='GT')
    ax3.plot(dr_pos[:, 0], dr_pos[:, 1], dr_pos[:, 2], color=DR_COLOR, linewidth=1.0, linestyle='--', label='DR', alpha=0.6)
    ax3.plot(opt_pos[:, 0], opt_pos[:, 1], opt_pos[:, 2], color=OPT_COLOR, linewidth=1.5, label='Opt')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z'); ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_drift(dr_errors, opt_errors, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.fill_between(range(len(dr_errors)), dr_errors, alpha=0.1, color=DR_COLOR)
    ax.plot(dr_errors, color=DR_COLOR, linewidth=1.2, label=f'Dead-Reckoned (mean {np.mean(dr_errors):.3f}m)')
    ax.fill_between(range(len(opt_errors)), opt_errors, alpha=0.15, color=OPT_COLOR)
    ax.plot(opt_errors, color=OPT_COLOR, linewidth=1.5, label=f'Optimized (mean {np.mean(opt_errors):.3f}m)')
    ax.set_xlabel('Frame'); ax.set_ylabel('ATE (m)')
    ax.set_title(title, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(all_results, save_path):
    """Generate comparison bar chart from results dict."""
    modes = ["vision", "imu", "visual_inertial"]
    mode_labels = ["Vision", "IMU", "Visual-Inertial"]

    # Collect data per split
    splits_found = sorted(set(s for (_, s, _) in all_results.keys()))
    methods = ["dr", "opt"]
    method_labels = ["Dead-Reckoned", "Optimized"]
    method_colors = [DR_COLOR, OPT_COLOR]

    fig, axes = plt.subplots(1, len(splits_found), figsize=(6*len(splits_found), 6), facecolor='white')
    if len(splits_found) == 1:
        axes = [axes]

    for ax_idx, split in enumerate(splits_found):
        ax = axes[ax_idx]
        x = np.arange(len(modes))
        w = 0.35

        for j, method in enumerate(methods):
            vals = []
            for mode in modes:
                key = (mode, split, method)
                vals.append(all_results.get(key, 0))
            bars = ax.bar(x + (j - 0.5)*w, vals, w, label=method_labels[j],
                         color=method_colors[j], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

        ax.set_ylabel('ATE RMSE (m)', fontsize=11)
        ax.set_title(f'{split.upper()} Set', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Dead-Reckoning vs Pose Graph Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison chart saved: {save_path}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_mode(mode, device, run_dir, split="test"):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode} on {split.upper()} (DR + Optimization)")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{mode}_best.pth")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return None, None

    if mode == "vision":
        model = VisionNet().to(device)
        ds = VisionOnlyDataset(split)
    elif mode == "imu":
        model = IMUNet().to(device)
        ds = IMUOnlyDataset(split)
    elif mode == "visual_inertial":
        model = VisualInertialNet().to(device)
        ds = VisualInertialDataset(split)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt['epoch']}")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    result = get_predictions(model, loader, device, mode)

    preds = result["preds"]
    samples = ds.samples
    scenes = list(dict.fromkeys(s["scene"] for s in samples))

    dr_ates, opt_ates = {}, {}
    offset = 0

    for scene in scenes:
        scene_count = sum(1 for s in samples if s["scene"] == scene)

        if mode == "vision":
            scene_t = preds[offset:offset + scene_count, :3]
            scene_rot = preds[offset:offset + scene_count, 3:].copy()
            mask = scene_rot[:, 3] < 0
            scene_rot[mask] *= -1
        else:
            scene_t = preds[offset:offset + scene_count]
            scene_rot = result["gyro_qs"][offset:offset + scene_count]

        gt_csv = os.path.join(DATASETS_DIR, scene, "poses.csv")

        # Dead-reckoning baseline
        dr_pos, gt_pos, _ = dead_reckon(scene_t, scene_rot, gt_csv)
        dr_ate, dr_errors = compute_ate_rmse(dr_pos, gt_pos)

        # Pose graph optimization
        opt_pos, _, _ = pose_graph_optimize(
            scene_t, scene_rot, gt_csv,
            loop_closure_weight=2.0,
            smoothness_weight=0.05,
            relative_weight=1.0,
        )
        opt_ate, opt_errors = compute_ate_rmse(opt_pos, gt_pos)

        improvement = (1 - opt_ate/dr_ate) * 100 if dr_ate > 0 else 0

        print(f"\n  Scene: {scene}")
        print(f"    DR  ATE: {dr_ate:.4f} m")
        print(f"    Opt ATE: {opt_ate:.4f} m  ({improvement:+.1f}%)")

        dr_ates[scene] = dr_ate
        opt_ates[scene] = opt_ate

        prefix = f"{split}_{mode}" if split != "test" else mode

        plot_trajectory_comparison(dr_pos, opt_pos, gt_pos,
            f"{mode.replace('_',' ').title()} — {scene} ({split})",
            os.path.join(run_dir, f"{prefix}_{scene}_trajectory.png"))

        plot_error_drift(dr_errors, opt_errors,
            f"{mode.replace('_',' ').title()} — {scene} — ATE Drift ({split})",
            os.path.join(run_dir, f"{prefix}_{scene}_error_drift.png"))

        offset += scene_count

    overall_dr = np.mean(list(dr_ates.values()))
    overall_opt = np.mean(list(opt_ates.values()))
    improvement = (1 - overall_opt/overall_dr) * 100

    print(f"\n  Overall DR:  {overall_dr:.4f} m")
    print(f"  Overall Opt: {overall_opt:.4f} m  ({improvement:+.1f}%)")

    # Save results CSV
    prefix = f"{split}_{mode}" if split != "test" else mode
    csv_path = os.path.join(run_dir, f"{prefix}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "dr_ate_rmse", "opt_ate_rmse", "improvement_pct"])
        for sc in dr_ates:
            imp = (1 - opt_ates[sc]/dr_ates[sc]) * 100
            writer.writerow([sc, f"{dr_ates[sc]:.6f}", f"{opt_ates[sc]:.6f}", f"{imp:.1f}"])
        writer.writerow(["OVERALL", f"{overall_dr:.6f}", f"{overall_opt:.6f}", f"{improvement:.1f}"])
    print(f"  Results saved: {csv_path}")

    return overall_dr, overall_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EVAL_DIR, f"run_{timestamp}_optimized")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Device: {device}\nResults: {run_dir}")

    modes = ["vision", "imu", "visual_inertial"] if args.all else ([args.mode] if args.mode else [])

    all_results = {}  # {(mode, split, method): ate}

    for mode in modes:
        for split in args.splits:
            dr_ate, opt_ate = evaluate_mode(mode, device, run_dir, split)
            if dr_ate is not None:
                all_results[(mode, split, "dr")] = dr_ate
                all_results[(mode, split, "opt")] = opt_ate

    if all_results:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")

        for split in args.splits:
            print(f"\n  [{split.upper()}]")
            print(f"  {'Model':<22s} {'Dead-Reckon':>12s} {'Optimized':>12s} {'Improve':>10s}")
            print(f"  {'-'*58}")
            for mode in modes:
                dr = all_results.get((mode, split, "dr"), 0)
                opt = all_results.get((mode, split, "opt"), 0)
                imp = (1 - opt/dr) * 100 if dr > 0 else 0
                print(f"  {mode:<22s} {dr:>10.4f} m {opt:>10.4f} m {imp:>+8.1f}%")

        # Save comparison CSV
        comp_path = os.path.join(run_dir, "comparison.csv")
        with open(comp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "split", "dr_ate_rmse", "opt_ate_rmse", "improvement_pct"])
            for (m, s, method), ate in sorted(all_results.items()):
                if method == "dr":
                    opt = all_results.get((m, s, "opt"), 0)
                    imp = (1 - opt/ate) * 100 if ate > 0 else 0
                    writer.writerow([m, s, f"{ate:.6f}", f"{opt:.6f}", f"{imp:.1f}"])
        print(f"\n  Comparison saved: {comp_path}")

        # Generate comparison bar chart
        plot_comparison_bar(all_results, os.path.join(run_dir, "comparison_bar.png"))


if __name__ == "__main__":
    main()