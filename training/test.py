"""
test.py
-------
Evaluate trained models on the test set.
Each run saves to a timestamped folder so previous results are preserved.

Usage:
    python test.py --mode vision
    python test.py --mode imu
    python test.py --mode visual_inertial
    python test.py --all
"""

import os
import csv
import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import VisionOnlyDataset, IMUOnlyDataset, VisualInertialDataset, load_split_csv

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "evaluation")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")


# ============================================================
# Compute physically plausible clamp limits from training data
# ============================================================

def compute_clamp_limits():
    """
    Compute per-axis translation and rotation clamp limits from training data.
    Uses max observed value * 1.5 as the clamp boundary.
    This prevents outlier predictions from causing catastrophic drift
    during dead-reckoning while allowing normal variation.
    """
    try:
        rows = load_split_csv("train")
        tx = np.array([float(r["tx"]) for r in rows])
        ty = np.array([float(r["ty"]) for r in rows])
        tz = np.array([float(r["tz"]) for r in rows])
        qx = np.array([float(r["qx"]) for r in rows])
        qy = np.array([float(r["qy"]) for r in rows])
        qz = np.array([float(r["qz"]) for r in rows])

        margin = 1.5  # allow 50% beyond observed max

        limits = {
            "tx": float(np.max(np.abs(tx)) * margin),
            "ty": float(np.max(np.abs(ty)) * margin),
            "tz": float(np.max(np.abs(tz)) * margin),
            "qx": float(np.max(np.abs(qx)) * margin),
            "qy": float(np.max(np.abs(qy)) * margin),
            "qz": float(np.max(np.abs(qz)) * margin),
        }
        print(f"  Clamp limits (from training data × {margin}):")
        print(f"    tx: ±{limits['tx']:.4f}m  ty: ±{limits['ty']:.4f}m  tz: ±{limits['tz']:.4f}m")
        print(f"    qx: ±{limits['qx']:.6f}  qy: ±{limits['qy']:.6f}  qz: ±{limits['qz']:.6f}")
        return limits
    except Exception as e:
        print(f"  Warning: Could not compute clamp limits: {e}")
        return None


CLAMP_LIMITS = compute_clamp_limits()


def normalize_quaternion(q):
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_to_rotmat(q):
    qx, qy, qz, qw = normalize_quaternion(q)
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotmat_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def dead_reckon(pred_rel_poses, gt_poses_csv):
    gt_all = []
    with open(gt_poses_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_all.append({
                "position": np.array([float(row["px"]), float(row["py"]), float(row["pz"])]),
                "quaternion": np.array([float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])])
            })

    frame_step = 10
    num_frames = len(pred_rel_poses) + 1
    gt_positions = []
    for k in range(num_frames):
        idx = k * frame_step
        if idx < len(gt_all):
            gt_positions.append(gt_all[idx]["position"])
    gt_positions = np.array(gt_positions)

    pred_positions = [gt_positions[0].copy()]
    current_pos = gt_positions[0].copy()
    current_q = gt_all[0]["quaternion"].copy()
    current_R = quat_to_rotmat(current_q)

    for rel in pred_rel_poses:
        t_rel = np.array(rel[:3])
        q_rel = np.array(rel[3:])

        # Canonicalize: ensure qw > 0 to prevent sign-flip drift
        if q_rel[3] < 0:
            q_rel = -q_rel

        # Clamp to physically plausible range from training data
        if CLAMP_LIMITS is not None:
            t_rel[0] = np.clip(t_rel[0], -CLAMP_LIMITS["tx"], CLAMP_LIMITS["tx"])
            t_rel[1] = np.clip(t_rel[1], -CLAMP_LIMITS["ty"], CLAMP_LIMITS["ty"])
            t_rel[2] = np.clip(t_rel[2], -CLAMP_LIMITS["tz"], CLAMP_LIMITS["tz"])
            q_rel[0] = np.clip(q_rel[0], -CLAMP_LIMITS["qx"], CLAMP_LIMITS["qx"])
            q_rel[1] = np.clip(q_rel[1], -CLAMP_LIMITS["qy"], CLAMP_LIMITS["qy"])
            q_rel[2] = np.clip(q_rel[2], -CLAMP_LIMITS["qz"], CLAMP_LIMITS["qz"])
            # Re-normalize quaternion after clamping
            q_rel = q_rel / (np.linalg.norm(q_rel) + 1e-8)

        R_rel = quat_to_rotmat(q_rel)

        t_world = current_R @ t_rel
        current_pos = current_pos + t_world
        current_R = current_R @ R_rel

        pred_positions.append(current_pos.copy())

    return np.array(pred_positions), gt_positions


def compute_ate_rmse(pred_positions, gt_positions):
    n = min(len(pred_positions), len(gt_positions))
    errors = np.linalg.norm(pred_positions[:n] - gt_positions[:n], axis=1)
    return np.sqrt(np.mean(errors ** 2)), errors


@torch.no_grad()
def get_predictions(model, loader, device, mode):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        if mode == "vision":
            x, label = batch
            x = x.to(device)
            pred = model(x)
        elif mode == "imu":
            x, label = batch
            x = x.to(device)
            pred = model(x)
        elif mode == "visual_inertial":
            img, imu, label = batch
            img = img.to(device)
            imu = imu.to(device)
            pred = model(img, imu)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_trajectory(pred_pos, gt_pos, title, save_path):
    fig = plt.figure(figsize=(20, 6))

    # Top view (X-Y)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=1.5, label='Predicted')
    ax1.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(gt_pos[-1, 0], gt_pos[-1, 1], 'ks', markersize=8, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (X-Y)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Side view (X-Z)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax2.plot(pred_pos[:, 0], pred_pos[:, 2], 'r--', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (X-Z)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='GT')
    ax3.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=1.5, label='Pred')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D View')
    ax3.legend(fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_over_time(ate_errors, title, save_path):
    """Plot how ATE grows over time (shows drift behavior)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ate_errors, 'r-', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position Error (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_mode(mode, device, run_dir):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode}")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{mode}_best.pth")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return None

    if mode == "vision":
        model = VisionNet().to(device)
        test_ds = VisionOnlyDataset("test")
    elif mode == "imu":
        model = IMUNet().to(device)
        test_ds = IMUOnlyDataset("test")
    elif mode == "visual_inertial":
        model = VisualInertialNet().to(device)
        test_ds = VisualInertialDataset("test")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded model from epoch {checkpoint['epoch']}")

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    print(f"  Test samples: {len(test_ds)}")

    # Get predictions
    preds, labels = get_predictions(model, test_loader, device, mode)

    # Per-sample errors
    trans_errors = np.linalg.norm(preds[:, :3] - labels[:, :3], axis=1)

    # Rotation error: geodesic angular distance (handles q/-q ambiguity)
    q_pred = preds[:, 3:]
    q_gt = labels[:, 3:]
    q_pred_norm = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
    q_gt_norm = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q_pred_norm * q_gt_norm, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    rot_errors_deg = 2.0 * np.arccos(dot) * 180.0 / np.pi

    print(f"\n  Per-sample errors:")
    print(f"    Translation RMSE: {np.sqrt(np.mean(trans_errors**2)):.6f} m")
    print(f"    Translation Mean: {np.mean(trans_errors):.6f} m")
    print(f"    Rotation Mean:    {np.mean(rot_errors_deg):.4f} deg")
    print(f"    Rotation Max:     {np.max(rot_errors_deg):.4f} deg")

    # Find all test scenes
    test_samples = test_ds.samples
    test_scenes = []
    seen = set()
    for s in test_samples:
        if s["scene"] not in seen:
            test_scenes.append(s["scene"])
            seen.add(s["scene"])

    print(f"  Test scenes: {test_scenes}")

    # Evaluate per scene
    all_results = {}
    offset = 0

    for scene in test_scenes:
        # Count samples for this scene
        scene_count = sum(1 for s in test_samples if s["scene"] == scene)
        scene_preds = preds[offset:offset + scene_count]
        offset += scene_count

        gt_poses_csv = os.path.join(DATASETS_DIR, scene, "poses.csv")
        pred_pos, gt_pos = dead_reckon(scene_preds.tolist(), gt_poses_csv)
        ate_rmse, ate_errors = compute_ate_rmse(pred_pos, gt_pos)

        scene_trans_errors = np.linalg.norm(scene_preds[:, :3] - labels[offset - scene_count:offset, :3], axis=1)

        print(f"\n  Scene: {scene}")
        print(f"    Samples:     {scene_count}")
        print(f"    Trans RMSE:  {np.sqrt(np.mean(scene_trans_errors**2)):.6f} m")
        print(f"    ATE RMSE:    {ate_rmse:.4f} m")
        print(f"    ATE Max:     {np.max(ate_errors):.4f} m")

        all_results[scene] = {
            "ate_rmse": ate_rmse,
            "ate_max": np.max(ate_errors),
            "ate_mean": np.mean(ate_errors),
            "trans_rmse": np.sqrt(np.mean(scene_trans_errors**2)),
            "samples": scene_count,
        }

        # Plot trajectory for this scene
        plot_path = os.path.join(run_dir, f"{mode}_{scene}_trajectory.png")
        plot_trajectory(pred_pos, gt_pos, f"{mode} — {scene}", plot_path)
        print(f"    Plot saved: {plot_path}")

        # Plot error over time
        err_path = os.path.join(run_dir, f"{mode}_{scene}_error_drift.png")
        plot_error_over_time(ate_errors, f"{mode} — {scene} — ATE Drift", err_path)

    # Overall ATE across all test scenes
    overall_ate = np.mean([r["ate_rmse"] for r in all_results.values()])

    # Save results CSV
    results_path = os.path.join(run_dir, f"{mode}_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "samples", "trans_rmse", "ate_rmse", "ate_max", "ate_mean"])
        for scene, r in all_results.items():
            writer.writerow([scene, r["samples"], f"{r['trans_rmse']:.6f}",
                           f"{r['ate_rmse']:.6f}", f"{r['ate_max']:.6f}", f"{r['ate_mean']:.6f}"])
        writer.writerow(["OVERALL", "", "", f"{overall_ate:.6f}", "", ""])
    print(f"\n  Results saved: {results_path}")

    return overall_ate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EVAL_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results dir: {run_dir}")

    if args.all:
        results = {}
        for mode in ["vision", "imu", "visual_inertial"]:
            ate = evaluate_mode(mode, device, run_dir)
            if ate is not None:
                results[mode] = ate

        if results:
            # Comparison summary
            print(f"\n{'='*60}")
            print(f"  COMPARISON SUMMARY")
            print(f"{'='*60}")
            for mode, ate in results.items():
                print(f"  {mode:>20s}:  ATE RMSE = {ate:.4f} m")

            # Save comparison
            comp_path = os.path.join(run_dir, "comparison.csv")
            with open(comp_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["mode", "overall_ate_rmse"])
                for mode, ate in results.items():
                    writer.writerow([mode, f"{ate:.6f}"])
            print(f"\n  Comparison saved: {comp_path}")

            # Comparison bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            modes = list(results.keys())
            ates = [results[m] for m in modes]
            colors = ['#4285f4', '#ea4335', '#34a853']
            bars = ax.bar(modes, ates, color=colors[:len(modes)], width=0.5)
            ax.set_ylabel('ATE RMSE (m)')
            ax.set_title('Model Comparison — ATE RMSE on Test Set')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, ates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}m', ha='center', fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "comparison_bar.png"), dpi=150, bbox_inches='tight')
            plt.close()

    elif args.mode:
        evaluate_mode(args.mode, device, run_dir)
    else:
        print("Specify --mode or --all")


if __name__ == "__main__":
    main()