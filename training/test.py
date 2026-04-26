"""
test.py
-------
Evaluate trained models on the test set.

Key fixes applied vs previous versions
---------------------------------------
1. Quaternion sign flip fix:
   All predictions have qw = -0.999 (antipodal quaternion).
   Before dead-reckoning, we flip any prediction where qw < 0
   to its equivalent positive-qw form. Single-step error is identical
   (q and -q represent the same rotation) but dead-reckoning over 999
   steps was accumulating wrong rotation compositions.

2. No unnormalize_translation call:
   dataloaders.py v6+ returns raw meter labels. No normalization to undo.

3. Beautiful trajectory plots:
   - Consistent color scheme (GT=blue, Pred=red/orange)
   - Per-scene ATE drift plot
   - Start/end markers
   - Equal aspect ratio for top view

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import VisionOnlyDataset, IMUOnlyDataset, VisualInertialDataset

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "evaluation")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")


# ============================================================
# Geometry utilities
# ============================================================

def normalize_quaternion(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else np.array([0.0, 0.0, 0.0, 1.0])


def canonical_quaternion(q):
    """
    Ensure qw >= 0 (canonical form).
    q and -q represent the same rotation. Dead-reckoning via R = quat_to_rotmat
    gives the same R for both, but if we accumulate via quaternion multiplication
    the sign must be consistent. Enforcing qw >= 0 everywhere removes the ambiguity.
    """
    q = normalize_quaternion(q)
    if q[3] < 0:
        q = -q
    return q


def quat_to_rotmat(q):
    qx, qy, qz, qw = canonical_quaternion(q)
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ])
    return R


def fix_quaternion_signs(preds):
    """
    Fix the antipodal quaternion problem in batch predictions.

    All predictions consistently have qw = -0.999... because the network
    converged to the antipodal hemisphere. This is geometrically equivalent
    for single-step error, but dead-reckoning R_composed = R1 @ R2 @ ... requires
    consistent sign convention.

    Flip any prediction where qw < 0 to -q (same rotation, positive qw).
    """
    preds = preds.copy()
    mask = preds[:, 6] < 0      # qw is index 6
    preds[mask, 3:7] *= -1
    return preds


# ============================================================
# Dead-reckoning
# ============================================================

def dead_reckon(pred_rel_poses, gt_poses_csv):
    """
    Compose predicted relative poses into an absolute trajectory.

    Labels are in the LOCAL frame of pose_k:
        t_world = R_current @ t_rel
        pos_new = pos + t_world
        R_new   = R_current @ R_rel

    The first pose is anchored to ground truth.
    """
    gt_all = []
    with open(gt_poses_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_all.append({
                "position": np.array([float(row["px"]), float(row["py"]), float(row["pz"])]),
                "quaternion": np.array([float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])]),
            })

    frame_step = 10
    num_frames = len(pred_rel_poses) + 1

    gt_positions = []
    for k in range(num_frames):
        idx = k * frame_step
        if idx < len(gt_all):
            gt_positions.append(gt_all[idx]["position"])
    gt_positions = np.array(gt_positions)

    # Start from GT first pose
    pred_positions = [gt_positions[0].copy()]
    current_pos = gt_positions[0].copy()
    current_R = quat_to_rotmat(gt_all[0]["quaternion"])

    for rel in pred_rel_poses:
        t_rel = np.array(rel[:3])
        q_rel = np.array(rel[3:])
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


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def get_predictions(model, loader, device, mode):
    model.eval()
    all_preds, all_labels = [], []

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
            img, imu = img.to(device), imu.to(device)
            pred = model(img, imu)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# Plotting — beautiful trajectories
# ============================================================

GT_COLOR = "#2563EB"       # blue
PRED_COLOR = "#DC2626"     # red
START_COLOR = "#16A34A"    # green
END_COLOR = "#7C3AED"      # purple
ALPHA_GRID = 0.2


def plot_trajectory(pred_pos, gt_pos, title, save_path):
    """
    3-panel trajectory plot: Top (XY), Side (XZ), and 3D.
    Colors consistent across panels, equal aspect on top view.
    """
    fig = plt.figure(figsize=(22, 7), facecolor='white')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    # --- Top view (XY) ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], color=GT_COLOR, linewidth=2.0,
             label='Ground Truth', zorder=2)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], color=PRED_COLOR, linewidth=1.5,
             linestyle='--', label='Predicted', zorder=3)
    ax1.plot(gt_pos[0, 0], gt_pos[0, 1], 'o', color=START_COLOR,
             markersize=10, label='Start', zorder=5)
    ax1.plot(gt_pos[-1, 0], gt_pos[-1, 1], 's', color=END_COLOR,
             markersize=8, label='End', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('Top view (X–Y)', fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=ALPHA_GRID)
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Side view (XZ) ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 2], color=GT_COLOR, linewidth=2.0, label='GT')
    ax2.plot(pred_pos[:, 0], pred_pos[:, 2], color=PRED_COLOR, linewidth=1.5,
             linestyle='--', label='Pred')
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Z (m)', fontsize=11)
    ax2.set_title('Side view (X–Z)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=ALPHA_GRID)
    ax2.spines[['top', 'right']].set_visible(False)

    # --- 3D ---
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
             color=GT_COLOR, linewidth=2.0, label='GT')
    ax3.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
             color=PRED_COLOR, linewidth=1.5, linestyle='--', label='Pred')
    ax3.scatter(*gt_pos[0], color=START_COLOR, s=80, zorder=5)
    ax3.scatter(*gt_pos[-1], color=END_COLOR, s=60, marker='s', zorder=5)
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('Z', fontsize=10)
    ax3.set_title('3D view', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=ALPHA_GRID)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_drift(ate_errors, title, save_path):
    """ATE over time — shows where drift accumulates."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    frames = np.arange(len(ate_errors))
    ax.fill_between(frames, ate_errors, alpha=0.15, color=PRED_COLOR)
    ax.plot(frames, ate_errors, color=PRED_COLOR, linewidth=1.5)
    ax.axhline(np.mean(ate_errors), color='gray', linewidth=1.0, linestyle='--',
               label=f'Mean {np.mean(ate_errors):.3f} m')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('ATE (m)', fontsize=11)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=ALPHA_GRID)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(results, save_path):
    """Grouped bar chart of ATE per scene per model."""
    modes = list(results.keys())
    # Collect all scenes across all modes
    all_scenes = []
    for r in results.values():
        for sc in r.keys():
            if sc not in all_scenes:
                all_scenes.append(sc)

    colors = {
        "vision": "#2563EB",
        "imu": "#DC2626",
        "visual_inertial": "#16A34A",
    }
    x = np.arange(len(all_scenes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(all_scenes) * 3), 5), facecolor='white')
    for i, mode in enumerate(modes):
        vals = [results[mode].get(sc, {}).get("ate_rmse", 0.0) for sc in all_scenes]
        bars = ax.bar(x + i * width, vals, width, label=mode,
                      color=colors.get(mode, "#888"), alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Scene', fontsize=11)
    ax.set_ylabel('ATE RMSE (m)', fontsize=11)
    ax.set_title('Model Comparison — ATE RMSE on Test Set', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace('scene_', '') for s in all_scenes], rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=ALPHA_GRID, axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Evaluation
# ============================================================

def evaluate_mode(mode, device, run_dir):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode}")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{mode}_best.pth")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return None, {}

    if mode == "vision":
        model = VisionNet().to(device)
        test_ds = VisionOnlyDataset("test")
    elif mode == "imu":
        model = IMUNet().to(device)
        test_ds = IMUOnlyDataset("test")
    elif mode == "visual_inertial":
        model = VisualInertialNet().to(device)
        test_ds = VisualInertialDataset("test")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded model from epoch {ckpt['epoch']}")

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    print(f"  Test samples: {len(test_ds)}")

    # Inference
    preds, labels = get_predictions(model, test_loader, device, mode)

    # FIX: canonicalize quaternion signs before dead-reckoning
    preds = fix_quaternion_signs(preds)
    # Labels are already stored with consistent qw signs from poses.csv

    # Per-sample errors (raw meters, no normalization in v6 dataloaders)
    trans_errors = np.linalg.norm(preds[:, :3] - labels[:, :3], axis=1)

    q_pred = preds[:, 3:] / (np.linalg.norm(preds[:, 3:], axis=1, keepdims=True) + 1e-8)
    q_gt = labels[:, 3:] / (np.linalg.norm(labels[:, 3:], axis=1, keepdims=True) + 1e-8)
    dot = np.clip(np.abs(np.sum(q_pred * q_gt, axis=1)), 0.0, 1.0)
    rot_errors_deg = 2.0 * np.degrees(np.arccos(dot))

    print(f"\n  Per-sample errors:")
    print(f"    Translation RMSE: {np.sqrt(np.mean(trans_errors**2)):.6f} m")
    print(f"    Translation Mean: {np.mean(trans_errors):.6f} m")
    print(f"    Rotation Mean:    {np.mean(rot_errors_deg):.4f} deg")
    print(f"    Rotation Max:     {np.max(rot_errors_deg):.4f} deg")

    # Per-scene trajectory evaluation
    test_samples = test_ds.samples
    test_scenes = []
    seen = set()
    for s in test_samples:
        if s["scene"] not in seen:
            test_scenes.append(s["scene"])
            seen.add(s["scene"])
    print(f"  Test scenes: {test_scenes}")

    scene_results = {}
    offset = 0

    for scene in test_scenes:
        scene_count = sum(1 for s in test_samples if s["scene"] == scene)
        scene_preds = preds[offset:offset + scene_count]
        scene_labels = labels[offset:offset + scene_count]
        offset += scene_count

        gt_poses_csv = os.path.join(DATASETS_DIR, scene, "poses.csv")
        pred_pos, gt_pos = dead_reckon(scene_preds.tolist(), gt_poses_csv)
        ate_rmse, ate_errors = compute_ate_rmse(pred_pos, gt_pos)

        scene_t_errors = np.linalg.norm(scene_preds[:, :3] - scene_labels[:, :3], axis=1)
        scene_trans_rmse = np.sqrt(np.mean(scene_t_errors ** 2))

        print(f"\n  Scene: {scene}")
        print(f"    Samples:     {scene_count}")
        print(f"    Trans RMSE:  {scene_trans_rmse:.6f} m")
        print(f"    ATE RMSE:    {ate_rmse:.4f} m")
        print(f"    ATE Max:     {np.max(ate_errors):.4f} m")

        scene_results[scene] = {
            "ate_rmse": ate_rmse,
            "ate_max": float(np.max(ate_errors)),
            "ate_mean": float(np.mean(ate_errors)),
            "trans_rmse": scene_trans_rmse,
            "samples": scene_count,
        }

        # Trajectory plot
        plot_path = os.path.join(run_dir, f"{mode}_{scene}_trajectory.png")
        plot_trajectory(pred_pos, gt_pos,
                        f"{mode.replace('_', ' ').title()} — {scene}", plot_path)
        print(f"    Plot saved: {plot_path}")

        # Error drift plot
        err_path = os.path.join(run_dir, f"{mode}_{scene}_error_drift.png")
        plot_error_drift(ate_errors,
                         f"{mode.replace('_', ' ').title()} — {scene} — ATE Drift",
                         err_path)

    # Overall
    overall_ate = np.mean([r["ate_rmse"] for r in scene_results.values()])

    # Save CSV
    results_path = os.path.join(run_dir, f"{mode}_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "samples", "trans_rmse", "ate_rmse", "ate_max", "ate_mean"])
        for sc, r in scene_results.items():
            writer.writerow([sc, r["samples"], f"{r['trans_rmse']:.6f}",
                             f"{r['ate_rmse']:.6f}", f"{r['ate_max']:.6f}", f"{r['ate_mean']:.6f}"])
        writer.writerow(["OVERALL", "", "", f"{overall_ate:.6f}", "", ""])
    print(f"\n  Results saved: {results_path}")

    return overall_ate, scene_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None,
                        choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EVAL_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results dir: {run_dir}")

    if args.all:
        all_ates = {}
        all_scene_results = {}
        for mode in ["vision", "imu", "visual_inertial"]:
            ate, sc_res = evaluate_mode(mode, device, run_dir)
            if ate is not None:
                all_ates[mode] = ate
                all_scene_results[mode] = sc_res

        if all_ates:
            print(f"\n{'='*60}")
            print(f"  COMPARISON SUMMARY")
            print(f"{'='*60}")
            for mode, ate in all_ates.items():
                print(f"  {mode:>20s}:  ATE RMSE = {ate:.4f} m")

            # Save comparison CSV
            comp_path = os.path.join(run_dir, "comparison.csv")
            with open(comp_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["mode", "overall_ate_rmse"])
                for mode, ate in all_ates.items():
                    writer.writerow([mode, f"{ate:.6f}"])
            print(f"\n  Comparison saved: {comp_path}")

            # Comparison bar chart (per scene per model)
            plot_comparison_bar(
                all_scene_results,
                os.path.join(run_dir, "comparison_bar.png")
            )

    elif args.mode:
        evaluate_mode(args.mode, device, run_dir)
    else:
        print("Specify --mode or --all")


if __name__ == "__main__":
    main()