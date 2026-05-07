"""
test.py — Precomputed rotation version
----------------------------------------
Vision:  network predicts 7D → dead-reckon with predicted rotation
IMU/VI:  network predicts 3D translation → dead-reckon with gyro rotation

This eliminates rotation prediction error for IMU and VI models.
The only source of trajectory error is translation prediction accuracy.
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
from mpl_toolkits.mplot3d import Axes3D

from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import (VisionOnlyDataset, IMUOnlyDataset, VisualInertialDataset,
                         load_split_csv)

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "evaluation")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")


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


def dead_reckon(pred_translations, rotations, gt_poses_csv):
    """
    Dead-reckon using predicted translations and given rotations.

    For vision: rotations come from network predictions
    For IMU/VI: rotations come from precomputed gyro integration

    pred_translations: (N, 3) array of [tx, ty, tz]
    rotations: (N, 4) array of [qx, qy, qz, qw]
    """
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

    for i in range(len(pred_translations)):
        t_rel = pred_translations[i]
        q_rel = rotations[i]
        R_rel = quat_to_rotmat(q_rel)

        t_world = current_R @ t_rel
        current_pos = current_pos + t_world
        current_R = current_R @ R_rel

        pred_positions.append(current_pos.copy())

    return np.array(pred_positions), gt_positions


def compute_ate_rmse(pred_pos, gt_pos):
    n = min(len(pred_pos), len(gt_pos))
    errors = np.linalg.norm(pred_pos[:n] - gt_pos[:n], axis=1)
    return np.sqrt(np.mean(errors ** 2)), errors


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

    result = {
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }
    if all_gyro_qs:
        result["gyro_qs"] = np.concatenate(all_gyro_qs)
        result["gt_qs"] = np.concatenate(all_gt_qs)
    return result

# Plotting (same as before)

GT_COLOR = "#2563EB"
PRED_COLOR = "#DC2626"
START_COLOR = "#16A34A"
END_COLOR = "#7C3AED"


def plot_trajectory(pred_pos, gt_pos, title, save_path):
    fig = plt.figure(figsize=(22, 7), facecolor='white')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], color=GT_COLOR, linewidth=2.0, label='Ground Truth')
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], color=PRED_COLOR, linewidth=1.5, linestyle='--', label='Predicted')
    ax1.plot(gt_pos[0, 0], gt_pos[0, 1], 'o', color=START_COLOR, markersize=10, label='Start')
    ax1.plot(gt_pos[-1, 0], gt_pos[-1, 1], 's', color=END_COLOR, markersize=8, label='End')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top view (X–Y)', fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 2], color=GT_COLOR, linewidth=2.0, label='GT')
    ax2.plot(pred_pos[:, 0], pred_pos[:, 2], color=PRED_COLOR, linewidth=1.5, linestyle='--', label='Pred')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side view (X–Z)', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], color=GT_COLOR, linewidth=2.0, label='GT')
    ax3.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], color=PRED_COLOR, linewidth=1.5, linestyle='--', label='Pred')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3D view', fontweight='bold'); ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_drift(errors, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.fill_between(range(len(errors)), errors, alpha=0.15, color=PRED_COLOR)
    ax.plot(errors, color=PRED_COLOR, linewidth=1.5)
    ax.axhline(np.mean(errors), color='gray', linestyle='--', label=f'Mean {np.mean(errors):.3f}m')
    ax.set_xlabel('Frame'); ax.set_ylabel('ATE (m)')
    ax.set_title(title, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Evaluation

def evaluate_mode(mode, device, run_dir, split="test"):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {mode} on {split.upper()}")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, f"{mode}_best.pth")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return None, {}

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

    preds = result["preds"]      # (N, 7) for vision, (N, 3) for IMU/VI
    labels = result["labels"]    # (N, 7) for vision, (N, 3) for IMU/VI

    # Translation error
    if mode == "vision":
        t_preds = preds[:, :3]
        t_labels = labels[:, :3]
    else:
        t_preds = preds
        t_labels = labels

    trans_errors = np.linalg.norm(t_preds - t_labels, axis=1)
    print(f"\n  Per-sample translation RMSE: {np.sqrt(np.mean(trans_errors**2)):.6f} m")

    # Per-scene evaluation
    samples = ds.samples
    scenes = list(dict.fromkeys(s["scene"] for s in samples))
    print(f"  Scenes: {scenes}")

    scene_results = {}
    offset = 0

    for scene in scenes:
        scene_count = sum(1 for s in samples if s["scene"] == scene)

        if mode == "vision":
            scene_t = preds[offset:offset + scene_count, :3]
            scene_rot = preds[offset:offset + scene_count, 3:]
            # Canonicalize vision rotations
            mask = scene_rot[:, 3] < 0
            scene_rot[mask] *= -1
        else:
            scene_t = preds[offset:offset + scene_count]
            scene_rot = result["gyro_qs"][offset:offset + scene_count]

        gt_csv = os.path.join(DATASETS_DIR, scene, "poses.csv")
        pred_pos, gt_pos = dead_reckon(scene_t, scene_rot, gt_csv)
        ate_rmse, ate_errors = compute_ate_rmse(pred_pos, gt_pos)

        print(f"\n  Scene: {scene}")
        print(f"    Samples:  {scene_count}")
        print(f"    ATE RMSE: {ate_rmse:.4f} m")
        print(f"    ATE Max:  {np.max(ate_errors):.4f} m")

        scene_results[scene] = {"ate_rmse": ate_rmse, "ate_max": float(np.max(ate_errors))}

        # Include split name in filenames so train/val/test don't overwrite each other
        prefix = f"{split}_{mode}" if split != "test" else mode

        plot_trajectory(pred_pos, gt_pos,
                        f"{mode.replace('_',' ').title()} — {scene} ({split})",
                        os.path.join(run_dir, f"{prefix}_{scene}_trajectory.png"))
        plot_error_drift(ate_errors,
                         f"{mode.replace('_',' ').title()} — {scene} — ATE Drift ({split})",
                         os.path.join(run_dir, f"{prefix}_{scene}_error_drift.png"))

        offset += scene_count

    overall_ate = np.mean([r["ate_rmse"] for r in scene_results.values()])
    print(f"\n  Overall ATE ({split}): {overall_ate:.4f} m")

    # Save results CSV
    csv_path = os.path.join(run_dir, f"{prefix}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "ate_rmse", "ate_max"])
        for sc, r in scene_results.items():
            writer.writerow([sc, f"{r['ate_rmse']:.6f}", f"{r['ate_max']:.6f}"])
        writer.writerow(["OVERALL", f"{overall_ate:.6f}", ""])
    print(f"  Results saved: {csv_path}")

    return overall_ate, scene_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        choices=["train", "val", "test"],
                        help="Which splits to evaluate (default: test only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EVAL_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Device: {device}\nResults: {run_dir}")

    modes = ["vision", "imu", "visual_inertial"] if args.all else ([args.mode] if args.mode else [])
    if not modes:
        print("Specify --mode or --all")
        return

    all_results = {}  # {(mode, split): ate}

    for mode in modes:
        for split in args.splits:
            ate, _ = evaluate_mode(mode, device, run_dir, split)
            if ate is not None:
                all_results[(mode, split)] = ate

    if all_results:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")

        # Group by split
        for split in args.splits:
            split_results = {m: a for (m, s), a in all_results.items() if s == split}
            if split_results:
                print(f"\n  [{split.upper()}]")
                for m, a in split_results.items():
                    print(f"    {m:>20s}: {a:.4f} m")

        # Save comparison CSV
        comp_path = os.path.join(run_dir, "comparison.csv")
        with open(comp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "split", "ate_rmse"])
            for (m, s), a in sorted(all_results.items()):
                writer.writerow([m, s, f"{a:.6f}"])
        print(f"\n  Comparison saved: {comp_path}")


if __name__ == "__main__":
    main()