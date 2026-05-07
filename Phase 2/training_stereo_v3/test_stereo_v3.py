"""
test_stereo_v3.py — Evaluation for V3 (6D rotation) models
============================================================

All networks output: translation (3) + rotation_6d (6)
At test time: 6D → rotation matrix → quaternion → dead reckoning

Usage:
    python test_stereo_v3.py --all --splits test
    python test_stereo_v3.py --mode stereo_vi --splits test
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
from scipy import sparse
from scipy.sparse.linalg import spsolve

from networks_stereo_v3 import (
    StereoVisionNetV3, IMUNetV3, StereoVisualInertialNetV3,
    rotation_6d_to_matrix, matrix_to_quaternion
)
from dataloaders_stereo_v2 import (
    StereoVisionOnlyV2Preloaded, IMUOnlyV2Preloaded, StereoVIv2Preloaded,
    load_split_csv
)

MODELS_DIR = "/mnt/data/models"
EVAL_DIR = "/mnt/data/evaluation"
DATASETS_DIR = "/mnt/data/datasets"

MODE_TO_FILE = {
    "stereo_vision": "stereo_vision_v3_best.pth",
    "imu_only": "imu_only_v3_best.pth",
    "stereo_vi": "stereo_vi_v3_best.pth",
}


def normalize_quaternion(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else np.array([0.0, 0.0, 0.0, 1.0])

def quat_to_rotmat(q):
    qx, qy, qz, qw = normalize_quaternion(q)
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ])


def load_gt_poses(gt_poses_csv):
    gt_all = []
    with open(gt_poses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_all.append({
                "position": np.array([float(row["px"]), float(row["py"]), float(row["pz"])]),
                "quaternion": np.array([float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])]),
            })
    return gt_all


def dead_reckon(translations, quaternions, gt_poses_csv):
    """Dead reckon from per-step translations and quaternions."""
    gt_all = load_gt_poses(gt_poses_csv)
    frame_step = 10
    num_frames = len(translations) + 1

    gt_positions = []
    for k in range(num_frames):
        idx = k * frame_step
        if idx < len(gt_all):
            gt_positions.append(gt_all[idx]["position"])
    gt_positions = np.array(gt_positions)

    pred_positions = [gt_positions[0].copy()]
    current_pos = gt_positions[0].copy()
    current_R = quat_to_rotmat(gt_all[0]["quaternion"])

    for i in range(len(translations)):
        t_world = current_R @ translations[i]
        current_pos = current_pos + t_world
        R_rel = quat_to_rotmat(quaternions[i])
        current_R = current_R @ R_rel
        pred_positions.append(current_pos.copy())

    return np.array(pred_positions), gt_positions


def compute_ate_rmse(pred_pos, gt_pos):
    n = min(len(pred_pos), len(gt_pos))
    errors = np.linalg.norm(pred_pos[:n] - gt_pos[:n], axis=1)
    return np.sqrt(np.mean(errors ** 2)), errors

def pose_graph_optimize(dr_positions, translations, quaternions, gt_poses_csv,
                        w_relative=1.0, w_loop=2.0, w_smooth=0.05):
    gt_all = load_gt_poses(gt_poses_csv)
    N = len(dr_positions)

    R_abs = [quat_to_rotmat(gt_all[0]["quaternion"])]
    t_world = []
    for i in range(len(translations)):
        t_world.append(R_abs[-1] @ translations[i])
        R_abs.append(R_abs[-1] @ quat_to_rotmat(quaternions[i]))
    t_world = np.array(t_world)

    p0 = dr_positions[0].copy()
    M = N - 1
    optimized = np.zeros((N, 3))
    optimized[0] = p0

    for axis in range(3):
        rows, cols, vals, rhs = [], [], [], []
        ri = 0
        for i in range(N - 1):
            target = t_world[i, axis] - (dr_positions[i+1, axis] - dr_positions[i, axis])
            if i == 0:
                rows.append(ri); cols.append(0); vals.append(w_relative)
                rhs.append(w_relative * target)
            else:
                rows.append(ri); cols.append(i); vals.append(w_relative)
                rows.append(ri); cols.append(i-1); vals.append(-w_relative)
                rhs.append(w_relative * target)
            ri += 1

        rows.append(ri); cols.append(M-1); vals.append(w_loop)
        rhs.append(w_loop * (p0[axis] - dr_positions[-1, axis]))
        ri += 1

        for i in range(1, N-1):
            ds = dr_positions[i+1, axis] - 2*dr_positions[i, axis] + dr_positions[i-1, axis]
            if i == 1:
                rows.append(ri); cols.append(1); vals.append(w_smooth)
                rows.append(ri); cols.append(0); vals.append(-2*w_smooth)
            else:
                rows.append(ri); cols.append(i); vals.append(w_smooth)
                rows.append(ri); cols.append(i-1); vals.append(-2*w_smooth)
                rows.append(ri); cols.append(i-2); vals.append(w_smooth)
            rhs.append(-w_smooth * ds)
            ri += 1

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(ri, M))
        b = np.array(rhs)
        delta = spsolve(A.T @ A, A.T @ b)
        for j in range(M):
            optimized[j+1, axis] = dr_positions[j+1, axis] + delta[j]

    return optimized

@torch.no_grad()
def get_predictions(model, loader, device, mode):
    """Returns translations (N,3) and quaternions (N,4) as numpy."""
    model.eval()
    all_trans, all_quats = [], []

    for batch in loader:
        if mode == "stereo_vision":
            img, label, depth = batch
            trans, rot_6d, _ = model(img.to(device))
        elif mode == "imu_only":
            noisy, clean, label = batch
            trans, rot_6d, _, _ = model(noisy.to(device))
        elif mode == "stereo_vi":
            img, noisy, clean, label, depth = batch
            trans, rot_6d, _, _, _ = model(img.to(device), noisy.to(device))

        # 6D → quaternion
        quats = matrix_to_quaternion(rotation_6d_to_matrix(rot_6d))

        all_trans.append(trans.cpu().numpy())
        all_quats.append(quats.cpu().numpy())

    return np.concatenate(all_trans), np.concatenate(all_quats)

GT_COLOR = "#2563EB"; DR_COLOR = "#DC2626"; OPT_COLOR = "#16A34A"; START_COLOR = "#F59E0B"

def plot_trajectory(dr, opt, gt, title, path):
    fig = plt.figure(figsize=(22, 7), facecolor='white')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(gt[:,0], gt[:,1], color=GT_COLOR, lw=2, label='GT')
    ax.plot(dr[:,0], dr[:,1], color=DR_COLOR, lw=1.2, ls='--', alpha=.6, label='DR')
    ax.plot(opt[:,0], opt[:,1], color=OPT_COLOR, lw=1.8, label='Opt')
    ax.plot(gt[0,0], gt[0,1], 'o', color=START_COLOR, ms=10, label='Start')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('Top (X-Y)', fontweight='bold')
    ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=.2)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gt[:,0], gt[:,2], color=GT_COLOR, lw=2, label='GT')
    ax.plot(dr[:,0], dr[:,2], color=DR_COLOR, lw=1.2, ls='--', alpha=.6, label='DR')
    ax.plot(opt[:,0], opt[:,2], color=OPT_COLOR, lw=1.8, label='Opt')
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_title('Side (X-Z)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=.2)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(gt[:,0], gt[:,1], gt[:,2], color=GT_COLOR, lw=2, label='GT')
    ax.plot(dr[:,0], dr[:,1], dr[:,2], color=DR_COLOR, lw=1, ls='--', alpha=.5, label='DR')
    ax.plot(opt[:,0], opt[:,1], opt[:,2], color=OPT_COLOR, lw=1.8, label='Opt')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D', fontweight='bold'); ax.legend(fontsize=8)

    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def plot_error(dr_err, opt_err, title, path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.fill_between(range(len(dr_err)), dr_err, alpha=.1, color=DR_COLOR)
    ax.plot(dr_err, color=DR_COLOR, lw=1.2, alpha=.7, label=f'DR ({np.mean(dr_err):.3f}m)')
    ax.fill_between(range(len(opt_err)), opt_err, alpha=.1, color=OPT_COLOR)
    ax.plot(opt_err, color=OPT_COLOR, lw=1.5, label=f'Opt ({np.mean(opt_err):.3f}m)')
    ax.set_xlabel('Frame'); ax.set_ylabel('ATE (m)'); ax.set_title(title, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=.2)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def evaluate_mode(mode, device, run_dir, split="test"):
    print(f"\n{'='*60}")
    print(f"  {mode} on {split.upper()} (6D rotation, pose graph opt)")
    print(f"{'='*60}")

    model_path = os.path.join(MODELS_DIR, MODE_TO_FILE[mode])
    if not os.path.exists(model_path):
        print(f"  ERROR: {model_path} not found"); return None, {}

    if mode == "stereo_vision":
        model = StereoVisionNetV3().to(device)
        ds = StereoVisionOnlyV2Preloaded(split, device)
    elif mode == "imu_only":
        model = IMUNetV3().to(device)
        ds = IMUOnlyV2Preloaded(split, device)
    else:
        model = StereoVisualInertialNetV3().to(device)
        ds = StereoVIv2Preloaded(split, device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded epoch {ckpt['epoch']}")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    translations, quaternions = get_predictions(model, loader, device, mode)

    samples = ds.samples
    scenes = list(dict.fromkeys(s["scene"] for s in samples))
    scene_results = {}
    offset = 0

    for scene in scenes:
        sc = sum(1 for s in samples if s["scene"] == scene)
        t = translations[offset:offset+sc]
        q = quaternions[offset:offset+sc]
        gt_csv = os.path.join(DATASETS_DIR, scene, "poses.csv")

        dr_pos, gt_pos = dead_reckon(t, q, gt_csv)
        dr_ate, dr_err = compute_ate_rmse(dr_pos, gt_pos)

        opt_pos = pose_graph_optimize(dr_pos, t, q, gt_csv)
        opt_ate, opt_err = compute_ate_rmse(opt_pos, gt_pos)

        imp = (1 - opt_ate/dr_ate)*100 if dr_ate > 0 else 0
        print(f"\n  {scene}: DR {dr_ate:.4f}m → Opt {opt_ate:.4f}m ({imp:+.1f}%)")

        scene_results[scene] = {"dr_ate": dr_ate, "opt_ate": opt_ate, "imp": imp}
        pfx = f"{split}_{mode}" if split != "test" else mode

        plot_trajectory(dr_pos, opt_pos, gt_pos,
                        f"{mode.replace('_',' ').title()} — {scene} ({split})",
                        os.path.join(run_dir, f"{pfx}_{scene}_trajectory.png"))
        plot_error(dr_err, opt_err,
                   f"{mode.replace('_',' ').title()} — {scene} ({split})",
                   os.path.join(run_dir, f"{pfx}_{scene}_error.png"))
        offset += sc

    dr_avg = np.mean([r["dr_ate"] for r in scene_results.values()])
    opt_avg = np.mean([r["opt_ate"] for r in scene_results.values()])
    imp_avg = (1 - opt_avg/dr_avg)*100 if dr_avg > 0 else 0
    print(f"\n  Overall: DR {dr_avg:.4f}m → Opt {opt_avg:.4f}m ({imp_avg:+.1f}%)")

    pfx = f"{split}_{mode}" if split != "test" else mode
    with open(os.path.join(run_dir, f"{pfx}_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "dr_ate", "opt_ate", "improvement"])
        for s, r in scene_results.items():
            w.writerow([s, f"{r['dr_ate']:.6f}", f"{r['opt_ate']:.6f}", f"{r['imp']:.1f}"])
        w.writerow(["OVERALL", f"{dr_avg:.6f}", f"{opt_avg:.6f}", f"{imp_avg:.1f}"])

    return opt_avg, scene_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stereo_vision", "imu_only", "stereo_vi"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train","val","test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EVAL_DIR, f"v3_eval_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Device: {device}\nResults: {run_dir}")

    modes = ["stereo_vision", "imu_only", "stereo_vi"] if args.all else ([args.mode] if args.mode else [])
    if not modes: print("Specify --mode or --all"); return

    results = {}
    for m in modes:
        for s in args.splits:
            ate, sr = evaluate_mode(m, device, run_dir, s)
            if ate is not None: results[(m,s)] = sr

    if results:
        print(f"\n{'='*60}\n  SUMMARY (DR → Optimized)\n{'='*60}")
        for s in args.splits:
            sr = {m: r for (m,sp), r in results.items() if sp == s}
            if sr:
                print(f"\n  [{s.upper()}]")
                for m, r in sr.items():
                    d = np.mean([v["dr_ate"] for v in r.values()])
                    o = np.mean([v["opt_ate"] for v in r.values()])
                    print(f"    {m:>20s}: {d:.4f} → {o:.4f}m ({(1-o/d)*100:+.1f}%)")

        with open(os.path.join(run_dir, "comparison.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode","split","dr_ate","opt_ate","improvement"])
            for (m,s), sr in sorted(results.items()):
                d = np.mean([v["dr_ate"] for v in sr.values()])
                o = np.mean([v["opt_ate"] for v in sr.values()])
                w.writerow([m, s, f"{d:.6f}", f"{o:.6f}", f"{(1-o/d)*100:.1f}"])


if __name__ == "__main__":
    main()
