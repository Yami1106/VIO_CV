"""
precompute_rotation_stereo.py

Usage:
    python precompute_rotation_stereo.py                          # all scenes
    python precompute_rotation_stereo.py --scene scene_001_circle # one scene
"""

import os
import csv
import argparse
import numpy as np

DATASETS_DIR = "/mnt/data/datasets"
FRAME_STEP = 10


def normalize_quaternion(q):
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    qx, qy, qz, qw = normalize_quaternion(q)
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
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


def load_imu(scene_name):
    csv_path = os.path.join(DATASETS_DIR, scene_name, "imu.csv")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "t": float(row["timestamp"]),
                "gyro": np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])]),
            })
    return rows


def integrate_gyro_window(imu_rows, start_idx, end_idx, dt):
    R_accum = np.eye(3)
    for idx in range(start_idx, min(end_idx, len(imu_rows))):
        omega = imu_rows[idx]["gyro"]
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-12:
            axis = omega / np.linalg.norm(omega)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        else:
            R_delta = np.eye(3)
        R_accum = R_accum @ R_delta
    q = rotmat_to_quat(R_accum)
    if q[3] < 0:
        q = -q
    return q


def process_scene(scene_name):
    scene_dir = os.path.join(DATASETS_DIR, scene_name)
    samples_csv = os.path.join(scene_dir, "samples.csv")

    if not os.path.exists(samples_csv):
        print(f"  SKIP: {samples_csv} not found")
        return

    imu_rows = load_imu(scene_name)
    if len(imu_rows) > 1:
        dt = imu_rows[1]["t"] - imu_rows[0]["t"]
    else:
        dt = 0.01

    samples = []
    with open(samples_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            samples.append(row)

    for sample in samples:
        start_idx = int(sample["imu_start_idx"])
        end_idx = int(sample["imu_end_idx"])
        q_gyro = integrate_gyro_window(imu_rows, start_idx, end_idx, dt)
        sample["gyro_qx"] = f"{q_gyro[0]:.10f}"
        sample["gyro_qy"] = f"{q_gyro[1]:.10f}"
        sample["gyro_qz"] = f"{q_gyro[2]:.10f}"
        sample["gyro_qw"] = f"{q_gyro[3]:.10f}"

    new_fieldnames = list(dict.fromkeys(list(fieldnames) + ["gyro_qx", "gyro_qy", "gyro_qz", "gyro_qw"]))

    with open(samples_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    # Verify
    poses_csv = os.path.join(scene_dir, "poses.csv")
    poses = []
    with open(poses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append(np.array([float(row["qx"]), float(row["qy"]),
                                   float(row["qz"]), float(row["qw"])]))

    errors_deg = []
    for i, sample in enumerate(samples[:10]):
        pose_idx_1 = i * FRAME_STEP
        pose_idx_2 = pose_idx_1 + FRAME_STEP
        if pose_idx_2 >= len(poses):
            break
        R1 = quat_to_rotmat(poses[pose_idx_1])
        R2 = quat_to_rotmat(poses[pose_idx_2])
        R_rel_gt = R1.T @ R2
        q_gyro = np.array([float(sample["gyro_qx"]), float(sample["gyro_qy"]),
                           float(sample["gyro_qz"]), float(sample["gyro_qw"])])
        R_gyro = quat_to_rotmat(q_gyro)
        R_err = R_gyro.T @ R_rel_gt
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        errors_deg.append(np.degrees(np.arccos(cos_a)))

    mean_err = np.mean(errors_deg) if errors_deg else 0
    print(f"  {scene_name}: {len(samples)} samples, gyro vs GT error: {mean_err:.6f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    if args.scene:
        process_scene(args.scene)
    else:
        for name in sorted(os.listdir(DATASETS_DIR)):
            scene_dir = os.path.join(DATASETS_DIR, name)
            if os.path.isdir(scene_dir) and os.path.exists(os.path.join(scene_dir, "samples.csv")):
                process_scene(name)

    print("\nDone. Now run split_dataset_stereo.py to include gyro_q* columns.")


if __name__ == "__main__":
    main()
