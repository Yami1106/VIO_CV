import os
import csv
import argparse
import numpy as np

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
FRAME_STEP = 10


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """
    q = [qx, qy, qz, qw]
    Returns rotation matrix from body/camera to world.
    """
    qx, qy, qz, qw = normalize_quaternion(q)

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)
    return R


def rotmat_to_quat(R):
    """
    Returns quaternion as [qx, qy, qz, qw]
    """
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

    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def load_poses(csv_path):
    poses = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "timestamp": float(row["timestamp"]),
                "position": np.array([
                    float(row["px"]),
                    float(row["py"]),
                    float(row["pz"])
                ], dtype=float),
                "quaternion": np.array([
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"])
                ], dtype=float)
            })
    return poses


def load_imu(csv_path):
    imu_rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imu_rows.append({
                "timestamp": float(row["timestamp"]),
                "gyro": np.array([
                    float(row["gx"]),
                    float(row["gy"]),
                    float(row["gz"])
                ], dtype=float),
                "acc": np.array([
                    float(row["ax"]),
                    float(row["ay"]),
                    float(row["az"])
                ], dtype=float)
            })
    return imu_rows


def find_imu_window_indices(imu_rows, t1, t2):
    start_idx = None
    end_idx = None

    for i, row in enumerate(imu_rows):
        ts = row["timestamp"]
        if start_idx is None and ts >= t1:
            start_idx = i
        if ts <= t2:
            end_idx = i

    return start_idx, end_idx


def compute_relative_pose(pose1, pose2):
    p1 = pose1["position"]
    p2 = pose2["position"]

    q1 = pose1["quaternion"]
    q2 = pose2["quaternion"]

    R1 = quat_to_rotmat(q1)
    R2 = quat_to_rotmat(q2)

    R_rel = R1.T @ R2
    t_rel = R1.T @ (p2 - p1)

    q_rel = rotmat_to_quat(R_rel)
    return t_rel, q_rel


def build_samples(poses, imu_rows, frame_step):
    samples = []

    for i in range(0, len(poses) - frame_step, frame_step):
        j = i + frame_step

        pose1 = poses[i]
        pose2 = poses[j]

        t1 = pose1["timestamp"]
        t2 = pose2["timestamp"]

        imu_start_idx, imu_end_idx = find_imu_window_indices(imu_rows, t1, t2)
        if imu_start_idx is None or imu_end_idx is None or imu_end_idx < imu_start_idx:
            continue

        t_rel, q_rel = compute_relative_pose(pose1, pose2)

        img1 = f"image_{i // frame_step:06d}.png"
        img2 = f"image_{j // frame_step:06d}.png"

        samples.append([
            img1,
            img2,
            t1,
            t2,
            imu_start_idx,
            imu_end_idx,
            t_rel[0], t_rel[1], t_rel[2],
            q_rel[0], q_rel[1], q_rel[2], q_rel[3]
        ])

    return samples


def save_samples(csv_path, samples):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img1", "img2", "t1", "t2",
            "imu_start_idx", "imu_end_idx",
            "tx", "ty", "tz",
            "qx", "qy", "qz", "qw"
        ])
        writer.writerows(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--frame_step", type=int, default=10)
    args = parser.parse_args()

    scene_dir = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", args.scene_name)
    poses_csv = os.path.join(scene_dir, "poses.csv")
    imu_csv = os.path.join(scene_dir, "imu.csv")
    samples_csv = os.path.join(scene_dir, "samples.csv")

    poses = load_poses(poses_csv)
    imu_rows = load_imu(imu_csv)

    samples = build_samples(poses, imu_rows, args.frame_step)
    save_samples(samples_csv, samples)

    print(f"Saved {len(samples)} samples to: {samples_csv}")


if __name__ == "__main__":
    main()