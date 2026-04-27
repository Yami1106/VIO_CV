import os
import csv
import argparse
import numpy as np

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"

GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=float)
ADD_NOISE = False
GYRO_NOISE_STD = 0.002
ACC_NOISE_STD = 0.05


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """
    q = [qx, qy, qz, qw]
    Returns rotation matrix from local/body to world.
    """
    qx, qy, qz, qw = normalize_quaternion(q)

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)
    return R


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


def central_difference(values, dt):
    n = len(values)
    deriv = np.zeros_like(values)

    if n < 2:
        return deriv

    deriv[0] = (values[1] - values[0]) / dt
    deriv[-1] = (values[-1] - values[-2]) / dt

    for i in range(1, n - 1):
        deriv[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)

    return deriv


def compute_body_angular_velocity(quaternions, dt):
    """
    Compute angular velocity from rotation matrices.
    Returns omega in body frame.
    """
    n = len(quaternions)
    omegas = np.zeros((n, 3), dtype=float)
    R_list = [quat_to_rotmat(q) for q in quaternions]

    for i in range(n - 1):
        R_i = R_list[i]
        R_j = R_list[i + 1]

        R_delta = R_i.T @ R_j
        skew_sym = (R_delta - R_delta.T) / (2.0 * dt)

        omega = np.array([
            skew_sym[2, 1],
            skew_sym[0, 2],
            skew_sym[1, 0]
        ], dtype=float)

        omegas[i] = omega

    omegas[-1] = omegas[-2]
    return omegas


def compute_imu_from_poses(poses):
    timestamps = np.array([p["timestamp"] for p in poses], dtype=float)
    positions = np.array([p["position"] for p in poses], dtype=float)
    quaternions = np.array([p["quaternion"] for p in poses], dtype=float)

    if len(timestamps) < 3:
        raise ValueError("Need at least 3 poses to compute IMU.")

    dt_values = np.diff(timestamps)
    dt = float(np.median(dt_values))

    velocities_world = central_difference(positions, dt)
    accelerations_world = central_difference(velocities_world, dt)

    angular_vel_body = compute_body_angular_velocity(quaternions, dt)

    imu_rows = []
    for i in range(len(poses)):
        q = quaternions[i]
        R_bw = quat_to_rotmat(q)      # body to world
        R_wb = R_bw.T                 # world to body

        # accelerometer measures specific force:
        # f_body = R_wb * (a_world - g_world)
        specific_force_body = R_wb @ (accelerations_world[i] - GRAVITY_WORLD)

        gyro = angular_vel_body[i].copy()
        accel = specific_force_body.copy()

        if ADD_NOISE:
            gyro += np.random.normal(0.0, GYRO_NOISE_STD, size=3)
            accel += np.random.normal(0.0, ACC_NOISE_STD, size=3)

        imu_rows.append([
            timestamps[i],
            gyro[0], gyro[1], gyro[2],
            accel[0], accel[1], accel[2]
        ])

    return imu_rows


def save_imu_csv(output_csv, imu_rows):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gx", "gy", "gz", "ax", "ay", "az"])
        writer.writerows(imu_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    args = parser.parse_args()

    scene_dir = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", args.scene_name)
    poses_csv = os.path.join(scene_dir, "poses.csv")
    imu_csv = os.path.join(scene_dir, "imu.csv")

    poses = load_poses(poses_csv)
    imu_rows = compute_imu_from_poses(poses)
    save_imu_csv(imu_csv, imu_rows)

    print(f"Saved {len(imu_rows)} IMU rows to: {imu_csv}")


if __name__ == "__main__":
    main()