import os
import csv
import math
import argparse
import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def rotation_matrix_to_quaternion(R):
    """
    Returns quaternion as [qx, qy, qz, qw]
    """
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def make_camera_rotation(position, target=np.array([0.0, 0.0, 0.0]), world_up=np.array([0.0, 1.0, 0.0])):
    """
    Blender camera:
      local -Z = viewing direction
      local +Y = camera up
    """
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    forward = normalize(target - position)
    right = normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([1.0, 0.0, 0.0])
        right = normalize(np.cross(forward, world_up))

    up = normalize(np.cross(right, forward))
    R = np.column_stack((right, up, -forward))
    return R


# ---------------------------------------------------------------------------
# 13 trajectory shapes — maximally diverse, no near-duplicates.
#
# Dropped (too similar to a kept shape):
#   thrice    ~ clover    (both 3-lobed radial)
#   ampersand ~ mouse     (both 2-lobed epicycloid, just rotated)
#   winter    ~ star      (both 4-fold radial symmetry)
#
# Shape families represented (one per family):
#   Simple closed : circle, oval, dice
#   Crossing      : figure8
#   Radial 3-lobe : clover
#   Radial 4-lobe : star
#   Radial 5-lobe : patrick
#   Loopy         : halfmoon, mouse, diamond, trefoil
#   Complex       : sid, picasso
# ---------------------------------------------------------------------------

TRAJECTORY_CHOICES = [
    "circle", "oval", "dice", "figure8",
    "clover", "star", "patrick",
    "halfmoon", "mouse", "diamond", "trefoil",
    "sid", "picasso",
]


def get_xy_from_trajectory(traj_type, theta, scale=1.0):

    # --- Simple closed curves ---

    if traj_type == "circle":
        x = scale * np.cos(theta)
        y = scale * np.sin(theta)

    elif traj_type == "oval":
        x = 1.6 * scale * np.cos(theta)
        y = 0.9 * scale * np.sin(theta)

    elif traj_type == "dice":
        # Rounded square (superellipse) — Blackbird dice shape
        n = 4.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        eps = 1e-10
        r = scale / (np.abs(cos_t)**n + np.abs(sin_t)**n + eps)**(1.0 / n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    # --- Crossing / figure shapes ---

    elif traj_type == "figure8":
        # Flat figure-8 / lemniscate
        x = scale * np.sin(theta)
        y = 0.5 * scale * np.sin(2.0 * theta)

    # --- Radial petal shapes (each has a different lobe count) ---

    elif traj_type == "clover":
        # 3 lobes
        r = scale * (1.0 + 0.5 * np.cos(3.0 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif traj_type == "star":
        # 4 lobes
        r = scale * (1.0 + 0.4 * np.cos(4.0 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif traj_type == "patrick":
        # 5 lobes — Blackbird patrick (Patrick Star)
        r = scale * (1.0 + 0.5 * np.cos(5.0 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    # --- Loopy / epicycloid shapes ---

    elif traj_type == "halfmoon":
        # Cardioid-like with offset loop
        x = scale * np.cos(theta) + 0.5 * scale * np.cos(2.0 * theta)
        y = scale * np.sin(theta)

    elif traj_type == "mouse":
        # 2-eared epicycloid
        x = scale * (np.cos(theta) - 0.5 * np.cos(2.0 * theta))
        y = scale * (np.sin(theta) + 0.5 * np.sin(2.0 * theta))

    elif traj_type == "diamond":
        # Pointy 3-fold shape (your original)
        x = scale * (np.sin(theta) + 0.6 * np.sin(3.0 * theta))
        y = scale * (np.cos(theta) - 0.6 * np.cos(3.0 * theta))

    elif traj_type == "trefoil":
        # Smooth 3-fold offset curve (your original)
        x = scale * (np.cos(theta) + 0.35 * np.cos(3.0 * theta))
        y = scale * (np.sin(theta) - 0.35 * np.sin(3.0 * theta))

    # --- Complex / irregular shapes ---

    elif traj_type == "sid":
        # Multi-harmonic irregular with small loops
        x = scale * (np.sin(theta) + 0.4 * np.sin(3.0 * theta) + 0.2 * np.sin(5.0 * theta))
        y = scale * (np.cos(theta) + 0.4 * np.cos(3.0 * theta))

    elif traj_type == "picasso":
        # Highly irregular, multiple harmonics
        x = scale * (np.sin(theta) + 0.3 * np.sin(2 * theta) + 0.2 * np.sin(5 * theta))
        y = scale * (np.cos(theta) + 0.4 * np.cos(3 * theta) + 0.15 * np.cos(7 * theta))

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return x, y


def generate_trajectory(
    traj_type="circle",
    duration=10.0,
    rate_hz=100,
    scale=3.0,
    height=6.0,
    z_variation=0.3,
    scene_center=(0.0, 0.0, 0.0)
):
    dt = 1.0 / rate_hz
    num_samples = int(duration * rate_hz)

    poses = []
    scene_center = np.array(scene_center, dtype=float)

    for i in range(num_samples):
        t = i * dt
        theta = 2.0 * math.pi * t / duration

        px, py = get_xy_from_trajectory(traj_type, theta, scale=scale)
        pz = height + z_variation * np.sin(0.5 * theta)

        position = np.array([px, py, pz], dtype=float)

        target = np.array([scene_center[0], scene_center[1], 0.0], dtype=float)

        R = make_camera_rotation(position, target)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

        poses.append([t, px, py, pz, qx, qy, qz, qw])

    return poses


def save_poses_csv(output_csv_path, poses):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"])
        writer.writerows(poses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--traj_type", type=str, required=True,
                        choices=TRAJECTORY_CHOICES)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate_hz", type=int, default=100)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--z_variation", type=float, default=0.3)
    args = parser.parse_args()

    project_root = "/home/yami/Downloads/Group9_p4"
    output_dir = os.path.join(project_root, "Code", "Phase2", "datasets", args.scene_name)
    output_csv = os.path.join(output_dir, "poses.csv")

    poses = generate_trajectory(
        traj_type=args.traj_type,
        duration=args.duration,
        rate_hz=args.rate_hz,
        scale=args.scale,
        height=args.height,
        z_variation=args.z_variation
    )

    save_poses_csv(output_csv, poses)
    print(f"Saved {len(poses)} poses to: {output_csv}")


if __name__ == "__main__":
    main()