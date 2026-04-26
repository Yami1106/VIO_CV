import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"


def load_poses(csv_path):
    timestamps = []
    positions = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            positions.append([
                float(row["px"]),
                float(row["py"]),
                float(row["pz"])
            ])

    timestamps = np.array(timestamps, dtype=float)
    positions = np.array(positions, dtype=float)
    return timestamps, positions


def plot_trajectory(scene_name, timestamps, positions, save_path=None):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    fig = plt.figure(figsize=(14, 4))

    # Top view
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x, y, linewidth=2)
    ax1.scatter(x[0], y[0], s=50, marker="o", label="start")
    ax1.scatter(x[-1], y[-1], s=50, marker="x", label="end")
    ax1.set_title(f"{scene_name} - Top View (X-Y)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()

    # Side view
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(x, z, linewidth=2)
    ax2.scatter(x[0], z[0], s=50, marker="o", label="start")
    ax2.scatter(x[-1], z[-1], s=50, marker="x", label="end")
    ax2.set_title(f"{scene_name} - Side View (X-Z)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.grid(True)
    ax2.legend()

    # 3D view
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot(x, y, z, linewidth=2)
    ax3.scatter(x[0], y[0], z[0], s=50, marker="o", label="start")
    ax3.scatter(x[-1], y[-1], z[-1], s=50, marker="x", label="end")
    ax3.set_title(f"{scene_name} - 3D View")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    scene_dir = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets", args.scene_name)
    poses_csv = os.path.join(scene_dir, "poses.csv")

    timestamps, positions = load_poses(poses_csv)

    save_path = None
    if args.save:
        save_path = os.path.join(scene_dir, "trajectory_plot.png")

    plot_trajectory(args.scene_name, timestamps, positions, save_path=save_path)


if __name__ == "__main__":
    main()