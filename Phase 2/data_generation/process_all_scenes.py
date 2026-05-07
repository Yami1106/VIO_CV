"""
process_all_scenes.py
---------------------
After rendering images, run this to compute IMU and build samples for all scenes.

Usage:
    python process_all_scenes.py
"""

import subprocess
import sys
import os

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")
DATA_GEN_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "data_generation")

IMU_SCRIPT = os.path.join(DATA_GEN_DIR, "compute_imu.py")
SAMPLES_SCRIPT = os.path.join(DATA_GEN_DIR, "build_samples.py")
VERIFY_SCRIPT = os.path.join(DATA_GEN_DIR, "verify_dataset.py")


def get_all_scenes():
    """Find all scene directories."""
    scenes = []
    for d in sorted(os.listdir(DATASETS_DIR)):
        scene_dir = os.path.join(DATASETS_DIR, d)
        if os.path.isdir(scene_dir) and d.startswith("scene_"):
            # Check it has poses.csv and images/
            has_poses = os.path.exists(os.path.join(scene_dir, "poses.csv"))
            has_images = os.path.isdir(os.path.join(scene_dir, "images"))
            if has_poses and has_images:
                scenes.append(d)
    return scenes


def run_cmd(cmd, label):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR in {label}: {result.stderr[:200]}")
        return False
    else:
        print(f"    {label}: {result.stdout.strip()}")
        return True


def main():
    scenes = get_all_scenes()
    print(f"Found {len(scenes)} scenes with poses + images\n")

    for i, scene in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] Processing {scene}")

        # Compute IMU
        run_cmd([sys.executable, IMU_SCRIPT, "--scene_name", scene], "IMU")

        # Build samples
        run_cmd([sys.executable, SAMPLES_SCRIPT, "--scene_name", scene], "Samples")

    # Verify all
    print(f"\n{'='*60}")
    print("Running verification on all scenes...")
    print(f"{'='*60}")
    subprocess.run([sys.executable, VERIFY_SCRIPT])


if __name__ == "__main__":
    main()
