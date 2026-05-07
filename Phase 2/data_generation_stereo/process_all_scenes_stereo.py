"""
process_all_scenes_stereo.py
-----------------------------

Usage:
    python process_all_scenes_stereo.py
"""

import subprocess
import sys
import os

DATASETS_DIR = "/mnt/data/datasets"

# Scripts live alongside this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMU_SCRIPT = os.path.join(SCRIPT_DIR, "compute_imu_stereo.py")
SAMPLES_SCRIPT = os.path.join(SCRIPT_DIR, "build_samples_stereo.py")
VERIFY_SCRIPT = os.path.join(SCRIPT_DIR, "verify_dataset_stereo.py")


def get_all_scenes():
    scenes = []
    for d in sorted(os.listdir(DATASETS_DIR)):
        scene_dir = os.path.join(DATASETS_DIR, d)
        if os.path.isdir(scene_dir) and d.startswith("scene_"):
            has_poses = os.path.exists(os.path.join(scene_dir, "poses.csv"))
            # For stereo, check images/left/ exists
            has_images = (
                os.path.isdir(os.path.join(scene_dir, "images", "left")) or
                os.path.isdir(os.path.join(scene_dir, "images"))
            )
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
    if os.path.exists(VERIFY_SCRIPT):
        print(f"\n{'='*60}")
        print("Running verification on all scenes...")
        print(f"{'='*60}")
        subprocess.run([sys.executable, VERIFY_SCRIPT])
    else:
        print(f"\nSkipping verification (verify_dataset_stereo.py not found)")

    print(f"\nNext steps:")
    print(f"  1. python precompute_rotation_stereo.py")
    print(f"  2. python split_dataset_stereo.py")


if __name__ == "__main__":
    main()
