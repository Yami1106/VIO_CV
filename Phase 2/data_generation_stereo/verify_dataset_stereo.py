"""
verify_dataset_stereo.py
-------------------------
Usage:
    python verify_dataset_stereo.py                          # all scenes
    python verify_dataset_stereo.py --scene_name scene_001_circle
"""

import os
import csv
import argparse
import glob

DATASETS_DIR = "/mnt/data/datasets"


def load_csv_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_scene(scene_dir, scene_name):
    print(f"\n{'='*60}")
    print(f"  Verifying: {scene_name}")
    print(f"{'='*60}")

    poses_csv = os.path.join(scene_dir, "poses.csv")
    imu_csv = os.path.join(scene_dir, "imu.csv")
    samples_csv = os.path.join(scene_dir, "samples.csv")
    left_dir = os.path.join(scene_dir, "images", "left")
    right_dir = os.path.join(scene_dir, "images", "right")

    errors = []

    # --- 1. Check files exist ---
    for f, name in [(poses_csv, "poses.csv"), (imu_csv, "imu.csv"),
                     (samples_csv, "samples.csv")]:
        if not os.path.exists(f):
            errors.append(f"MISSING: {name}")
            print(f"  [FAIL] {name} not found")
        else:
            print(f"  [OK]   {name} exists")

    # Check stereo dirs
    for d, name in [(left_dir, "images/left"), (right_dir, "images/right")]:
        if not os.path.isdir(d):
            errors.append(f"MISSING: {name}/ directory")
            print(f"  [FAIL] {name}/ directory not found")
        else:
            print(f"  [OK]   {name}/ directory exists")

    if errors:
        print(f"\n  Cannot continue — missing files.")
        return errors

    # --- 2. Load data ---
    poses = load_csv_rows(poses_csv)
    imu = load_csv_rows(imu_csv)
    samples = load_csv_rows(samples_csv)

    left_files = sorted(glob.glob(os.path.join(left_dir, "image_*.png")))
    right_files = sorted(glob.glob(os.path.join(right_dir, "image_*.png")))
    left_basenames = set(os.path.basename(f) for f in left_files)
    right_basenames = set(os.path.basename(f) for f in right_files)

    print(f"\n  Counts:")
    print(f"    Poses:        {len(poses)}")
    print(f"    IMU:          {len(imu)}")
    print(f"    Samples:      {len(samples)}")
    print(f"    Left images:  {len(left_files)}")
    print(f"    Right images: {len(right_files)}")

    # --- 3. Check left/right image counts match ---
    if len(left_files) != len(right_files):
        errors.append(f"Left ({len(left_files)}) != Right ({len(right_files)}) image count")
        print(f"  [FAIL] Left and right image counts differ")
    else:
        print(f"  [OK]   Left and right image counts match")

    # --- 4. Check left/right have same filenames ---
    if left_basenames != right_basenames:
        diff = left_basenames.symmetric_difference(right_basenames)
        errors.append(f"{len(diff)} images only in left or right")
        print(f"  [FAIL] {len(diff)} images not in both left/ and right/")
    else:
        print(f"  [OK]   Left and right have identical filenames")

    # --- 5. Check poses and IMU row counts ---
    if len(poses) != len(imu):
        errors.append(f"poses ({len(poses)}) != imu ({len(imu)}) row count")
        print(f"  [WARN] poses and imu have different row counts")
    else:
        print(f"  [OK]   poses and imu row counts match")

    # --- 6. Check timestamps match ---
    mismatched_ts = 0
    for i in range(min(len(poses), len(imu))):
        if abs(float(poses[i]["timestamp"]) - float(imu[i]["timestamp"])) > 1e-6:
            mismatched_ts += 1
    if mismatched_ts > 0:
        errors.append(f"{mismatched_ts} timestamp mismatches")
        print(f"  [FAIL] {mismatched_ts} timestamps differ between poses and imu")
    else:
        print(f"  [OK]   All timestamps match")

    # --- 7. Check samples reference existing images (in both left and right) ---
    missing_images = []
    for s_idx, sample in enumerate(samples):
        for img_key in ["img1", "img2"]:
            img = sample[img_key]
            if img not in left_basenames:
                missing_images.append((s_idx, img, "left"))
            if img not in right_basenames:
                missing_images.append((s_idx, img, "right"))

    if missing_images:
        errors.append(f"{len(missing_images)} image references not found")
        print(f"  [FAIL] {len(missing_images)} image(s) missing:")
        for s_idx, img, side in missing_images[:10]:
            print(f"         sample {s_idx}: {img} not in {side}/")
    else:
        print(f"  [OK]   All images referenced in samples.csv exist in both left/ and right/")

    # --- 8. Check IMU index ranges ---
    imu_idx_errors = 0
    for sample in samples:
        imu_start = int(sample["imu_start_idx"])
        imu_end = int(sample["imu_end_idx"])
        if imu_start < 0 or imu_end >= len(imu) or imu_start > imu_end:
            imu_idx_errors += 1
    if imu_idx_errors > 0:
        errors.append(f"{imu_idx_errors} invalid IMU index ranges")
        print(f"  [FAIL] {imu_idx_errors} samples have out-of-range IMU indices")
    else:
        print(f"  [OK]   All IMU index ranges are valid")

    # --- 9. Check relative pose not all zeros ---
    zero_count = 0
    for sample in samples:
        tx, ty, tz = float(sample["tx"]), float(sample["ty"]), float(sample["tz"])
        if abs(tx) < 1e-10 and abs(ty) < 1e-10 and abs(tz) < 1e-10:
            zero_count += 1
    if zero_count > 0:
        print(f"  [WARN] {zero_count} samples have near-zero translation")
    else:
        print(f"  [OK]   All samples have non-zero relative translation")

    # --- Summary ---
    print(f"\n  {'='*40}")
    if errors:
        print(f"  RESULT: {len(errors)} issue(s) found")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"  RESULT: ALL CHECKS PASSED")
    print(f"  {'='*40}\n")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default=None)
    args = parser.parse_args()

    if args.scene_name:
        scene_dir = os.path.join(DATASETS_DIR, args.scene_name)
        check_scene(scene_dir, args.scene_name)
    else:
        scenes = sorted([d for d in os.listdir(DATASETS_DIR)
                        if os.path.isdir(os.path.join(DATASETS_DIR, d)) and d.startswith("scene_")])
        print(f"Found {len(scenes)} scenes to verify")
        all_errors = {}
        for scene_name in scenes:
            errs = check_scene(os.path.join(DATASETS_DIR, scene_name), scene_name)
            if errs:
                all_errors[scene_name] = errs

        print(f"\n{'='*60}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*60}")
        if all_errors:
            print(f"  {len(all_errors)} scene(s) have issues:")
            for name, errs in all_errors.items():
                print(f"    {name}: {len(errs)} issue(s)")
        else:
            print(f"  ALL {len(scenes)} SCENES PASSED ALL CHECKS")


if __name__ == "__main__":
    main()
