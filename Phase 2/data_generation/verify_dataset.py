"""
verify_dataset.py
-----------------
Run this to check that images, poses.csv, imu.csv, and samples.csv
are all consistent with each other.

Usage:
    python verify_dataset.py --scene_name scene_001_circle
"""

import os
import csv
import argparse
import glob

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"


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
    images_dir = os.path.join(scene_dir, "images")

    errors = []

    # --- 1. Check files exist ---
    for f, name in [(poses_csv, "poses.csv"), (imu_csv, "imu.csv"),
                     (samples_csv, "samples.csv")]:
        if not os.path.exists(f):
            errors.append(f"MISSING: {name}")
            print(f"  [FAIL] {name} not found")
        else:
            print(f"  [OK]   {name} exists")

    if not os.path.isdir(images_dir):
        errors.append("MISSING: images/ directory")
        print(f"  [FAIL] images/ directory not found")
    else:
        print(f"  [OK]   images/ directory exists")

    if errors:
        print(f"\n  Cannot continue — missing files.")
        return errors

    # --- 2. Load data ---
    poses = load_csv_rows(poses_csv)
    imu = load_csv_rows(imu_csv)
    samples = load_csv_rows(samples_csv)
    image_files = sorted(glob.glob(os.path.join(images_dir, "image_*.png")))
    image_basenames = set(os.path.basename(f) for f in image_files)

    print(f"\n  Counts:")
    print(f"    Poses:   {len(poses)}")
    print(f"    IMU:     {len(imu)}")
    print(f"    Samples: {len(samples)}")
    print(f"    Images:  {len(image_files)}")

    # --- 3. Check poses and IMU have same number of rows ---
    if len(poses) != len(imu):
        errors.append(f"poses ({len(poses)}) != imu ({len(imu)}) row count")
        print(f"  [WARN] poses and imu have different row counts")
    else:
        print(f"  [OK]   poses and imu row counts match")

    # --- 4. Check timestamps match between poses and imu ---
    mismatched_ts = 0
    for i in range(min(len(poses), len(imu))):
        pt = float(poses[i]["timestamp"])
        it = float(imu[i]["timestamp"])
        if abs(pt - it) > 1e-6:
            mismatched_ts += 1
    if mismatched_ts > 0:
        errors.append(f"{mismatched_ts} timestamp mismatches between poses and imu")
        print(f"  [FAIL] {mismatched_ts} timestamps differ between poses.csv and imu.csv")
    else:
        print(f"  [OK]   All timestamps match between poses.csv and imu.csv")

    # --- 5. Check samples reference existing images ---
    print(f"\n  Sample checks:")
    missing_images = []
    for s_idx, sample in enumerate(samples):
        img1 = sample["img1"]
        img2 = sample["img2"]
        if img1 not in image_basenames:
            missing_images.append((s_idx, img1))
        if img2 not in image_basenames:
            missing_images.append((s_idx, img2))

    if missing_images:
        errors.append(f"{len(missing_images)} image references in samples.csv not found on disk")
        print(f"  [FAIL] {len(missing_images)} image(s) referenced in samples.csv don't exist:")
        for s_idx, img in missing_images[:10]:  # show first 10
            print(f"         sample {s_idx}: {img}")
        if len(missing_images) > 10:
            print(f"         ... and {len(missing_images)-10} more")

        # Diagnose the naming mismatch
        print(f"\n  --- DIAGNOSIS ---")
        print(f"  Images on disk (first 5):  {sorted(list(image_basenames))[:5]}")
        print(f"  Images in samples (first 5): {[samples[i]['img1'] for i in range(min(5, len(samples)))]}")
        print(f"\n  This usually means render_full_sequence.py and build_samples.py")
        print(f"  use different naming conventions. See fix below.")
    else:
        print(f"  [OK]   All images referenced in samples.csv exist on disk")

    # --- 6. Check sample timestamps match poses ---
    ts_errors = 0
    pose_timestamps = [float(p["timestamp"]) for p in poses]
    for s_idx, sample in enumerate(samples):
        t1 = float(sample["t1"])
        t2 = float(sample["t2"])
        # Check t1 and t2 exist in poses
        t1_found = any(abs(t - t1) < 1e-6 for t in pose_timestamps)
        t2_found = any(abs(t - t2) < 1e-6 for t in pose_timestamps)
        if not t1_found or not t2_found:
            ts_errors += 1

    if ts_errors > 0:
        errors.append(f"{ts_errors} sample timestamps not found in poses.csv")
        print(f"  [FAIL] {ts_errors} sample timestamp(s) don't match any pose timestamp")
    else:
        print(f"  [OK]   All sample timestamps found in poses.csv")

    # --- 7. Check IMU index ranges are valid ---
    imu_idx_errors = 0
    for s_idx, sample in enumerate(samples):
        imu_start = int(sample["imu_start_idx"])
        imu_end = int(sample["imu_end_idx"])
        if imu_start < 0 or imu_end >= len(imu) or imu_start > imu_end:
            imu_idx_errors += 1

    if imu_idx_errors > 0:
        errors.append(f"{imu_idx_errors} samples have invalid IMU index ranges")
        print(f"  [FAIL] {imu_idx_errors} sample(s) have out-of-range IMU indices")
    else:
        print(f"  [OK]   All IMU index ranges are valid")

    # --- 8. Check relative pose is not all zeros ---
    zero_pose_count = 0
    for sample in samples:
        tx, ty, tz = float(sample["tx"]), float(sample["ty"]), float(sample["tz"])
        if abs(tx) < 1e-10 and abs(ty) < 1e-10 and abs(tz) < 1e-10:
            zero_pose_count += 1

    if zero_pose_count > 0:
        print(f"  [WARN] {zero_pose_count} samples have near-zero relative translation")
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
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Verify one scene. If omitted, verifies all scenes.")
    args = parser.parse_args()

    datasets_dir = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")

    if args.scene_name:
        scene_dir = os.path.join(datasets_dir, args.scene_name)
        check_scene(scene_dir, args.scene_name)
    else:
        # Verify all scenes
        scenes = sorted([d for d in os.listdir(datasets_dir)
                        if os.path.isdir(os.path.join(datasets_dir, d))])
        print(f"Found {len(scenes)} scenes to verify")
        all_errors = {}
        for scene_name in scenes:
            scene_dir = os.path.join(datasets_dir, scene_name)
            errs = check_scene(scene_dir, scene_name)
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
