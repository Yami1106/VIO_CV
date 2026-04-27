"""
split_dataset.py
----------------
Splits scenes into train / val / test by WHOLE SCENE.

Usage:
    python split_dataset.py
"""

import os
import csv

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")

# ============================================================
# 7 train / 1 val / 2 test
# Training has BOTH simple and complex shapes
# Test has complex unseen shapes
# ============================================================
SPLIT = {
    "train": [
        "scene_001_circle", "scene_002_oval", "scene_003_diamond",
        "scene_004_figure8", "scene_005_star", "scene_006_clover",
        "scene_007_mouse",
        "scene_011_circle", "scene_012_oval", "scene_013_diamond",
        "scene_014_figure8", "scene_015_star", "scene_016_clover",
        "scene_017_mouse",
    ],
    "val": [
        "scene_008_halfmoon",
        "scene_018_halfmoon",
    ],
    "test": [
        "scene_009_sid", "scene_010_picasso",
        "scene_019_sid", "scene_020_picasso",
    ],
}


def load_samples_for_scene(scene_name):
    samples_csv = os.path.join(DATASETS_DIR, scene_name, "samples.csv")
    if not os.path.exists(samples_csv):
        print(f"  WARNING: {samples_csv} not found, skipping")
        return []

    rows = []
    with open(samples_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["scene"] = scene_name
            rows.append(row)
    return rows


def save_combined_csv(output_path, all_rows):
    if not all_rows:
        print(f"  WARNING: no rows to save for {output_path}")
        return

    fieldnames = ["scene", "img1", "img2", "t1", "t2",
                  "imu_start_idx", "imu_end_idx",
                  "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    print("=" * 60)
    print("  Dataset Split: by Scene")
    print("=" * 60)

    all_scenes = SPLIT["train"] + SPLIT["val"] + SPLIT["test"]
    for scene in all_scenes:
        scene_dir = os.path.join(DATASETS_DIR, scene)
        if not os.path.isdir(scene_dir):
            print(f"  ERROR: {scene} not found")
            print(f"  Available:")
            for d in sorted(os.listdir(DATASETS_DIR)):
                if os.path.isdir(os.path.join(DATASETS_DIR, d)):
                    print(f"    - {d}")
            return

    for split_name, scenes in SPLIT.items():
        print(f"\n  {split_name.upper()} split: {scenes}")
        all_rows = []
        for scene in scenes:
            rows = load_samples_for_scene(scene)
            print(f"    {scene}: {len(rows)} samples")
            all_rows.extend(rows)

        output_path = os.path.join(DATASETS_DIR, f"{split_name}_samples.csv")
        save_combined_csv(output_path, all_rows)
        print(f"    -> Total: {len(all_rows)} samples")

    print(f"\n  DONE.")


if __name__ == "__main__":
    main()