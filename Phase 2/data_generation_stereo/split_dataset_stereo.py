"""
split_dataset_stereo.py
------------------------
Splits scenes into train / val / test by WHOLE SCENE.
Uses /mnt/data/datasets.

Usage:
    python split_dataset_stereo.py
"""

import os
import csv

DATASETS_DIR = "/mnt/data/datasets"

SPLIT = {
    "train": [
        "scene_001_circle",     # data.blend
        "scene_002_oval",       # data.blend
        "scene_003_diamond",    # data.blend
        "scene_004_figure8",    # data.blend
        "scene_006_clover",     # texture2.blend
        "scene_007_mouse",      # texture2.blend
        "scene_008_halfmoon",   # texture2.blend
    ],
    "val": [
        "scene_005_star",       # data.blend
    ],
    "test": [
        "scene_009_sid",        # texture2.blend
        "scene_010_picasso",    # texture2.blend
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

    fieldnames = list(all_rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def main():
    print("=" * 60)
    print("  Dataset Split: by Scene (stereo)")
    print(f"  Root: {DATASETS_DIR}")
    print("=" * 60)

    all_scenes = SPLIT["train"] + SPLIT["val"] + SPLIT["test"]
    for scene in all_scenes:
        scene_dir = os.path.join(DATASETS_DIR, scene)
        if not os.path.isdir(scene_dir):
            print(f"  ERROR: {scene} not found at {scene_dir}")
            available = [d for d in sorted(os.listdir(DATASETS_DIR))
                        if os.path.isdir(os.path.join(DATASETS_DIR, d))]
            if available:
                print(f"  Available: {available}")
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
