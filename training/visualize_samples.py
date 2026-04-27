"""
visualize_samples.py
--------------------
Visualize image pairs from the dataset with their labels.

Usage:
    python visualize_samples.py --split train --start 0 --n 5
    python visualize_samples.py --split test --start 0 --n 5 --scene scene_009_sid
"""

import os
import argparse
import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")
EVAL_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "evaluation")


def load_split_csv(split_name):
    csv_path = os.path.join(DATASETS_DIR, f"{split_name}_samples.csv")
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--scene", type=str, default=None, help="Filter to specific scene")
    args = parser.parse_args()

    samples = load_split_csv(args.split)

    if args.scene:
        samples = [s for s in samples if s["scene"] == args.scene]
        print(f"Filtered to {args.scene}: {len(samples)} samples")

    n = min(args.n, len(samples) - args.start)

    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        idx = args.start + i
        row = samples[idx]
        scene = row["scene"]
        img_dir = os.path.join(DATASETS_DIR, scene, "images")

        img1_path = os.path.join(img_dir, row["img1"])
        img2_path = os.path.join(img_dir, row["img2"])

        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))

        # Compute difference image
        if len(img1.shape) == 3:
            diff = np.abs(img1.astype(float) - img2.astype(float)).mean(axis=2)
        else:
            diff = np.abs(img1.astype(float) - img2.astype(float))

        tx = float(row["tx"])
        ty = float(row["ty"])
        tz = float(row["tz"])
        t_mag = np.sqrt(tx**2 + ty**2 + tz**2)

        # Image 1
        axes[i, 0].imshow(img1, cmap='gray' if len(img1.shape) == 2 else None)
        axes[i, 0].set_title(f"{row['img1']}", fontsize=9)
        axes[i, 0].axis('off')

        # Image 2
        axes[i, 1].imshow(img2, cmap='gray' if len(img2.shape) == 2 else None)
        axes[i, 1].set_title(f"{row['img2']}", fontsize=9)
        axes[i, 1].axis('off')

        # Difference
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(
            f"|diff|  tx={tx:.4f} ty={ty:.4f} tz={tz:.4f}\n"
            f"|t|={t_mag:.4f}m  {scene}",
            fontsize=8
        )
        axes[i, 2].axis('off')

    col_labels = ["Frame k", "Frame k+1", "Absolute Difference + Label"]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label + "\n" + axes[0, j].get_title(), fontsize=9)

    plt.suptitle(f"Dataset Samples: {args.split} (idx {args.start} to {args.start + n - 1})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(EVAL_DIR, exist_ok=True)
    save_path = os.path.join(EVAL_DIR, f"samples_{args.split}_{args.start}_{args.start+n-1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
