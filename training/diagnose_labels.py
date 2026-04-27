"""
diagnose_labels.py
------------------
Run this to understand what's in your labels and detect bias.
Put this in the training/ folder and run:
    python diagnose_labels.py
"""

import os
import csv
import numpy as np

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze_split(split_name):
    csv_path = os.path.join(DATASETS_DIR, f"{split_name}_samples.csv")
    if not os.path.exists(csv_path):
        print(f"  {split_name}_samples.csv not found")
        return

    rows = load_csv(csv_path)

    tx = np.array([float(r["tx"]) for r in rows])
    ty = np.array([float(r["ty"]) for r in rows])
    tz = np.array([float(r["tz"]) for r in rows])
    qx = np.array([float(r["qx"]) for r in rows])
    qy = np.array([float(r["qy"]) for r in rows])
    qz = np.array([float(r["qz"]) for r in rows])
    qw = np.array([float(r["qw"]) for r in rows])

    print(f"\n  {split_name.upper()} ({len(rows)} samples)")
    print(f"  {'':>8} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    print(f"  {'tx':>8} {tx.mean():12.6f} {tx.std():12.6f} {tx.min():12.6f} {tx.max():12.6f}")
    print(f"  {'ty':>8} {ty.mean():12.6f} {ty.std():12.6f} {ty.min():12.6f} {ty.max():12.6f}")
    print(f"  {'tz':>8} {tz.mean():12.6f} {tz.std():12.6f} {tz.min():12.6f} {tz.max():12.6f}")
    print(f"  {'qx':>8} {qx.mean():12.6f} {qx.std():12.6f} {qx.min():12.6f} {qx.max():12.6f}")
    print(f"  {'qy':>8} {qy.mean():12.6f} {qy.std():12.6f} {qy.min():12.6f} {qy.max():12.6f}")
    print(f"  {'qz':>8} {qz.mean():12.6f} {qz.std():12.6f} {qz.min():12.6f} {qz.max():12.6f}")
    print(f"  {'qw':>8} {qw.mean():12.6f} {qw.std():12.6f} {qw.min():12.6f} {qw.max():12.6f}")

    # Translation magnitude per step
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    print(f"\n  Translation magnitude per step:")
    print(f"    mean: {t_mag.mean():.6f} m")
    print(f"    std:  {t_mag.std():.6f} m")
    print(f"    min:  {t_mag.min():.6f} m")
    print(f"    max:  {t_mag.max():.6f} m")

    # Check Z bias specifically
    print(f"\n  Z-translation (tz) bias check:")
    print(f"    mean tz: {tz.mean():.8f} m")
    print(f"    If this is not ~0, there's a systematic Z bias")
    print(f"    Over 999 steps, bias accumulates to: {tz.mean() * 999:.4f} m")

    # Quaternion: check if qw is mostly ~1 (small rotations)
    angle_per_step = 2 * np.arccos(np.clip(np.abs(qw), 0, 1)) * 180 / np.pi
    print(f"\n  Rotation per step:")
    print(f"    mean angle: {angle_per_step.mean():.4f} deg")
    print(f"    max angle:  {angle_per_step.max():.4f} deg")

    return tx, ty, tz


def main():
    print("=" * 60)
    print("  Label Diagnostics")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        analyze_split(split)

    # Also check per-scene for test
    print(f"\n{'='*60}")
    print(f"  Per-scene test analysis")
    print(f"{'='*60}")

    test_rows = load_csv(os.path.join(DATASETS_DIR, "test_samples.csv"))
    scenes = {}
    for r in test_rows:
        s = r["scene"]
        if s not in scenes:
            scenes[s] = []
        scenes[s].append(r)

    for scene, rows in scenes.items():
        tz = np.array([float(r["tz"]) for r in rows])
        print(f"\n  {scene}:")
        print(f"    tz mean: {tz.mean():.8f} m")
        print(f"    tz accumulated over 999 steps: {tz.mean() * 999:.4f} m")
        print(f"    This explains Z drift of ~{abs(tz.mean() * 999):.1f}m")


if __name__ == "__main__":
    main()
