"""
dataloaders.py
--------------
Phase 2 VIO dataloaders.

Key design decisions
--------------------
1. IMU normalization (channel-wise mean/std from training set)
   Gyro ~0.001 rad/s vs accel ~9.8 m/s² — without normalization the
   LSTM wastes capacity on scale differences.

2. NO translation label normalization (v6 behaviour).
   The v7 attempt at normalizing translation labels made val loss jump
   from 0.0002 → 1.6 and worsened ATE from 2.42 m → 2.94 m.
   The tz drift is only -0.048 m over 999 steps which is negligible.
   The network handles raw meter-scale labels fine with the decoupled
   loss (alpha=70, beta=30).

3. IMU_SEQ_LEN = 20
   Increased from 10 to give the BiLSTM richer temporal dynamics.
   Each frame interval spans ~100ms at 100 Hz IMU → 10 raw samples.
   We upsample to 20 via linear interpolation so the LSTM sees a denser
   sequence without losing information.

4. GPU preloaded variants for fast training (~2s/epoch vs ~50s).
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "datasets")

IMU_SEQ_LEN = 20  # increased from 10 for richer temporal context


def load_split_csv(split_name):
    csv_path = os.path.join(DATASETS_DIR, f"{split_name}_samples.csv")
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_scene_imu(scene_name):
    imu_csv = os.path.join(DATASETS_DIR, scene_name, "imu.csv")
    imu_data = []
    with open(imu_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imu_data.append([
                float(row["gx"]), float(row["gy"]), float(row["gz"]),
                float(row["ax"]), float(row["ay"]), float(row["az"]),
            ])
    return np.array(imu_data, dtype=np.float32)


# ============================================================
# IMU normalization stats (training set only)
# ============================================================

def compute_imu_stats():
    stats_cache = os.path.join(DATASETS_DIR, "imu_stats.npz")
    if os.path.exists(stats_cache):
        data = np.load(stats_cache)
        return data["mean"], data["std"]

    train_csv = os.path.join(DATASETS_DIR, "train_samples.csv")
    if not os.path.exists(train_csv):
        return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)

    train_scenes = set()
    with open(train_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_scenes.add(row["scene"])

    all_imu = []
    for scene in sorted(train_scenes):
        imu = load_scene_imu(scene)
        all_imu.append(imu)

    all_imu = np.concatenate(all_imu, axis=0)
    mean = all_imu.mean(axis=0)
    std = all_imu.std(axis=0)
    std = np.maximum(std, 1e-6)

    np.savez(stats_cache, mean=mean, std=std)
    print(f"  IMU stats computed from {len(all_imu)} samples across {len(train_scenes)} scenes")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    return mean, std


# Compute at import time
IMU_MEAN, IMU_STD = compute_imu_stats()


def get_imu_segment(imu_array, start_idx, end_idx, seq_len=IMU_SEQ_LEN):
    """Extract and resample IMU segment to fixed seq_len."""
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    segment = imu_array[start_idx:end_idx + 1]

    n = len(segment)
    if n == 0:
        segment = np.zeros((seq_len, 6), dtype=np.float32)
    elif n == seq_len:
        pass
    else:
        # Linear interpolation to exactly seq_len timesteps
        # This is better than nearest-neighbour for seq_len > n (upsampling)
        old_t = np.linspace(0, 1, n)
        new_t = np.linspace(0, 1, seq_len)
        segment = np.array([
            np.interp(new_t, old_t, segment[:, ch])
            for ch in range(segment.shape[1])
        ]).T.astype(np.float32)

    # Normalize
    segment = (segment - IMU_MEAN) / IMU_STD
    return segment.astype(np.float32)


def get_label(row):
    """
    Returns raw (un-normalized) 7D pose label.
    Translation is in METERS (raw from samples.csv).
    Quaternion is unit norm.
    """
    tx = float(row["tx"])
    ty = float(row["ty"])
    tz = float(row["tz"])
    qx = float(row["qx"])
    qy = float(row["qy"])
    qz = float(row["qz"])
    qw = float(row["qw"])
    return np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32)


DEFAULT_IMG_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ============================================================
# Regular (lazy-loading) datasets
# ============================================================

class VisionOnlyDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.samples = load_split_csv(split)
        self.transform = transform or DEFAULT_IMG_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        scene = row["scene"]
        img_dir = os.path.join(DATASETS_DIR, scene, "images")
        img1 = self.transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
        img2 = self.transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
        img_pair = torch.cat([img1, img2], dim=0)
        label = torch.tensor(get_label(row), dtype=torch.float32)
        return img_pair, label


class IMUOnlyDataset(Dataset):
    def __init__(self, split="train", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        self.seq_len = seq_len
        self._imu_cache = {}

    def _get_imu(self, scene):
        if scene not in self._imu_cache:
            self._imu_cache[scene] = load_scene_imu(scene)
        return self._imu_cache[scene]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        imu_array = self._get_imu(row["scene"])
        segment = get_imu_segment(imu_array, row["imu_start_idx"], row["imu_end_idx"], self.seq_len)
        imu_tensor = torch.tensor(segment, dtype=torch.float32)
        label = torch.tensor(get_label(row), dtype=torch.float32)
        return imu_tensor, label


class VisualInertialDataset(Dataset):
    def __init__(self, split="train", transform=None, seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        self.transform = transform or DEFAULT_IMG_TRANSFORM
        self.seq_len = seq_len
        self._imu_cache = {}

    def _get_imu(self, scene):
        if scene not in self._imu_cache:
            self._imu_cache[scene] = load_scene_imu(scene)
        return self._imu_cache[scene]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        scene = row["scene"]
        img_dir = os.path.join(DATASETS_DIR, scene, "images")
        img1 = self.transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
        img2 = self.transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
        img_pair = torch.cat([img1, img2], dim=0)
        imu_array = self._get_imu(scene)
        segment = get_imu_segment(imu_array, row["imu_start_idx"], row["imu_end_idx"], self.seq_len)
        imu_tensor = torch.tensor(segment, dtype=torch.float32)
        label = torch.tensor(get_label(row), dtype=torch.float32)
        return img_pair, imu_tensor, label


# ============================================================
# GPU-preloaded datasets (fast training)
# ============================================================

class VisionOnlyPreloaded(Dataset):
    def __init__(self, split="train", device="cuda"):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        print(f"  Preloading {len(self.samples)} image pairs to {device}...")
        pairs, labels = [], []
        for i, row in enumerate(self.samples):
            img_dir = os.path.join(DATASETS_DIR, row["scene"], "images")
            img1 = transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
            img2 = transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
            pairs.append(torch.cat([img1, img2], dim=0))
            labels.append(torch.tensor(get_label(row), dtype=torch.float32))
            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")
        self.img_pairs = torch.stack(pairs).to(device)
        self.labels = torch.stack(labels).to(device)
        print(f"  Done. {self.img_pairs.nbytes / 1e6:.0f}MB on {device}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.img_pairs[idx], self.labels[idx]


class IMUOnlyPreloaded(Dataset):
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        imu_cache = {}
        print(f"  Preloading {len(self.samples)} IMU segments to {device}...")
        segments, labels = [], []
        for row in self.samples:
            scene = row["scene"]
            if scene not in imu_cache:
                imu_cache[scene] = load_scene_imu(scene)
            seg = get_imu_segment(imu_cache[scene], row["imu_start_idx"], row["imu_end_idx"], seq_len)
            segments.append(torch.tensor(seg, dtype=torch.float32))
            labels.append(torch.tensor(get_label(row), dtype=torch.float32))
        self.imu_segs = torch.stack(segments).to(device)
        self.labels = torch.stack(labels).to(device)
        print(f"  Done. {self.imu_segs.nbytes / 1e6:.1f}MB on {device}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.imu_segs[idx], self.labels[idx]


class VisualInertialPreloaded(Dataset):
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        imu_cache = {}
        print(f"  Preloading {len(self.samples)} VI pairs to {device}...")
        pairs, segments, labels = [], [], []
        for i, row in enumerate(self.samples):
            img_dir = os.path.join(DATASETS_DIR, row["scene"], "images")
            img1 = transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
            img2 = transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
            pairs.append(torch.cat([img1, img2], dim=0))
            scene = row["scene"]
            if scene not in imu_cache:
                imu_cache[scene] = load_scene_imu(scene)
            seg = get_imu_segment(imu_cache[scene], row["imu_start_idx"], row["imu_end_idx"], seq_len)
            segments.append(torch.tensor(seg, dtype=torch.float32))
            labels.append(torch.tensor(get_label(row), dtype=torch.float32))
            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")
        self.img_pairs = torch.stack(pairs).to(device)
        self.imu_segs = torch.stack(segments).to(device)
        self.labels = torch.stack(labels).to(device)
        mb = (self.img_pairs.nbytes + self.imu_segs.nbytes) / 1e6
        print(f"  Done. {mb:.0f}MB on {device}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.img_pairs[idx], self.imu_segs[idx], self.labels[idx]


if __name__ == "__main__":
    print(f"IMU Mean: {IMU_MEAN}")
    print(f"IMU Std:  {IMU_STD}")
    print(f"IMU_SEQ_LEN: {IMU_SEQ_LEN}")
    rows = load_split_csv("train")
    label = get_label(rows[0])
    print(f"\nSample label (raw meters): {label}")
    print("Dataloaders OK.")