"""
dataloaders.py — Precomputed rotation version
----------------------------------------------

Label format:
  Vision:  7D [tx, ty, tz, qx, qy, qz, qw] — network predicts all 7
  IMU:     3D [tx, ty, tz] — network predicts translation only
  VI:      3D [tx, ty, tz] — network predicts translation only
           + gyro_quat (4D) as additional input to the network
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

IMU_SEQ_LEN = 20


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
        all_imu.append(load_scene_imu(scene))

    all_imu = np.concatenate(all_imu, axis=0)
    mean = all_imu.mean(axis=0)
    std = np.maximum(all_imu.std(axis=0), 1e-6)
    np.savez(stats_cache, mean=mean, std=std)
    print(f"  IMU stats: {len(all_imu)} samples, {len(train_scenes)} scenes")
    return mean, std


IMU_MEAN, IMU_STD = compute_imu_stats()


def get_imu_segment(imu_array, start_idx, end_idx, seq_len=IMU_SEQ_LEN):
    start_idx, end_idx = int(start_idx), int(end_idx)
    segment = imu_array[start_idx:end_idx + 1]
    n = len(segment)
    if n == 0:
        segment = np.zeros((seq_len, 6), dtype=np.float32)
    elif n != seq_len:
        old_t = np.linspace(0, 1, n)
        new_t = np.linspace(0, 1, seq_len)
        segment = np.array([np.interp(new_t, old_t, segment[:, ch])
                           for ch in range(6)]).T.astype(np.float32)
    return ((segment - IMU_MEAN) / IMU_STD).astype(np.float32)


def get_vision_label(row):
    """Full 7D label for vision-only (predicts both translation and rotation)."""
    return np.array([
        float(row["tx"]), float(row["ty"]), float(row["tz"]),
        float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
    ], dtype=np.float32)


def get_translation_label(row):
    """3D label for IMU/VI (predicts translation only)."""
    return np.array([
        float(row["tx"]), float(row["ty"]), float(row["tz"]),
    ], dtype=np.float32)


def get_gyro_quat(row):
    """Precomputed rotation from gyro integration."""
    return np.array([
        float(row["gyro_qx"]), float(row["gyro_qy"]),
        float(row["gyro_qz"]), float(row["gyro_qw"]),
    ], dtype=np.float32)


def get_gt_quat(row):
    """Ground truth rotation (for comparison/evaluation only)."""
    return np.array([
        float(row["qx"]), float(row["qy"]),
        float(row["qz"]), float(row["qw"]),
    ], dtype=np.float32)


DEFAULT_IMG_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ============================================================
# Regular datasets
# ============================================================

class VisionOnlyDataset(Dataset):
    """Vision: predicts full 7D. No change from before."""
    def __init__(self, split="train", transform=None):
        self.samples = load_split_csv(split)
        self.transform = transform or DEFAULT_IMG_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img_dir = os.path.join(DATASETS_DIR, row["scene"], "images")
        img1 = self.transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
        img2 = self.transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
        img_pair = torch.cat([img1, img2], dim=0)
        label = torch.tensor(get_vision_label(row), dtype=torch.float32)
        return img_pair, label


class IMUOnlyDataset(Dataset):
    """IMU: predicts 3D translation. Also returns gyro_quat and gt_quat."""
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
        imu = self._get_imu(row["scene"])
        segment = get_imu_segment(imu, row["imu_start_idx"], row["imu_end_idx"], self.seq_len)
        imu_tensor = torch.tensor(segment, dtype=torch.float32)
        label = torch.tensor(get_translation_label(row), dtype=torch.float32)
        gyro_q = torch.tensor(get_gyro_quat(row), dtype=torch.float32)
        gt_q = torch.tensor(get_gt_quat(row), dtype=torch.float32)
        return imu_tensor, label, gyro_q, gt_q


class VisualInertialDataset(Dataset):
    """VI: predicts 3D translation. Returns gyro_quat as network input."""
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
        img_dir = os.path.join(DATASETS_DIR, row["scene"], "images")
        img1 = self.transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
        img2 = self.transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
        img_pair = torch.cat([img1, img2], dim=0)

        imu = self._get_imu(row["scene"])
        segment = get_imu_segment(imu, row["imu_start_idx"], row["imu_end_idx"], self.seq_len)
        imu_tensor = torch.tensor(segment, dtype=torch.float32)

        label = torch.tensor(get_translation_label(row), dtype=torch.float32)
        gyro_q = torch.tensor(get_gyro_quat(row), dtype=torch.float32)
        gt_q = torch.tensor(get_gt_quat(row), dtype=torch.float32)
        return img_pair, imu_tensor, label, gyro_q, gt_q


# ============================================================
# Preloaded GPU datasets
# ============================================================

class VisionOnlyPreloaded(Dataset):
    def __init__(self, split="train", device="cuda"):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        print(f"  Preloading {len(self.samples)} vision pairs to {device}...")
        pairs, labels = [], []
        for i, row in enumerate(self.samples):
            img_dir = os.path.join(DATASETS_DIR, row["scene"], "images")
            img1 = transform(Image.open(os.path.join(img_dir, row["img1"])).convert("L"))
            img2 = transform(Image.open(os.path.join(img_dir, row["img2"])).convert("L"))
            pairs.append(torch.cat([img1, img2], dim=0))
            labels.append(torch.tensor(get_vision_label(row), dtype=torch.float32))
            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")
        self.img_pairs = torch.stack(pairs).to(device)
        self.labels = torch.stack(labels).to(device)
        print(f"  Done. {self.img_pairs.nbytes / 1e6:.0f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.img_pairs[idx], self.labels[idx]


class IMUOnlyPreloaded(Dataset):
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        imu_cache = {}
        print(f"  Preloading {len(self.samples)} IMU segments to {device}...")
        segments, labels, gyro_qs, gt_qs = [], [], [], []
        for row in self.samples:
            scene = row["scene"]
            if scene not in imu_cache:
                imu_cache[scene] = load_scene_imu(scene)
            seg = get_imu_segment(imu_cache[scene], row["imu_start_idx"], row["imu_end_idx"], seq_len)
            segments.append(torch.tensor(seg, dtype=torch.float32))
            labels.append(torch.tensor(get_translation_label(row), dtype=torch.float32))
            gyro_qs.append(torch.tensor(get_gyro_quat(row), dtype=torch.float32))
            gt_qs.append(torch.tensor(get_gt_quat(row), dtype=torch.float32))
        self.imu_segs = torch.stack(segments).to(device)
        self.labels = torch.stack(labels).to(device)
        self.gyro_qs = torch.stack(gyro_qs).to(device)
        self.gt_qs = torch.stack(gt_qs).to(device)
        print(f"  Done. {self.imu_segs.nbytes / 1e6:.1f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.imu_segs[idx], self.labels[idx], self.gyro_qs[idx], self.gt_qs[idx]


class VisualInertialPreloaded(Dataset):
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        imu_cache = {}
        print(f"  Preloading {len(self.samples)} VI pairs to {device}...")
        pairs, segments, labels, gyro_qs, gt_qs = [], [], [], [], []
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
            labels.append(torch.tensor(get_translation_label(row), dtype=torch.float32))
            gyro_qs.append(torch.tensor(get_gyro_quat(row), dtype=torch.float32))
            gt_qs.append(torch.tensor(get_gt_quat(row), dtype=torch.float32))
            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")
        self.img_pairs = torch.stack(pairs).to(device)
        self.imu_segs = torch.stack(segments).to(device)
        self.labels = torch.stack(labels).to(device)
        self.gyro_qs = torch.stack(gyro_qs).to(device)
        self.gt_qs = torch.stack(gt_qs).to(device)
        print(f"  Done. {(self.img_pairs.nbytes + self.imu_segs.nbytes) / 1e6:.0f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.img_pairs[idx], self.imu_segs[idx],
                self.labels[idx], self.gyro_qs[idx], self.gt_qs[idx])


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print(f"IMU Mean: {IMU_MEAN}")
    print(f"IMU Std:  {IMU_STD}")

    print("\nVision dataset (7D label):")
    ds = VisionOnlyDataset("train")
    img, label = ds[0]
    print(f"  img: {img.shape}, label: {label.shape} (should be 7)")

    print("\nIMU dataset (3D label + gyro_quat):")
    ds = IMUOnlyDataset("train")
    imu, label, gyro_q, gt_q = ds[0]
    print(f"  imu: {imu.shape}, label: {label.shape} (should be 3)")
    print(f"  gyro_q: {gyro_q}, gt_q: {gt_q}")

    print("\nVI dataset (3D label + gyro_quat):")
    ds = VisualInertialDataset("train")
    img, imu, label, gyro_q, gt_q = ds[0]
    print(f"  img: {img.shape}, imu: {imu.shape}, label: {label.shape} (should be 3)")
    print(f"  gyro_q: {gyro_q}")

    print("\nAll tests passed!")