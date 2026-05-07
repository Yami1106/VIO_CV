"""
dataloaders_stereo_v2.py — Stereo V2 Dataloaders
==================================================

1. IMU noise injection — adds realistic noise + bias at load time
2. Returns BOTH noisy and clean IMU segments 
3. Returns full 7D labels for VI
4. Returns depth (pz) values for auxiliary depth supervision
5. No gyro_quat — everything is learned end-to-end

Noise model (based on typical MEMS IMU specs):
- Gyro:  white noise σ=0.01 rad/s, bias drift σ=0.001 rad/s
- Accel: white noise σ=0.1 m/s², bias drift σ=0.01 m/s²
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

DATASETS_DIR = "/mnt/data/datasets"

IMU_SEQ_LEN = 20

# Realistic IMU noise parameters (MEMS grade)
GYRO_NOISE_STD = 0.01      # rad/s white noise
GYRO_BIAS_STD = 0.001      # rad/s slowly varying bias
ACCEL_NOISE_STD = 0.1       # m/s² white noise
ACCEL_BIAS_STD = 0.01       # m/s² slowly varying bias


def load_split_csv(split_name):
    csv_path = os.path.join(DATASETS_DIR, f"{split_name}_samples.csv")
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_scene_imu(scene_name):
    """Load clean IMU data from csv."""
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


def load_scene_poses(scene_name):
    """Load poses to get absolute pz (height) values."""
    poses_csv = os.path.join(DATASETS_DIR, scene_name, "poses.csv")
    poses = []
    with open(poses_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "pz": float(row["pz"]),
            })
    return poses


def compute_imu_stats():
    """Compute normalization stats from CLEAN IMU data."""
    stats_cache = os.path.join(DATASETS_DIR, "imu_stats_v2.npz")
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
    return mean, std


IMU_MEAN, IMU_STD = compute_imu_stats()


def get_imu_segment(imu_array, start_idx, end_idx, seq_len=IMU_SEQ_LEN):
    """Extract and resample a clean IMU segment (unnormalized)."""
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
    return segment


def add_imu_noise(clean_segment):
    """
    Add realistic MEMS IMU noise to a clean segment.

    Noise model:
    - Per-sample white noise (measurement noise)
    - Per-segment constant bias (slowly varying bias, constant within one window)

    Returns: noisy_segment, noise_applied (both same shape as input)
    """
    T_len, _ = clean_segment.shape

    # Gyro noise: channels 0,1,2
    gyro_white = np.random.normal(0, GYRO_NOISE_STD, (T_len, 3)).astype(np.float32)
    gyro_bias = np.random.normal(0, GYRO_BIAS_STD, (1, 3)).astype(np.float32)
    gyro_bias = np.tile(gyro_bias, (T_len, 1))  # constant across window

    # Accel noise: channels 3,4,5
    accel_white = np.random.normal(0, ACCEL_NOISE_STD, (T_len, 3)).astype(np.float32)
    accel_bias = np.random.normal(0, ACCEL_BIAS_STD, (1, 3)).astype(np.float32)
    accel_bias = np.tile(accel_bias, (T_len, 1))

    noise = np.concatenate([gyro_white + gyro_bias, accel_white + accel_bias], axis=1)
    noisy = clean_segment + noise

    return noisy, noise


def normalize_imu(segment):
    """Normalize IMU segment using precomputed stats."""
    return ((segment - IMU_MEAN) / IMU_STD).astype(np.float32)


def get_full_label(row):
    """Full 7D label: translation + quaternion."""
    return np.array([
        float(row["tx"]), float(row["ty"]), float(row["tz"]),
        float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
    ], dtype=np.float32)


def get_depth_label(poses, img_name, frame_step=10):
    """
    Get absolute pz (height) from poses for the given image frame.
    img_name like 'image_000005.png' → pose index 5 * frame_step = 50
    """
    frame_idx = int(img_name.split("_")[1].split(".")[0])
    pose_idx = frame_idx * frame_step
    if pose_idx < len(poses):
        return np.float32(poses[pose_idx]["pz"])
    return np.float32(6.0)  # fallback


DEFAULT_IMG_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


def load_stereo_pair(scene, img_name, transform):
    left_dir = os.path.join(DATASETS_DIR, scene, "images", "left")
    right_dir = os.path.join(DATASETS_DIR, scene, "images", "right")
    left_img = transform(Image.open(os.path.join(left_dir, img_name)).convert("L"))
    right_img = transform(Image.open(os.path.join(right_dir, img_name)).convert("L"))
    return left_img, right_img


# ============================================================
# IMU-only dataset (noisy + clean, full 7D labels)
# ============================================================

class IMUOnlyV2Preloaded(Dataset):
    """
    IMU-only: noisy IMU + clean IMU for denoiser + full 7D labels.
    No images, no depth.

    Returns: noisy_imu (T,6), clean_imu (T,6), label_7d (7,)
    """
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        imu_cache = {}
        print(f"  Preloading {len(self.samples)} IMU-only v2 to {device}...")

        noisy_segs, clean_segs, labels = [], [], []

        for i, row in enumerate(self.samples):
            scene = row["scene"]
            if scene not in imu_cache:
                imu_cache[scene] = load_scene_imu(scene)

            clean_raw = get_imu_segment(imu_cache[scene], row["imu_start_idx"],
                                        row["imu_end_idx"], seq_len)
            noisy_raw, _ = add_imu_noise(clean_raw)

            clean_segs.append(torch.tensor(normalize_imu(clean_raw), dtype=torch.float32))
            noisy_segs.append(torch.tensor(normalize_imu(noisy_raw), dtype=torch.float32))
            labels.append(torch.tensor(get_full_label(row), dtype=torch.float32))

            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")

        self.noisy_imu = torch.stack(noisy_segs).to(device)
        self.clean_imu = torch.stack(clean_segs).to(device)
        self.labels = torch.stack(labels).to(device)
        print(f"  Done. {(self.noisy_imu.nbytes + self.clean_imu.nbytes) / 1e6:.0f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.noisy_imu[idx], self.clean_imu[idx], self.labels[idx]


# ============================================================
# Vision-only dataset (unchanged except adds depth label)
# ============================================================

class StereoVisionOnlyV2Preloaded(Dataset):
    """
    Returns: img_quad (4,128,128), label_7d (7,), depth_t1 (scalar)
    """
    def __init__(self, split="train", device="cuda"):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        print(f"  Preloading {len(self.samples)} stereo vision v2 quads to {device}...")

        poses_cache = {}
        quads, labels, depths = [], [], []

        for i, row in enumerate(self.samples):
            left1, right1 = load_stereo_pair(row["scene"], row["img1"], transform)
            left2, right2 = load_stereo_pair(row["scene"], row["img2"], transform)
            quads.append(torch.cat([left1, right1, left2, right2], dim=0))
            labels.append(torch.tensor(get_full_label(row), dtype=torch.float32))

            # Depth from first frame
            scene = row["scene"]
            if scene not in poses_cache:
                poses_cache[scene] = load_scene_poses(scene)
            depth = get_depth_label(poses_cache[scene], row["img1"])
            depths.append(torch.tensor(depth, dtype=torch.float32))

            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")

        self.img_quads = torch.stack(quads).to(device)
        self.labels = torch.stack(labels).to(device)
        self.depths = torch.stack(depths).to(device)
        print(f"  Done. {self.img_quads.nbytes / 1e6:.0f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.img_quads[idx], self.labels[idx], self.depths[idx]


# ============================================================
# Visual-Inertial dataset (noisy IMU + clean IMU + full 7D + depth)
# ============================================================

class StereoVIv2Preloaded(Dataset):
    """
    Returns:
        img_quad     (4, 128, 128)  — stereo image pairs
        noisy_imu    (T, 6)         — noisy normalized IMU
        clean_imu    (T, 6)         — clean normalized IMU (supervision for denoiser)
        label_7d     (7,)           — full relative pose [tx,ty,tz,qx,qy,qz,qw]
        depth        (scalar)       — absolute pz of first frame
    """
    def __init__(self, split="train", device="cuda", seq_len=IMU_SEQ_LEN):
        self.samples = load_split_csv(split)
        transform = DEFAULT_IMG_TRANSFORM
        imu_cache = {}
        poses_cache = {}
        print(f"  Preloading {len(self.samples)} stereo VI v2 to {device}...")

        quads = []
        noisy_segs, clean_segs = [], []
        labels, depths = [], []

        for i, row in enumerate(self.samples):
            # Images
            left1, right1 = load_stereo_pair(row["scene"], row["img1"], transform)
            left2, right2 = load_stereo_pair(row["scene"], row["img2"], transform)
            quads.append(torch.cat([left1, right1, left2, right2], dim=0))

            # IMU: get clean segment, add noise, normalize both
            scene = row["scene"]
            if scene not in imu_cache:
                imu_cache[scene] = load_scene_imu(scene)
            clean_raw = get_imu_segment(imu_cache[scene], row["imu_start_idx"],
                                        row["imu_end_idx"], seq_len)
            noisy_raw, _ = add_imu_noise(clean_raw)

            clean_norm = normalize_imu(clean_raw)
            noisy_norm = normalize_imu(noisy_raw)

            clean_segs.append(torch.tensor(clean_norm, dtype=torch.float32))
            noisy_segs.append(torch.tensor(noisy_norm, dtype=torch.float32))

            # Full 7D label
            labels.append(torch.tensor(get_full_label(row), dtype=torch.float32))

            # Depth
            if scene not in poses_cache:
                poses_cache[scene] = load_scene_poses(scene)
            depth = get_depth_label(poses_cache[scene], row["img1"])
            depths.append(torch.tensor(depth, dtype=torch.float32))

            if (i + 1) % 3000 == 0:
                print(f"    {i+1}/{len(self.samples)}")

        self.img_quads = torch.stack(quads).to(device)
        self.noisy_imu = torch.stack(noisy_segs).to(device)
        self.clean_imu = torch.stack(clean_segs).to(device)
        self.labels = torch.stack(labels).to(device)
        self.depths = torch.stack(depths).to(device)
        mem = (self.img_quads.nbytes + self.noisy_imu.nbytes + self.clean_imu.nbytes) / 1e6
        print(f"  Done. {mem:.0f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.img_quads[idx], self.noisy_imu[idx], self.clean_imu[idx],
                self.labels[idx], self.depths[idx])


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("Testing noise injection...")
    clean = np.random.randn(20, 6).astype(np.float32)
    noisy, noise = add_imu_noise(clean)
    print(f"  Clean shape: {clean.shape}, Noisy shape: {noisy.shape}")
    print(f"  Gyro noise std:  {noise[:, :3].std():.4f} (expect ~{GYRO_NOISE_STD})")
    print(f"  Accel noise std: {noise[:, 3:].std():.4f} (expect ~{ACCEL_NOISE_STD})")
    print("  Noise injection OK!")
