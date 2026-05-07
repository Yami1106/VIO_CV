"""
train_stereo_v3.py — Training with 6D Rotation Representation
===============================================================

Usage:
    python train_stereo_v3.py --mode stereo_vision
    python train_stereo_v3.py --mode imu_only
    python train_stereo_v3.py --mode stereo_vi
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks_stereo_v3 import (
    StereoVisionNetV3, IMUNetV3, StereoVisualInertialNetV3,
    rotation_6d_to_matrix
)
from dataloaders_stereo_v2 import (
    StereoVisionOnlyV2Preloaded, IMUOnlyV2Preloaded, StereoVIv2Preloaded
)

MODELS_DIR = "/mnt/data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

TZ_WEIGHT = 4.0


def quaternion_to_matrix(q):
    """
    Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.
    Input: (B, 4), Output: (B, 3, 3)
    """
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw),
        2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw),
        2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2),
    ], dim=-1).view(-1, 3, 3)

    return R

class TranslationLoss(nn.Module):
    def __init__(self, alpha=70.0, wz=TZ_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('w', torch.tensor([1.0, 1.0, wz]))

    def forward(self, pred, gt):
        loss = ((pred - gt) ** 2 * self.w.to(pred.device)).mean()
        return self.alpha * loss, loss.item()


class GeodesicRotationLoss(nn.Module):
    """
    Geodesic distance on SO(3): the actual angle between two rotations.
    L = arccos( (trace(R_pred^T @ R_gt) - 1) / 2 )

    This is in radians. For small rotations (~0.01 rad per step),
    this gives clean gradients unlike quaternion dot product.
    """
    def __init__(self, beta=50.0):
        super().__init__()
        self.beta = beta

    def forward(self, rot_6d_pred, q_gt):
        R_pred = rotation_6d_to_matrix(rot_6d_pred)         # (B, 3, 3)
        R_gt = quaternion_to_matrix(q_gt)                     # (B, 3, 3)

        # R_diff = R_pred^T @ R_gt
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)     # (B, 3, 3)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

        # Clamp for numerical stability
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cos_angle)  # radians

        loss = angle.mean()
        return self.beta * loss, loss.item()


class DepthLoss(nn.Module):
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, gt):
        loss = torch.mean(torch.abs(pred - gt))
        return self.weight * loss, loss.item()


class IMUDenoiseLoss(nn.Module):
    def __init__(self, weight=5.0):
        super().__init__()
        self.weight = weight

    def forward(self, denoised, clean):
        gyro_loss = ((denoised[:, :, :3] - clean[:, :, :3]) ** 2).mean()
        accel_loss = ((denoised[:, :, 3:] - clean[:, :, 3:]) ** 2).mean()
        total = self.weight * (gyro_loss + accel_loss)
        return total, gyro_loss.item(), accel_loss.item()


def rollout_loss(t_pred, t_gt, steps=5, wz=TZ_WEIGHT):
    B = t_pred.shape[0]
    n = B // steps
    if n == 0:
        return torch.tensor(0.0, device=t_pred.device)
    p = t_pred[:n * steps].view(n, steps, 3)
    g = t_gt[:n * steps].view(n, steps, 3)
    diff = p.sum(1) - g.sum(1)
    return (diff[:, :2] ** 2).mean() + (diff[:, 2] ** 2).mean() * wz


def train_vision(model, loader, opt, t_loss, r_loss, d_loss, device, gamma):
    model.train()
    total = 0.0; n = 0
    for img, label, depth_gt in loader:
        img, label, depth_gt = img.to(device), label.to(device), depth_gt.to(device)
        trans, rot_6d, depth = model(img)

        lt, _ = t_loss(trans, label[:, :3])
        lr, _ = r_loss(rot_6d, label[:, 3:])
        ld, _ = d_loss(depth, depth_gt)
        rl = rollout_loss(trans, label[:, :3])
        loss = lt + lr + ld + gamma * rl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item(); n += 1
    return total / n


def train_imu(model, loader, opt, t_loss, r_loss, dn_loss, device, gamma):
    model.train()
    total = 0.0; n = 0
    for noisy, clean, label in loader:
        noisy, clean, label = noisy.to(device), clean.to(device), label.to(device)
        trans, rot_6d, denoised, noise = model(noisy)

        lt, _ = t_loss(trans, label[:, :3])
        lr, _ = r_loss(rot_6d, label[:, 3:])
        ldn, _, _ = dn_loss(denoised, clean)
        rl = rollout_loss(trans, label[:, :3])
        loss = lt + lr + ldn + gamma * rl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item(); n += 1
    return total / n


def train_vi(model, loader, opt, t_loss, r_loss, d_loss, dn_loss, device, gamma):
    model.train()
    total = 0.0; n = 0
    for img, noisy, clean, label, depth_gt in loader:
        img, noisy, clean = img.to(device), noisy.to(device), clean.to(device)
        label, depth_gt = label.to(device), depth_gt.to(device)
        trans, rot_6d, depth, denoised, noise = model(img, noisy)

        lt, _ = t_loss(trans, label[:, :3])
        lr, _ = r_loss(rot_6d, label[:, 3:])
        ld, _ = d_loss(depth, depth_gt)
        ldn, _, _ = dn_loss(denoised, clean)
        rl = rollout_loss(trans, label[:, :3])
        loss = lt + lr + ld + ldn + gamma * rl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item(); n += 1
    return total / n


@torch.no_grad()
def evaluate(model, loader, t_loss, r_loss, device, mode, d_loss=None, dn_loss=None):
    model.eval()
    total = 0.0; n = 0
    all_t_pred, all_t_gt = [], []
    all_rot_err = []

    for batch in loader:
        if mode == "stereo_vision":
            img, label, depth_gt = [x.to(device) for x in batch]
            trans, rot_6d, depth = model(img)
            ld_val = d_loss(depth, depth_gt)[0].item() if d_loss else 0
        elif mode == "imu_only":
            noisy, clean, label = [x.to(device) for x in batch]
            trans, rot_6d, denoised, noise = model(noisy)
            ld_val = 0
        elif mode == "stereo_vi":
            img, noisy, clean, label, depth_gt = [x.to(device) for x in batch]
            trans, rot_6d, depth, denoised, noise = model(img, noisy)
            ld_val = d_loss(depth, depth_gt)[0].item() if d_loss else 0

        lt, _ = t_loss(trans, label[:, :3])
        lr, rot_err = r_loss(rot_6d, label[:, 3:])
        total += (lt + lr).item()
        all_t_pred.append(trans.cpu().numpy())
        all_t_gt.append(label[:, :3].cpu().numpy())
        all_rot_err.append(rot_err)
        n += 1

    all_t_pred = np.concatenate(all_t_pred)
    all_t_gt = np.concatenate(all_t_gt)
    err = all_t_pred - all_t_gt
    tx = np.sqrt(np.mean(err[:, 0] ** 2))
    ty = np.sqrt(np.mean(err[:, 1] ** 2))
    tz = np.sqrt(np.mean(err[:, 2] ** 2))
    mean_rot = np.mean(all_rot_err)  # mean geodesic error in radians
    rot_deg = np.degrees(mean_rot)

    return total / n, tx, ty, tz, rot_deg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["stereo_vision", "imu_only", "stereo_vi"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--depth_w", type=float, default=10.0)
    parser.add_argument("--denoise_w", type=float, default=5.0)
    parser.add_argument("--wz", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_loss_fn = TranslationLoss(alpha=args.alpha, wz=args.wz)
    r_loss_fn = GeodesicRotationLoss(beta=args.beta)
    d_loss_fn = DepthLoss(weight=args.depth_w)
    dn_loss_fn = IMUDenoiseLoss(weight=args.denoise_w)

    if args.mode == "stereo_vision":
        print("Mode: stereo_vision v3 — 6D rotation + depth")
        train_ds = StereoVisionOnlyV2Preloaded("train", device)
        val_ds = StereoVisionOnlyV2Preloaded("val", device)
        model = StereoVisionNetV3().to(device)
        save_name = "stereo_vision_v3_best.pth"
    elif args.mode == "imu_only":
        print("Mode: imu_only v3 — 6D rotation + denoiser")
        train_ds = IMUOnlyV2Preloaded("train", device)
        val_ds = IMUOnlyV2Preloaded("val", device)
        model = IMUNetV3().to(device)
        save_name = "imu_only_v3_best.pth"
    else:
        print("Mode: stereo_vi v3 — 6D rotation + depth + denoiser")
        train_ds = StereoVIv2Preloaded("train", device)
        val_ds = StereoVIv2Preloaded("val", device)
        model = StereoVisualInertialNetV3().to(device)
        save_name = "stereo_vi_v3_best.pth"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    params = sum(p.numel() for p in model.parameters())
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)

    best_val = float('inf')
    patience_counter = 0
    save_path = os.path.join(MODELS_DIR, save_name)

    print(f"\n{'Ep':>4}  {'Train':>10}  {'Val':>10}  {'tx':>8}  {'ty':>8}  {'tz':>8}  "
          f"{'rot°':>7}  {'LR':>12}  {'s':>5}")
    print("-" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if args.mode == "stereo_vision":
            tr = train_vision(model, train_loader, optimizer,
                              t_loss_fn, r_loss_fn, d_loss_fn, device, args.gamma)
        elif args.mode == "imu_only":
            tr = train_imu(model, train_loader, optimizer,
                           t_loss_fn, r_loss_fn, dn_loss_fn, device, args.gamma)
        else:
            tr = train_vi(model, train_loader, optimizer,
                          t_loss_fn, r_loss_fn, d_loss_fn, dn_loss_fn, device, args.gamma)

        vl, tx, ty, tz, rot_deg = evaluate(
            model, val_loader, t_loss_fn, r_loss_fn, device, args.mode,
            d_loss=d_loss_fn, dn_loss=dn_loss_fn)

        scheduler.step(vl)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:4d}  {tr:10.6f}  {vl:10.6f}  "
              f"{tx:8.5f}  {ty:8.5f}  {tz:8.5f}  "
              f"{rot_deg:6.3f}°  {lr:12.2e}  {elapsed:4.1f}s")

        if vl < best_val:
            best_val = vl
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': vl, 'mode': args.mode}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
