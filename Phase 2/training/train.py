"""
train.py — Precomputed rotation version
-----------------------------------------
Vision:  predicts 7D, loss = alpha*L_trans + beta*L_rot
IMU/VI:  predicts 3D translation only, loss = alpha*L_trans (no rotation loss)

- No beta warmup needed
- No rotation oscillation
- Network focuses 100% on translation accuracy
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import (VisionOnlyPreloaded, IMUOnlyPreloaded, VisualInertialPreloaded)

PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TZ_WEIGHT = 4.0


class TranslationLoss(nn.Module):
    """Translation-only loss with per-axis weighting."""
    def __init__(self, alpha=70.0, wz=TZ_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('axis_weights',
                             torch.tensor([1.0, 1.0, wz], dtype=torch.float32))

    def forward(self, pred, target):
        sq_err = (pred - target) ** 2
        loss = (sq_err * self.axis_weights.to(pred.device)).mean()
        return self.alpha * loss, loss.item()


class FullPoseLoss(nn.Module):
    """Full 7D loss for vision-only (still predicts rotation)."""
    def __init__(self, alpha=70.0, beta=50.0, wz=TZ_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer('axis_weights',
                             torch.tensor([1.0, 1.0, wz], dtype=torch.float32))

    def forward(self, pred, target):
        t_pred, t_gt = pred[:, :3], target[:, :3]
        loss_trans = ((t_pred - t_gt) ** 2 * self.axis_weights.to(t_pred.device)).mean()

        q_pred = pred[:, 3:] / (torch.norm(pred[:, 3:], dim=1, keepdim=True) + 1e-8)
        q_gt = target[:, 3:] / (torch.norm(target[:, 3:], dim=1, keepdim=True) + 1e-8)
        dot = torch.clamp(torch.abs(torch.sum(q_pred * q_gt, dim=1)), 0.0, 1.0)
        loss_rot = torch.mean(1.0 - dot)

        total = self.alpha * loss_trans + self.beta * loss_rot
        return total, loss_trans.item(), loss_rot.item()


def rollout_loss_trans(preds, labels, steps=5, wz=TZ_WEIGHT):
    """Rollout loss for translation only."""
    B = preds.shape[0]
    n = B // steps
    if n == 0:
        return torch.tensor(0.0, device=preds.device)
    p = preds[:n * steps].view(n, steps, 3)
    g = labels[:n * steps].view(n, steps, 3)
    diff = p.sum(1) - g.sum(1)
    return (diff[:, :2] ** 2).mean() + (diff[:, 2] ** 2).mean() * wz


def train_one_epoch(model, loader, optimizer, criterion, device, mode, gamma=0.5):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if mode == "vision":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss, t_loss, r_loss = criterion(pred, label)
            rl = rollout_loss_trans(pred[:, :3], label[:, :3])
            loss = loss + gamma * rl
        elif mode == "imu":
            x, label, gyro_q, gt_q = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)  # (B, 3) translation only
            loss, t_loss = criterion(pred, label)
            rl = rollout_loss_trans(pred, label)
            loss = loss + gamma * rl
        elif mode == "visual_inertial":
            img, imu, label, gyro_q, gt_q = batch
            img, imu, label = img.to(device), imu.to(device), label.to(device)
            gyro_q = gyro_q.to(device)
            pred = model(img, imu, gyro_q)  # (B, 3) translation only
            loss, t_loss = criterion(pred, label)
            rl = rollout_loss_trans(pred, label)
            loss = loss + gamma * rl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        if mode == "vision":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss, _, _ = criterion(pred, label)
            all_preds.append(pred[:, :3].cpu().numpy())  # only translation for RMSE
        elif mode == "imu":
            x, label, gyro_q, gt_q = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss, _ = criterion(pred, label)
            all_preds.append(pred.cpu().numpy())
        elif mode == "visual_inertial":
            img, imu, label, gyro_q, gt_q = batch
            img, imu, label = img.to(device), imu.to(device), label.to(device)
            gyro_q = gyro_q.to(device)
            pred = model(img, imu, gyro_q)
            loss, _ = criterion(pred, label)
            all_preds.append(pred.cpu().numpy())

        if mode == "vision":
            all_labels.append(label[:, :3].cpu().numpy())
        else:
            all_labels.append(label.cpu().numpy())

        total_loss += loss.item()
        n += 1

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    t_err = all_preds - all_labels
    tx_rmse = np.sqrt(np.mean(t_err[:, 0] ** 2))
    ty_rmse = np.sqrt(np.mean(t_err[:, 1] ** 2))
    tz_rmse = np.sqrt(np.mean(t_err[:, 2] ** 2))

    return total_loss / n, tx_rmse, ty_rmse, tz_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--beta", type=float, default=50.0, help="Rotation weight (vision only)")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--wz", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "vision":
        print(f"Mode: vision — predicts full 7D (translation + rotation)")
    else:
        print(f"Mode: {args.mode} — predicts 3D translation only (rotation from gyro)")

    if args.mode == "vision":
        train_ds = VisionOnlyPreloaded("train", device)
        val_ds = VisionOnlyPreloaded("val", device)
        model = VisionNet().to(device)
        criterion = FullPoseLoss(alpha=args.alpha, beta=args.beta, wz=args.wz)
    elif args.mode == "imu":
        train_ds = IMUOnlyPreloaded("train", device)
        val_ds = IMUOnlyPreloaded("val", device)
        model = IMUNet().to(device)
        criterion = TranslationLoss(alpha=args.alpha, wz=args.wz)
    else:
        train_ds = VisualInertialPreloaded("train", device)
        val_ds = VisualInertialPreloaded("val", device)
        model = VisualInertialNet().to(device)
        criterion = TranslationLoss(alpha=args.alpha, wz=args.wz)

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
    save_path = os.path.join(MODELS_DIR, f"{args.mode}_best.pth")

    print(f"\n{'Ep':>4}  {'Train':>10}  {'Val':>10}  {'tx':>8}  {'ty':>8}  {'tz':>8}  {'LR':>12}  {'s':>5}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, args.mode, gamma=args.gamma)
        val_loss, tx, ty, tz = evaluate(model, val_loader, criterion, device, args.mode)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:4d}  {tr_loss:10.6f}  {val_loss:10.6f}  "
              f"{tx:8.5f}  {ty:8.5f}  {tz:8.5f}  {lr:12.2e}  {elapsed:4.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'mode': args.mode}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()