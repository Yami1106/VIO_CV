"""
train.py
--------
Training with per-axis weighted loss + Z-focused rollout regularization.

Root cause of 2.5m ATE with 0.005m per-step RMSE
--------------------------------------------------
The raw labels have:
    tx std ~ 0.019m
    ty std ~ 0.018m
    tz std ~ 0.008m   <-- 2.5x smaller than tx/ty

Standard MSE treats all three axes equally, so the network learns
tx/ty well but systematically underpredicts tz magnitude.
With 999 steps, a 0.003m/step Z bias = ~3m of vertical drift.

Fixes applied
-------------
1. Per-axis loss weights: wz=4.0 boosts Z supervision by 4x.
   This equates effective gradient for tz to match tx/ty despite
   the smaller absolute scale.

2. Z-weighted rollout loss: 5-step accumulated Z error gets 4x penalty.
   Directly penalizes the bias that causes dead-reckoning drift.

3. gamma=0.5 (up from 0.1): stronger rollout regularization needed
   because the per-step Z bias is the dominant source of ATE.

Why VIO should beat Vision
--------------------------
IMU double-integrates acceleration to get velocity → position.
For Z, this gives direct altitude change measurements every 10ms.
Vision only sees Z change through parallax in a downward camera,
which is poor at estimating depth (altitude). So IMU fusion is
theoretically better for Z — but ONLY if the network learns to use it.
The per-axis weighting forces it to.

Usage:
    python train.py --mode vision             --epochs 150 --alpha 70 --beta 30
    python train.py --mode imu                --epochs 150 --alpha 70 --beta 30
    python train.py --mode visual_inertial    --epochs 150 --alpha 70 --beta 30
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

TX_WEIGHT = 1.0
TY_WEIGHT = 1.0
TZ_WEIGHT = 4.0   # KEY FIX: boost Z supervision 4x


# ============================================================
# Loss functions
# ============================================================

class DecoupledPoseLoss(nn.Module):
    """
    L_total = alpha * L_trans + beta * L_rot

    L_trans = per-axis weighted MSE:
        wx*(tx-tx*)^2 + wy*(ty-ty*)^2 + wz*(tz-tz*)^2
        with wz=4.0 to compensate for tz being 2.5x smaller than tx/ty.

    L_rot = mean(1 - |q_pred . q_gt|)  geodesic quaternion distance.
    """
    def __init__(self, alpha=70.0, beta=30.0,
                 wx=TX_WEIGHT, wy=TY_WEIGHT, wz=TZ_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer('axis_weights',
                             torch.tensor([wx, wy, wz], dtype=torch.float32))

    def forward(self, pred, target):
        t_pred = pred[:, :3]
        t_gt = target[:, :3]
        axis_sq_err = (t_pred - t_gt) ** 2
        loss_trans = (axis_sq_err * self.axis_weights.to(t_pred.device)).mean()

        q_pred = pred[:, 3:] / (torch.norm(pred[:, 3:], dim=1, keepdim=True) + 1e-8)
        q_gt = target[:, 3:] / (torch.norm(target[:, 3:], dim=1, keepdim=True) + 1e-8)
        dot = torch.clamp(torch.abs(torch.sum(q_pred * q_gt, dim=1)), 0.0, 1.0)
        loss_rot = torch.mean(1.0 - dot)

        total = self.alpha * loss_trans + self.beta * loss_rot
        return total, loss_trans.item(), loss_rot.item()


def rollout_loss(preds, labels, steps=5, wz=TZ_WEIGHT):
    """
    Short-horizon rollout: penalize accumulated translation bias.
    Z gets the same 4x boost as in the main loss.
    """
    B = preds.shape[0]
    n_groups = B // steps
    if n_groups == 0:
        return torch.tensor(0.0, device=preds.device)

    pred_t = preds[:n_groups * steps, :3].view(n_groups, steps, 3)
    gt_t = labels[:n_groups * steps, :3].view(n_groups, steps, 3)
    diff = pred_t.sum(dim=1) - gt_t.sum(dim=1)   # (n_groups, 3)

    xy_loss = (diff[:, :2] ** 2).mean()
    z_loss = (diff[:, 2] ** 2).mean() * wz
    return xy_loss + z_loss


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, mode,
                    gamma=0.5, rollout_steps=5, wz=TZ_WEIGHT):
    model.train()
    total_loss = total_t = total_r = 0.0
    n = 0

    for batch in loader:
        if mode == "vision":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
        elif mode == "imu":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
        else:
            img, imu, label = batch
            img, imu, label = img.to(device), imu.to(device), label.to(device)
            pred = model(img, imu)

        loss, t_loss, r_loss = criterion(pred, label)
        rl = rollout_loss(pred, label, steps=rollout_steps, wz=wz)
        loss = loss + gamma * rl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_t += t_loss
        total_r += r_loss
        n += 1

    return total_loss / n, total_t / n, total_r / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode):
    model.eval()
    total_loss = total_t = total_r = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        if mode == "vision":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
        elif mode == "imu":
            x, label = batch
            x, label = x.to(device), label.to(device)
            pred = model(x)
        else:
            img, imu, label = batch
            img, imu, label = img.to(device), imu.to(device), label.to(device)
            pred = model(img, imu)

        loss, t_loss, r_loss = criterion(pred, label)
        total_loss += loss.item()
        total_t += t_loss
        total_r += r_loss
        n += 1
        all_preds.append(pred.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    t_errs = all_preds[:, :3] - all_labels[:, :3]
    tx_rmse = np.sqrt(np.mean(t_errs[:, 0] ** 2))
    ty_rmse = np.sqrt(np.mean(t_errs[:, 1] ** 2))
    tz_rmse = np.sqrt(np.mean(t_errs[:, 2] ** 2))
    trans_rmse = np.sqrt(np.mean(np.sum(t_errs ** 2, axis=1)))

    return total_loss / n, total_t / n, total_r / n, trans_rmse, tx_rmse, ty_rmse, tz_rmse


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Rollout loss weight. 0.5 strongly penalizes Z bias accumulation.")
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--wz", type=float, default=4.0,
                        help="Z-axis loss weight. 4.0 compensates for tz being 2.5x smaller than tx/ty.")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode:   {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch:  {args.batch_size}")
    print(f"LR:     {args.lr}")
    print(f"Loss:   alpha={args.alpha} (trans), beta={args.beta} (rot), gamma={args.gamma} (rollout)")
    print(f"Axis weights: tx={TX_WEIGHT:.1f}, ty={TY_WEIGHT:.1f}, tz={args.wz:.1f}  "
          f"<-- Z gets {args.wz:.0f}x weight to fight Z drift")
    print()

    if args.mode == "vision":
        train_ds = VisionOnlyPreloaded("train", device)
        val_ds = VisionOnlyPreloaded("val", device)
        model = VisionNet().to(device)
    elif args.mode == "imu":
        train_ds = IMUOnlyPreloaded("train", device)
        val_ds = IMUOnlyPreloaded("val", device)
        model = IMUNet().to(device)
    else:
        train_ds = VisualInertialPreloaded("train", device)
        val_ds = VisualInertialPreloaded("val", device)
        model = VisualInertialNet().to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print()

    criterion = DecoupledPoseLoss(alpha=args.alpha, beta=args.beta,
                                  wx=TX_WEIGHT, wy=TY_WEIGHT, wz=args.wz)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(MODELS_DIR, f"{args.mode}_best.pth")

    # Header shows per-axis RMSE — watch tz_rmse converge separately
    print(f"{'Ep':>4}  {'Train':>10}  {'T_loss':>8}  {'R_loss':>8}  "
          f"{'Val':>10}  {'tx_rmse':>8}  {'ty_rmse':>8}  {'tz_rmse':>8}  {'LR':>9}  {'Time':>5}")
    print("-" * 110)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_t, tr_r = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mode,
            gamma=args.gamma, rollout_steps=args.rollout_steps, wz=args.wz
        )
        val_loss, val_t, val_r, val_rmse, tx_rmse, ty_rmse, tz_rmse = evaluate(
            model, val_loader, criterion, device, args.mode
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(f"{epoch:4d}  {tr_loss:10.6f}  {tr_t:8.5f}  {tr_r:8.5f}  "
              f"{val_loss:10.6f}  {tx_rmse:8.5f}  {ty_rmse:8.5f}  {tz_rmse:8.5f}  "
              f"{lr:9.2e}  {elapsed:4.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_trans_rmse': val_rmse,
                'mode': args.mode,
                'alpha': args.alpha, 'beta': args.beta,
                'gamma': args.gamma, 'wz': args.wz,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()