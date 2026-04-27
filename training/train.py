"""
train.py
--------
Training with per-axis weighted loss + Z-focused rollout regularization.

Fixes applied
-------------
1. Per-axis loss weights: wz=4.0 boosts Z supervision by 4x.
2. Z-weighted rollout loss: 5-step accumulated translation error with Z boost.
3. Beta warmup: rotation loss weight ramps from 5→beta over 10 epochs.
4. Gradient clipping at 0.5 for stability.

Usage:
    python train.py --mode vision             --epochs 200 --alpha 70 --beta 50
    python train.py --mode imu                --epochs 200 --alpha 70 --beta 50
    python train.py --mode visual_inertial    --epochs 200 --alpha 70 --beta 50
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
TZ_WEIGHT = 4.0


# ============================================================
# Loss functions
# ============================================================

class DecoupledPoseLoss(nn.Module):
    def __init__(self, alpha=70.0, beta=30.0,
                 wx=TX_WEIGHT, wy=TY_WEIGHT, wz=TZ_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer('axis_weights',
                             torch.tensor([wx, wy, wz], dtype=torch.float32))

    def forward(self, pred, target, beta_override=None):
        beta = beta_override if beta_override is not None else self.beta

        t_pred = pred[:, :3]
        t_gt = target[:, :3]
        axis_sq_err = (t_pred - t_gt) ** 2
        loss_trans = (axis_sq_err * self.axis_weights.to(t_pred.device)).mean()

        q_pred = pred[:, 3:] / (torch.norm(pred[:, 3:], dim=1, keepdim=True) + 1e-8)
        q_gt = target[:, 3:] / (torch.norm(target[:, 3:], dim=1, keepdim=True) + 1e-8)
        dot = torch.clamp(torch.abs(torch.sum(q_pred * q_gt, dim=1)), 0.0, 1.0)
        loss_rot = torch.mean(1.0 - dot)

        total = self.alpha * loss_trans + beta * loss_rot
        return total, loss_trans.item(), loss_rot.item()


def rollout_loss(preds, labels, steps=5, wz=TZ_WEIGHT):
    """
    Short-horizon rollout: penalize accumulated translation bias.
    Z gets same boost as main loss.
    """
    B = preds.shape[0]
    n_groups = B // steps
    if n_groups == 0:
        return torch.tensor(0.0, device=preds.device)

    pred_t = preds[:n_groups * steps, :3].view(n_groups, steps, 3)
    gt_t = labels[:n_groups * steps, :3].view(n_groups, steps, 3)
    diff = pred_t.sum(dim=1) - gt_t.sum(dim=1)

    xy_loss = (diff[:, :2] ** 2).mean()
    z_loss = (diff[:, 2] ** 2).mean() * wz
    return xy_loss + z_loss


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, mode,
                    gamma=0.5, rollout_steps=5, wz=TZ_WEIGHT, beta_override=None,
                    grad_clip=0.5):
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

        loss, t_loss, r_loss = criterion(pred, label, beta_override=beta_override)
        rl = rollout_loss(pred, label, steps=rollout_steps, wz=wz)
        loss = loss + gamma * rl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_t += t_loss
        total_r += r_loss
        n += 1

    return total_loss / n, total_t / n, total_r / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode, beta_override=None):
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

        loss, t_loss, r_loss = criterion(pred, label, beta_override=beta_override)
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
    tz_bias = np.mean(t_errs[:, 2])

    # Rotation error in degrees
    q_pred = all_preds[:, 3:]
    q_gt = all_labels[:, 3:]
    q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-8)
    dot = np.clip(np.abs(np.sum(q_pred * q_gt, axis=1)), 0.0, 1.0)
    rot_deg = 2.0 * np.degrees(np.arccos(dot))
    rot_mean = np.mean(rot_deg)
    rot_max = np.max(rot_deg)

    return (total_loss / n, total_t / n, total_r / n,
            tx_rmse, ty_rmse, tz_rmse, tz_bias, rot_mean, rot_max)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--wz", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--beta_warmup", type=int, default=10,
                        help="Epochs to warm up beta from 5 to target")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: {args.mode}  Epochs: {args.epochs}")
    print(f"Loss: alpha={args.alpha}, beta={args.beta} (warmup: 5.0→{args.beta} over {args.beta_warmup} epochs)")
    print(f"      gamma={args.gamma} (rollout), wz={args.wz} (Z weight)")
    print(f"Gradient clip: {args.grad_clip}  Patience: {args.patience}")
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

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Params: {sum(p.numel() for p in model.parameters()):,}")
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

    print(f"{'Ep':>4}  {'Train':>10}  {'T_loss':>8}  {'R_loss':>8}  "
          f"{'Val':>10}  {'tx':>8}  {'ty':>8}  {'tz':>8}  {'tz_b':>8}  "
          f"{'rot°':>5}  {'r_max':>5}  {'beta':>6}  {'LR':>12}  {'s':>5}")
    print("-" * 125)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Beta warmup
        if epoch <= args.beta_warmup:
            frac = epoch / args.beta_warmup
            current_beta = 5.0 + (args.beta - 5.0) * frac
        else:
            current_beta = args.beta

        tr_loss, tr_t, tr_r = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mode,
            gamma=args.gamma, rollout_steps=args.rollout_steps, wz=args.wz,
            beta_override=current_beta, grad_clip=args.grad_clip
        )
        (val_loss, val_t, val_r, tx_rmse, ty_rmse, tz_rmse,
         tz_bias, rot_mean, rot_max) = evaluate(
            model, val_loader, criterion, device, args.mode,
            beta_override=current_beta
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        tz_b_str = f"{tz_bias:+.5f}"
        print(f"{epoch:4d}  {tr_loss:10.6f}  {tr_t:8.5f}  {tr_r:8.5f}  "
              f"{val_loss:10.6f}  {tx_rmse:8.5f}  {ty_rmse:8.5f}  {tz_rmse:8.5f}  "
              f"{tz_b_str:>8}  {rot_mean:5.3f}  {rot_max:5.2f}  {current_beta:6.1f}  "
              f"{lr:12.2e}  {elapsed:4.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
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