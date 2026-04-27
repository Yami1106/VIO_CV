"""
quick_test.py
-------------
Test trained models on individual samples and see exact predictions vs ground truth.

Usage:
    python quick_test.py --mode vision --sample 0
    python quick_test.py --mode visual_inertial --sample 42 --split train
    python quick_test.py --mode vision --sample 0 --split test
"""

import argparse
import numpy as np
import torch
from networks import VisionNet, IMUNet, VisualInertialNet
from dataloaders import (VisionOnlyDataset, IMUOnlyDataset, VisualInertialDataset,
                         unnormalize_translation, TRANS_MEAN, TRANS_STD)

import os
PROJECT_ROOT = "/home/yami/Downloads/Group9_p4"
MODELS_DIR = os.path.join(PROJECT_ROOT, "Code", "Phase2", "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vision", "imu", "visual_inertial"])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.mode == "vision":
        model = VisionNet().to(device)
        ds = VisionOnlyDataset(args.split)
    elif args.mode == "imu":
        model = IMUNet().to(device)
        ds = IMUOnlyDataset(args.split)
    elif args.mode == "visual_inertial":
        model = VisualInertialNet().to(device)
        ds = VisualInertialDataset(args.split)

    ckpt = torch.load(os.path.join(MODELS_DIR, f"{args.mode}_best.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model: {args.mode} (epoch {ckpt['epoch']})")
    print(f"Split: {args.split} ({len(ds)} samples)")
    print(f"Trans normalization - Mean: {TRANS_MEAN}, Std: {TRANS_STD}")
    print()

    print(f"{'Idx':>5}  {'':>8}  {'tx':>10}  {'ty':>10}  {'tz':>10}  {'qx':>10}  {'qy':>10}  {'qz':>10}  {'qw':>10}")
    print("-" * 100)

    for i in range(args.sample, min(args.sample + args.n, len(ds))):
        with torch.no_grad():
            if args.mode == "vision":
                x, label = ds[i]
                pred = model(x.unsqueeze(0).to(device))
            elif args.mode == "imu":
                x, label = ds[i]
                pred = model(x.unsqueeze(0).to(device))
            elif args.mode == "visual_inertial":
                img, imu, label = ds[i]
                pred = model(img.unsqueeze(0).to(device), imu.unsqueeze(0).to(device))

        pred = pred.cpu().numpy()[0]
        label = label.numpy()

        # Unnormalize translation
        pred_unnorm = pred.copy()
        label_unnorm = label.copy()
        pred_unnorm[:3] = unnormalize_translation(pred[:3].reshape(1, -1))[0]
        label_unnorm[:3] = unnormalize_translation(label[:3].reshape(1, -1))[0]

        scene = ds.samples[i]["scene"]

        print(f"{i:5d}  {'GT':>8}  {label_unnorm[0]:10.6f}  {label_unnorm[1]:10.6f}  {label_unnorm[2]:10.6f}  {label_unnorm[3]:10.6f}  {label_unnorm[4]:10.6f}  {label_unnorm[5]:10.6f}  {label_unnorm[6]:10.6f}")
        print(f"{'':>5}  {'PRED':>8}  {pred_unnorm[0]:10.6f}  {pred_unnorm[1]:10.6f}  {pred_unnorm[2]:10.6f}  {pred_unnorm[3]:10.6f}  {pred_unnorm[4]:10.6f}  {pred_unnorm[5]:10.6f}  {pred_unnorm[6]:10.6f}")

        t_err = np.linalg.norm(pred_unnorm[:3] - label_unnorm[:3])
        print(f"{'':>5}  {'ERR':>8}  trans={t_err:.6f}m  scene={scene}")
        print()


if __name__ == "__main__":
    main()
