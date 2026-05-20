<div align="center">

# Visual-Inertial Odometry — Classical & Deep Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Two complete approaches to VIO: a classical Kalman filter achieving 0.12 m RMSE, and a deep learned fused network reaching 1.18 m ATE after optimisation.*

</div>

---

## Overview

Visual-Inertial Odometry estimates a robot's trajectory by fusing camera images with IMU measurements. This project implements the problem twice — once with a classical probabilistic filter, and once with deep neural networks — and compares both on real and synthetic data.

---

## Phase 1 — Classical S-MSCKF

**Stereo Multi-State Constraint Kalman Filter** on the EuRoC MAV dataset (`MH_01_easy`).

| Component | Detail |
|---|---|
| IMU propagation | 3rd-order matrix exponential + RK4 nominal integration |
| Feature tracking | Sliding window of camera poses with multi-view constraints |
| Initialisation | Gravity + bias from first 200 IMU samples |
| Covariance update | Joseph form for numerical stability |
| **Result** | **0.1201 m RMSE ATE** after SE(3) alignment with Vicon ground truth |

---

## Phase 2 — Deep Learning VIO

Custom **synthetic dataset**: 20 scenes, 13 trajectory shapes, rendered in Blender at 100 Hz.

Three networks trained in PyTorch:

| Network | Architecture | ATE |
|---|---|---|
| Vision-only | Multi-scale 3-stream CNN encoder | 4.71 m |
| IMU-only | Bidirectional LSTM + temporal attention | 2.73 m |
| Visual-Inertial (fused) | Learned sigmoid gate: `g·fᵥ + (1-g)·fᵢ` | 1.94 m |
| VI + global optimisation | Loop closure + smoothness | **1.18 m** |

Loss: weighted MSE (altitude weight 4.0) + quaternion geodesic loss + 5-step rollout penalty.

---

## Key findings

- Precomputed gyro rotation is a strong prior — removing it degraded ATE from 1.94 m → 3.69 m
- Global optimisation (loop closure + path smoothness) gave a 39% trajectory improvement
- Gated sensor fusion consistently outperformed either modality alone

---

## Tech stack

`Python` · `PyTorch` · `NumPy` · `Blender` · `EuRoC Dataset`

---

<div align="center">
Part of the WPI Computer Vision course · <a href="https://github.com/Yami1106">Ashish Sukumar</a>
</div>
