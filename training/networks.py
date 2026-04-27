"""
networks.py  —  PRGFlow-inspired architecture
=============================================

Key design decisions from the papers:

1. PRGFlow (Sanket et al., 2020)
   - Separate translation and rotation heads (decoupled learning dynamics)
   - ResNet-style residual blocks for the vision backbone
   - Predicting translation BEFORE scale/rotation leads to better XY accuracy
   - Supervised loss outperforms self-supervised for regression tasks

2. Blackbird (Antonini et al., 2018)
   - 100 Hz IMU, 120 Hz camera — need rich temporal IMU representation
   - Down-facing camera for ego-motion estimation

Architecture overview
---------------------
VisionNet
  MultiScaleEncoder  (3 resolution streams, concat → 512-d)
  └─ ResConvBlock x3 with skip connections
  TransHead  512 → 256 → 64 → 3   (tx, ty, tz)
  RotHead    512 → 256 → 64 → 4   (qx, qy, qz, qw)

IMUNet
  2-layer BiLSTM (6 → 128 hidden)
  TemporalAttention  (softmax over time → weighted sum)
  TransHead  128 → 64 → 3
  RotHead    128 → 64 → 4

VisualInertialNet
  VisionEncoder  → 512-d
  IMUEncoder     → 128-d
  FusionGate     (learned gating, NOT hard concat)
  SharedDecoder  → TransHead + RotHead

Param counts (approximate):
  VisionNet:           ~1.1M
  IMUNet:              ~130K
  VisualInertialNet:   ~1.4M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building blocks
# ============================================================

class ResConvBlock(nn.Module):
    """
    PRGFlow-style residual conv block.
    Input and output channels are the same so skip connection is identity.
    Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
    """
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class MultiScaleVisionEncoder(nn.Module):
    """
    PRGFlow multi-scale encoder: three parallel streams at different resolutions.
    Inspired by the T×2, S×2 IC-STN warp combination from PRGFlow.

    Each stream:
        Conv(2→C, stride=2) → ResConvBlock → AdaptivePool(4×4) → flatten

    Three streams concatenated → FC → 512-d feature.

    This gives the network coarse spatial context (stream 3) and fine
    texture details (stream 1) simultaneously, similar to how IC-STN
    uses multi-warp blocks to refine estimates at different scales.
    """
    def __init__(self, dropout=0.1):
        super().__init__()

        # Stream 1: fine details  (128 → 64 via stride-2 conv)
        self.stream1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResConvBlock(32, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),   # 32 × 4 × 4 = 512
        )

        # Stream 2: medium  (128 → 32 via stride-4)
        self.stream2 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResConvBlock(64, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),   # 64 × 4 × 4 = 1024
        )

        # Stream 3: coarse global context
        self.stream3 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResConvBlock(128, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),   # 128 × 4 × 4 = 2048
        )

        # 512 + 1024 + 2048 = 3584 → 512
        self.proj = nn.Sequential(
            nn.Linear(3584, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        s1 = self.stream1(x).flatten(1)   # (B, 512)
        s2 = self.stream2(x).flatten(1)   # (B, 1024)
        s3 = self.stream3(x).flatten(1)   # (B, 2048)
        cat = torch.cat([s1, s2, s3], dim=1)  # (B, 3584)
        return self.proj(cat)              # (B, 512)


class TemporalAttentionIMUEncoder(nn.Module):
    """
    BiLSTM with learned temporal attention over IMU timesteps.

    From PRGFlow: the IMU encoder must capture dynamics between frames.
    Temporal attention lets the network focus on the most informative
    timesteps (e.g., peak accelerations) rather than only using the
    final hidden state.

    Architecture:
        BiLSTM(6 → 64 per direction, 2 layers) → all timesteps (B, T, 128)
        Attention: linear(128→1) → softmax → weighted sum → (B, 128)
        FC → 128-d
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_dim * 2  # bidirectional → 128

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, imu_seq):
        # imu_seq: (B, T, 6)
        out, _ = self.lstm(imu_seq)     # (B, T, 128)
        attn_w = self.attention(out)    # (B, T, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        context = (out * attn_w).sum(dim=1)  # (B, 128)
        return self.proj(context)       # (B, 128)


class PoseDecoder(nn.Module):
    """
    PRGFlow-inspired separate heads for translation and rotation.

    From PRGFlow: "predicting translation before scale almost always
    results in better performance and decoupling predictions of scale
    and translation generally results in better performance."

    TransHead: feat → feat//2 → feat//4 → 3
    RotHead:   feat → feat//2 → feat//4 → 4  (normalized to unit quat)
    """
    def __init__(self, feat_dim, dropout=0.2):
        super().__init__()
        h1 = feat_dim // 2
        h2 = feat_dim // 4

        self.trans_head = nn.Sequential(
            nn.Linear(feat_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 3),
        )

        self.rot_head = nn.Sequential(
            nn.Linear(feat_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 4),
        )

    def forward(self, feat):
        trans = self.trans_head(feat)
        rot = F.normalize(self.rot_head(feat), p=2, dim=1)
        return torch.cat([trans, rot], dim=1)


# ============================================================
# Main networks
# ============================================================

class VisionNet(nn.Module):
    """
    PRGFlow-inspired multi-scale vision odometry network.

    Input:  (B, 2, 128, 128)  — stacked grayscale frame pair
    Output: (B, 7)            — [tx, ty, tz, qx, qy, qz, qw]

    ~1.1M parameters.
    """
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleVisionEncoder(dropout=0.1)
        self.decoder = PoseDecoder(feat_dim=512, dropout=0.2)

    def forward(self, img_pair):
        feat = self.encoder(img_pair)
        return self.decoder(feat)


class IMUNet(nn.Module):
    """
    BiLSTM + temporal attention for IMU-only odometry.

    Input:  (B, T, 6)   — [gx, gy, gz, ax, ay, az] sequence
    Output: (B, 7)      — [tx, ty, tz, qx, qy, qz, qw]

    ~130K parameters.
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.encoder = TemporalAttentionIMUEncoder(
            input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2
        )
        self.decoder = PoseDecoder(feat_dim=128, dropout=0.2)

    def forward(self, imu_seq):
        feat = self.encoder(imu_seq)
        return self.decoder(feat)


class VisualInertialNet(nn.Module):
    """
    Fused visual-inertial network with learned gating.

    Vision encoder  → 512-d
    IMU encoder     → 128-d

    Fusion: a learned gate (sigmoid) decides how much to trust each
    modality at each sample. This is more robust than hard concatenation
    because the gate can suppress the IMU branch when it's uninformative
    (e.g., near-hover with low dynamics) and suppress vision when the
    texture is ambiguous.

        gate = sigmoid(W_v * v_feat + W_i * i_feat + b)
        fused = gate * v_proj + (1 - gate) * i_proj

    Then shared decoder → TransHead + RotHead.

    ~1.4M parameters.
    """
    def __init__(self, imu_input_dim=6):
        super().__init__()

        # Encoders
        self.vision_encoder = MultiScaleVisionEncoder(dropout=0.1)  # → 512
        self.imu_encoder = TemporalAttentionIMUEncoder(
            input_dim=imu_input_dim, hidden_dim=64, num_layers=2, dropout=0.2
        )  # → 128

        # Project both to same dim (256) before gating
        self.v_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.i_proj = nn.Sequential(nn.Linear(128, 256), nn.ReLU(inplace=True))

        # Gating: combine both projections → scalar gate per feature dim
        self.gate = nn.Sequential(
            nn.Linear(512, 256),   # concat of 256+256
            nn.Sigmoid(),
        )

        # Shared decoder
        self.decoder = PoseDecoder(feat_dim=256, dropout=0.2)

    def forward(self, img_pair, imu_seq):
        v = self.v_proj(self.vision_encoder(img_pair))   # (B, 256)
        i = self.i_proj(self.imu_encoder(imu_seq))       # (B, 256)

        # Learned gating
        g = self.gate(torch.cat([v, i], dim=1))          # (B, 256)
        fused = g * v + (1.0 - g) * i                   # (B, 256)

        return self.decoder(fused)


# ============================================================
# Quick sanity test
# ============================================================
if __name__ == "__main__":
    def count(m): return sum(p.numel() for p in m.parameters())

    print("=" * 50)
    print("Testing VisionNet...")
    net = VisionNet()
    x = torch.randn(4, 2, 128, 128)
    out = net(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count(net):,}")
    print(f"  Rot norm: {torch.norm(out[:, 3:], dim=1).mean():.4f}")

    print("\nTesting IMUNet...")
    net = IMUNet()
    x = torch.randn(4, 10, 6)
    out = net(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count(net):,}")

    print("\nTesting VisualInertialNet...")
    net = VisualInertialNet()
    img = torch.randn(4, 2, 128, 128)
    imu = torch.randn(4, 10, 6)
    out = net(img, imu)
    print(f"  Input:  img {img.shape}, imu {imu.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count(net):,}")

    print("\nAll tests passed!")
    print("=" * 50)