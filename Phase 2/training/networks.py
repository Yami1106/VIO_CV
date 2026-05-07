"""
networks.py — Precomputed rotation architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResConvBlock(nn.Module):
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
    def __init__(self, dropout=0.1):
        super().__init__()
        self.stream1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResConvBlock(32, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResConvBlock(64, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),
        )
        self.stream3 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResConvBlock(128, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Linear(3584, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        s1 = self.stream1(x).flatten(1)
        s2 = self.stream2(x).flatten(1)
        s3 = self.stream3(x).flatten(1)
        return self.proj(torch.cat([s1, s2, s3], dim=1))


class TemporalAttentionIMUEncoder(nn.Module):
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
        feat_dim = hidden_dim * 2
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
        out, _ = self.lstm(imu_seq)
        attn_w = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.proj(context)


# Decoders
class PoseDecoder(nn.Module):
    """Full 7D decoder for vision-only (still needs to predict rotation)."""
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


class TranslationOnlyDecoder(nn.Module):
    """
    Translation-only decoder for IMU and VI networks.
    Only predicts [tx, ty, tz]. Rotation comes from gyro integration.
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

    def forward(self, feat):
        return self.trans_head(feat)  # (B, 3) — translation only

# Networks

class VisionNet(nn.Module):
    """
    Vision-only: still predicts full 7D since no IMU available.
    Input:  (B, 2, 128, 128)
    Output: (B, 7) — [tx, ty, tz, qx, qy, qz, qw]
    """
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleVisionEncoder(dropout=0.1)
        self.decoder = PoseDecoder(feat_dim=512, dropout=0.2)

    def forward(self, img_pair):
        return self.decoder(self.encoder(img_pair))


class IMUNet(nn.Module):
    """
    IMU-only: predicts ONLY translation. Rotation is precomputed from gyro.

    Input:  (B, T, 6)
    Output: (B, 3) — [tx, ty, tz] only

    At evaluation time, the precomputed gyro quaternion is appended
    to make the full 7D pose for dead-reckoning.
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.encoder = TemporalAttentionIMUEncoder(
            input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2
        )
        self.decoder = TranslationOnlyDecoder(feat_dim=128, dropout=0.2)

    def forward(self, imu_seq):
        return self.decoder(self.encoder(imu_seq))


class VisualInertialNet(nn.Module):
    """
    Visual-inertial: predicts ONLY translation. Rotation is precomputed.

    Input:  img (B, 2, 128, 128), imu (B, T, 6)
    Output: (B, 3) — [tx, ty, tz] only

    The precomputed rotation is also fed as an additional input to the
    fusion layer (4 extra dims) so the network knows the current rotation
    state when predicting translation.
    """
    def __init__(self, imu_input_dim=6):
        super().__init__()
        self.vision_encoder = MultiScaleVisionEncoder(dropout=0.1)
        self.imu_encoder = TemporalAttentionIMUEncoder(
            input_dim=imu_input_dim, hidden_dim=64, num_layers=2, dropout=0.2
        )

        # Project both to same dim (256) before gating
        # Vision: 512 → 256
        # IMU: 128 + 4 (gyro_quat) = 132 → 256
        self.v_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.i_proj = nn.Sequential(nn.Linear(132, 256), nn.ReLU(inplace=True))

        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )

        self.decoder = TranslationOnlyDecoder(feat_dim=256, dropout=0.2)

    def forward(self, img_pair, imu_seq, gyro_quat=None):
        v = self.vision_encoder(img_pair)  # (B, 512)
        i = self.imu_encoder(imu_seq)      # (B, 128)

        # Concatenate precomputed rotation to IMU features
        if gyro_quat is not None:
            i = torch.cat([i, gyro_quat], dim=1)  # (B, 132)
        else:
            # Fallback: pad with zeros if no gyro_quat provided
            i = torch.cat([i, torch.zeros(i.size(0), 4, device=i.device)], dim=1)

        v = self.v_proj(v)   # (B, 256)
        i = self.i_proj(i)   # (B, 256)

        g = self.gate(torch.cat([v, i], dim=1))
        fused = g * v + (1.0 - g) * i

        return self.decoder(fused)  # (B, 3)

# Quick test
if __name__ == "__main__":
    def count(m): return sum(p.numel() for p in m.parameters())

    print("=" * 50)
    print("Testing VisionNet (full 7D)...")
    net = VisionNet()
    x = torch.randn(4, 2, 128, 128)
    out = net(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} — should be (4, 7)")
    print(f"  Params: {count(net):,}")

    print("\nTesting IMUNet (translation only)...")
    net = IMUNet()
    x = torch.randn(4, 10, 6)
    out = net(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} — should be (4, 3)")
    print(f"  Params: {count(net):,}")

    print("\nTesting VisualInertialNet (translation only + gyro_quat input)...")
    net = VisualInertialNet()
    img = torch.randn(4, 2, 128, 128)
    imu = torch.randn(4, 10, 6)
    gyro_q = torch.randn(4, 4)
    out = net(img, imu, gyro_q)
    print(f"  Input:  img {img.shape}, imu {imu.shape}, gyro_q {gyro_q.shape}")
    print(f"  Output: {out.shape} — should be (4, 3)")
    print(f"  Params: {count(net):,}")

    print("\nAll tests passed!")
    print("=" * 50)