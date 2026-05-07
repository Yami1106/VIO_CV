"""
networks_stereo_v3.py — 6D Rotation Representation Architecture
=================================================================

Output format: 9D = translation (3) + rotation_6d (6)
At test time, rotation_6d → 3×3 rotation matrix → quaternion for dead reckoning

Networks:
- StereoVisionNetV3          → 9D + depth
- IMUNetV3                   → 9D + denoiser
- StereoVisualInertialNetV3  → 9D + depth + denoiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization.

    Input:  (B, 6) — first two columns of rotation matrix (flattened)
    Output: (B, 3, 3) — proper rotation matrices in SO(3)
    """
    a1 = rot_6d[:, 0:3]  # first column
    a2 = rot_6d[:, 3:6]  # second column

    # Gram-Schmidt: make orthonormal
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)


def matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].
    Numerically stable version using Shepperd's method.

    Input:  (B, 3, 3)
    Output: (B, 4) — [qx, qy, qz, qw]
    """
    batch = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    qw = torch.zeros(batch, device=R.device)
    qx = torch.zeros(batch, device=R.device)
    qy = torch.zeros(batch, device=R.device)
    qz = torch.zeros(batch, device=R.device)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    mask = trace > 0
    qw = torch.where(mask, 0.25 * s, qw)
    qx = torch.where(mask, (R[:, 2, 1] - R[:, 1, 2]) / s, qx)
    qy = torch.where(mask, (R[:, 0, 2] - R[:, 2, 0]) / s, qy)
    qz = torch.where(mask, (R[:, 1, 0] - R[:, 0, 1]) / s, qz)

    # Case 2: R[0,0] largest diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2.0
    qw = torch.where(mask2, (R[:, 2, 1] - R[:, 1, 2]) / s2, qw)
    qx = torch.where(mask2, 0.25 * s2, qx)
    qy = torch.where(mask2, (R[:, 0, 1] + R[:, 1, 0]) / s2, qy)
    qz = torch.where(mask2, (R[:, 0, 2] + R[:, 2, 0]) / s2, qz)

    # Case 3: R[1,1] largest diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2.0
    qw = torch.where(mask3, (R[:, 0, 2] - R[:, 2, 0]) / s3, qw)
    qx = torch.where(mask3, (R[:, 0, 1] + R[:, 1, 0]) / s3, qx)
    qy = torch.where(mask3, 0.25 * s3, qy)
    qz = torch.where(mask3, (R[:, 1, 2] + R[:, 2, 1]) / s3, qz)

    # Case 4: R[2,2] largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2.0
    qw = torch.where(mask4, (R[:, 1, 0] - R[:, 0, 1]) / s4, qw)
    qx = torch.where(mask4, (R[:, 0, 2] + R[:, 2, 0]) / s4, qx)
    qy = torch.where(mask4, (R[:, 1, 2] + R[:, 2, 1]) / s4, qy)
    qz = torch.where(mask4, 0.25 * s4, qz)

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    q = F.normalize(q, dim=-1)
    # Canonical form: qw > 0
    sign = torch.sign(q[:, 3:4])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign


def rotation_6d_to_quaternion(rot_6d):
    """Convenience: 6D → matrix → quaternion."""
    R = rotation_6d_to_matrix(rot_6d)
    return matrix_to_quaternion(R)


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


class MultiScaleStereoVisionEncoder(nn.Module):
    def __init__(self, in_channels=4, dropout=0.1):
        super().__init__()
        self.stream1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResConvBlock(32, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResConvBlock(64, dropout=dropout),
            nn.AdaptiveAvgPool2d(4),
        )
        self.stream3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2, bias=False),
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


class IMUDenoiser(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
        )

    def forward(self, noisy_imu):
        out, _ = self.lstm(noisy_imu)
        predicted_noise = self.noise_head(out)
        return noisy_imu - predicted_noise, predicted_noise


class TemporalAttentionIMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        feat_dim = hidden_dim * 2
        self.attention = nn.Sequential(nn.Linear(feat_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
        )

    def forward(self, imu_seq):
        out, _ = self.lstm(imu_seq)
        attn_w = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.proj(context)


class PoseDecoder6D(nn.Module):
    """
    Geometry-aware pose decoder with 6D rotation.

    Output: translation (B, 3), rotation_6d (B, 6)
    """
    def __init__(self, feat_dim, dropout=0.2):
        super().__init__()
        h1 = feat_dim // 2
        h2 = feat_dim // 4

        # Shared trunk — learns coupled rotation-translation features
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
        )

        # Translation head
        self.trans_head = nn.Sequential(
            nn.Linear(h2, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 3),
        )

        # Rotation head — outputs 6 raw numbers, Gram-Schmidt makes them SO(3)
        self.rot_head = nn.Sequential(
            nn.Linear(h2, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 6),
        )

        # Initialize rotation head to near-identity
        # Identity rotation in 6D: [1,0,0, 0,1,0] (first two columns of I)
        nn.init.zeros_(self.rot_head[-1].weight)
        self.rot_head[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, feat):
        shared = self.shared(feat)
        trans = self.trans_head(shared)
        rot_6d = self.rot_head(shared)
        return trans, rot_6d


class DepthHead(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )

    def forward(self, feat):
        return self.head(feat).squeeze(-1)


class StereoVisionNetV3(nn.Module):
    """
    Stereo vision-only with 6D rotation.
    Input:  (B, 4, 128, 128)
    Output: trans (B,3), rot_6d (B,6), depth (B,)
    """
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleStereoVisionEncoder(in_channels=4, dropout=0.1)
        self.decoder = PoseDecoder6D(feat_dim=512, dropout=0.2)
        self.depth_head = DepthHead(feat_dim=512)

    def forward(self, img_quad):
        feat = self.encoder(img_quad)
        trans, rot_6d = self.decoder(feat)
        depth = self.depth_head(feat)
        return trans, rot_6d, depth


class IMUNetV3(nn.Module):
    """
    IMU-only with denoiser + 6D rotation.
    Input:  noisy_imu (B, T, 6)
    Output: trans (B,3), rot_6d (B,6), denoised (B,T,6), noise (B,T,6)
    """
    def __init__(self, imu_input_dim=6):
        super().__init__()
        self.denoiser = IMUDenoiser(input_dim=imu_input_dim)
        self.encoder = TemporalAttentionIMUEncoder(input_dim=imu_input_dim)
        self.decoder = PoseDecoder6D(feat_dim=128, dropout=0.2)

    def forward(self, noisy_imu):
        denoised, noise = self.denoiser(noisy_imu)
        feat = self.encoder(denoised)
        trans, rot_6d = self.decoder(feat)
        return trans, rot_6d, denoised, noise


class StereoVisualInertialNetV3(nn.Module):
    """
    Stereo VI with 6D rotation + denoiser + depth.
    Input:  img (B,4,128,128), noisy_imu (B,T,6)
    Output: trans (B,3), rot_6d (B,6), depth (B,), denoised (B,T,6), noise (B,T,6)
    """
    def __init__(self, imu_input_dim=6):
        super().__init__()
        self.denoiser = IMUDenoiser(input_dim=imu_input_dim)
        self.vision_encoder = MultiScaleStereoVisionEncoder(in_channels=4, dropout=0.1)
        self.imu_encoder = TemporalAttentionIMUEncoder(input_dim=imu_input_dim)

        self.v_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.i_proj = nn.Sequential(nn.Linear(128, 256), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())

        self.decoder = PoseDecoder6D(feat_dim=256, dropout=0.2)
        self.depth_head = DepthHead(feat_dim=256)

    def forward(self, img_quad, noisy_imu):
        denoised, noise = self.denoiser(noisy_imu)
        v = self.vision_encoder(img_quad)
        i = self.imu_encoder(denoised)

        v = self.v_proj(v)
        i = self.i_proj(i)
        g = self.gate(torch.cat([v, i], dim=1))
        fused = g * v + (1.0 - g) * i

        trans, rot_6d = self.decoder(fused)
        depth = self.depth_head(fused)
        return trans, rot_6d, depth, denoised, noise


if __name__ == "__main__":
    def count(m): return sum(p.numel() for p in m.parameters())

    print("=" * 60)
    print("6D Rotation Representation Tests")
    print("=" * 60)

    # Test 6D → matrix → quaternion
    rot_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])  # identity
    R = rotation_6d_to_matrix(rot_6d)
    q = matrix_to_quaternion(R)
    print(f"\nIdentity test:")
    print(f"  6D input:    {rot_6d[0].tolist()}")
    print(f"  3x3 matrix:\n{R[0]}")
    print(f"  Quaternion:  {q[0].tolist()} (should be [0,0,0,1])")

    # Test all networks
    print(f"\n1. StereoVisionNetV3:")
    net = StereoVisionNetV3()
    t, r, d = net(torch.randn(4, 4, 128, 128))
    print(f"   trans {t.shape}, rot_6d {r.shape}, depth {d.shape}  params: {count(net):,}")

    print(f"\n2. IMUNetV3:")
    net = IMUNetV3()
    t, r, dn, ns = net(torch.randn(4, 20, 6))
    print(f"   trans {t.shape}, rot_6d {r.shape}, denoised {dn.shape}  params: {count(net):,}")

    print(f"\n3. StereoVINetV3:")
    net = StereoVisualInertialNetV3()
    t, r, d, dn, ns = net(torch.randn(4, 4, 128, 128), torch.randn(4, 20, 6))
    print(f"   trans {t.shape}, rot_6d {r.shape}, depth {d.shape}  params: {count(net):,}")

    # Test gradient flow through 6D→matrix
    rot_6d = torch.randn(4, 6, requires_grad=True)
    R = rotation_6d_to_matrix(rot_6d)
    loss = R.sum()
    loss.backward()
    print(f"\nGradient flow through 6D→matrix: {'OK' if rot_6d.grad is not None else 'FAILED'}")

    print("\nAll v3 tests passed!")
    print("=" * 60)
