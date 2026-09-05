"""
src/forecasting/multimodal_forecaster.py
End-to-End Multimodal Spatiotemporal Forecaster fusing:
  1. Continuous Satellite Sequences (Thermal IR + Water Vapor Video)
  2. 500 hPa & 700 hPa Environmental Steering Flow (Annulus 3°-6° Average)
  3. Deep-layer Vertical Wind Shear (VWS: 200 hPa vs 850 hPa Differential Vector & Magnitude)
  4. 2D Subtropical Ridge Grid (20° x 20° at 500 hPa)
Anchored by Physical Persistence Kinematics to guarantee trajectory stability.
"""

import torch
import torch.nn as nn
from src.forecasting.forecaster import torch_haversine

class SatelliteFrameEncoder(nn.Module):
    """
    Encodes spatial satellite imagery per timestep.
    Input: (B, C=2, H=128, W=128) -> Output: (B, 128)
    """
    def __init__(self, in_channels=2, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),         # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),        # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x):
        return self.conv(x)

class SubtropicalRidgeEncoder(nn.Module):
    """
    Encodes the 2D Subtropical Ridge Geopotential Height Grid (20° x 20° at 500 hPa).
    Input: (B, C=1, H=16, W=16) -> Output: (B, 64)
    Directly extracts synoptic ridge axis orientation, pressure gradients, and recurvature weakness.
    """
    def __init__(self, in_channels=1, feat_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, feat_dim)
        )

    def forward(self, x):
        return self.conv(x)

class MultimodalCycloneForecaster(nn.Module):
    """
    Fuses Satellite Video Sequences with Upper-Air Steering, Deep-Layer Vertical Wind Shear (VWS),
    and 2D Subtropical Ridge Grid, predicting neural corrections on top of physical kinematic persistence.
    """
    def __init__(self, img_channels=2, steering_dim=10, ridge_dim=64, hidden_dim=128):
        super().__init__()
        self.steering_dim = steering_dim
        self.ridge_dim = ridge_dim
        self.hidden_dim = hidden_dim

        self.frame_encoder = SatelliteFrameEncoder(in_channels=img_channels, feat_dim=hidden_dim)
        self.ridge_encoder = SubtropicalRidgeEncoder(in_channels=1, feat_dim=ridge_dim)

        # Video Temporal Aggregator (over T=5 timesteps)
        self.video_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Steering & VWS Temporal Aggregator (over T=5 timesteps)
        self.steering_mlp = nn.Sequential(
            nn.Linear(steering_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.steering_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Cross-modal fusion: Video (128) + Steering/VWS (128) + Ridge (64) = 320
        fused_dim = hidden_dim + hidden_dim + ridge_dim

        # Multi-Horizon Decoders (+6h, +12h, +24h)
        self.drop = nn.Dropout(0.3)
        self.dec_6h = nn.Sequential(nn.Linear(fused_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.dec_12h = nn.Sequential(nn.Linear(fused_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.dec_24h = nn.Sequential(nn.Linear(fused_dim, 64), nn.GELU(), nn.Linear(64, 3))

        # Initialize final projection to zero so predictions start strictly at the physical persistence baseline
        for head in [self.dec_6h, self.dec_12h, self.dec_24h]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x_video, x_steering, x_ridge=None, curr_coords=None, prev_coords=None):
        """
        x_video: (B, T=5, C=2, H=128, W=128)
        x_steering: (B, T=5, F=10 or 5)
        x_ridge: (B, 1, 16, 16) or (B, 16, 16)
        curr_coords: (B, 3) -> [lat_t0, lon_t0, wind_t0]
        prev_coords: (B, 3) -> [lat_t-3, lon_t-3, wind_t-3]
        """
        B, T, C, H, W = x_video.shape

        # 1. Normalize steering fields
        st_norm = x_steering.clone()
        if st_norm.shape[-1] < self.steering_dim:
            # Pad with zeros if older 5-dim steering passed
            pad_size = self.steering_dim - st_norm.shape[-1]
            st_norm = torch.cat([st_norm, torch.zeros(B, T, pad_size, device=st_norm.device)], dim=-1)

        st_norm[:, :, 0] = (st_norm[:, :, 0] - 5850.0) / 100.0 # geopotential height anomaly
        st_norm[:, :, 1:] = st_norm[:, :, 1:] / 10.0          # steering velocity & shear in 10 m/s units

        # 2. Encode satellite video frames
        video_flat = x_video.view(B * T, C, H, W)
        frame_feats = self.frame_encoder(video_flat)
        frame_feats = frame_feats.view(B, T, -1) # (B, T, hidden_dim)
        _, h_video = self.video_gru(frame_feats)
        v_ctx = h_video.squeeze(0) # (B, hidden_dim)

        # 3. Encode environmental steering & vertical wind shear currents
        st_feats = self.steering_mlp(st_norm) # (B, T, 64)
        _, h_steering = self.steering_gru(st_feats)
        s_ctx = h_steering.squeeze(0) # (B, hidden_dim)

        # 4. Encode 2D Subtropical Ridge Grid
        if x_ridge is not None:
            if x_ridge.ndim == 3:
                x_r = x_ridge.unsqueeze(1) # (B, 1, 16, 16)
            else:
                x_r = x_ridge
            # Ridge height anomaly: (z - 5850) / 100
            r_norm = (x_r - 5850.0) / 100.0
            r_ctx = self.ridge_encoder(r_norm) # (B, ridge_dim)
        else:
            r_ctx = torch.zeros(B, self.ridge_dim, device=x_video.device)

        # 5. Cross-modal fusion with dropout regularization
        fused = torch.cat([v_ctx, s_ctx, r_ctx], dim=1) # (B, fused_dim=320)
        fused = self.drop(fused)

        c6 = self.dec_6h(fused)
        c12 = self.dec_12h(fused)
        c24 = self.dec_24h(fused)

        lat0 = curr_coords[:, 0:1]
        lon0 = curr_coords[:, 1:2]
        wind0 = curr_coords[:, 2:3]

        # 6. Physical Kinematic Persistence Base
        if prev_coords is not None:
            dt = 3.0
            v_lat = (lat0 - prev_coords[:, 0:1]) / dt
            v_lon = (lon0 - prev_coords[:, 1:2]) / dt
        else:
            v_lat = torch.zeros_like(lat0)
            v_lon = torch.zeros_like(lon0)

        base_6h_lat = lat0 + v_lat * 6.0
        base_6h_lon = lon0 + v_lon * 6.0
        base_12h_lat = lat0 + v_lat * 12.0
        base_12h_lon = lon0 + v_lon * 12.0
        base_24h_lat = lat0 + v_lat * 24.0
        base_24h_lon = lon0 + v_lon * 24.0

        pred_6h = torch.cat([base_6h_lat + c6[:, 0:1], base_6h_lon + c6[:, 1:2], torch.relu(wind0 + c6[:, 2:3])], dim=1)
        pred_12h = torch.cat([base_12h_lat + c12[:, 0:1], base_12h_lon + c12[:, 1:2], torch.relu(wind0 + c12[:, 2:3])], dim=1)
        pred_24h = torch.cat([base_24h_lat + c24[:, 0:1], base_24h_lon + c24[:, 1:2], torch.relu(wind0 + c24[:, 2:3])], dim=1)

        return torch.stack([pred_6h, pred_12h, pred_24h], dim=1)
