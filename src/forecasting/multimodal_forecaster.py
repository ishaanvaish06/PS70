"""
src/forecasting/multimodal_forecaster.py
End-to-End Multimodal Spatiotemporal Forecaster fusing:
  1. Continuous Satellite Sequences (Thermal IR + Visible Video)
  2. 500 hPa & 700 hPa Atmospheric Steering Flow (Subtropical Ridge & Steering Currents)
Outputs multi-horizon trajectory coordinates and wind speed (+6h, +12h, +24h).
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

class MultimodalCycloneForecaster(nn.Module):
    """
    Fuses Satellite Video Sequences with 500 hPa / 700 hPa Atmospheric Steering currents.
    """
    def __init__(self, img_channels=2, steering_dim=5, hidden_dim=128):
        super().__init__()
        self.frame_encoder = SatelliteFrameEncoder(in_channels=img_channels, feat_dim=hidden_dim)

        # Video Temporal Aggregator (over T=5 timesteps)
        self.video_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Steering Temporal Aggregator (over T=5 timesteps)
        self.steering_mlp = nn.Sequential(
            nn.Linear(steering_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.steering_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Cross-modal fusion
        fused_dim = hidden_dim + hidden_dim

        # Multi-Horizon Decoders (+6h, +12h, +24h)
        self.dec_6h = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.dec_12h = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.dec_24h = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x_video, x_steering, curr_coords):
        """
        x_video: (B, T=5, C=2, H=128, W=128)
        x_steering: (B, T=5, F=5) -> [z_500, u_500, v_500, u_700, v_700]
        curr_coords: (B, 3) -> [lat_t0, lon_t0, wind_t0]
        """
        B, T, C, H, W = x_video.shape

        # 1. Encode satellite video frames
        video_flat = x_video.view(B * T, C, H, W)
        frame_feats = self.frame_encoder(video_flat)
        frame_feats = frame_feats.view(B, T, -1) # (B, T, hidden_dim)
        _, h_video = self.video_gru(frame_feats)
        v_ctx = h_video.squeeze(0) # (B, hidden_dim)

        # 2. Encode 500hPa steering currents
        st_feats = self.steering_mlp(x_steering) # (B, T, 64)
        _, h_steering = self.steering_gru(st_feats)
        s_ctx = h_steering.squeeze(0) # (B, hidden_dim)

        # 3. Fuse video motion vectors + upper-air steering flow
        fused = torch.cat([v_ctx, s_ctx], dim=1) # (B, fused_dim)

        c6 = self.dec_6h(fused)
        c12 = self.dec_12h(fused)
        c24 = self.dec_24h(fused)

        lat0, lon0, wind0 = curr_coords[:, 0:1], curr_coords[:, 1:2], curr_coords[:, 2:3]

        pred_6h = torch.cat([lat0 + c6[:, 0:1], lon0 + c6[:, 1:2], torch.relu(wind0 + c6[:, 2:3])], dim=1)
        pred_12h = torch.cat([lat0 + c12[:, 0:1], lon0 + c12[:, 1:2], torch.relu(wind0 + c12[:, 2:3])], dim=1)
        pred_24h = torch.cat([lat0 + c24[:, 0:1], lon0 + c24[:, 1:2], torch.relu(wind0 + c24[:, 2:3])], dim=1)

        return torch.stack([pred_6h, pred_12h, pred_24h], dim=1)
