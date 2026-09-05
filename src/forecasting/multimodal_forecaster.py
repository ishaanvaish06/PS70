"""
src/forecasting/multimodal_forecaster.py
End-to-End Multimodal Spatiotemporal Forecaster v2 — All Improvements:
  1. Pretrained EfficientNet-B0 satellite encoder (replaces 3-layer CNN)
  2. Temporal attention mechanism (replaces GRU-only compression)
  3. Full-history velocity + acceleration features (5-step regression)
  4. Cross-modal attention fusion
  5. Ridge + VWS + steering flow integration
Anchored by Physical Persistence Kinematics to guarantee trajectory stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.forecasting.forecaster import torch_haversine


class SatelliteFrameEncoder(nn.Module):
    """
    Pretrained EfficientNet-B0 backbone for spatial satellite imagery.
    Input: (B, C=2, H=128, W=128) -> Output: (B, feat_dim=128)
    The first conv layer is replaced to accept 2-channel input (TIR + WV).
    Only the feature extraction layers are used (no classifier head).
    """
    def __init__(self, in_channels=2, feat_dim=128):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        orig_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )
        with torch.no_grad():
            if in_channels <= 3:
                new_conv.weight.copy_(orig_conv.weight[:, :in_channels])
            else:
                nn.init.kaiming_normal_(new_conv.weight)

        base.features[0][0] = new_conv
        # Only keep the feature extraction backbone
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, feat_dim)
        )

    def forward(self, x):
        feat = self.features(x)     # (B, 1280, H', W')
        feat = self.pool(feat)      # (B, 1280, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 1280)
        return self.proj(feat)


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over T timesteps.
    Replaces GRU compression with attention-weighted aggregation.
    Input: (B, T, D) -> Output: (B, D)
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable query token for global aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, D)

        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T + 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T + 1, D)
        out = self.proj(out)

        out = out + residual
        # Return only the CLS token output (global summary)
        return out[:, 0]


class SubtropicalRidgeEncoder(nn.Module):
    """
    Encodes the 2D Subtropical Ridge Geopotential Height Grid (20x20 at 500 hPa).
    Input: (B, C=1, H=16, W=16) -> Output: (B, feat_dim)
    """
    def __init__(self, in_channels=1, feat_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, feat_dim)
        )

    def forward(self, x):
        return self.conv(x)


class KinematicFeatureExtractor(nn.Module):
    """
    Extracts velocity and acceleration features from the full 5-step history
    using differentiable linear regression, rather than just 2-point difference.
    Input: (B, 5, 2) lat/lon history -> Output: (B, 6) [v_lat, v_lon, a_lat, a_lon, curve_lat, curve_lon]
    """
    def __init__(self):
        super().__init__()
        # Time indices for 5 steps: t-24, t-18, t-12, t-6, t (hours ago)
        t = torch.tensor([-24.0, -18.0, -12.0, -6.0, 0.0])
        self.register_buffer('t', t)
        self.register_buffer('t_mean', t.mean())
        self.register_buffer('t_std', t.std())

    def forward(self, coords):
        """
        coords: (B, 5, 2) -> [lat, lon] at each timestep
        Returns: (B, 6) -> [v_lat, v_lon, a_lat, a_lon, curve_lat, curve_lon]
        """
        B = coords.shape[0]
        lat = coords[:, :, 0]  # (B, 5)
        lon = coords[:, :, 1]  # (B, 5)

        # Normalize time for numerical stability
        t_norm = (self.t - self.t_mean) / self.t_std  # (5,)

        features = []
        for coord_seq in [lat, lon]:
            # Linear regression: coord = v * t + b  (velocity)
            t_batch = t_norm.unsqueeze(0).expand(B, -1)  # (B, 5)
            t_mean = t_batch.mean(dim=1, keepdim=True)
            c_mean = coord_seq.mean(dim=1, keepdim=True)

            t_centered = t_batch - t_mean
            c_centered = coord_seq - c_mean

            # Least-squares velocity: v = sum((t-t_mean)*(c-c_mean)) / sum((t-t_mean)^2)
            numerator = (t_centered * c_centered).sum(dim=1)
            denominator = (t_centered ** 2).sum(dim=1).clamp(min=1e-6)
            velocity = numerator / denominator  # (B,)
            features.append(velocity)

            # Residual = quadratic component (acceleration proxy)
            linear_pred = velocity.unsqueeze(1) * t_centered + c_mean
            residual = coord_seq - linear_pred
            # Acceleration: difference between recent and early residuals
            accel = (residual[:, -1] - residual[:, 0]) / 6.0  # (B,)
            features.append(accel)

            # Curvature: max deviation from linear fit (normalized)
            curve = residual.abs().max(dim=1).values  # (B,)
            features.append(curve / 10.0)

        return torch.stack(features, dim=1)  # (B, 6)


class CrossModalAttention(nn.Module):
    """
    Cross-attention between video context and steering/ridge context.
    Allows steering features to attend to relevant satellite patterns.
    """
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        # query: (B, 1, D), key_value: (B, S, D)
        residual = query
        q = self.norm1(query)
        kv = self.norm1(key_value)
        attn_out, _ = self.attn(q, kv, kv)
        query = residual + attn_out
        query = query + self.ffn(self.norm2(query))
        return query.squeeze(1)


class MultimodalCycloneForecaster(nn.Module):
    """
    v2 Forecaster with:
      - Pretrained EfficientNet-B0 satellite encoder
      - Temporal attention (replaces GRU)
      - Kinematic features (velocity + acceleration from full history)
      - Cross-modal attention fusion
      - Ridge + VWS + steering integration
    """
    def __init__(self, img_channels=2, steering_dim=10, ridge_dim=64, hidden_dim=128):
        super().__init__()
        self.steering_dim = steering_dim
        self.ridge_dim = ridge_dim
        self.hidden_dim = hidden_dim

        # --- Satellite Encoder: Pretrained EfficientNet-B0 ---
        self.frame_encoder = SatelliteFrameEncoder(in_channels=img_channels, feat_dim=hidden_dim)

        # --- Temporal Attention (replaces GRU) ---
        self.video_temporal_attn = TemporalAttention(dim=hidden_dim, num_heads=4, dropout=0.15)

        # --- Steering MLP + Temporal Attention ---
        self.steering_mlp = nn.Sequential(
            nn.Linear(steering_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.steering_temporal_attn = TemporalAttention(dim=64, num_heads=4, dropout=0.15)

        # --- Ridge Encoder ---
        self.ridge_encoder = SubtropicalRidgeEncoder(in_channels=1, feat_dim=ridge_dim)

        # --- Kinematic Feature Extractor (velocity + acceleration from full history) ---
        self.kinematic_extractor = KinematicFeatureExtractor()
        self.kinematic_dim = 6  # v_lat, v_lon, a_lat, a_lon, curve_lat, curve_lon

        # --- Cross-Modal Attention (project steering/ridge to hidden_dim first) ---
        self.steering_proj = nn.Linear(64, hidden_dim)
        self.cross_attn_v_s = CrossModalAttention(dim=hidden_dim, num_heads=2, dropout=0.15)

        # --- Fusion ---
        # video(128) + steering(64) + ridge(64) + kinematic(6) + curr_coords(3) = 265
        fused_dim = hidden_dim + 64 + ridge_dim + self.kinematic_dim + 3

        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_drop = nn.Dropout(0.3)

        # --- Multi-Horizon Decoders with residual blocks ---
        decoder_hidden = 128
        self.dec_6h = self._make_decoder(fused_dim, decoder_hidden)
        self.dec_12h = self._make_decoder(fused_dim, decoder_hidden)
        self.dec_24h = self._make_decoder(fused_dim, decoder_hidden)

        # Initialize final projections to zero (start at persistence)
        for head in [self.dec_6h, self.dec_12h, self.dec_24h]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _make_decoder(self, in_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )

    def _compute_kinematic_base(self, curr_coords, x_video):
        """
        Compute persistence base using full-history velocity regression
        (not just 2-point difference).
        """
        lat0 = curr_coords[:, 0:1]
        lon0 = curr_coords[:, 1:2]
        wind0 = curr_coords[:, 2:3]

        B = x_video.shape[0]
        device = x_video.device

        # Extract full lat/lon history from input features
        # x_video doesn't contain coords directly; use curr_coords and prev_coords approach
        # But we have kinematic_extractor for when coords_history is available
        # For now, use curr_coords and prev_coords (passed separately)
        return lat0, lon0, wind0

    def forward(self, x_video, x_steering, x_ridge=None, curr_coords=None, prev_coords=None,
                coords_history=None):
        """
        x_video: (B, T=5, C=2, H=128, W=128)
        x_steering: (B, T=5, F=10 or 5)
        x_ridge: (B, 1, 16, 16) or (B, 16, 16)
        curr_coords: (B, 3) -> [lat_t0, lon_t0, wind_t0]
        prev_coords: (B, 3) -> [lat_t-3, lon_t-3, wind_t-3] (kept for compatibility)
        coords_history: (B, 5, 2) -> full [lat, lon] history (optional, preferred)
        """
        B, T, C, H, W = x_video.shape

        # --- 1. Normalize steering ---
        st_norm = x_steering.clone()
        if st_norm.shape[-1] < self.steering_dim:
            pad_size = self.steering_dim - st_norm.shape[-1]
            st_norm = torch.cat([st_norm, torch.zeros(B, T, pad_size, device=st_norm.device)], dim=-1)

        st_norm[:, :, 0] = (st_norm[:, :, 0] - 5850.0) / 100.0
        st_norm[:, :, 1:] = st_norm[:, :, 1:] / 10.0

        # --- 2. Encode satellite frames + temporal attention ---
        video_flat = x_video.view(B * T, C, H, W)
        frame_feats = self.frame_encoder(video_flat)  # (B*T, hidden_dim)
        frame_feats = frame_feats.view(B, T, -1)  # (B, T, hidden_dim)
        v_ctx = self.video_temporal_attn(frame_feats)  # (B, hidden_dim)

        # --- 3. Encode steering + temporal attention ---
        st_feats = self.steering_mlp(st_norm)  # (B, T, 64)
        s_ctx = self.steering_temporal_attn(st_feats)  # (B, 64)
        # Project to hidden_dim for cross-attention
        s_ctx_expanded = s_ctx.unsqueeze(1)  # (B, 1, 64)

        # --- 4. Encode ridge ---
        if x_ridge is not None:
            if x_ridge.ndim == 3:
                x_r = x_ridge.unsqueeze(1)
            else:
                x_r = x_ridge
            r_norm = (x_r - 5850.0) / 100.0
            r_ctx = self.ridge_encoder(r_norm)  # (B, ridge_dim)
        else:
            r_ctx = torch.zeros(B, self.ridge_dim, device=x_video.device)

        # --- 5. Cross-modal attention: video attends to steering ---
        v_ctx_2d = v_ctx.unsqueeze(1)  # (B, 1, hidden_dim)
        s_feats_proj = self.steering_proj(st_feats)  # (B, T, hidden_dim)
        v_enhanced = self.cross_attn_v_s(v_ctx_2d, s_feats_proj)  # (B, hidden_dim)

        # --- 6. Kinematic features ---
        if coords_history is not None:
            kin_feat = self.kinematic_extractor(coords_history)  # (B, 6)
        elif prev_coords is not None:
            # Fallback: compute from curr and prev (2-point)
            lat0 = curr_coords[:, 0:1]
            lon0 = curr_coords[:, 1:2]
            v_lat = (lat0 - prev_coords[:, 0:1]) / 3.0
            v_lon = (lon0 - prev_coords[:, 1:2]) / 3.0
            kin_feat = torch.cat([
                v_lat, v_lon,
                torch.zeros_like(v_lat), torch.zeros_like(v_lon),  # no acceleration
                torch.zeros_like(v_lat), torch.zeros_like(v_lon)   # no curvature
            ], dim=1)  # (B, 6)
        else:
            kin_feat = torch.zeros(B, self.kinematic_dim, device=x_video.device)

        # --- 7. Fusion ---
        fused = torch.cat([v_enhanced if v_enhanced.dim() == 2 else v_enhanced.squeeze(1),
                           s_ctx, r_ctx, kin_feat, curr_coords], dim=1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_drop(fused)

        # --- 8. Decode corrections ---
        c6 = self.dec_6h(fused)
        c12 = self.dec_12h(fused)
        c24 = self.dec_24h(fused)

        lat0 = curr_coords[:, 0:1]
        lon0 = curr_coords[:, 1:2]
        wind0 = curr_coords[:, 2:3]

        # --- 9. Physical Kinematic Persistence Base (using regression velocity) ---
        if coords_history is not None:
            kin = self.kinematic_extractor(coords_history)
            v_lat = kin[:, 0:1]
            v_lon = kin[:, 1:2]
        elif prev_coords is not None:
            v_lat = (lat0 - prev_coords[:, 0:1]) / 3.0
            v_lon = (lon0 - prev_coords[:, 1:2]) / 3.0
        else:
            v_lat = torch.zeros_like(lat0)
            v_lon = torch.zeros_like(lon0)

        base_6h_lat = lat0 + v_lat * 6.0
        base_6h_lon = lon0 + v_lon * 6.0
        base_12h_lat = lat0 + v_lat * 12.0
        base_12h_lon = lon0 + v_lon * 12.0
        base_24h_lat = lat0 + v_lat * 24.0
        base_24h_lon = lon0 + v_lon * 24.0

        pred_6h = torch.cat([
            base_6h_lat + c6[:, 0:1],
            base_6h_lon + c6[:, 1:2],
            torch.relu(wind0 + c6[:, 2:3])
        ], dim=1)
        pred_12h = torch.cat([
            base_12h_lat + c12[:, 0:1],
            base_12h_lon + c12[:, 1:2],
            torch.relu(wind0 + c12[:, 2:3])
        ], dim=1)
        pred_24h = torch.cat([
            base_24h_lat + c24[:, 0:1],
            base_24h_lon + c24[:, 1:2],
            torch.relu(wind0 + c24[:, 2:3])
        ], dim=1)

        return torch.stack([pred_6h, pred_12h, pred_24h], dim=1)
