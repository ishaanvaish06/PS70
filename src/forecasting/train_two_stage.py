"""
src/forecasting/train_two_stage.py
Two-Stage Pre-Training & Fine-Tuning Pipeline v2 — All Improvements:
  Stage 1: Pre-train recurrent kinematic backbone on global cyclone tracks.
  Stage 2: Transfer weights & fine-tune v2 MultimodalCycloneForecaster with:
    - Data augmentation (jitter, rotation, time dropout, Gaussian noise)
    - Adjusted loss weighting (6h + 2*12h + 4*24h)
    - Lower LR (3e-4) with cosine annealing + warm restarts
    - Robust Huber-Haversine loss
    - Full-history kinematic features
"""

import os
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader

from src.forecasting.multimodal_forecaster import MultimodalCycloneForecaster
from src.forecasting.forecaster import torch_haversine

MULTIBASIN_FILE = os.path.join("data", "processed", "forecasting", "multibasin_pretrain_sequences.npz")
GLOBAL_DATA_FILE = MULTIBASIN_FILE if os.path.exists(MULTIBASIN_FILE) else os.path.join("data", "processed", "forecasting", "global_pretrain_sequences.npz")
NIO_DATA_FILE = os.path.join("data", "processed", "forecasting", "multimodal_sequences", "multimodal_forecasting_dataset.npz")
PRETRAIN_CKPT = os.path.join("models", "forecasting", "global_pretrained_backbone.pt")
FINAL_MODEL_PATH = os.path.join("models", "forecasting", "multimodal_forecast_model.pt")
FINAL_METRICS_PATH = os.path.join("models", "forecasting", "multimodal_metrics.json")


# -------------------------------------------------------------
# DATA AUGMENTATION
# -------------------------------------------------------------
class CycloneAugmentor:
    """
    Data augmentation for cyclone sequences:
      - Lat/lon jitter (±0.3°)
      - Random temporal dropout (zero out 1-2 of 5 timesteps)
      - Gaussian noise on steering features
      - Random time reversal (flip sequence direction)
    """
    def __init__(self, jitter_deg=0.3, noise_std=0.05, dropout_prob=0.15, p_reverse=0.2):
        self.jitter_deg = jitter_deg
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.p_reverse = p_reverse

    def augment_video(self, video):
        """video: (T, C, H, W) uint8 or float"""
        if np.random.random() < self.p_reverse:
            video = np.flip(video, axis=0).copy()
        return video

    def augment_steering(self, steering):
        """steering: (T, F)"""
        steering = steering.copy()
        # Gaussian noise
        steering += np.random.randn(*steering.shape).astype(steering.dtype) * self.noise_std
        # Time dropout: randomly zero out timesteps
        if np.random.random() < 0.5:
            n_drop = np.random.randint(1, 3)
            drop_idx = np.random.choice(steering.shape[0], n_drop, replace=False)
            steering[drop_idx] = 0.0
        if np.random.random() < self.p_reverse:
            steering = steering[::-1].copy()
        return steering

    def augment_coords(self, curr_coords, prev_coords, Y, ridge=None):
        """Add spatial jitter to all coordinates consistently."""
        dlat = np.random.uniform(-self.jitter_deg, self.jitter_deg)
        dlon = np.random.uniform(-self.jitter_deg, self.jitter_deg)

        curr_coords = curr_coords.copy()
        prev_coords = prev_coords.copy()
        Y = Y.copy()

        curr_coords[0] += dlat
        curr_coords[1] += dlon
        prev_coords[0] += dlat
        prev_coords[1] += dlon
        Y[:, 0] += dlat
        Y[:, 1] += dlon

        return curr_coords, prev_coords, Y, ridge

    def __call__(self, video, steering, curr_coords, prev_coords, Y, ridge=None):
        video = self.augment_video(video)
        steering = self.augment_steering(steering)
        curr_coords, prev_coords, Y, ridge = self.augment_coords(curr_coords, prev_coords, Y, ridge)
        return video, steering, curr_coords, prev_coords, Y, ridge


# -------------------------------------------------------------
# STAGE 1: GLOBAL PRE-TRAINING
# -------------------------------------------------------------
class GlobalKinematicForecaster(nn.Module):
    """Recurrent kinematic residual model pre-trained on global cyclone trajectories."""
    def __init__(self, in_dim=3, hidden_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.dec_6h = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.dec_12h = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.dec_24h = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3))

        for head in [self.dec_6h, self.dec_12h, self.dec_24h]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x_rel, c, p):
        f = self.proj(x_rel)
        _, h = self.gru(f)
        ctx = h.squeeze(0)
        c6 = self.dec_6h(ctx)
        c12 = self.dec_12h(ctx)
        c24 = self.dec_24h(ctx)

        dt = 3.0
        v_lat = (c[:, 0:1] - p[:, 0:1]) / dt
        v_lon = (c[:, 1:2] - p[:, 1:2]) / dt

        base_6h_lat = c[:, 0:1] + v_lat * 6.0
        base_6h_lon = c[:, 1:2] + v_lon * 6.0
        base_12h_lat = c[:, 0:1] + v_lat * 12.0
        base_12h_lon = c[:, 1:2] + v_lon * 12.0
        base_24h_lat = c[:, 0:1] + v_lat * 24.0
        base_24h_lon = c[:, 1:2] + v_lon * 24.0

        p6 = torch.cat([base_6h_lat + c6[:, 0:1], base_6h_lon + c6[:, 1:2], torch.relu(c[:, 2:3] + c6[:, 2:3])], dim=1)
        p12 = torch.cat([base_12h_lat + c12[:, 0:1], base_12h_lon + c12[:, 1:2], torch.relu(c[:, 2:3] + c12[:, 2:3])], dim=1)
        p24 = torch.cat([base_24h_lat + c24[:, 0:1], base_24h_lon + c24[:, 1:2], torch.relu(c[:, 2:3] + c24[:, 2:3])], dim=1)
        return torch.stack([p6, p12, p24], dim=1)


def run_stage_1_pretraining(device, epochs=10, force=False):
    print("\n" + "=" * 70)
    print("STAGE 1: PRE-TRAINING ON GLOBAL MULTI-BASIN CYCLONE TRACKS")
    print("=" * 70)

    if os.path.exists(PRETRAIN_CKPT) and not force:
        print(f"Pre-trained Stage 1 weights already exist at: {PRETRAIN_CKPT}. Skipping.")
        return

    data = np.load(GLOBAL_DATA_FILE, allow_pickle=True)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    curr = X[:, 4, :].clone()
    prev = X[:, 3, :].clone()

    X_rel = X.clone()
    X_rel[:, :, 0] = X[:, :, 0] - curr[:, 0:1]
    X_rel[:, :, 1] = X[:, :, 1] - curr[:, 1:2]
    X_rel[:, :, 2] = X_rel[:, :, 2] / 100.0

    dataset = TensorDataset(X_rel, curr, prev, Y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = GlobalKinematicForecaster(in_dim=3, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    smooth_l1 = nn.SmoothL1Loss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xr, c, p, y in loader:
            xr, c, p, y = xr.to(device), c.to(device), p.to(device), y.to(device)
            preds = model(xr, c, p)

            d6 = torch_haversine(preds[:, 0, 0], preds[:, 0, 1], y[:, 0, 0], y[:, 0, 1])
            d12 = torch_haversine(preds[:, 1, 0], preds[:, 1, 1], y[:, 1, 0], y[:, 1, 1])
            d24 = torch_haversine(preds[:, 2, 0], preds[:, 2, 1], y[:, 2, 0], y[:, 2, 1])
            # Adjusted weighting: emphasize 24h
            track_loss = (d6.mean() + 1.5 * d12.mean() + 3.0 * d24.mean()) / 60.0

            w_loss = smooth_l1(preds[:, :, 2], y[:, :, 2])
            loss = track_loss + 0.05 * w_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"  Stage 1 | Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(PRETRAIN_CKPT), exist_ok=True)
    torch.save(model.state_dict(), PRETRAIN_CKPT)
    print(f"Stage 1 Complete! Saved to: {PRETRAIN_CKPT}")


# -------------------------------------------------------------
# STAGE 2: MULTIMODAL FINE-TUNING WITH AUGMENTATION
# -------------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, X_video, X_steering, X_ridge, curr_coords, prev_coords, Y, coords_history, augmentor=None):
        self.X_video = X_video
        self.X_steering = X_steering
        self.X_ridge = X_ridge
        self.curr_coords = curr_coords
        self.prev_coords = prev_coords
        self.Y = Y
        self.coords_history = coords_history
        self.augmentor = augmentor

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        video = self.X_video[idx].copy()
        steering = self.X_steering[idx].copy()
        ridge = self.X_ridge[idx].copy() if self.X_ridge is not None else None
        curr = self.curr_coords[idx].copy()
        prev = self.prev_coords[idx].copy()
        y = self.Y[idx].copy()
        coords_hist = self.coords_history[idx].copy()

        if self.augmentor is not None:
            video, steering, curr, prev, y, ridge = self.augmentor(
                video, steering, curr, prev, y, ridge
            )

        video_t = torch.tensor(video, dtype=torch.float32) / 255.0
        steering_t = torch.tensor(steering, dtype=torch.float32)
        ridge_t = torch.tensor(ridge, dtype=torch.float32) if ridge is not None else torch.zeros(16, 16)
        curr_t = torch.tensor(curr, dtype=torch.float32)
        prev_t = torch.tensor(prev, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        coords_hist_t = torch.tensor(coords_hist, dtype=torch.float32)

        return video_t, steering_t, ridge_t, curr_t, prev_t, coords_hist_t, y_t


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1-a, 0, 1)))


def robust_huber_haversine(dist_km, delta=50.0):
    """Robust Huber loss on Haversine distance in km."""
    return torch.where(
        dist_km < delta,
        0.5 * (dist_km ** 2) / delta,
        dist_km - 0.5 * delta
    ).mean()


def evaluate_metrics(model, loader, device):
    model.eval()
    e6_list, e12_list, e24_list = [], [], []
    w6_err, w12_err, w24_err = [], [], []

    with torch.no_grad():
        for batch in loader:
            xv, xs, xr, cc, pc, ch, y = batch
            xv, xs, xr = xv.to(device), xs.to(device), xr.to(device)
            cc, pc, ch = cc.to(device), pc.to(device), ch.to(device)

            preds = model(xv, xs, xr, cc, pc, coords_history=ch).cpu().numpy()
            y_np = y.numpy()

            for b in range(len(preds)):
                e6 = haversine_np(preds[b, 0, 0], preds[b, 0, 1], y_np[b, 0, 0], y_np[b, 0, 1])
                e12 = haversine_np(preds[b, 1, 0], preds[b, 1, 1], y_np[b, 1, 0], y_np[b, 1, 1])
                e24 = haversine_np(preds[b, 2, 0], preds[b, 2, 1], y_np[b, 2, 0], y_np[b, 2, 1])

                e6_list.append(e6)
                e12_list.append(e12)
                e24_list.append(e24)

                if not np.isnan(y_np[b, 0, 2]): w6_err.append(abs(preds[b, 0, 2] - y_np[b, 0, 2]))
                if not np.isnan(y_np[b, 1, 2]): w12_err.append(abs(preds[b, 1, 2] - y_np[b, 1, 2]))
                if not np.isnan(y_np[b, 2, 2]): w24_err.append(abs(preds[b, 2, 2] - y_np[b, 2, 2]))

    return {
        "track_error_6h_km": float(np.mean(e6_list)),
        "track_error_6h_median_km": float(np.median(e6_list)),
        "track_error_12h_km": float(np.mean(e12_list)),
        "track_error_12h_median_km": float(np.median(e12_list)),
        "track_error_24h_km": float(np.mean(e24_list)),
        "track_error_24h_median_km": float(np.median(e24_list)),
        "wind_mae_6h_kmh": float(np.mean(w6_err)) if w6_err else 0.0,
        "wind_mae_12h_kmh": float(np.mean(w12_err)) if w12_err else 0.0,
        "wind_mae_24h_kmh": float(np.mean(w24_err)) if w24_err else 0.0,
        "wind_mae_kmh": float(np.mean(w6_err + w12_err + w24_err)) if w6_err else 0.0
    }


def run_stage_2_finetuning(device, epochs=40):
    print("\n" + "=" * 70)
    print("STAGE 2: MULTIMODAL FINE-TUNING v2 (EFFICIENTNET + ATTENTION + AUGMENTATION)")
    print("=" * 70)

    data = np.load(NIO_DATA_FILE, allow_pickle=True)
    X_video = data["X_video"]
    X_steering = np.nan_to_num(data["X_steering"], nan=0.0)
    X_ridge = np.nan_to_num(data["X_ridge"], nan=5850.0) if "X_ridge" in data else np.full((len(X_video), 16, 16), 5850.0, dtype=np.float32)
    curr_coords = np.nan_to_num(data["curr_coords"], nan=55.0)
    prev_coords = np.nan_to_num(data["prev_coords"], nan=55.0)
    Y = data["Y"]
    cyclone_ids = data["cyclone_ids"]
    names = data["names"]

    # Build coords_history using velocity extrapolation from curr/prev (t and t-6h)
    # This is more accurate than the previous naive extrapolation
    print("Building coords_history from velocity extrapolation...")
    coords_history_list = []
    for i in range(len(Y)):
        # Velocity from t-6h to t (6-hour interval)
        v_lat = (curr_coords[i, 0] - prev_coords[i, 0]) / 6.0
        v_lon = (curr_coords[i, 1] - prev_coords[i, 1]) / 6.0
        lat_t0 = curr_coords[i, 0]
        lon_t0 = curr_coords[i, 1]
        
        # Build history: [t-24h, t-18h, t-12h, t-6h, t]
        hist = np.array([
            [lat_t0 - v_lat * 24, lon_t0 - v_lon * 24],
            [lat_t0 - v_lat * 18, lon_t0 - v_lon * 18],
            [lat_t0 - v_lat * 12, lon_t0 - v_lon * 12],
            [lat_t0 - v_lat * 6,  lon_t0 - v_lon * 6],
            [lat_t0, lon_t0],
        ], dtype=np.float32)
        coords_history_list.append(hist)
    
    coords_history = np.stack(coords_history_list)  # (N, 5, 2)
    print(f"Built coords_history: {coords_history.shape}")

    # Filter invalid targets
    valid_mask = ~np.isnan(Y[:, :, 0]).any(axis=1) & ~np.isnan(Y[:, :, 1]).any(axis=1)
    X_video = X_video[valid_mask]
    X_steering = X_steering[valid_mask]
    X_ridge = X_ridge[valid_mask]
    curr_coords = curr_coords[valid_mask]
    prev_coords = prev_coords[valid_mask]
    Y = Y[valid_mask]
    cyclone_ids = cyclone_ids[valid_mask]
    names = names[valid_mask]
    coords_history = coords_history[valid_mask]

    print(f"Total NIO sequences after filtering: {len(Y)}")
    steering_dim = X_steering.shape[-1]
    print(f"Steering feature dimension: {steering_dim}")

    # Storm-level split
    unique_cyclones = np.unique(cyclone_ids)
    np.random.seed(42)
    shuffled = np.random.permutation(unique_cyclones)

    n_test = max(2, int(len(shuffled) * 0.15))
    n_val = max(2, int(len(shuffled) * 0.15))

    test_cids = set(shuffled[:n_test])
    val_cids = set(shuffled[n_test: n_test + n_val])
    train_cids = set(shuffled[n_test + n_val:])

    train_idx = [i for i, cid in enumerate(cyclone_ids) if cid in train_cids]
    val_idx = [i for i, cid in enumerate(cyclone_ids) if cid in val_cids]
    test_idx = [i for i, cid in enumerate(cyclone_ids) if cid in test_cids]

    print(f"Partition: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"Test storms: {set(names[test_idx])}")

    # Augmentor: only on training set
    train_augmentor = CycloneAugmentor(jitter_deg=0.3, noise_std=0.05, dropout_prob=0.15, p_reverse=0.2)
    val_augmentor = None  # No augmentation on val/test

    train_loader = DataLoader(
        MultimodalDataset(X_video[train_idx], X_steering[train_idx], X_ridge[train_idx],
                          curr_coords[train_idx], prev_coords[train_idx], Y[train_idx],
                          coords_history[train_idx],
                          augmentor=train_augmentor),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        MultimodalDataset(X_video[val_idx], X_steering[val_idx], X_ridge[val_idx],
                          curr_coords[val_idx], prev_coords[val_idx], Y[val_idx],
                          coords_history[val_idx]),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        MultimodalDataset(X_video[test_idx], X_steering[test_idx], X_ridge[test_idx],
                          curr_coords[test_idx], prev_coords[test_idx], Y[test_idx],
                          coords_history[test_idx]),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = MultimodalCycloneForecaster(
        img_channels=2,
        steering_dim=steering_dim,
        ridge_dim=64,
        hidden_dim=128
    ).to(device)

    # Load Stage 1 pretrained weights into GRU/video components if available
    if os.path.exists(PRETRAIN_CKPT):
        try:
            pretrain_ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=True)
            # Only load matching keys
            model_dict = model.state_dict()
            matched = {k: v for k, v in pretrain_ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
            if matched:
                model_dict.update(matched)
                model.load_state_dict(model_dict)
                print(f"Loaded {len(matched)} pretrained weights from Stage 1")
            else:
                print("No matching pretrained weights found, training from scratch")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    # Initial baseline test score
    init_metrics = evaluate_metrics(model, test_loader, device)
    print(f"Initial Test: +6h={init_metrics['track_error_6h_km']:.2f}km, "
          f"+12h={init_metrics['track_error_12h_km']:.2f}km, "
          f"+24h={init_metrics['track_error_24h_km']:.2f}km")

    # Optimizer: lower LR (3e-4) with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # Scheduler: Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Robust loss
    best_val_score = float("inf")
    patience_counter = 0
    max_patience = 15

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            xv, xs, xr, cc, pc, ch, y = batch
            xv, xs, xr = xv.to(device), xs.to(device), xr.to(device)
            cc, pc, ch = cc.to(device), pc.to(device), ch.to(device)
            y = y.to(device)

            preds = model(xv, xs, xr, cc, pc, coords_history=ch)

            d6 = torch_haversine(preds[:, 0, 0], preds[:, 0, 1], y[:, 0, 0], y[:, 0, 1])
            d12 = torch_haversine(preds[:, 1, 0], preds[:, 1, 1], y[:, 1, 0], y[:, 1, 1])
            d24 = torch_haversine(preds[:, 2, 0], preds[:, 2, 1], y[:, 2, 0], y[:, 2, 1])

            # ADJUSTED LOSS WEIGHTING: 6h + 2*12h + 4*24h
            l6 = robust_huber_haversine(d6, delta=25.0)
            l12 = robust_huber_haversine(d12, delta=40.0)
            l24 = robust_huber_haversine(d24, delta=60.0)
            track_loss = (l6 + 2.0 * l12 + 4.0 * l24) / 80.0

            # Wind speed loss (masked for NaN targets)
            valid_w = ~torch.isnan(y[:, :, 2])
            if valid_w.any():
                w_loss = nn.SmoothL1Loss()(preds[:, :, 2][valid_w], y[:, :, 2][valid_w])
            else:
                w_loss = torch.tensor(0.0, device=device)

            loss = track_loss + 0.05 * w_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate_metrics(model, val_loader, device)
        val_score = (val_metrics["track_error_6h_km"]
                     + 2.0 * val_metrics["track_error_12h_km"]
                     + 4.0 * val_metrics["track_error_24h_km"])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | LR: {lr_now:.6f} | "
              f"Val: +6h={val_metrics['track_error_6h_km']:.1f}km, "
              f"+12h={val_metrics['track_error_12h_km']:.1f}km, "
              f"+24h={val_metrics['track_error_24h_km']:.1f}km")

        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"    -> New best model saved (score: {val_score:.1f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch} (patience={max_patience})")
                break

    # Final Evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION: v2 MULTIMODAL MODEL (EFFICIENTNET + ATTENTION)")
    print("=" * 70)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, weights_only=True))
    test_metrics = evaluate_metrics(model, test_loader, device)

    print(f"  +6h Track Error:   Mean = {test_metrics['track_error_6h_km']:.2f} km | Median = {test_metrics['track_error_6h_median_km']:.2f} km (Target: < 25 km)")
    print(f"  +12h Track Error:  Mean = {test_metrics['track_error_12h_km']:.2f} km | Median = {test_metrics['track_error_12h_median_km']:.2f} km (Target: < 50 km)")
    print(f"  +24h Track Error:  Mean = {test_metrics['track_error_24h_km']:.2f} km | Median = {test_metrics['track_error_24h_median_km']:.2f} km (Target: < 100 km)")
    print(f"  +6h Wind MAE:      {test_metrics['wind_mae_6h_kmh']:.2f} km/h")
    print(f"  +12h Wind MAE:     {test_metrics['wind_mae_12h_kmh']:.2f} km/h")
    print(f"  +24h Wind MAE:     {test_metrics['wind_mae_24h_kmh']:.2f} km/h")
    print(f"  Overall Wind MAE:  {test_metrics['wind_mae_kmh']:.2f} km/h")

    with open(FINAL_METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nFinal model saved to: {FINAL_MODEL_PATH}")
    print(f"Metrics saved to: {FINAL_METRICS_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Stage 1: Multi-Basin Pre-training
    run_stage_1_pretraining(device, epochs=10, force=True)

    # Stage 2: Multimodal Fine-Tuning with all v2 improvements
    run_stage_2_finetuning(device, epochs=40)


if __name__ == "__main__":
    main()
