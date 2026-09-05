"""
src/forecasting/train_two_stage.py
Two-Stage Pre-Training & Fine-Tuning Pipeline:
  Stage 1: Pre-train recurrent kinematic backbone on 44,251 global cyclone tracks (WP + NA basins).
  Stage 2: Transfer weights & fine-tune MultimodalCycloneForecaster on North Indian Ocean dataset
           incorporating 3°-6° annulus environmental steering flow, deep-layer Vertical Wind Shear (VWS),
           2D Subtropical Ridge Grid (20° x 20° at 500 hPa), and continuous satellite video.
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
    print("STAGE 1: PRE-TRAINING ON 76,100 MULTI-BASIN CYCLONE TRACKS (WP + SI + NA BASINS)")
    print("=" * 70)

    if os.path.exists(PRETRAIN_CKPT) and not force:
        print(f"Pre-trained Stage 1 weights already exist at: {PRETRAIN_CKPT}. Skipping re-training.")
        return

    data = np.load(GLOBAL_DATA_FILE, allow_pickle=True)
    X = torch.tensor(data["X"], dtype=torch.float32) # (44251, 5, 3)
    Y = torch.tensor(data["Y"], dtype=torch.float32) # (44251, 3, 3)
    curr = X[:, 4, :].clone()
    prev = X[:, 3, :].clone()

    # Relative displacement normalization
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
            track_loss = (d6.mean() + d12.mean() + 1.5 * d24.mean()) / 50.0

            w_loss = smooth_l1(preds[:, :, 2], y[:, :, 2])
            loss = track_loss + 0.05 * w_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"  Stage 1 | Epoch {epoch:02d}/{epochs} | Global Pre-Training Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(PRETRAIN_CKPT), exist_ok=True)
    torch.save(model.state_dict(), PRETRAIN_CKPT)
    print(f"Stage 1 Complete! Pre-trained kinematic weights saved to: {PRETRAIN_CKPT}")

# -------------------------------------------------------------
# STAGE 2: MULTIMODAL FINE-TUNING ON NORTH INDIAN OCEAN
# -------------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, X_video, X_steering, X_ridge, curr_coords, prev_coords, Y):
        self.X_video = X_video
        self.X_steering = torch.tensor(X_steering, dtype=torch.float32)
        self.X_ridge = torch.tensor(X_ridge, dtype=torch.float32)
        self.curr_coords = torch.tensor(curr_coords, dtype=torch.float32)
        self.prev_coords = torch.tensor(prev_coords, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        video = torch.tensor(self.X_video[idx], dtype=torch.float32) / 255.0
        return video, self.X_steering[idx], self.X_ridge[idx], self.curr_coords[idx], self.prev_coords[idx], self.Y[idx]

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1-a, 0, 1)))

def robust_huber_haversine(dist_km, delta=50.0):
    """
    Robust Huber loss on Haversine distance in km.
    Quadratic when error <= delta km, linear when error > delta km.
    Prevents rare stalling anomalies (like Shaheen or Asna) from dominating gradients.
    """
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
        for xv, xs, xr, cc, pc, y in loader:
            xv, xs, xr, cc, pc = xv.to(device), xs.to(device), xr.to(device), cc.to(device), pc.to(device)
            preds = model(xv, xs, xr, cc, pc).cpu().numpy()
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

def run_stage_2_finetuning(device, epochs=25):
    print("\n" + "=" * 70)
    print("STAGE 2: MULTIMODAL FINE-TUNING (RIDGE + VWS + STEERING + SATELLITE)")
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

    print(f"Total North Indian Ocean sequences: {len(Y)}")
    steering_dim = X_steering.shape[-1]
    print(f"Steering feature dimension: {steering_dim}")

    # Storm-level split
    unique_cyclones = np.unique(cyclone_ids)
    np.random.seed(42)
    shuffled = np.random.permutation(unique_cyclones)

    n_test = max(2, int(len(shuffled) * 0.15))
    n_val = max(2, int(len(shuffled) * 0.15))

    test_cids = set(shuffled[:n_test])
    val_cids = set(shuffled[n_test : n_test + n_val])
    train_cids = set(shuffled[n_test + n_val :])

    train_idx = [i for i, cid in enumerate(cyclone_ids) if cid in train_cids]
    val_idx = [i for i, cid in enumerate(cyclone_ids) if cid in val_cids]
    test_idx = [i for i, cid in enumerate(cyclone_ids) if cid in test_cids]

    print(f"Partition: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"Test storms: {set(names[test_idx])}")

    train_loader = DataLoader(
        MultimodalDataset(X_video[train_idx], X_steering[train_idx], X_ridge[train_idx], curr_coords[train_idx], prev_coords[train_idx], Y[train_idx]),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        MultimodalDataset(X_video[val_idx], X_steering[val_idx], X_ridge[val_idx], curr_coords[val_idx], prev_coords[val_idx], Y[val_idx]),
        batch_size=16,
        shuffle=False
    )
    test_loader = DataLoader(
        MultimodalDataset(X_video[test_idx], X_steering[test_idx], X_ridge[test_idx], curr_coords[test_idx], prev_coords[test_idx], Y[test_idx]),
        batch_size=16,
        shuffle=False
    )

    model = MultimodalCycloneForecaster(
        img_channels=2,
        steering_dim=steering_dim,
        ridge_dim=64,
        hidden_dim=128
    ).to(device)

    # Initial baseline test score
    init_metrics = evaluate_metrics(model, test_loader, device)
    print(f"Initial Persistence Anchor on Test Set: +6h={init_metrics['track_error_6h_km']:.2f}km, +12h={init_metrics['track_error_12h_km']:.2f}km, +24h={init_metrics['track_error_24h_km']:.2f}km")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    smooth_l1 = nn.SmoothL1Loss()

    best_val_score = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xv, xs, xr, cc, pc, y in train_loader:
            xv, xs, xr, cc, pc, y = xv.to(device), xs.to(device), xr.to(device), cc.to(device), pc.to(device), y.to(device)
            preds = model(xv, xs, xr, cc, pc)

            d6 = torch_haversine(preds[:, 0, 0], preds[:, 0, 1], y[:, 0, 0], y[:, 0, 1])
            d12 = torch_haversine(preds[:, 1, 0], preds[:, 1, 1], y[:, 1, 0], y[:, 1, 1])
            d24 = torch_haversine(preds[:, 2, 0], preds[:, 2, 1], y[:, 2, 0], y[:, 2, 1])

            # Robust Huber loss: prevents extreme stalling outliers (like Shaheen or Asna) from dominating gradients
            l6 = robust_huber_haversine(d6, delta=30.0)
            l12 = robust_huber_haversine(d12, delta=50.0)
            l24 = robust_huber_haversine(d24, delta=80.0)
            track_loss = (l6 + l12 + 1.5 * l24) / 50.0

            valid_w = ~torch.isnan(y[:, :, 2])
            if valid_w.any():
                w_loss = smooth_l1(preds[:, :, 2][valid_w], y[:, :, 2][valid_w])
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
        val_score = val_metrics["track_error_6h_km"] + val_metrics["track_error_12h_km"] + val_metrics["track_error_24h_km"]

        print(f"  Stage 2 | Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Track: +6h={val_metrics['track_error_6h_km']:.1f}km, +12h={val_metrics['track_error_12h_km']:.1f}km, +24h={val_metrics['track_error_24h_km']:.1f}km")

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # Final Evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION: TWO-STAGE PRETRAINED MULTIMODAL MODEL (WITH RIDGE & VWS)")
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

    # Stage 1: Multi-Basin Pre-training (76,100 sequences from WP + SI + NA)
    run_stage_1_pretraining(device, epochs=10, force=True)

    # Stage 2: Multimodal Fine-Tuning with Annulus Environmental Steering, VWS, and 2D Subtropical Ridge (3,966 sequences)
    run_stage_2_finetuning(device, epochs=25)

if __name__ == "__main__":
    main()
