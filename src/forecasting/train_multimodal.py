"""
src/forecasting/train_multimodal.py
Trains the MultimodalCycloneForecaster fusing:
  1. Continuous INSAT Satellite Sequences (T=5, C=2, H=128, W=128)
  2. ERA5 500 hPa & 700 hPa Atmospheric Steering Flow (T=5, F=5)
Strictly partitioned by cyclone_id to prevent data leakage.
"""

import os
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.forecasting.multimodal_forecaster import MultimodalCycloneForecaster
from src.forecasting.forecaster import torch_haversine

DATA_FILE = os.path.join("data", "processed", "forecasting", "multimodal_sequences", "multimodal_forecasting_dataset.npz")
MODEL_DIR = os.path.join("models", "forecasting")
MODEL_PATH = os.path.join(MODEL_DIR, "multimodal_forecast_model.pt")
METRICS_PATH = os.path.join(MODEL_DIR, "multimodal_metrics.json")

class MultimodalDataset(Dataset):
    def __init__(self, X_video, X_steering, curr_coords, Y):
        self.X_video = X_video # uint8 (N, 5, 2, 128, 128)
        self.X_steering = torch.tensor(X_steering, dtype=torch.float32)
        self.curr_coords = torch.tensor(curr_coords, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # Convert video to float in range [0, 1] on the fly
        video = torch.tensor(self.X_video[idx], dtype=torch.float32) / 255.0
        return video, self.X_steering[idx], self.curr_coords[idx], self.Y[idx]

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1)))
    return R * c

def evaluate(model, loader, device):
    model.eval()
    err_6h, err_12h, err_24h = [], [], []
    wind_errs = []

    with torch.no_grad():
        for x_v, x_s, curr_c, y in loader:
            x_v = x_v.to(device)
            x_s = x_s.to(device)
            curr_c = curr_c.to(device)

            preds = model(x_v, x_s, curr_c).cpu().numpy() # (B, 3, 3)
            y_np = y.numpy()

            for b in range(len(preds)):
                # +6h
                e6 = haversine_np(preds[b, 0, 0], preds[b, 0, 1], y_np[b, 0, 0], y_np[b, 0, 1])
                # +12h
                e12 = haversine_np(preds[b, 1, 0], preds[b, 1, 1], y_np[b, 1, 0], y_np[b, 1, 1])
                # +24h
                e24 = haversine_np(preds[b, 2, 0], preds[b, 2, 1], y_np[b, 2, 0], y_np[b, 2, 1])

                err_6h.append(e6)
                err_12h.append(e12)
                err_24h.append(e24)

                # Wind MAE (ignoring nans in ground truth)
                for h in range(3):
                    if not np.isnan(y_np[b, h, 2]):
                        wind_errs.append(abs(preds[b, h, 2] - y_np[b, h, 2]))

    return {
        "track_error_6h_km": float(np.mean(err_6h)),
        "track_error_12h_km": float(np.mean(err_12h)),
        "track_error_24h_km": float(np.mean(err_24h)),
        "wind_mae_kmh": float(np.mean(wind_errs)) if wind_errs else 0.0
    }

def main():
    print("=" * 70)
    print("TRAINING MULTIMODAL FORECASTER (CONTINUOUS SATELLITE + 500hPa STEERING)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_FILE)
    X_video = data["X_video"]
    X_steering = data["X_steering"]
    curr_coords = data["curr_coords"]
    Y = data["Y"]
    cyclone_ids = data["cyclone_ids"]
    names = data["names"]

    # Fill NaNs in steering and current coordinates if any
    X_steering = np.nan_to_num(X_steering, nan=0.0)
    curr_coords = np.nan_to_num(curr_coords, nan=55.0)

    # Clean target coordinates (lat/lon must not be NaN)
    valid_mask = ~np.isnan(Y[:, :, 0]).any(axis=1) & ~np.isnan(Y[:, :, 1]).any(axis=1)
    X_video = X_video[valid_mask]
    X_steering = X_steering[valid_mask]
    curr_coords = curr_coords[valid_mask]
    Y = Y[valid_mask]
    cyclone_ids = cyclone_ids[valid_mask]
    names = names[valid_mask]

    print(f"Valid sequences after target quality filtering: {len(Y)}")

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
        MultimodalDataset(X_video[train_idx], X_steering[train_idx], curr_coords[train_idx], Y[train_idx]),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        MultimodalDataset(X_video[val_idx], X_steering[val_idx], curr_coords[val_idx], Y[val_idx]),
        batch_size=16,
        shuffle=False
    )
    test_loader = DataLoader(
        MultimodalDataset(X_video[test_idx], X_steering[test_idx], curr_coords[test_idx], Y[test_idx]),
        batch_size=16,
        shuffle=False
    )

    model = MultimodalCycloneForecaster(img_channels=2, steering_dim=5, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    smooth_l1 = nn.SmoothL1Loss()

    best_val_err = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    EPOCHS = 15
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x_v, x_s, curr_c, y in train_loader:
            x_v = x_v.to(device)
            x_s = x_s.to(device)
            curr_c = curr_c.to(device)
            y = y.to(device)

            preds = model(x_v, x_s, curr_c) # (B, 3, 3)

            # Haversine track distance loss
            d6 = torch_haversine(preds[:, 0, 0], preds[:, 0, 1], y[:, 0, 0], y[:, 0, 1])
            d12 = torch_haversine(preds[:, 1, 0], preds[:, 1, 1], y[:, 1, 0], y[:, 1, 1])
            d24 = torch_haversine(preds[:, 2, 0], preds[:, 2, 1], y[:, 2, 0], y[:, 2, 1])
            track_loss = (d6.mean() + d12.mean() + 1.5 * d24.mean()) / 50.0

            # Wind speed loss
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
        val_metrics = evaluate(model, val_loader, device)
        val_total_track = (val_metrics["track_error_6h_km"] + val_metrics["track_error_12h_km"] + val_metrics["track_error_24h_km"]) / 3.0

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Track: +6h={val_metrics['track_error_6h_km']:.1f}km, +12h={val_metrics['track_error_12h_km']:.1f}km, +24h={val_metrics['track_error_24h_km']:.1f}km")

        if val_total_track < best_val_err:
            best_val_err = val_total_track
            torch.save(model.state_dict(), MODEL_PATH)

    # Load best model and evaluate on unseen test set
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION ON UNSEEN CYCLONES")
    print("=" * 70)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    print(f"Test Track Error (+6h):   {test_metrics['track_error_6h_km']:.2f} km   (Benchmark Target: < 25 km)")
    print(f"Test Track Error (+12h):  {test_metrics['track_error_12h_km']:.2f} km   (Benchmark Target: < 50 km)")
    print(f"Test Track Error (+24h):  {test_metrics['track_error_24h_km']:.2f} km   (Benchmark Target: < 100 km)")
    print(f"Test Wind Speed MAE:      {test_metrics['wind_mae_kmh']:.2f} km/h")

    with open(METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
