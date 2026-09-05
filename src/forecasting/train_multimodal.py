"""
src/forecasting/train_multimodal.py
Trains the MultimodalCycloneForecaster v2 (EfficientNet + Attention) on NIO data.
Single-stage training for quick iteration. For full pipeline, use train_two_stage.py.
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
from src.forecasting.train_two_stage import CycloneAugmentor

DATA_FILE = os.path.join("data", "processed", "forecasting", "multimodal_sequences", "multimodal_forecasting_dataset.npz")
MODEL_DIR = os.path.join("models", "forecasting")
MODEL_PATH = os.path.join(MODEL_DIR, "multimodal_forecast_model.pt")
METRICS_PATH = os.path.join(MODEL_DIR, "multimodal_metrics.json")


class MultimodalDataset(Dataset):
    def __init__(self, X_video, X_steering, X_ridge, curr_coords, prev_coords, Y, augmentor=None):
        self.X_video = X_video
        self.X_steering = X_steering
        self.X_ridge = X_ridge
        self.curr_coords = curr_coords
        self.prev_coords = prev_coords
        self.Y = Y
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

        # Build coords_history from curr and prev
        v_lat = (curr[0] - prev[0]) / 6.0
        v_lon = (curr[1] - prev[1]) / 6.0
        lat_t0, lon_t0 = curr[0], curr[1]
        coords_hist = torch.tensor([
            [lat_t0 - v_lat * 24, lon_t0 - v_lon * 24],
            [lat_t0 - v_lat * 18, lon_t0 - v_lon * 18],
            [lat_t0 - v_lat * 12, lon_t0 - v_lon * 12],
            [lat_t0 - v_lat * 6,  lon_t0 - v_lon * 6],
            [lat_t0, lon_t0],
        ], dtype=torch.float32)

        return video_t, steering_t, ridge_t, curr_t, prev_t, coords_hist, y_t


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
                err_6h.append(e6)
                err_12h.append(e12)
                err_24h.append(e24)

                for h in range(3):
                    if not np.isnan(y_np[b, h, 2]):
                        wind_errs.append(abs(preds[b, h, 2] - y_np[b, h, 2]))

    return {
        "track_error_6h_km": float(np.mean(err_6h)),
        "track_error_6h_median_km": float(np.median(err_6h)),
        "track_error_12h_km": float(np.mean(err_12h)),
        "track_error_12h_median_km": float(np.median(err_12h)),
        "track_error_24h_km": float(np.mean(err_24h)),
        "track_error_24h_median_km": float(np.median(err_24h)),
        "wind_mae_kmh": float(np.mean(wind_errs)) if wind_errs else 0.0
    }


def robust_huber_haversine(dist_km, delta=50.0):
    return torch.where(
        dist_km < delta,
        0.5 * (dist_km ** 2) / delta,
        dist_km - 0.5 * delta
    ).mean()


def main():
    print("=" * 70)
    print("TRAINING MULTIMODAL FORECASTER v2 (EFFICIENTNET + ATTENTION + AUGMENTATION)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_FILE)
    X_video = data["X_video"]
    X_steering = np.nan_to_num(data["X_steering"], nan=0.0)
    X_ridge = np.nan_to_num(data["X_ridge"], nan=5850.0) if "X_ridge" in data else np.full((len(X_video), 16, 16), 5850.0, dtype=np.float32)
    curr_coords = np.nan_to_num(data["curr_coords"], nan=55.0)
    prev_coords = np.nan_to_num(data["prev_coords"], nan=55.0)
    Y = data["Y"]
    cyclone_ids = data["cyclone_ids"]
    names = data["names"]

    valid_mask = ~np.isnan(Y[:, :, 0]).any(axis=1) & ~np.isnan(Y[:, :, 1]).any(axis=1)
    X_video = X_video[valid_mask]
    X_steering = X_steering[valid_mask]
    X_ridge = X_ridge[valid_mask]
    curr_coords = curr_coords[valid_mask]
    prev_coords = prev_coords[valid_mask]
    Y = Y[valid_mask]
    cyclone_ids = cyclone_ids[valid_mask]
    names = names[valid_mask]

    print(f"Valid sequences: {len(Y)}")

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

    train_aug = CycloneAugmentor(jitter_deg=0.3, noise_std=0.05, dropout_prob=0.15, p_reverse=0.2)

    train_loader = DataLoader(
        MultimodalDataset(X_video[train_idx], X_steering[train_idx], X_ridge[train_idx],
                          curr_coords[train_idx], prev_coords[train_idx], Y[train_idx],
                          augmentor=train_aug),
        batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        MultimodalDataset(X_video[val_idx], X_steering[val_idx], X_ridge[val_idx],
                          curr_coords[val_idx], prev_coords[val_idx], Y[val_idx]),
        batch_size=16, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        MultimodalDataset(X_video[test_idx], X_steering[test_idx], X_ridge[test_idx],
                          curr_coords[test_idx], prev_coords[test_idx], Y[test_idx]),
        batch_size=16, shuffle=False, num_workers=0
    )

    steering_dim = X_steering.shape[-1]
    model = MultimodalCycloneForecaster(img_channels=2, steering_dim=steering_dim,
                                         ridge_dim=64, hidden_dim=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_err = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
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

            l6 = robust_huber_haversine(d6, delta=25.0)
            l12 = robust_huber_haversine(d12, delta=40.0)
            l24 = robust_huber_haversine(d24, delta=60.0)
            track_loss = (l6 + 2.0 * l12 + 4.0 * l24) / 80.0

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
        val_metrics = evaluate(model, val_loader, device)
        val_total = (val_metrics["track_error_6h_km"]
                     + 2.0 * val_metrics["track_error_12h_km"]
                     + 4.0 * val_metrics["track_error_24h_km"])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | LR: {lr_now:.6f} | "
              f"Val: +6h={val_metrics['track_error_6h_km']:.1f}km, "
              f"+12h={val_metrics['track_error_12h_km']:.1f}km, "
              f"+24h={val_metrics['track_error_24h_km']:.1f}km")

        if val_total < best_val_err:
            best_val_err = val_total
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> New best (score: {val_total:.1f})")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    print(f"  +6h Track Error:  Mean = {test_metrics['track_error_6h_km']:.2f} km | Median = {test_metrics['track_error_6h_median_km']:.2f} km")
    print(f"  +12h Track Error: Mean = {test_metrics['track_error_12h_km']:.2f} km | Median = {test_metrics['track_error_12h_median_km']:.2f} km")
    print(f"  +24h Track Error: Mean = {test_metrics['track_error_24h_km']:.2f} km | Median = {test_metrics['track_error_24h_median_km']:.2f} km")
    print(f"  Overall Wind MAE: {test_metrics['wind_mae_kmh']:.2f} km/h")

    with open(METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
