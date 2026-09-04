"""
src/forecasting/train.py
Training pipeline for the Spatiotemporal Cyclone Trajectory & Intensity Forecaster.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.forecasting.forecaster import CycloneForecaster, CombinedForecastingLoss, torch_haversine
from src.data.forecasting_dataset import CycloneForecastingDataset

def clean_and_normalize(X, Y):
    # Impute any NaNs (such as SST over land) with feature means
    X_clean = np.copy(X)
    for feat_idx in range(X_clean.shape[2]):
        col = X_clean[:, :, feat_idx]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            mean_val = np.nanmean(col)
            col[nan_mask] = mean_val
            X_clean[:, :, feat_idx] = col
    return torch.from_numpy(X_clean.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))

def train_forecaster(epochs=40, batch_size=64, lr=1e-3, save_dir="models/forecasting"):
    print("=" * 60)
    print("TRAINING CYCLONE TRAJECTORY & INTENSITY FORECASTER")
    print("=" * 60)

    train_ds = CycloneForecastingDataset(split="train")
    val_ds = CycloneForecastingDataset(split="val")

    X_train_t, Y_train_t = clean_and_normalize(train_ds.X, train_ds.Y)
    X_val_t, Y_val_t = clean_and_normalize(val_ds.X, val_ds.Y)

    print(f"Train sequences: {len(X_train_t)}, Validation sequences: {len(X_val_t)}")
    print(f"Input shape: {X_train_t.shape}, Target shape: {Y_train_t.shape}")

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dim = X_train_t.shape[2]
    model = CycloneForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2).to(device)

    criterion = CombinedForecastingLoss(track_weight=1.0, wind_weight=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_val_track_km = float("inf")
    best_checkpoint_path = os.path.join(save_dir, "forecast_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_track = 0.0
        train_wind = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss, track_l, wind_l = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * len(bx)
            train_track += track_l.item() * len(bx)
            train_wind += wind_l.item() * len(bx)

        scheduler.step()
        train_loss /= len(X_train_t)
        train_track /= len(X_train_t)
        train_wind /= len(X_train_t)

        # Validation
        model.eval()
        val_track_6h, val_track_12h, val_track_24h = [], [], []
        val_wind_errors = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                dist_6h = torch_haversine(pred[:, 0, 0], pred[:, 0, 1], by[:, 0, 0], by[:, 0, 1])
                dist_12h = torch_haversine(pred[:, 1, 0], pred[:, 1, 1], by[:, 1, 0], by[:, 1, 1])
                dist_24h = torch_haversine(pred[:, 2, 0], pred[:, 2, 1], by[:, 2, 0], by[:, 2, 1])

                val_track_6h.extend(dist_6h.cpu().numpy())
                val_track_12h.extend(dist_12h.cpu().numpy())
                val_track_24h.extend(dist_24h.cpu().numpy())

                w_err = torch.abs(pred[:, :, 2] - by[:, :, 2])
                val_wind_errors.extend(w_err.cpu().numpy().flatten())

        m_6h = np.mean(val_track_6h)
        m_12h = np.mean(val_track_12h)
        m_24h = np.mean(val_track_24h)
        avg_track_km = (m_6h + m_12h + m_24h) / 3.0
        w_mae = np.mean(val_wind_errors)

        if epoch % 5 == 0 or epoch == 1 or avg_track_km < best_val_track_km:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.2f} | "
                  f"Val Track: +6h={m_6h:.1f}km, +12h={m_12h:.1f}km, +24h={m_24h:.1f}km (Avg: {avg_track_km:.1f}km) | "
                  f"Wind MAE: {w_mae:.1f} km/h")

        if avg_track_km < best_val_track_km:
            best_val_track_km = avg_track_km
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "best_val_track_km": best_val_track_km,
                "val_6h_km": m_6h,
                "val_12h_km": m_12h,
                "val_24h_km": m_24h
            }, best_checkpoint_path)

    print(f"\nTraining Complete! Best model saved to: {best_checkpoint_path} (Avg Val Track: {best_val_track_km:.1f} km)")
    return best_checkpoint_path

if __name__ == "__main__":
    train_forecaster()
