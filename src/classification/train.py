"""
src/classification/train.py
Training script for Multi-Source Cyclone Classification & Intensity Estimation.
Trains:
  1. MultispectralCycloneCNN: 2-channel CNN (IR + Visible imagery)
  2. MultisourceTabularModel: ERA5 reanalysis + IBTrACS tabular features
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classification.classifier import MultispectralCycloneCNN, MultisourceTabularModel
from src.data.classification_dataset import MultispectralSatelliteDataset, MultisourceClassificationDataset

def train_multispectral_cnn(epochs=20, batch_size=16, lr=3e-4, save_dir="models/classification"):
    print("\n" + "=" * 60)
    print("1. TRAINING MULTISPECTRAL SATELLITE CNN (IR + VISIBLE)")
    print("=" * 60)

    train_ds = MultispectralSatelliteDataset(split="train")
    val_ds = MultispectralSatelliteDataset(split="val")

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultispectralCycloneCNN(in_channels=2, num_classes=7).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(save_dir, "multispectral_cnn.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, cat_idx, wind in train_loader:
            imgs, cat_idx, wind = imgs.to(device), cat_idx.to(device), wind.to(device)
            optimizer.zero_grad()
            logits, pred_wind = model(imgs)

            l_cls = criterion_cls(logits, cat_idx)
            l_reg = criterion_reg(pred_wind, wind)
            loss = l_cls + 0.02 * l_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        wind_errors = []

        with torch.no_grad():
            for imgs, cat_idx, wind in val_loader:
                imgs, cat_idx, wind = imgs.to(device), cat_idx.to(device), wind.to(device)
                logits, pred_wind = model(imgs)

                l_cls = criterion_cls(logits, cat_idx)
                l_reg = criterion_reg(pred_wind, wind)
                loss = l_cls + 0.02 * l_reg
                val_loss += loss.item() * len(imgs)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == cat_idx).sum().item()
                total += len(imgs)
                wind_errors.extend(torch.abs(pred_wind - wind).cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        val_wind_mae = np.mean(wind_errors)

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.1f}% | Wind MAE: {val_wind_mae:.1f} km/h")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"Saved best multispectral CNN to: {best_path}")
    return best_path

def train_tabular_model(save_dir="models/classification"):
    print("\n" + "=" * 60)
    print("2. TRAINING MULTI-SOURCE TABULAR MODEL (ERA5 + IBTrACS)")
    print("=" * 60)

    train_ds = MultisourceClassificationDataset(split="train")
    val_ds = MultisourceClassificationDataset(split="val")

    print(f"Train rows: {len(train_ds)}, Val rows: {len(val_ds)}")
    model = MultisourceTabularModel()
    model.fit(train_ds.X, train_ds.category_indices, train_ds.wind_speeds, train_ds.pressures)

    # Validate
    pred_cat, pred_wind, pred_pres, _ = model.predict(val_ds.X)
    acc = np.mean(pred_cat == val_ds.category_indices) * 100.0
    wind_mae = np.mean(np.abs(pred_wind - val_ds.wind_speeds))
    pres_mask = val_ds.pressures > 800.0
    pres_mae = np.mean(np.abs(pred_pres[pres_mask] - val_ds.pressures[pres_mask]))

    print(f"Validation Results:")
    print(f"  Category Accuracy: {acc:.2f}%")
    print(f"  Wind Speed MAE:    {wind_mae:.2f} km/h")
    print(f"  Pressure MAE:      {pres_mae:.2f} hPa")

    save_path = os.path.join(save_dir, "tabular_multisource_model.pkl")
    model.save(save_path)
    print(f"Saved tabular model to: {save_path}")
    return save_path

def main():
    train_tabular_model()
    train_multispectral_cnn()

if __name__ == "__main__":
    main()
