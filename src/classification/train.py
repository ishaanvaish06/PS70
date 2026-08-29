"""
src/classification/train.py
Training script for Person 3 (Cyclone Classification & Intensity Estimation).

Trains:
1. Model B: Multi-Source Tabular Model (ERA5 + Location data) saved to models/classification/tabular_multisource_model.pkl
2. Model A: Image-Only PyTorch CNN Baseline saved to models/classification/image_only_model.pt
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classification.classifier import (
    MultisourceTabularModel,
    ImageOnlyIntensityModel,
    TORCH_AVAILABLE,
    CLASS_TO_IDX
)
from src.data.classification_dataset import MultisourceClassificationDataset, CycloneImageDataset

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

def train_tabular_model(save_dir="models/classification"):
    print("=" * 60)
    print("1. TRAINING MULTI-SOURCE TABULAR MODEL (MODEL B)")
    print("=" * 60)

    train_ds = MultisourceClassificationDataset(split="train")
    val_ds = MultisourceClassificationDataset(split="val")

    X_train = train_ds.X
    y_cat_train = train_ds.category_indices
    y_wind_train = train_ds.wind_speeds
    y_pres_train = train_ds.pressures

    X_val = val_ds.X
    y_cat_val = val_ds.category_indices
    y_wind_val = val_ds.wind_speeds
    y_pres_val = val_ds.pressures

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Features (6): {train_ds.feature_cols}")

    model = MultisourceTabularModel(use_lightgbm=True)
    model.fit(X_train, y_cat_train, y_wind_train, y_pres_train)

    # Evaluate on validation split
    pred_cat, pred_wind, pred_pres, confs = model.predict(X_val)
    acc = np.mean(pred_cat == y_cat_val) * 100.0
    wind_mae = np.mean(np.abs(pred_wind - y_wind_val))

    print(f"--> Val Category Accuracy: {acc:.2f}%")
    print(f"--> Val Wind Speed MAE:   {wind_mae:.2f} km/h")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tabular_multisource_model.pkl")
    model.save(save_path)
    print(f"Successfully saved Model B to: {save_path}")
    return model

def train_image_model(save_dir="models/classification", epochs=15, batch_size=16, lr=1e-4):
    print("\n" + "=" * 60)
    print("2. TRAINING IMAGE-ONLY PYTORCH CNN MODEL (MODEL A)")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Skipping Image-Only CNN training.")
        return None

    train_ds = CycloneImageDataset(split="train")
    val_ds = CycloneImageDataset(split="val")

    print(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImageOnlyIntensityModel(num_classes=7, backbone_name="resnet18").to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "image_only_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, cat_idx, wind_speed in train_loader:
            imgs = imgs.to(device)
            cat_idx = cat_idx.to(device)
            wind_speed = wind_speed.to(device)

            optimizer.zero_grad()
            logits, pred_wind = model(imgs)

            loss_cls = criterion_cls(logits, cat_idx)
            loss_reg = criterion_reg(pred_wind, wind_speed)
            # Combine losses: classification cross-entropy + scaled wind regression loss
            loss = loss_cls + 0.02 * loss_reg

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)

        train_loss = running_loss / len(train_ds)

        # Validation pass
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        wind_errors = []

        with torch.no_grad():
            for imgs, cat_idx, wind_speed in val_loader:
                imgs = imgs.to(device)
                cat_idx = cat_idx.to(device)
                wind_speed = wind_speed.to(device)

                logits, pred_wind = model(imgs)
                loss_cls = criterion_cls(logits, cat_idx)
                loss_reg = criterion_reg(pred_wind, wind_speed)
                loss = loss_cls + 0.02 * loss_reg

                val_loss += loss.item() * len(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == cat_idx).sum().item()
                total += len(cat_idx)
                wind_errors.extend(torch.abs(pred_wind - wind_speed).cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        val_wind_mae = np.mean(wind_errors) if len(wind_errors) > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | Val Wind MAE: {val_wind_mae:.2f} km/h")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"Successfully saved Model A to: {save_path}")
    return model

def main():
    train_tabular_model()
    train_image_model()
    print("\n" + "=" * 60)
    print("PERSON 3 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
