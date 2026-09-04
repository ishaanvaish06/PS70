"""
src/detection/train.py
Training pipeline for Cyclone Detection, Structural Pattern Tagging, and Category Estimation.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.detection.detector import CycloneDetector, PATTERN_TO_IDX, CAT_TO_IDX
from src.data.detection_dataset import CycloneDetectionDataset

def train_detection_model(epochs=12, batch_size=16, lr=1e-4, save_dir="models/detection"):
    print("=" * 60)
    print("TRAINING MULTI-TASK CYCLONE DETECTOR (PRESENCE & PATTERN)")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CycloneDetectionDataset(split="train", transform=transform)
    val_ds = CycloneDetectionDataset(split="val", transform=transform)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CycloneDetector(num_patterns=len(PATTERN_TO_IDX), num_categories=len(CAT_TO_IDX)).to(device)

    crit_pres = nn.BCEWithLogitsLoss()
    crit_pat = nn.CrossEntropyLoss()
    crit_cat = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "model_weights.pt")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            detected = batch["detected"].float().to(device)
            pat_indices = torch.tensor([PATTERN_TO_IDX.get(p, 0) for p in batch["structural_pattern"]], dtype=torch.long).to(device)
            cat_indices = torch.tensor([CAT_TO_IDX.get(c, 0) for c in batch["category"]], dtype=torch.long).to(device)

            optimizer.zero_grad()
            out = model(imgs)

            l_pres = crit_pres(out["presence_logit"], detected)
            l_pat = crit_pat(out["pattern_logits"], pat_indices)
            l_cat = crit_cat(out["category_logits"], cat_indices)

            loss = 1.5 * l_pres + 1.0 * l_pat + 0.5 * l_cat
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        pres_correct, pat_correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                detected = batch["detected"].float().to(device)
                pat_indices = torch.tensor([PATTERN_TO_IDX.get(p, 0) for p in batch["structural_pattern"]], dtype=torch.long).to(device)
                cat_indices = torch.tensor([CAT_TO_IDX.get(c, 0) for c in batch["category"]], dtype=torch.long).to(device)

                out = model(imgs)
                l_pres = crit_pres(out["presence_logit"], detected)
                l_pat = crit_pat(out["pattern_logits"], pat_indices)
                l_cat = crit_cat(out["category_logits"], cat_indices)
                loss = 1.5 * l_pres + 1.0 * l_pat + 0.5 * l_cat
                val_loss += loss.item() * len(imgs)

                pres_pred = (torch.sigmoid(out["presence_logit"]) >= 0.5).float()
                pres_correct += (pres_pred == detected).sum().item()

                pat_pred = torch.argmax(out["pattern_logits"], dim=1)
                pat_correct += (pat_pred == pat_indices).sum().item()
                total += len(imgs)

        val_loss /= len(val_ds)
        pres_acc = (pres_correct / total) * 100.0 if total > 0 else 0.0
        pat_acc = (pat_correct / total) * 100.0 if total > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Presence Acc: {pres_acc:.1f}% | Pattern Acc: {pat_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"\nTraining Complete! Best model saved to: {best_path}")
    return best_path

if __name__ == "__main__":
    train_detection_model()
