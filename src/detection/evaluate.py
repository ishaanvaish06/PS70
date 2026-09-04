"""
src/detection/evaluate.py
Evaluates Cyclone Detection & Structural Pattern Recognition on the held-out test set.
Computes Precision, Recall, F1 score, and Accuracy.
"""

import os
import sys
import json
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.detection.detector import CycloneDetector, PATTERN_TO_IDX, IDX_TO_PATTERN, CAT_TO_IDX
from src.data.detection_dataset import CycloneDetectionDataset

def evaluate_detection(checkpoint_path="models/detection/model_weights.pt",
                       out_json="metrics/detection_metrics.json"):
    print("=" * 60)
    print("EVALUATING CYCLONE DETECTION & STRUCTURAL PATTERN MODEL")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = CycloneDetectionDataset(split="test", transform=transform)
    print(f"Test samples: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycloneDetector(num_patterns=len(PATTERN_TO_IDX), num_categories=len(CAT_TO_IDX)).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    y_pres_true, y_pres_pred = [], []
    y_pat_true, y_pat_pred = [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            detected = 1 if sample["detected"] else 0
            pat_idx = PATTERN_TO_IDX.get(sample["structural_pattern"], 0)

            out = model(img)
            p_score = torch.sigmoid(out["presence_logit"]).item()
            p_pred = 1 if p_score >= 0.5 else 0

            pat_pred = torch.argmax(out["pattern_logits"], dim=1).item()

            y_pres_true.append(detected)
            y_pres_pred.append(p_pred)
            y_pat_true.append(pat_idx)
            y_pat_pred.append(pat_pred)

    # Calculate metrics
    pres_acc = accuracy_score(y_pres_true, y_pres_pred)
    pres_prec = precision_score(y_pres_true, y_pres_pred, zero_division=0)
    pres_rec = recall_score(y_pres_true, y_pres_pred, zero_division=0)
    pres_f1 = f1_score(y_pres_true, y_pres_pred, zero_division=0)

    pat_acc = accuracy_score(y_pat_true, y_pat_pred)
    pat_f1 = f1_score(y_pat_true, y_pat_pred, average="macro", zero_division=0)

    results = {
        "presence_detection": {
            "accuracy": round(float(pres_acc), 4),
            "precision": round(float(pres_prec), 4),
            "recall": round(float(pres_rec), 4),
            "f1_score": round(float(pres_f1), 4)
        },
        "structural_pattern": {
            "accuracy": round(float(pat_acc), 4),
            "macro_f1": round(float(pat_f1), 4)
        }
    }

    print("\nDetection Test Results:")
    print(f"  Presence Accuracy:  {pres_acc*100:.1f}%")
    print(f"  Presence Precision: {pres_prec:.4f}")
    print(f"  Presence Recall:    {pres_rec:.4f}")
    print(f"  Presence F1-Score:  {pres_f1:.4f}")
    print(f"  Pattern Accuracy:   {pat_acc*100:.1f}%")
    print(f"  Pattern Macro F1:   {pat_f1:.4f}")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {out_json}")
    return results

if __name__ == "__main__":
    evaluate_detection()
