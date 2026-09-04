"""
src/classification/evaluate.py
Evaluates Multi-Source Tabular and Multispectral Satellite Models on test splits.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classification.classifier import MultispectralCycloneCNN, MultisourceTabularModel, CLASS_TO_IDX, IDX_TO_CLASS
from src.data.classification_dataset import MultisourceClassificationDataset, MultispectralSatelliteDataset

def evaluate_classification(out_json="models/classification/metrics_comparison.json"):
    print("=" * 60)
    print("EVALUATING MULTI-SOURCE CYCLONE INTENSITY MODELS")
    print("=" * 60)

    # 1. Evaluate Tabular Model (ERA5 + IBTrACS)
    tab_model_path = "models/classification/tabular_multisource_model.pkl"
    tabular_metrics = {}
    if os.path.exists(tab_model_path):
        tab_model = MultisourceTabularModel.load(tab_model_path)
        test_ds = MultisourceClassificationDataset(split="test")

        pred_cat, pred_wind, pred_pres, confs = tab_model.predict(test_ds.X)
        acc = accuracy_score(test_ds.category_indices, pred_cat) * 100.0
        f1 = f1_score(test_ds.category_indices, pred_cat, average="macro")
        wind_mae = np.mean(np.abs(pred_wind - test_ds.wind_speeds))
        pres_mask = test_ds.pressures > 800.0
        pres_mae = np.mean(np.abs(pred_pres[pres_mask] - test_ds.pressures[pres_mask]))

        tabular_metrics = {
            "category_accuracy_percent": round(float(acc), 2),
            "category_macro_f1": round(float(f1), 4),
            "wind_speed_mae_kmh": round(float(wind_mae), 2),
            "pressure_mae_hpa": round(float(pres_mae), 2)
        }
        print(f"Tabular Model (ERA5 + Location) Test Results:")
        print(f"  Category Accuracy: {acc:.2f}%")
        print(f"  Macro F1:          {f1:.4f}")
        print(f"  Wind Speed MAE:    {wind_mae:.2f} km/h")
        print(f"  Pressure MAE:      {pres_mae:.2f} hPa")

    # 2. Evaluate Multispectral CNN (IR + VIS)
    cnn_path = "models/classification/multispectral_cnn.pt"
    cnn_metrics = {}
    if os.path.exists(cnn_path):
        test_img_ds = MultispectralSatelliteDataset(split="test")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MultispectralCycloneCNN(in_channels=2, num_classes=7).to(device)
        model.load_state_dict(torch.load(cnn_path, map_location=device))
        model.eval()

        all_preds, all_true, wind_errs = [], [], []
        with torch.no_grad():
            for idx in range(len(test_img_ds)):
                stacked, cat, wind = test_img_ds[idx]
                stacked = stacked.unsqueeze(0).to(device)
                logits, pred_w = model(stacked)
                pred_c = torch.argmax(logits, dim=1).item()
                all_preds.append(pred_c)
                all_true.append(int(cat))
                wind_errs.append(abs(float(pred_w.item()) - float(wind)))

        cnn_acc = accuracy_score(all_true, all_preds) * 100.0
        cnn_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        cnn_wind_mae = np.mean(wind_errs)

        cnn_metrics = {
            "category_accuracy_percent": round(float(cnn_acc), 2),
            "category_macro_f1": round(float(cnn_f1), 4),
            "wind_speed_mae_kmh": round(float(cnn_wind_mae), 2)
        }
        print(f"\nMultispectral CNN (IR + Visible) Test Results:")
        print(f"  Category Accuracy: {cnn_acc:.2f}%")
        print(f"  Macro F1:          {cnn_f1:.4f}")
        print(f"  Wind Speed MAE:    {cnn_wind_mae:.2f} km/h")

    payload = {
        "tabular_multisource_model": tabular_metrics,
        "multispectral_satellite_cnn": cnn_metrics
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMetrics saved to {out_json}")
    return payload

if __name__ == "__main__":
    evaluate_classification()
