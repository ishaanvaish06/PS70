"""
src/forecasting/evaluate.py
Evaluation benchmark comparing the Deep Learning Forecaster against the
Physical Movement-Vector Persistence Baseline on the held-out test set.
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.forecasting.forecaster import CycloneForecaster, torch_haversine
from src.forecasting.baseline import haversine_distance_km, evaluate_persistence
from src.forecasting.train import clean_and_normalize
from src.data.forecasting_dataset import CycloneForecastingDataset

def run_evaluation(checkpoint_path="models/forecasting/forecast_model.pt",
                   out_json="models/forecasting/forecast_metrics.json"):
    print("=" * 70)
    print("CYCLONE FORECASTING BENCHMARK: AI FORECASTER VS. PERSISTENCE BASELINE")
    print("=" * 70)

    test_ds = CycloneForecastingDataset(split="test")
    X_test, Y_test = test_ds.X, test_ds.Y
    print(f"Held-out test set: {len(X_test)} sequence windows across 24 test cyclones.")

    # 1. Evaluate Persistence Baseline
    baseline_results, baseline_preds = evaluate_persistence(X_test, Y_test)

    # 2. Evaluate AI Forecaster
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = checkpoint.get("input_dim", X_test.shape[2])

    model = CycloneForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test_t, Y_test_t = clean_and_normalize(X_test, Y_test)
    with torch.no_grad():
        pred_t = model(X_test_t)
        Y_ai_pred = pred_t.numpy()

    lead_times = [6, 12, 24]
    ai_results = {}
    comparison = {}

    for i, dt in enumerate(lead_times):
        true_lat, true_lon = Y_test[:, i, 0], Y_test[:, i, 1]
        true_wind = Y_test[:, i, 2]

        pred_lat, pred_lon = Y_ai_pred[:, i, 0], Y_ai_pred[:, i, 1]
        pred_wind = Y_ai_pred[:, i, 2]

        track_dist = haversine_distance_km(pred_lat, pred_lon, true_lat, true_lon)
        wind_mae = np.mean(np.abs(pred_wind - true_wind))

        mean_err = float(np.mean(track_dist))
        med_err = float(np.median(track_dist))
        p90_err = float(np.percentile(track_dist, 90))

        base_mean = baseline_results[f"+{dt}h"]["mean_track_error_km"]
        base_wind_mae = baseline_results[f"+{dt}h"]["wind_speed_mae_kmh"]
        track_reduction_pct = ((base_mean - mean_err) / base_mean) * 100.0

        key = f"+{dt}h"
        ai_results[key] = {
            "mean_track_error_km": round(mean_err, 2),
            "median_track_error_km": round(med_err, 2),
            "p90_track_error_km": round(p90_err, 2),
            "wind_speed_mae_kmh": round(float(wind_mae), 2)
        }

        comparison[key] = {
            "persistence_track_error_km": round(base_mean, 2),
            "ai_forecaster_track_error_km": round(mean_err, 2),
            "track_error_reduction_pct": f"{track_reduction_pct:+.1f}%",
            "persistence_wind_mae_kmh": round(base_wind_mae, 2),
            "ai_wind_mae_kmh": round(float(wind_mae), 2)
        }

    # Print summary table
    print("\n" + "-" * 75)
    print(f"{'Lead Time':<10} | {'Baseline Track (km)':<20} | {'AI Forecaster (km)':<20} | {'Error Lift (%)':<15}")
    print("-" * 75)
    for dt in lead_times:
        k = f"+{dt}h"
        b_km = comparison[k]["persistence_track_error_km"]
        ai_km = comparison[k]["ai_forecaster_track_error_km"]
        lift = comparison[k]["track_error_reduction_pct"]
        print(f"{k:<10} | {b_km:<20} | {ai_km:<20} | {lift:<15}")
    print("-" * 75)

    payload = {
        "persistence_baseline": baseline_results,
        "ai_forecaster": ai_results,
        "comparative_lift": comparison
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved benchmark metrics to: {out_json}")
    return payload

if __name__ == "__main__":
    run_evaluation()
