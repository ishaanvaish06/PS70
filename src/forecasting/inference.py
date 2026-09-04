"""
src/forecasting/inference.py
Standalone inference module for Cyclone Trajectory & Wind Forecasting.
Conforms directly to the SIH team API contract schema.
"""

import os
import torch
import numpy as np

from src.forecasting.forecaster import CycloneForecaster

CHECKPOINT_PATH = os.path.join("models", "forecasting", "forecast_model.pt")

_MODEL = None

def get_model(checkpoint_path=CHECKPOINT_PATH):
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        input_dim = cp.get("input_dim", 7)
        model = CycloneForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()
        _MODEL = model
    return _MODEL

def forecast_cyclone(history_sequence, checkpoint_path=CHECKPOINT_PATH):
    """
    Predicts cyclone trajectory coordinates and wind speed for +6h, +12h, +24h.
    history_sequence: array of shape (5, 7) or (1, 5, 7)
      features: [lat, lon, wind_speed, pressure, sst, wind_u, wind_v]
    Returns list of prediction dicts.
    """
    seq = np.array(history_sequence, dtype=np.float32)
    if seq.ndim == 2:
        seq = np.expand_dims(seq, axis=0) # (1, 5, 7)

    # Impute NaNs if any
    nan_mask = np.isnan(seq)
    if np.any(nan_mask):
        seq = np.nan_to_num(seq, nan=0.0)

    model = get_model(checkpoint_path)
    seq_t = torch.from_numpy(seq)

    with torch.no_grad():
        preds = model(seq_t).squeeze(0).numpy() # (3, 3)

    lead_times = [6, 12, 24]
    forecast_results = []
    for i, dt in enumerate(lead_times):
        forecast_results.append({
            "lead_time_hours": dt,
            "latitude": round(float(preds[i, 0]), 2),
            "longitude": round(float(preds[i, 1]), 2),
            "wind_speed_kmh": round(float(preds[i, 2]), 1)
        })

    return forecast_results

if __name__ == "__main__":
    # Test with dummy sequence
    dummy_seq = np.array([
        [14.0, 85.0, 65.0, 996.0, 29.0, 2.0, 3.0],
        [14.3, 84.6, 75.0, 992.0, 29.1, 2.2, 3.1],
        [14.8, 84.1, 85.0, 988.0, 29.0, 2.4, 3.3],
        [15.4, 83.5, 100.0, 980.0, 28.9, 2.7, 3.6],
        [16.1, 82.8, 120.0, 970.0, 28.8, 3.0, 4.0],
    ], dtype=np.float32)
    print("Testing dummy inference...")
    res = forecast_cyclone(dummy_seq)
    print("Forecast output:")
    for step in res:
        print(f"  +{step['lead_time_hours']:02d}h -> Lat: {step['latitude']}, Lon: {step['longitude']}, Wind: {step['wind_speed_kmh']} km/h")
