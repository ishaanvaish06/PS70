"""
src/forecasting/inference.py
Standalone inference module for Multimodal Cyclone Trajectory & Wind Forecasting.
Fuses:
  - Continuous Dual-Channel Satellite Video (Thermal IR + Water Vapor)
  - 500 hPa & 700 hPa Atmospheric Steering & Deep-Layer Vertical Wind Shear (VWS)
  - 2D Subtropical Ridge Geopotential Height Grid (20° x 20° at 500 hPa)
Conforms directly to the SIH team API contract schema.
"""

import os
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

from src.forecasting.forecaster import CycloneForecaster
from src.forecasting.multimodal_forecaster import MultimodalCycloneForecaster

CHECKPOINT_PATH = os.path.join("models", "forecasting", "forecast_model.pt")
MULTIMODAL_CHECKPOINT_PATH = os.path.join("models", "forecasting", "multimodal_forecast_model.pt")

_MODEL = None
_MULTIMODAL_MODEL = None

def get_multimodal_model(checkpoint_path=MULTIMODAL_CHECKPOINT_PATH):
    global _MULTIMODAL_MODEL
    if _MULTIMODAL_MODEL is None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Multimodal checkpoint not found at {checkpoint_path}. Train the model first.")
        # Load weights to inspect shape
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Determine steering_dim from first layer of steering_mlp
        steering_dim = cp["steering_mlp.0.weight"].shape[1] if "steering_mlp.0.weight" in cp else 10
        ridge_dim = 64
        model = MultimodalCycloneForecaster(img_channels=2, steering_dim=steering_dim, ridge_dim=ridge_dim, hidden_dim=128)
        model.load_state_dict(cp)
        model.eval()
        _MULTIMODAL_MODEL = model
    return _MULTIMODAL_MODEL

def forecast_cyclone_multimodal(video_seq, steering_seq, curr_coords, prev_coords=None, ridge_grid=None, checkpoint_path=MULTIMODAL_CHECKPOINT_PATH):
    """
    Multimodal Cyclone Trajectory & Intensity Prediction using:
      1. Continuous Satellite Video (T=5, C=2, H=128, W=128)
      2. Upper-Air Steering & Deep-Layer Vertical Wind Shear (T=5, F=10 or 5)
      3. 2D Subtropical Ridge Grid (H=16, W=16) [optional]
      4. Current Coordinates & Intensity [lat, lon, wind_kmh]
      5. Previous Coordinates [lat, lon, wind_kmh] (optional, defaults to curr_coords)
    """
    model = get_multimodal_model(checkpoint_path)
    
    # Process video: (1, 5, 2, 128, 128)
    v_arr = np.array(video_seq, dtype=np.float32)
    if v_arr.max() > 1.0:
        v_arr = v_arr / 255.0
    if v_arr.ndim == 4:
        v_arr = np.expand_dims(v_arr, axis=0)
    v_t = torch.from_numpy(v_arr)
    
    # Process steering & VWS: (1, 5, F)
    s_arr = np.array(steering_seq, dtype=np.float32)
    if s_arr.ndim == 2:
        s_arr = np.expand_dims(s_arr, axis=0)
    
    # Pad to model steering_dim if needed
    if s_arr.shape[-1] < model.steering_dim:
        pad = np.zeros((s_arr.shape[0], s_arr.shape[1], model.steering_dim - s_arr.shape[-1]), dtype=np.float32)
        s_arr = np.concatenate([s_arr, pad], axis=-1)
    elif s_arr.shape[-1] > model.steering_dim:
        s_arr = s_arr[:, :, :model.steering_dim]
    s_t = torch.from_numpy(np.nan_to_num(s_arr, nan=0.0))
    
    # Process ridge grid: (1, 1, 16, 16)
    if ridge_grid is not None:
        r_arr = np.array(ridge_grid, dtype=np.float32)
        if r_arr.ndim == 2:
            r_arr = np.expand_dims(np.expand_dims(r_arr, axis=0), axis=0)
        elif r_arr.ndim == 3:
            r_arr = np.expand_dims(r_arr, axis=0)
        r_t = torch.from_numpy(r_arr)
    else:
        # Default climatological 500 hPa tropical height
        r_t = torch.full((1, 1, 16, 16), 5850.0, dtype=torch.float32)
    
    # Process coords: (1, 3)
    c_arr = np.array(curr_coords, dtype=np.float32)
    if c_arr.ndim == 1:
        c_arr = np.expand_dims(c_arr, axis=0)
    c_t = torch.from_numpy(np.nan_to_num(c_arr, nan=55.0))
    
    if prev_coords is not None:
        p_arr = np.array(prev_coords, dtype=np.float32)
        if p_arr.ndim == 1:
            p_arr = np.expand_dims(p_arr, axis=0)
        p_t = torch.from_numpy(np.nan_to_num(p_arr, nan=55.0))
    else:
        p_t = c_t.clone()
    
    with torch.no_grad():
        preds = model(v_t, s_t, r_t, c_t, p_t).squeeze(0).numpy() # (3, 3)
        
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
    history_sequence: shape (seq_len, 7) or DataFrame
    """
    model = get_model(checkpoint_path)
    if hasattr(history_sequence, "values"):
        seq_arr = history_sequence.values
    else:
        seq_arr = np.array(history_sequence)
    
    if seq_arr.ndim == 2:
        seq_arr = np.expand_dims(seq_arr, axis=0)
    
    x_t = torch.from_numpy(seq_arr).float()
    with torch.no_grad():
        preds = model(x_t).squeeze(0).numpy()
    
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
