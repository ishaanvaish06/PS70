"""
src/forecasting/inference.py
Standalone inference module for Multimodal Cyclone Trajectory & Wind Forecasting v2.
Supports:
  - Single model inference
  - Multi-seed ensembling (average predictions from N models)
  - Test-time augmentation (TTA): average predictions over coordinate perturbations
  - Full v2 architecture with EfficientNet + Temporal Attention
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
_ENSEMBLE_MODELS = []


def _build_model_from_checkpoint(checkpoint_path):
    """Build MultimodalCycloneForecaster from a checkpoint file."""
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    steering_dim = cp["steering_mlp.0.weight"].shape[1] if "steering_mlp.0.weight" in cp else 10
    ridge_dim = 64
    model = MultimodalCycloneForecaster(img_channels=2, steering_dim=steering_dim, ridge_dim=ridge_dim, hidden_dim=128)
    model.load_state_dict(cp)
    model.eval()
    return model


def get_multimodal_model(checkpoint_path=MULTIMODAL_CHECKPOINT_PATH):
    global _MULTIMODAL_MODEL
    if _MULTIMODAL_MODEL is None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Multimodal checkpoint not found at {checkpoint_path}. Train first.")
        _MULTIMODAL_MODEL = _build_model_from_checkpoint(checkpoint_path)
    return _MULTIMODAL_MODEL


def load_ensemble(checkpoint_paths):
    """
    Load multiple model checkpoints for ensembling.
    checkpoint_paths: list of str paths to .pt files
    """
    global _ENSEMBLE_MODELS
    _ENSEMBLE_MODELS = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            model = _build_model_from_checkpoint(path)
            _ENSEMBLE_MODELS.append(model)
    print(f"Loaded {len(_ENSEMBLE_MODELS)} ensemble models")


def _predict_single(model, v_t, s_t, r_t, c_t, p_t, ch_t=None):
    """Run a single forward pass through the model."""
    with torch.no_grad():
        preds = model(v_t, s_t, r_t, c_t, p_t, coords_history=ch_t).squeeze(0).numpy()
    return preds


def _build_coords_history(curr_coords, prev_coords):
    """Build approximate 5-step coords history from curr and prev."""
    lat0, lon0 = curr_coords[0], curr_coords[1]
    plat, plon = prev_coords[0], prev_coords[1]
    v_lat = (lat0 - plat) / 6.0
    v_lon = (lon0 - plon) / 6.0
    history = np.array([
        [lat0 - v_lat * 24, lon0 - v_lon * 24],
        [lat0 - v_lat * 18, lon0 - v_lon * 18],
        [lat0 - v_lat * 12, lon0 - v_lon * 12],
        [lat0 - v_lat * 6,  lon0 - v_lon * 6],
        [lat0, lon0],
    ], dtype=np.float32)
    return history


def forecast_cyclone_multimodal(video_seq, steering_seq, curr_coords, prev_coords=None,
                                 ridge_grid=None, checkpoint_path=MULTIMODAL_CHECKPOINT_PATH,
                                 n_tta=5, tta_jitter=0.2):
    """
    Multimodal forecast with optional Test-Time Augmentation (TTA).
    TTA: averages predictions over N random coordinate perturbations for more robust output.
    """
    model = get_multimodal_model(checkpoint_path)

    # Prepare tensors
    v_arr = np.array(video_seq, dtype=np.float32)
    if v_arr.max() > 1.0:
        v_arr = v_arr / 255.0
    if v_arr.ndim == 4:
        v_arr = np.expand_dims(v_arr, axis=0)

    s_arr = np.array(steering_seq, dtype=np.float32)
    if s_arr.ndim == 2:
        s_arr = np.expand_dims(s_arr, axis=0)
    if s_arr.shape[-1] < model.steering_dim:
        pad = np.zeros((s_arr.shape[0], s_arr.shape[1], model.steering_dim - s_arr.shape[-1]), dtype=np.float32)
        s_arr = np.concatenate([s_arr, pad], axis=-1)
    elif s_arr.shape[-1] > model.steering_dim:
        s_arr = s_arr[:, :, :model.steering_dim]
    s_arr = np.nan_to_num(s_arr, nan=0.0)

    if ridge_grid is not None:
        r_arr = np.array(ridge_grid, dtype=np.float32)
        if r_arr.ndim == 2:
            r_arr = np.expand_dims(np.expand_dims(r_arr, axis=0), axis=0)
        elif r_arr.ndim == 3:
            r_arr = np.expand_dims(r_arr, axis=0)
    else:
        r_arr = np.full((1, 1, 16, 16), 5850.0, dtype=np.float32)

    c_arr = np.array(curr_coords, dtype=np.float32)
    if c_arr.ndim == 1:
        c_arr = np.expand_dims(c_arr, axis=0)
    c_arr = np.nan_to_num(c_arr, nan=55.0)

    if prev_coords is not None:
        p_arr = np.array(prev_coords, dtype=np.float32)
        if p_arr.ndim == 1:
            p_arr = np.expand_dims(p_arr, axis=0)
        p_arr = np.nan_to_num(p_arr, nan=55.0)
    else:
        p_arr = c_arr.copy()

    # Build coords history
    ch_arr = _build_coords_history(c_arr[0], p_arr[0])
    ch_arr = np.expand_dims(ch_arr, axis=0)

    v_t = torch.from_numpy(v_arr)
    s_t = torch.from_numpy(s_arr)
    r_t = torch.from_numpy(r_arr)
    c_t = torch.from_numpy(c_arr)
    p_t = torch.from_numpy(p_arr)
    ch_t = torch.from_numpy(ch_arr)

    # TTA: average over perturbed inputs
    all_preds = []
    for tta_idx in range(max(1, n_tta)):
        if tta_idx == 0:
            # Original
            all_preds.append(_predict_single(model, v_t, s_t, r_t, c_t, p_t, ch_t))
        else:
            # Perturbed
            jitter = np.random.uniform(-tta_jitter, tta_jitter, size=c_arr.shape).astype(np.float32)
            c_t_pert = c_t + torch.from_numpy(jitter)
            p_t_pert = p_t + torch.from_numpy(jitter)
            ch_pert = ch_t + torch.from_numpy(jitter[0, :, :2].reshape(1, 5, 2))
            all_preds.append(_predict_single(model, v_t, s_t, r_t, c_t_pert, p_t_pert, ch_pert))

    preds = np.mean(all_preds, axis=0)

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


def forecast_cyclone_multimodal_ensemble(video_seq, steering_seq, curr_coords, prev_coords=None,
                                          ridge_grid=None, n_tta=3, tta_jitter=0.15):
    """
    Ensemble forecast: average predictions from all loaded ensemble models + TTA.
    """
    if not _ENSEMBLE_MODELS:
        raise RuntimeError("No ensemble models loaded. Call load_ensemble() first.")

    all_preds = []
    for model in _ENSEMBLE_MODELS:
        # Same tensor prep as single model
        v_arr = np.array(video_seq, dtype=np.float32)
        if v_arr.max() > 1.0:
            v_arr = v_arr / 255.0
        if v_arr.ndim == 4:
            v_arr = np.expand_dims(v_arr, axis=0)

        s_arr = np.array(steering_seq, dtype=np.float32)
        if s_arr.ndim == 2:
            s_arr = np.expand_dims(s_arr, axis=0)
        if s_arr.shape[-1] < model.steering_dim:
            pad = np.zeros((s_arr.shape[0], s_arr.shape[1], model.steering_dim - s_arr.shape[-1]), dtype=np.float32)
            s_arr = np.concatenate([s_arr, pad], axis=-1)
        s_arr = np.nan_to_num(s_arr, nan=0.0)

        if ridge_grid is not None:
            r_arr = np.array(ridge_grid, dtype=np.float32)
            if r_arr.ndim == 2:
                r_arr = np.expand_dims(np.expand_dims(r_arr, axis=0), axis=0)
            elif r_arr.ndim == 3:
                r_arr = np.expand_dims(r_arr, axis=0)
        else:
            r_arr = np.full((1, 1, 16, 16), 5850.0, dtype=np.float32)

        c_arr = np.array(curr_coords, dtype=np.float32)
        if c_arr.ndim == 1:
            c_arr = np.expand_dims(c_arr, axis=0)
        c_arr = np.nan_to_num(c_arr, nan=55.0)

        if prev_coords is not None:
            p_arr = np.array(prev_coords, dtype=np.float32)
            if p_arr.ndim == 1:
                p_arr = np.expand_dims(p_arr, axis=0)
            p_arr = np.nan_to_num(p_arr, nan=55.0)
        else:
            p_arr = c_arr.copy()

        ch_arr = _build_coords_history(c_arr[0], p_arr[0])
        ch_arr = np.expand_dims(ch_arr, axis=0)

        v_t = torch.from_numpy(v_arr)
        s_t = torch.from_numpy(s_arr)
        r_t = torch.from_numpy(r_arr)
        c_t = torch.from_numpy(c_arr)
        p_t = torch.from_numpy(p_arr)
        ch_t = torch.from_numpy(ch_arr)

        # TTA per model
        for tta_idx in range(max(1, n_tta)):
            if tta_idx == 0:
                all_preds.append(_predict_single(model, v_t, s_t, r_t, c_t, p_t, ch_t))
            else:
                jitter = np.random.uniform(-tta_jitter, tta_jitter, size=c_arr.shape).astype(np.float32)
                c_t_pert = c_t + torch.from_numpy(jitter)
                p_t_pert = p_t + torch.from_numpy(jitter)
                ch_pert = ch_t + torch.from_numpy(jitter[0, :, :2].reshape(1, 5, 2))
                all_preds.append(_predict_single(model, v_t, s_t, r_t, c_t_pert, p_t_pert, ch_pert))

    preds = np.mean(all_preds, axis=0)

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
    Predicts cyclone trajectory and wind speed for +6h, +12h, +24h.
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
