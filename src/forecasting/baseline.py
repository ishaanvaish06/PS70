"""
src/forecasting/baseline.py
Implements the Physical Movement-Vector Persistence Baseline and
spherical Great-Circle Haversine distance evaluation.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Computes Great-Circle Haversine distance in kilometers between two points
    or arrays of points on Earth's surface.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c

def predict_persistence(X):
    """
    Kinematic persistence baseline using recent 6-hour movement vector.
    X: shape (N, 5, num_features) where:
       - index 4 is t=0 (current observation)
       - index 3 is t=-6h (previous 6-hour observation)
       - index 0 is latitude, index 1 is longitude, index 2 is wind_speed
    Returns:
       Y_pred: shape (N, 3, 3) for lead times (+6h, +12h, +24h)
               [lat, lon, wind_speed]
    """
    curr_lat = X[:, 4, 0]
    curr_lon = X[:, 4, 1]
    curr_wind = X[:, 4, 2]

    prev_lat = X[:, 3, 0]
    prev_lon = X[:, 3, 1]

    # 6-hourly velocity vector (deg/hour)
    v_lat = (curr_lat - prev_lat) / 6.0
    v_lon = (curr_lon - prev_lon) / 6.0

    lead_times = [6.0, 12.0, 24.0]
    N = len(X)
    Y_pred = np.zeros((N, len(lead_times), 3), dtype=np.float32)

    for i, dt in enumerate(lead_times):
        pred_lat = curr_lat + v_lat * dt
        pred_lon = curr_lon + v_lon * dt
        pred_wind = curr_wind  # Intensity persistence
        Y_pred[:, i, 0] = pred_lat
        Y_pred[:, i, 1] = pred_lon
        Y_pred[:, i, 2] = pred_wind

    return Y_pred

def evaluate_persistence(X, Y):
    """
    Evaluates persistence baseline on dataset (X, Y).
    Returns dict of errors per lead time.
    """
    Y_pred = predict_persistence(X)
    lead_times = [6, 12, 24]
    results = {}

    for i, dt in enumerate(lead_times):
        track_err = haversine_distance_km(Y_pred[:, i, 0], Y_pred[:, i, 1],
                                          Y[:, i, 0], Y[:, i, 1])
        wind_err = np.abs(Y_pred[:, i, 2] - Y[:, i, 2])

        results[f"+{dt}h"] = {
            "mean_track_error_km": float(np.mean(track_err)),
            "median_track_error_km": float(np.median(track_err)),
            "p90_track_error_km": float(np.percentile(track_err, 90)),
            "wind_speed_mae_kmh": float(np.mean(wind_err))
        }

    return results, Y_pred
