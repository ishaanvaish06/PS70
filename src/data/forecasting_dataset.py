"""
src/data/forecasting_dataset.py
Data loading module for Person 4 (Forecasting).
Loads multi-step temporal sequence arrays for cyclone track & intensity prediction.
Compatible with PyTorch Dataset if torch is installed, or standalone NumPy/Python.
"""

import os
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

class CycloneForecastingDataset(Dataset):
    """
    Sequence dataset for cyclone forecasting.
    Input X: shape (N, 5, 7) -> 5 timesteps (t-24, t-18, t-12, t-6, t), 7 features
             [lat, lon, wind_speed, pressure, sst, wind_u, wind_v]
    Target Y: shape (N, 3, 3) -> 3 lead times (+6h, +12h, +24h), 3 targets
             [lat, lon, wind_speed]
    """
    def __init__(self, split="train", data_dir="data/processed/forecasting", transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform

        npz_file = os.path.join(data_dir, f"{split}_sequences.npz")
        meta_file = os.path.join(data_dir, f"{split}_sequences_metadata.csv")

        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"Sequence archive not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
        self.feature_names = list(data["features"])
        self.target_names = list(data["targets"])

        if os.path.exists(meta_file):
            self.metadata = pd.read_csv(meta_file)
        else:
            self.metadata = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform is not None:
            x, y = self.transform(x, y)

        if TORCH_AVAILABLE and isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        return x, y

    def get_feature_stats(self):
        """Returns mean and std of input features for normalization."""
        # Reshape to (N*5, 7) across all steps
        flat = self.X.reshape(-1, self.X.shape[-1])
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
        return mean, std

def get_forecasting_data(split="train", data_dir="data/processed/forecasting"):
    """Quick helper returning raw numpy arrays (X, Y, metadata)."""
    ds = CycloneForecastingDataset(split=split, data_dir=data_dir)
    return ds.X, ds.Y, ds.metadata, ds.feature_names, ds.target_names
