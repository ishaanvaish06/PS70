"""
src/data/classification_dataset.py
Data loading module for Person 3 (Classification & Intensity).
Provides loaders for:
1. Multi-source Tabular Dataset (ERA5 environmental + IBTrACS features)
2. Image-only INSAT Dataset (133 IR satellite images)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

IMD_CLASSES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(IMD_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(IMD_CLASSES)}

class MultisourceClassificationDataset(Dataset):
    """
    Tabular multi-source dataset (IBTrACS + ERA5).
    Features: [lat, lon, sst, pressure_msl, wind_u, wind_v]
    Targets: category_idx (0-6), wind_speed_kmh, pressure_hpa
    """
    def __init__(self, split="train", data_dir="data/processed/classification"):
        self.split = split
        csv_file = os.path.join(data_dir, f"multisource_{split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Classification file not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.feature_cols = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
        
        # Extract features and targets
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.categories = self.df["category"].values
        self.category_indices = np.array([CLASS_TO_IDX.get(c, 0) for c in self.categories], dtype=np.int64)
        self.wind_speeds = self.df["wind_speed"].values.astype(np.float32)
        self.pressures = self.df["pressure"].fillna(-1.0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx]
        cat = self.category_indices[idx]
        wind = self.wind_speeds[idx]
        pres = self.pressures[idx]

        if TORCH_AVAILABLE:
            x = torch.from_numpy(x)
            cat = torch.tensor(cat, dtype=torch.long)
            wind = torch.tensor(wind, dtype=torch.float32)
            pres = torch.tensor(pres, dtype=torch.float32)

        return x, cat, wind, pres

class CycloneImageDataset(Dataset):
    """
    Image-only INSAT IR dataset (133 images).
    Outputs: image tensor/array, category_idx, wind_speed_kmh
    """
    def __init__(self, split="train", data_dir="data/processed/classification/image_only_kaggle", transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform

        csv_file = os.path.join(data_dir, f"{split}_labels.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Image labels not found: {csv_file}")

        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["filename"])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, "images", row["filename"])
        
        img = Image.open(img_path).convert("RGB")
        
        cat_idx = CLASS_TO_IDX.get(row["category"], 0)
        wind_kmh = float(row["wind_speed_kmh"])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img, dtype=np.float32) / 255.0

        if TORCH_AVAILABLE:
            if isinstance(img, np.ndarray):
                # Convert HWC -> CHW
                img = torch.from_numpy(img).permute(2, 0, 1)
            cat_idx = torch.tensor(cat_idx, dtype=torch.long)
            wind_kmh = torch.tensor(wind_kmh, dtype=torch.float32)

        return img, cat_idx, wind_kmh

class MultispectralSatelliteDataset(Dataset):
    """
    Dual-sensor multi-source satellite dataset pairing INSAT-3D
    Thermal Infrared (IR) and Visible (VIS) channels for each cyclone.
    Returns: (img_ir, img_vis), category_idx, wind_speed_kmh
    """
    def __init__(self, split="train", data_dir="data/processed/classification/multisource_satellite", transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        csv_file = os.path.join(data_dir, f"{split}_pairs.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Pairs file not found: {csv_file}")
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_ir = Image.open(row["path_ir"]).convert("L").resize((256, 256))
        img_vis = Image.open(row["path_vis"]).convert("L").resize((256, 256))

        arr_ir = np.array(img_ir, dtype=np.float32) / 255.0
        arr_vis = np.array(img_vis, dtype=np.float32) / 255.0
        # Stack channels: (2, 256, 256)
        stacked = np.stack([arr_ir, arr_vis], axis=0)

        cat_idx = int(row["category_idx"])
        wind_kmh = float(row["wind_speed_kmh"])

        if TORCH_AVAILABLE:
            stacked = torch.from_numpy(stacked)
            cat_idx = torch.tensor(cat_idx, dtype=torch.long)
            wind_kmh = torch.tensor(wind_kmh, dtype=torch.float32)

        return stacked, cat_idx, wind_kmh
