"""
src/data/detection_dataset.py
Data loading module for Person 2 (Cyclone Detection & Presence).
Provides loaders for cyclone image presence classification and structural pattern tagging.
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

class CycloneDetectionDataset(Dataset):
    """
    Dataset for Cyclone Presence & Structural Pattern Classification.
    """
    def __init__(self, split="train", data_dir="data/processed/detection", transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform

        csv_file = os.path.join(data_dir, f"{split}_detection.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Detection manifest not found: {csv_file}")

        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        img = Image.open(img_path).convert("RGB")
        detected = bool(row["cyclone_detected"])
        pattern = str(row["structural_pattern"])
        category = str(row["category"])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img, dtype=np.float32) / 255.0

        sample = {
            "image": img,
            "detected": detected,
            "category": category,
            "structural_pattern": pattern,
            "filename": row["filename"]
        }
        return sample
