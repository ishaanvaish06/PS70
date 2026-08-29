"""
src/data/dataloader_example.py
Demonstration script showing how downstream models (Persons 2, 3, 4) load and iterate over datasets.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.forecasting_dataset import CycloneForecastingDataset, get_forecasting_data
from src.data.classification_dataset import MultisourceClassificationDataset, CycloneImageDataset
from src.data.detection_dataset import CycloneDetectionDataset
from src.data.preprocess_satellite import preprocess_single_frame

def main():
    print("=" * 60)
    print("DATA LOADERS & PREPROCESSING VERIFICATION")
    print("=" * 60)

    # 1. Near-Real-Time Satellite Preprocessing (Single Frame Ingestion)
    print("\n--- 1. Testing Near-Real-Time Single Frame Ingestion (Spec Page 3) ---")
    sample_img = "data/processed/classification/image_only_kaggle/images/101.jpg"
    frame_res = preprocess_single_frame(sample_img, target_size=(256, 256), normalize=True)
    print(f"Processed incoming frame -> Shape: {frame_res['image'].shape}, Timestamp: {frame_res['timestamp']}, Status: {frame_res['status']}")

    # 2. Forecasting Dataset (Person 4)
    print("\n--- 2. Testing Forecasting Dataset (Person 4) ---")
    train_fc = CycloneForecastingDataset(split="train")
    print(f"Loaded train sequences: {len(train_fc)}")
    x, y = train_fc[0]
    print(f"Sample 0 Input X shape: {x.shape} (5 timesteps x {len(train_fc.feature_names)} features)")
    print(f"Features: {train_fc.feature_names}")
    print(f"Sample 0 Target Y shape: {y.shape} (3 lead times: +6h, +12h, +24h x 3 targets)")
    print(f"Targets: {train_fc.target_names}")
    mean, std = train_fc.get_feature_stats()
    print(f"Feature means: {np.round(mean, 2)}")

    # 3. Multi-source Classification Dataset (Person 3)
    print("\n--- 3. Testing Multi-Source Classification Dataset (Person 3) ---")
    train_ms = MultisourceClassificationDataset(split="train")
    print(f"Loaded multi-source classification samples: {len(train_ms)}")
    x_ms, cat_idx, wind, pres = train_ms[0]
    print(f"Sample 0 Features: {x_ms.shape}, Category Index: {cat_idx}, Wind: {wind:.1f} km/h, Pressure: {pres:.1f} hPa")

    # 4. Image-only Classification Dataset (Person 3)
    print("\n--- 4. Testing Image Classification Dataset (Person 3) ---")
    train_img = CycloneImageDataset(split="train")
    print(f"Loaded image classification samples: {len(train_img)}")
    img, cat_idx, wind = train_img[0]
    print(f"Sample 0 Image shape: {img.shape if hasattr(img, 'shape') else type(img)}, Cat Index: {cat_idx}, Wind: {wind} km/h")

    # 5. Detection Dataset (Person 2)
    print("\n--- 5. Testing Detection Dataset (Person 2) ---")
    train_det = CycloneDetectionDataset(split="train")
    print(f"Loaded detection samples: {len(train_det)}")
    det_sample = train_det[0]
    print(f"Sample 0 Detected: {det_sample['detected']}, Pattern: {det_sample['structural_pattern']}, Category: {det_sample['category']}")

    print("\n" + "=" * 60)
    print("ALL DATA LOADERS & PREPROCESSING PIPELINES VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    import numpy as np
    main()
