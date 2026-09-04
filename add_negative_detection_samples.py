"""
add_negative_detection_samples.py
Generates negative ocean / non-cyclonic background image samples and adds them
to Dataset A (Detection & Presence) across train, validation, and test splits.
This enables the binary presence classifier to learn true positive vs. negative discrimination.
"""

import os
import numpy as np
from PIL import Image
import pandas as pd

OUT_IMG_DIR = os.path.join("data", "processed", "detection", "negative_samples")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

def generate_ocean_patches(num_samples=100):
    np.random.seed(123)
    created_files = []
    
    for i in range(num_samples):
        fname = f"neg_ocean_{i:03d}.jpg"
        fpath = os.path.join(OUT_IMG_DIR, fname)
        
        # Vary background types: clear dark sea, scattered clouds, hazy sea
        scene_type = i % 3
        if scene_type == 0:
            # Dark open sea (low IR radiance)
            base = np.random.normal(loc=35, scale=5, size=(256, 256))
        elif scene_type == 1:
            # Low marine stratocumulus clouds
            base = np.random.normal(loc=70, scale=15, size=(256, 256))
        else:
            # Gradient sea-surface background
            y = np.linspace(30, 60, 256)[:, None]
            base = y + np.random.normal(loc=0, scale=8, size=(256, 256))
            
        base = np.clip(base, 0, 255).astype(np.uint8)
        img = Image.fromarray(base)
        img.save(fpath, format="JPEG", quality=90)
        created_files.append((fname, fpath.replace("\\", "/")))
        
    return created_files

def update_detection_splits(neg_files):
    det_dir = os.path.join("data", "processed", "detection")
    
    # 70% train, 15% val, 15% test
    n_total = len(neg_files)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    neg_train = neg_files[:n_train]
    neg_val = neg_files[n_train:n_train + n_val]
    neg_test = neg_files[n_train + n_val:]
    
    splits = [
        ("train_detection.csv", neg_train),
        ("val_detection.csv", neg_val),
        ("test_detection.csv", neg_test),
    ]
    
    all_rows = []
    for split_file, neg_subset in splits:
        csv_path = os.path.join(det_dir, split_file)
        pos_df = pd.read_csv(csv_path)
        # Remove any previously added negative samples
        pos_df = pos_df[pos_df["cyclone_detected"] == True]
        
        neg_rows = []
        for fname, fpath in neg_subset:
            neg_rows.append({
                "filename": fname,
                "wind_speed_kt": 0,
                "wind_speed_kmh": 0.0,
                "category": "No Cyclone",
                "cyclone_detected": False,
                "image_path": fpath,
                "mock_bbox": "None",
                "structural_pattern": "no_cyclone"
            })
            
        neg_df = pd.DataFrame(neg_rows)
        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
        # Shuffle
        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        combined_df.to_csv(csv_path, index=False)
        print(f"Updated {split_file}: {len(pos_df)} positive + {len(neg_df)} negative = {len(combined_df)} samples.")
        all_rows.append(combined_df)
        
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(det_dir, "detection_all.csv"), index=False)
    print(f"Updated detection_all.csv: {len(all_df)} total samples.")

def main():
    print("Generating negative background ocean samples for Dataset A...")
    neg_files = generate_ocean_patches(num_samples=100)
    print(f"Generated {len(neg_files)} negative samples in {OUT_IMG_DIR}")
    update_detection_splits(neg_files)
    print("Negative sample integration complete!")

if __name__ == "__main__":
    main()
