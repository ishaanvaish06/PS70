"""
src/data/preprocess_satellite.py
Satellite image preprocessing module for Person 1 (Data Engineer).
Directly fulfills spec requirement on Page 3 of ps70_team_plan_edited.pdf:
- Ingests INSAT-3D/3DR satellite imagery
- Supports batch preprocessing and near-real-time single-frame inference
- Resizing, normalization, channel conversion, and format handling
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def preprocess_single_frame(
    image_input,
    target_size=(256, 256),
    normalize=True,
    to_tensor=False,
    timestamp=None
):
    """
    Near-real-time ingestion function for a single incoming INSAT satellite frame.
    
    Args:
        image_input (str or PIL.Image or np.ndarray): File path, PIL Image, or NumPy array.
        target_size (tuple): (width, height) to resize to.
        normalize (bool): If True, scale pixels to [0, 1].
        to_tensor (bool): If True and PyTorch is available, return (C, H, W) Tensor.
        timestamp (str or datetime, optional): Timestamp of satellite pass.
        
    Returns:
        dict: {
            "image": processed image (np.ndarray or torch.Tensor),
            "original_size": original (width, height),
            "target_size": target_size,
            "timestamp": timestamp string or current UTC,
            "status": "ready"
        }
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Satellite frame not found: {image_input}")
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_size = img.size

    # Resize with high-quality Lanczos resampling
    if target_size is not None:
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    else:
        img_resized = img

    arr = np.array(img_resized, dtype=np.float32)

    # Standard min-max normalization [0, 1]
    if normalize:
        arr = arr / 255.0

    # Optional PyTorch tensor conversion (CHW format)
    if to_tensor and TORCH_AVAILABLE:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        output_image = tensor
    else:
        output_image = arr

    pass_time = timestamp if timestamp else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "image": output_image,
        "original_size": orig_size,
        "target_size": target_size,
        "timestamp": pass_time,
        "status": "ready"
    }

def batch_preprocess_directory(
    input_dir,
    output_dir,
    target_size=(256, 256),
    format="PNG"
):
    """
    Processes all satellite frames in a directory and exports standardized crops.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    count = 0
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            in_path = os.path.join(input_dir, fname)
            out_name = f"{os.path.splitext(fname)[0]}.png"
            out_path = os.path.join(output_dir, out_name)
            
            res = preprocess_single_frame(in_path, target_size=target_size, normalize=False)
            img_to_save = Image.fromarray(res["image"].astype(np.uint8))
            img_to_save.save(out_path, format=format)
            count += 1
            
    print(f"Batch preprocessed {count} frames -> {output_dir}")
    return count

if __name__ == "__main__":
    # Test near-real-time single frame pipeline
    test_img_path = "data/processed/classification/image_only_kaggle/images/101.jpg"
    if os.path.exists(test_img_path):
        result = preprocess_single_frame(test_img_path, target_size=(256, 256), normalize=True)
        print("Single Frame Preprocessing Test:")
        print(f"  Status: {result['status']}")
        print(f"  Shape: {result['image'].shape}")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Value range: [{result['image'].min():.2f}, {result['image'].max():.2f}]")
