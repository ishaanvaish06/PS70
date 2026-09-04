"""
src/detection/inference.py
Inference module for Cyclone Detection & Structural Pattern Recognition.
Conforms directly to the SIH team API contract schema.
"""

import os
import torch
from PIL import Image
from torchvision import transforms

from src.detection.detector import CycloneDetector, PATTERN_TO_IDX, IDX_TO_PATTERN, IDX_TO_CAT, CAT_TO_IDX

CHECKPOINT_PATH = os.path.join("models", "detection", "model_weights.pt")

_MODEL = None
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model(checkpoint_path=CHECKPOINT_PATH):
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = CycloneDetector(num_patterns=len(PATTERN_TO_IDX), num_categories=len(CAT_TO_IDX))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        _MODEL = model
    return _MODEL

def detect_cyclone(image_input, checkpoint_path=CHECKPOINT_PATH):
    """
    Detects cyclone presence, structural pattern, and approximate center/bbox from a satellite image.
    image_input: filepath (str) or PIL Image
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    w, h = img.size
    t_img = _TRANSFORM(img).unsqueeze(0)

    model = get_model(checkpoint_path)
    with torch.no_grad():
        out = model(t_img)
        pres_prob = torch.sigmoid(out["presence_logit"]).item()
        detected = bool(pres_prob >= 0.5)

        pat_probs = torch.softmax(out["pattern_logits"], dim=1).numpy()[0]
        pat_idx = int(pat_probs.argmax())
        pattern_name = IDX_TO_PATTERN[pat_idx]
        pat_conf = float(pat_probs[pat_idx])

        cat_probs = torch.softmax(out["category_logits"], dim=1).numpy()[0]
        cat_idx = int(cat_probs.argmax())
        cat_name = IDX_TO_CAT[cat_idx]

    # Center bounding box estimate (scaled to image dimensions)
    if detected:
        bbox = [int(0.25 * h), int(0.25 * w), int(0.75 * h), int(0.75 * w)]
    else:
        bbox = None

    return {
        "detected": detected,
        "confidence": round(pres_prob if detected else 1.0 - pres_prob, 3),
        "structural_pattern": pattern_name,
        "pattern_confidence": round(pat_conf, 3),
        "category": cat_name,
        "bbox": bbox
    }

if __name__ == "__main__":
    test_img = "data/processed/classification/image_only_kaggle/images/101.jpg"
    if os.path.exists(test_img):
        res = detect_cyclone(test_img)
        print("Sample Detection Result:", res)
