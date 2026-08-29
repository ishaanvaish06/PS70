# Person 2 — Cyclone Detection & Pattern Recognition: Implementation Guide
**SIH 2026 — PS 26070: Tropical Cyclone AI/ML System**  
**Role:** Person 2 (Cyclone Detection)

---

## 🎯 1. Your Role & Responsibility
Given an incoming satellite image (INSAT-3D/3DR Infrared or Visible), your model must:
1. **Identify Cyclone Presence:** Detect whether a tropical cyclone exists in the satellite frame (`detected: True/False` and `confidence`).
2. **Tag Structural Pattern (Covers PS "Patterns" Requirement):** Classify the cyclone's cloud organization pattern according to standard meteorological / Dvorak conventions:
   * `"eye_visible"` (Well-defined central eye)
   * `"curved_band"` (Curved convective spiral bands wrapping towards center)
   * `"shear_pattern"` (Convective mass displaced from circulation center)
   * `"no_organized_center"` (Weak or disorganized depression)
3. **Estimate Bounding Box / Center Location:** Return coordinates `[x1, y1, x2, y2]` or approximate latitude/longitude center.

---

## 📦 2. What Person 1 Has Prepared for You

All data assets are already preprocessed and ready to load:
* **Dataset Files (`data/processed/detection/`):**
  * `train_detection.csv` (93 training samples)
  * `val_detection.csv` (19 validation samples)
  * `test_detection.csv` (21 test samples)
  * `detection_all.csv` (133 total annotated samples)
* **Preprocessed Satellite Images:**
  * High-resolution infrared crops located in: `data/processed/classification/image_only_kaggle/images/`
* **Ready-to-Use DataLoaders:**
  * `from src.data.detection_dataset import CycloneDetectionDataset`
  * `from src.data.preprocess_satellite import preprocess_single_frame`

---

## 🚀 3. Step-by-Step Implementation Workflow

### Step 3.1: Data Ingestion & Preprocessing
Use the prepared DataLoader or near-real-time single-frame function:

```python
from src.data.detection_dataset import CycloneDetectionDataset
from src.data.preprocess_satellite import preprocess_single_frame

# Load training data
train_dataset = CycloneDetectionDataset(split="train")
sample = train_dataset[0]

print("Image shape:", sample["image"].shape)       # (H, W, 3) normalized [0, 1]
print("Detected:", sample["detected"])             # True
print("Structural Pattern:", sample["structural_pattern"]) # 'curved_band'
print("IMD Category:", sample["category"])         # 'Cyclonic Storm'
```

### Step 3.2: Model Architecture Strategy
Do **NOT** train an object detector from scratch. Use a pretrained backbone:
* **Recommended Architecture:** Fine-tuned lightweight CNN (e.g., `ResNet18`, `MobileNetV3`, or `EfficientNet-B0`) or fine-tuned `YOLOv8-nano` / `YOLOv8-cls`.
* **Multi-Task Heads:**
  1. **Presence Head:** Binary classification (Sigmoid $\rightarrow$ Cyclone Present / Not Present).
  2. **Pattern Head:** Multi-class classification (Softmax over 4 structural patterns).
  3. **Auxiliary Category Head:** Multi-class classification over 7 IMD categories.

```python
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class CycloneDetectorNet(nn.Module):
    def __init__(self, num_patterns=4, num_classes=7):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        self.presence_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_patterns)
        )

    def forward(self, x):
        features = self.backbone(x)
        presence = self.presence_head(features)
        pattern = self.pattern_head(features)
        return presence, pattern
```

### Step 3.3: Training & Augmentation
* **Augmentations:** Random horizontal/vertical flip, slight rotation ($\pm 15^\circ$), subtle brightness/contrast shifts. (Do not distort central cyclone structure excessively).
* **Loss Functions:** Binary Cross-Entropy (BCE) for presence, Cross-Entropy for structural pattern.
* **Optimizer:** AdamW ($\text{lr} = 10^{-4}$), batch size 16, 20–30 epochs.

### Step 3.4: Geographic Coordinate Conversion Function
Implement a utility that maps detection pixel coordinates into geographic bounding coordinates:

```python
def pixel_to_geo_coords(bbox, image_geo_bounds):
    """
    Converts pixel bounding box [x1, y1, x2, y2] to latitude/longitude bounds.
    """
    # image_geo_bounds: {"min_lat": 0, "max_lat": 30, "min_lon": 50, "max_lon": 100}
    # Return estimated cyclone center { "latitude": ..., "longitude": ... }
    pass
```

---

## 📊 4. Evaluation & Metrics to Report
You MUST evaluate your model on the test split (`test_detection.csv`) and output:
1. **Detection Metrics:** Precision, Recall, F1-Score, ROC-AUC.
2. **Structural Pattern Classification:** Multi-class Accuracy, Confusion Matrix.
3. **Visual Predictions:** Save 4–6 visual sample plots showing the satellite frame with detection status, confidence, and predicted structural pattern badge.

---

## 🔌 5. Inference Function & Integration Contract
Person 5 (Frontend) will invoke your model through a standardized inference function.

### Required Inference Function (`src/detection/inference.py`):
```python
def detect_cyclone(image_input):
    """
    Args:
        image_input: File path, PIL Image, or NumPy array.
    Returns:
        dict conforming to integration contract.
    """
    # Preprocess using Person 1 pipeline
    frame_data = preprocess_single_frame(image_input, target_size=(256, 256))
    
    # Run model forward pass...
    
    return {
        "detected": True,
        "confidence": 0.94,
        "center": {"latitude": 16.52, "longitude": 82.31},
        "bbox": [420, 190, 600, 370],
        "structural_pattern": "eye_visible"
    }
```

---

## 📁 6. Required Handoff Package Checklist

When your task is complete, ensure the following artifacts exist:
- [ ] `models/detection/model_weights.pt` (Trained model weights)
- [ ] `src/detection/detector.py` (PyTorch model definition)
- [ ] `src/detection/inference.py` (Stand-alone inference function)
- [ ] `src/detection/evaluate.py` (Evaluation script printing metrics)
- [ ] `models/detection/metrics.json` (Precision, recall, F1 scores)
- [ ] `models/detection/sample_predictions.png` (Visual test results)
- [ ] `src/detection/README.md` (Quickstart and usage guide)

---

## 📅 7. Day-by-Day Roadmap (per Team Plan)

| Day | Target Milestone |
|---|---|
| **Day 1** | Inspect `data/processed/detection/`, test data loader, set up PyTorch pretrained backbone. |
| **Day 2** | Train baseline presence and structural pattern classification model. |
| **Day 3** | Implement data augmentations and evaluate initial validation metrics. |
| **Day 4** | Add auxiliary IMD category head / refine pattern classifier. |
| **Day 5** | Implement geographic coordinate conversion utility. |
| **Day 6** | Build standardized `detect_cyclone()` inference function in `src/detection/inference.py`. |
| **Day 7** | Generate final test set evaluation metrics, confusion matrix, and visual samples. |
| **Day 8** | **FINAL HANDOFF:** Deliver weights, inference scripts, metrics, and documentation. |
