# Dataset A — Cyclone Detection & Presence (Person 2 Handoff)

## Overview
Dataset A delivers the image manifests and structural pattern annotations for Person 2 (Detection & Presence).

---

## 1. Files & Splits

* `train_detection.csv` (93 records)
* `val_detection.csv` (19 records)
* `test_detection.csv` (21 records)
* `detection_all.csv` (133 total records)

---

## 2. Schema

* `filename`: Base image filename
* `image_path`: Path to infrared satellite image
* `cyclone_detected`: Boolean presence label (`True`)
* `category`: IMD Category
* `wind_speed_kmh`: Sustained wind speed
* `structural_pattern`: Dvorak-aligned structural pattern tag (`eye_visible`, `curved_band`, `shear_pattern`)
* `mock_bbox`: Mock bounding box coordinates `[420, 190, 600, 370]` conforming to integration contract

---

## 3. Quickstart

```python
from src.data.detection_dataset import CycloneDetectionDataset

det_train = CycloneDetectionDataset(split="train")
sample = det_train[0]
print(sample["detected"], sample["structural_pattern"], sample["category"])
```
