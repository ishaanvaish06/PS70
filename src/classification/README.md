# Person 3 — Cyclone Classification & Intensity Estimation Subsystem

**SIH 2026 — PS 26070: Tropical Cyclone AI/ML System**

---

## 📌 Overview

The `src/classification/` module estimates tropical cyclone intensity and severity in accordance with the official **India Meteorological Department (IMD / RSMC New Delhi)** 7-tier scale.

It delivers:
1. **7-Tier IMD Intensity Classification:** (*Depression* $\rightarrow$ *Super Cyclonic Storm*)
2. **Continuous Maximum Sustained Wind Speed Regression:** ($\text{km/h}$)
3. **Continuous Minimum Central Pressure Regression:** ($\text{hPa}$)
4. **Multi-Source Advantage Proof:** Quantitative comparison demonstrating performance lift from fusing ECMWF ERA5 environmental reanalysis (`sst`, `pressure_msl`, `wind_u`, `wind_v`) and location (`lat`, `lon`) over single-sensor satellite IR images.

---

## 🏗️ Architecture & Model Pipelines

- **Model A (Image-Only PyTorch Baseline):** Fine-tuned `ResNet18` / `MobileNetV3` trained on INSAT IR satellite crop imagery with dual output heads (Cross-Entropy classification + Smooth L1 wind regression).
- **Model B (Multi-Source Tabular Model):** Gradient Boosted Decision Tree ensemble (`LightGBM` / `RandomForest`) trained on 4,208 synoptic environmental records.
- **Model C (Multimodal Fusion Model):** Dual-branch neural network combining 256x256 image features with tabular ERA5 feature embeddings.

---

## 🚀 Quickstart & Usage

### 1. Execute Model Training
Train both Model B (Multi-Source) and Model A (Image-Only PyTorch CNN):
```bash
python src/classification/train.py
```
*Outputs saved to `models/classification/`.*

### 2. Run Comparative Evaluation Pipeline
Evaluate models on the held-out test splits and generate metrics and plots:
```bash
python src/classification/evaluate.py
```
*Generates `models/classification/metrics_comparison.json` and `models/classification/confusion_matrix.png`.*

### 3. Programmatic Inference Usage
```python
from src.classification.inference import classify_cyclone

# Multi-source environmental inference (Recommended)
result = classify_cyclone(
    environmental_data={
        "lat": 16.5,
        "lon": 82.3,
        "sst": 29.5,
        "pressure_msl": 985.0,
        "wind_u": 12.5,
        "wind_v": 3.2
    }
)
print(result)
# Output:
# {
#   "category": "Severe Cyclonic Storm",
#   "imd_code": "SCS",
#   "wind_speed": 145.0,
#   "pressure": 950.0,
#   "confidence": 0.91
# }
```

---

## 📊 Handoff Checklist
- [x] `src/classification/classifier.py` — Model architecture definitions
- [x] `src/classification/train.py` — Multi-source & image training pipeline
- [x] `src/classification/evaluate.py` — Test evaluation & confusion matrix generator
- [x] `src/classification/inference.py` — Standardized unified API inference function
- [x] `models/classification/metrics_comparison.json` — Quantified performance lift report
- [x] `models/classification/confusion_matrix.png` — Visual IMD category confusion matrix
