# Person 3 — Cyclone Classification & Intensity: Implementation Guide
**SIH 2026 — PS 26070: Tropical Cyclone AI/ML System**  
**Role:** Person 3 (Classification & Intensity Estimation)

---

## 🎯 1. Your Role & Responsibility
Your core mission is to estimate the **intensity and severity** of a detected tropical cyclone:
1. **Classify IMD Intensity Scale (MUST):** Assign the cyclone to one of the 7 official India Meteorological Department (IMD / RSMC New Delhi) categories.
2. **Estimate Maximum Sustained Wind Speed (MUST):** Regression predicting wind speed in $\text{km/h}$ (or knots).
3. **Estimate Central Pressure (NICE TO HAVE):** Regression predicting minimum central pressure in $\text{hPa}$.
4. **Demonstrate Multi-Source Advantage (CORE REQUIREMENT):** Compare an **Image-Only Model** vs. a **Multi-Source Model (Atmospheric/ERA5 + Location)** to show how environmental data enhances accuracy.

---

## 🏛️ 2. Official IMD Intensity Scale (Strict Compliance)

Do NOT use the Saffir-Simpson (US) scale. All labels are aligned to the official IMD / RSMC New Delhi criteria:

| Index | Official IMD Category | Wind Speed Band (knots) | Wind Speed Band ($\text{km/h}$) |
|:---:|---|:---:|:---:|
| **0** | **Depression** | $17 - 27\text{ kt}$ | $31 - 49\text{ km/h}$ |
| **1** | **Deep Depression** | $28 - 33\text{ kt}$ | $50 - 61\text{ km/h}$ |
| **2** | **Cyclonic Storm** | $34 - 47\text{ kt}$ | $62 - 88\text{ km/h}$ |
| **3** | **Severe Cyclonic Storm** | $48 - 63\text{ kt}$ | $89 - 117\text{ km/h}$ |
| **4** | **Very Severe Cyclonic Storm** | $64 - 89\text{ kt}$ | $118 - 166\text{ km/h}$ |
| **5** | **Extremely Severe Cyclonic Storm** | $90 - 119\text{ kt}$ | $167 - 221\text{ km/h}$ |
| **6** | **Super Cyclonic Storm** | $\ge 120\text{ kt}$ | $\ge 222\text{ km/h}$ |

---

## 📦 3. What Person 1 Has Prepared for You

You have two clean data streams partitioned by unique cyclones (zero data leakage):

### Stream 1: Multi-Source Tabular Dataset (`data/processed/classification/`)
* `multisource_train.csv` (3,039 observations across 105 cyclones)
* `multisource_val.csv` (518 observations across 22 cyclones)
* `multisource_test.csv` (651 observations across 24 cyclones)
* **Features:** `lat`, `lon`, `sst` (Sea Surface Temp, $^\circ\text{C}$), `pressure_msl` (MSLP, $\text{hPa}$), `wind_u` (Zonal wind, $\text{m/s}$), `wind_v` (Meridional wind, $\text{m/s}$), `pre_genesis_favorable`
* **Targets:** `category` (IMD string), `wind_speed` ($\text{km/h}$), `pressure` ($\text{hPa}$)

### Stream 2: Image-Only INSAT Dataset (`data/processed/classification/image_only_kaggle/`)
* `train_labels.csv` (93 images)
* `val_labels.csv` (19 images)
* `test_labels.csv` (21 images)
* `images/` directory with 133 preprocessed infrared crops.

### Python / PyTorch DataLoaders:
```python
from src.data.classification_dataset import MultisourceClassificationDataset, CycloneImageDataset

# Multi-source tabular loader
ms_train = MultisourceClassificationDataset(split="train")
x_features, category_idx, wind_speed, pressure = ms_train[0]

# Image-only loader
img_train = CycloneImageDataset(split="train")
img_tensor, category_idx, wind_speed = img_train[0]
```

---

## 🚀 4. Step-by-Step Implementation Workflow

### Step 4.1: Build Model A (Image-Only Baseline)
Train a CNN on the 133 INSAT IR crops to establish the single-sensor baseline:
* **Architecture:** `ResNet18` or `MobileNetV3` with dual output heads:
  1. Classification Head (7 classes, Cross-Entropy Loss)
  2. Wind Speed Head (Linear, Smooth L1 / MSE Loss)

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageOnlyIntensityModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_feats, num_classes)
        self.wind_regressor = nn.Linear(in_feats, 1)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        wind = self.wind_regressor(feat)
        return logits, wind
```

### Step 4.2: Build Model B (Multi-Source Tabular Model)
Train an gradient-boosted tree ensemble (LightGBM / XGBoost) or an MLP on the 4,208 atmospheric + location records:

```python
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

# Load pre-split datasets
train_df = pd.read_csv("data/processed/classification/multisource_train.csv")
test_df = pd.read_csv("data/processed/classification/multisource_test.csv")

features = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
X_train, y_train_cat, y_train_wind = train_df[features], train_df["category"], train_df["wind_speed"]
X_test, y_test_cat, y_test_wind = test_df[features], test_df["category"], test_df["wind_speed"]

# Train Category Classifier
clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
clf.fit(X_train, y_train_cat)

# Train Wind Speed Regressor
reg = LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
reg.fit(X_train, y_train_wind)
```

### Step 4.3: Model C (Optional Feature Fusion Multimodal Architecture)
Fuse image embeddings with environmental numerical vectors into a unified multi-source network:
$$\text{Fused Feature} = [\text{CNN\_Embedding}(\text{Image}) \,\|\, \text{MLP}(\text{SST}, \text{MSLP}, U, V, \text{Lat}, \text{Lon})]$$

---

## 📊 5. Evaluation & Comparison Requirements
You MUST report metrics on the test set for both models to prove the multi-source hypothesis:

| Metric | Image-Only Model | Multi-Source Model | Performance Delta |
|---|:---:|:---:|:---:|
| **Category Accuracy** | $X.X\%$ | $Y.Y\%$ | $+Z.Z\%$ |
| **Category Macro F1** | $0.XX$ | $0.YY$ | $+0.ZZ$ |
| **Wind Speed MAE ($\text{km/h}$)** | $XX.X$ | $YY.Y$ | $-W.W\text{ km/h (Improvement)}$ |
| **Wind Speed RMSE ($\text{km/h}$)** | $XX.X$ | $YY.Y$ | $-W.W\text{ km/h (Improvement)}$ |

---

## 🔌 6. Inference Function & Integration Contract

### Required Inference Function (`src/classification/inference.py`):
```python
def classify_cyclone(image_input=None, environmental_data=None):
    """
    Args:
        image_input: Satellite image frame (optional if tabular available)
        environmental_data: dict with keys {'lat', 'lon', 'sst', 'pressure_msl', 'wind_u', 'wind_v'}
    Returns:
        dict conforming to integration contract.
    """
    # Execute prediction...
    
    return {
        "category": "Severe Cyclonic Storm",
        "wind_speed": 145.0,
        "pressure": 950.0,
        "confidence": 0.91
    }
```

---

## 📁 7. Required Handoff Package Checklist

- [ ] `models/classification/classifier.pt` or `lgbm_classifier.pkl`
- [ ] `models/classification/intensity_model.pt` or `lgbm_regressor.pkl`
- [ ] `src/classification/classifier.py` (Model definitions)
- [ ] `src/classification/inference.py` (Standardized inference function)
- [ ] `src/classification/evaluate.py` (Evaluation script comparing Image vs Multi-Source)
- [ ] `models/classification/metrics_comparison.json` (Comparative metric tables)
- [ ] `models/classification/confusion_matrix.png` (IMD category confusion matrix)
- [ ] `src/classification/README.md` (Documentation and quickstart)

---

## 📅 8. Day-by-Day Roadmap (per Team Plan)

| Day | Target Milestone |
|---|---|
| **Day 1** | Inspect `data/processed/classification/`, verify IMD 7-class encoding, set up baseline scripts. |
| **Day 2** | Train Image-Only baseline (ResNet/MobileNet) on 133 IR images. |
| **Day 3** | Evaluate Image-Only classification accuracy and wind MAE. |
| **Day 4** | Train Multi-Source tabular models (LightGBM/XGBoost/MLP) on ERA5+IBTrACS records. |
| **Day 5** | Evaluate multi-source model and tune hyperparameters. |
| **Day 6** | Build comparative evaluation table (Image-Only vs. Multi-Source performance lift). |
| **Day 7** | Implement standardized `classify_cyclone()` inference function and export confusion matrix. |
| **Day 8** | **FINAL HANDOFF:** Deliver models, inference functions, metrics comparison, and README. |
