# SIH 2026 — PS 26070: Tropical Cyclone AI/ML System

AI/ML system for identification, classification, and prediction of tropical cyclone patterns using multi-source satellite and meteorological data (Ministry of Earth Sciences / IMD).

---

## 🏗️ Repository Architecture

```
PS70/
├── build_datasets.py            # Master dataset assembly & sequence generator
├── clean_kaggle_intensity.py    # Kaggle image preprocessor
├── download_era5.py             # ERA5 ECMWF CDS API downloader
├── extract_era5_at_points.py    # ERA5 nearest-neighbor spatial/temporal extractor
├── get_ibtracs.py               # NOAA IBTrACS downloader and IMD scaler
├── data/
│   ├── metadata/                # Master datasets and cyclone-level split manifests
│   ├── processed/
│   │   ├── detection/           # Dataset A: Presence & pattern detection
│   │   ├── classification/      # Dataset B: Multi-source tabular & image-only
│   │   └── forecasting/         # Dataset C: Multi-step temporal sequence arrays (.npz)
│   ├── qa_reports/              # QA summary report and validation figures
│   └── raw/                     # Raw IBTrACS CSV, ERA5 NetCDF, and INSAT imagery
├── docs/
│   └── ERA5_ALIGNMENT.md        # Technical specification of ERA5 environmental alignment
├── scripts/
│   └── qa_visualization.py      # Quality assurance visualization pipeline
└── src/
    └── data/                    # Reusable PyTorch / NumPy Datasets & DataLoaders
        ├── forecasting_dataset.py
        ├── classification_dataset.py
        ├── detection_dataset.py
        └── dataloader_example.py
```

---

## 🚀 Quickstart for Downstream Team Members

### 1. Person 4 (Forecasting)
```python
from src.data.forecasting_dataset import CycloneForecastingDataset

# Loads (N, 5, 7) history sequences and (N, 3, 3) forecast targets (+6h, +12h, +24h)
train_dataset = CycloneForecastingDataset(split="train")
x, y = train_dataset[0]
```

### 2. Person 3 (Classification & Intensity)
```python
from src.data.classification_dataset import MultisourceClassificationDataset, CycloneImageDataset

# Multi-source tabular loader (IBTrACS + ERA5)
ms_dataset = MultisourceClassificationDataset(split="train")
features, category_idx, wind_speed, pressure = ms_dataset[0]

# Image-only INSAT loader
img_dataset = CycloneImageDataset(split="train")
image_tensor, category_idx, wind_speed = img_dataset[0]
```

### 3. Person 2 (Detection & Presence)
```python
from src.data.detection_dataset import CycloneDetectionDataset

# Presence and structural pattern loader
det_dataset = CycloneDetectionDataset(split="train")
sample = det_dataset[0]
# sample contains: "image", "detected", "structural_pattern", "category"
```

---

## 📊 Verification & QA
To verify all datasets and generate QA visualization reports:
```bash
python src/data/dataloader_example.py
python scripts/qa_visualization.py
```
Outputs are stored in `data/qa_reports/`.
