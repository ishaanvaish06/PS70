# SIH 2026 — PS 26070: Team AI/ML Roadmap & Implementation Guide
**AI/ML System for Tropical Cyclone Identification, Classification, and Prediction**  
*Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)*

---

## 🎯 Executive Overview & Current State

Person 1 (Data Engineer) has completed the entire data engineering foundation. All datasets, temporal sequences, tabular reanalysis, satellite images, cyclone-level splits, and PyTorch/Python data loaders are **100% verified and ready on disk**.

This document outlines the workflow, architecture, and responsibilities for **Person 2 (Detection)**, **Person 3 (Classification & Intensity)**, and **Person 4 (Forecasting)**.

```
                                  +---------------------------------------+
                                  |  Person 1: Data Engineering (DONE)   |
                                  |  - NOAA IBTrACS + ERA5 Reanalysis    |
                                  |  - Datasets A, B, C & PyTorch Loaders |
                                  +---------------------------------------+
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                   |                                   |
                  v                                   v                                   v
+-----------------------------------+ +-----------------------------------+ +-----------------------------------+
|  Person 2: Cyclone Detection      | | Person 3: Classification/Intensity| |  Person 4: Cyclone Forecasting    |
|  - Task Guide: tasks2.md          | | - Task Guide: tasks3.md           | |  - Task Guide: tasks4.md          |
|  - Ingests: Dataset A (INSAT IR)  | | - Ingests: Dataset B (Multi+Img)  | |  - Ingests: Dataset C (Sequences) |
|  - Output: Presence + Dvorak Tag  | | - Output: IMD Scale + Wind Speed  | |  - Output: +6h, +12h, +24h Track  |
+-----------------------------------+ +-----------------------------------+ +-----------------------------------+
                  \                                   |                                   /
                   \                                  |                                  /
                    +---------------------------------v---------------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |   Person 5: Frontend / UI     |
                                      |   - Unified /api/analyze API  |
                                      |   - Leaflet Map & Dashboards  |
                                      +-------------------------------+
```

---

## 📁 Directory Structure & Data Asset Catalog

```
PS70/
├── future_task/
│   ├── tasks.md                 <-- Master Team Guide (this file)
│   ├── tasks2.md                <-- Person 2 Implementation Guide (Detection)
│   ├── tasks3.md                <-- Person 3 Implementation Guide (Classification)
│   └── tasks4.md                <-- Person 4 Implementation Guide (Forecasting)
├── data/
│   ├── metadata/                # Master datasets and cyclone splits
│   │   ├── master_dataset.csv
│   │   ├── train.csv, validation.csv, test.csv
│   │   └── train_cyclones.csv, validation_cyclones.csv, test_cyclones.csv
│   ├── processed/
│   │   ├── detection/           # Dataset A for Person 2
│   │   ├── classification/      # Dataset B for Person 3
│   │   └── forecasting/         # Dataset C for Person 4 (.npz sequences)
│   ├── qa_reports/              # Validation figures & QA report
│   └── raw/                     # Raw IBTrACS, ERA5 NetCDF, and INSAT imagery
├── docs/
│   └── ERA5_ALIGNMENT.md        # Technical specs of atmospheric reanalysis
└── src/
    └── data/                    # Reusable PyTorch / NumPy Loaders
        ├── forecasting_dataset.py
        ├── classification_dataset.py
        ├── detection_dataset.py
        ├── preprocess_satellite.py
        └── dataloader_example.py
```

---

## 👥 Summary of Roles & Next Tasks

| Team Member | Domain | Key Task | Primary Dataset | Target Architecture | Guide File |
|---|---|---|---|---|:---:|
| **Person 2** | Detection & Presence | Detect cyclone presence, tag Dvorak structural pattern (`eye_visible`, `curved_band`, etc.), provide pixel/geo coords. | `data/processed/detection/` | Fine-tuned ResNet / MobileNet / YOLO | [`tasks2.md`](file:///D:/AVV/SIH2026/PS70/future_task/tasks2.md) |
| **Person 3** | Classification & Intensity | Classify cyclone into 7 official IMD intensity bands and predict maximum sustained wind speed (km/h). Compare Image-Only vs. Multi-Source. | `data/processed/classification/` | Tabular LightGBM/XGBoost + CNN (ResNet18) | [`tasks3.md`](file:///D:/AVV/SIH2026/PS70/future_task/tasks3.md) |
| **Person 4** | Trajectory & Wind Forecasting | Predict future coordinates (lat, lon) and wind speed at +6h, +12h, and +24h lead times. Benchmark against persistence baseline. | `data/processed/forecasting/` | Multi-Step LSTM / GRU / 1D-CNN | [`tasks4.md`](file:///D:/AVV/SIH2026/PS70/future_task/tasks4.md) |
| **Person 5** | UI & Integration | Build React + Leaflet dashboard consuming `/api/analyze` combining outputs from Person 2, 3, 4 into real-time monitoring cards. | Mock / Model APIs | React, Leaflet, Tailwind CSS | Integration Contract |

---

## 🔗 Unified Team API Integration Contract (for Person 5)

Person 5 (Frontend) will consume the combined `/api/analyze` payload constructed by merging outputs from Persons 2, 3, and 4:

```json
{
  "timestamp": "2026-08-29T12:00:00Z",
  "detection": {
    "detected": true,
    "confidence": 0.94,
    "center": { "latitude": 16.52, "longitude": 82.31 },
    "bbox": [420, 190, 600, 370],
    "structural_pattern": "eye_visible"
  },
  "classification": {
    "category": "Severe Cyclonic Storm",
    "wind_speed": 145.0,
    "pressure": 950.0,
    "confidence": 0.91
  },
  "forecast": [
    { "hours": 6, "latitude": 17.10, "longitude": 81.60, "wind_speed": 150.0 },
    { "hours": 12, "latitude": 17.80, "longitude": 80.90, "wind_speed": 155.0 },
    { "hours": 24, "latitude": 19.10, "longitude": 79.50, "wind_speed": 160.0 }
  ],
  "landfall": {
    "estimated": true,
    "latitude": 18.42,
    "longitude": 84.91,
    "estimated_time": "2026-08-30T06:00:00Z"
  },
  "risk": {
    "score": 82,
    "level": "HIGH"
  }
}
```

---

## ⚡ Quick Testing Command

To verify that your environment can load all datasets:
```bash
python src/data/dataloader_example.py
```
