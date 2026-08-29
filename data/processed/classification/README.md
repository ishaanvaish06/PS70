# Dataset B — Cyclone Classification & Intensity (Person 3 Handoff)

## Overview
Dataset B provides two parallel data streams for Person 3 (Classification & Intensity):
1. **Multi-Source Tabular Dataset:** ERA5 reanalysis (`sst`, `pressure_msl`, `wind_u`, `wind_v`) + geographic position (`lat`, `lon`) paired with IMD category and sustained wind speed.
2. **Image-Only INSAT Dataset:** 133 IR satellite images paired with IMD category and wind speed.

---

## 1. Multi-Source Tabular Data

* **Files:**
  * `multisource_train.csv` (3,039 records)
  * `multisource_val.csv` (518 records)
  * `multisource_test.csv` (651 records)
* **Features:** `lat`, `lon`, `sst`, `pressure_msl`, `wind_u`, `wind_v`
* **Targets:**
  * `category`: Official IMD Scale (*Depression* $\rightarrow$ *Super Cyclonic Storm*)
  * `wind_speed`: Maximum sustained wind ($\text{km/h}$)
  * `pressure`: Central minimum pressure ($\text{hPa}$)

---

## 2. Image-Only Dataset (`image_only_kaggle/`)

* **Files:**
  * `train_labels.csv` (93 images)
  * `val_labels.csv` (19 images)
  * `test_labels.csv` (21 images)
  * `images/`: 133 high-resolution infrared crops
* **Targets:** `category`, `wind_speed_kmh`, `wind_speed_kt`

---

## 3. Quickstart

```python
from src.data.classification_dataset import MultisourceClassificationDataset, CycloneImageDataset

# Multi-source tabular
ms_train = MultisourceClassificationDataset(split="train")
x, cat_idx, wind_kmh, pres = ms_train[0]

# Image-only
img_train = CycloneImageDataset(split="train")
img, cat_idx, wind_kmh = img_train[0]
```
