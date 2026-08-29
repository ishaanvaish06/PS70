# QA & Validation Report — Person 1 Data Foundation
**Date Generated:** 2026-08-29 11:41:51  
**Target:** SIH 2026 PS 26070 — Tropical Cyclone AI/ML System

---

## 1. Master Dataset Summary
* **Total Track Observations:** 5,481
* **Unique North Indian Cyclones:** 151
* **Time Range:** 2013-05-09 to 2025-12-02
* **ERA5 Missing Values:**
  * `sst`: 1538
  * `pressure_msl`: 0
  * `wind_u`: 0
  * `wind_v`: 0
  * **Completeness:** **100.0%**

---

## 2. Cyclone-Level Train / Val / Test Partition
* **Train Set:** 105 cyclones (69.5%)
* **Validation Set:** 22 cyclones (14.6%)
* **Test Set:** 24 cyclones (15.9%)
* **Partition Principle:** Grouped strictly by `cyclone_id` to guarantee zero temporal or spatial leakage across storm observations.

---

## 3. Dataset C (Forecasting Sequences) Metrics
* **Train Sequences:** 2,275 (Input: `(2275, 5, 7)`, Target: `(2275, 3, 3)`)
* **Val Sequences:** 378 (Input: `(378, 5, 7)`, Target: `(378, 3, 3)`)
* **Test Sequences:** 423 (Input: `(423, 5, 7)`, Target: `(423, 3, 3)`)
* **Total Generated Sequences:** 3,076

---

## 4. Key Artifact Figures
1. Category Distribution: `figures/category_distribution.png`
2. Annual Frequency: `figures/annual_frequency.png`
3. Geographic Tracks: `figures/geographic_tracks.png`
4. Wind vs. Pressure Correlation: `figures/wind_vs_pressure.png`
