# ERA5 Environmental Reanalysis Alignment Documentation
**SIH 2026 — PS 26070: Tropical Cyclone AI/ML System**  
**Role:** Person 1 (Data Engineer)

---

## 1. Overview & Objectives
To provide atmospheric and oceanic environmental context for tropical cyclone detection, intensity classification, and trajectory forecasting, ECMWF ERA5 hourly reanalysis on single levels was aligned with historical cyclone tracks from NOAA IBTrACS.

This document details the variables extracted, spatial and temporal resolutions, coordinate bounds, interpolation methodology, unit conversions, and validation metrics.

---

## 2. Atmospheric & Oceanic Variables

| Variable Name | ERA5 Parameter | Original Units | Cleaned / Target Units | Physical Significance |
|---|---|---|---|---|
| **SST** | `sea_surface_temperature` | Kelvin ($K$) | Celsius ($^\circ\text{C}$) | Ocean thermal energy driving tropical cyclogenesis and intensification. ($T_{^\circ\text{C}} = T_K - 273.15$) |
| **MSLP** | `mean_sea_level_pressure` | Pascal ($\text{Pa}$) | Hectopascals ($\text{hPa}$) | Core atmospheric pressure indicator; lower values denote stronger cyclone intensity. ($P_{\text{hPa}} = P_{\text{Pa}} / 100$) |
| **U10** | `10m_u_component_of_wind` | $\text{m/s}$ | $\text{m/s}$ | Zonal (East-West) surface wind component; steering flow & shear component. |
| **V10** | `10m_v_component_of_wind` | $\text{m/s}$ | $\text{m/s}$ | Meridional (North-South) surface wind component; steering flow & shear component. |

---

## 3. Spatial Resolution & Geographic Bounding Box

* **Native ERA5 Resolution:** $0.25^\circ \times 0.25^\circ$ ($\approx 28 \text{ km} \times 28 \text{ km}$ at the equator).
* **Downloaded Bounding Box:**
  * **North:** $30.0^\circ\text{N}$
  * **South:** $-5.0^\circ\text{S}$
  * **West:** $50.0^\circ\text{E}$
  * **East:** $105.0^\circ\text{E}$
* **Coverage Scope:** Fully encloses the North Indian Ocean basin (Bay of Bengal and Arabian Sea), exceeding the spec's minimum requirements ($[\text{N}27.5^\circ, \text{S}-5.0^\circ, \text{W}56.0^\circ, \text{E}102.0^\circ]$).

---

## 4. Temporal Resolution & Alignment Methodology

* **Temporal Frequency:** 3-hourly intervals (`00:00`, `03:00`, `06:00`, `09:00`, `12:00`, `15:00`, `18:00`, `21:00` UTC) matching standard synoptic reporting cycles.
* **Storage Format:** 94 monthly NetCDF (`.nc`) files partitioned by active cyclone months.
* **Alignment / Interpolation Strategy:**
  * **Spatial & Temporal Selection:** Nearest-neighbor lookup using Xarray (`.sel(latitude=lat, longitude=lon, time=timestamp, method="nearest")`).
  * **Copernicus 2024 Platform Migration Handling:** NetCDF dimensions and coordinate structures (`valid_time`, `number`, `expver`) were normalized to maintain seamless backward and forward compatibility.

---

## 5. Completeness & Validation Statistics

* **Total IBTrACS Observations Processed:** 5,481
* **Successfully Matched to ERA5:** 5,481
* **Unmatched / Missing Records:** 0 (100.0% match rate)
* **Missing Value Imputation:** Zero missing value imputation required due to exact bounding box superset coverage.
* **Master Dataset Location:**
  * Primary: `data/processed/master_dataset.csv`
  * Mirror: `data/metadata/master_dataset.csv`

---

## 6. Downstream Feature Usage
* **Person 3 (Classification & Intensity):** `sst`, `pressure_msl`, `wind_u`, `wind_v` used as multi-source tabular features fused with cyclone location.
* **Person 4 (Forecasting):** `sst`, `wind_u`, `wind_v`, and `pressure_msl` included at every history timestep ($t-24\text{h} \dots t$) to inform neural trajectory and wind speed regression.
