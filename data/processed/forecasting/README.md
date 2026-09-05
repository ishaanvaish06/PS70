# Dataset C — Cyclone Forecasting Sequences (Person 4 Handoff)

## Overview
Dataset C provides multi-step temporal sequence arrays tailored for Person 4 (Forecasting). It enables trajectory ($\text{lat}, \text{lon}$) and intensity ($\text{wind\_speed}$) prediction at $+6\text{h}$, $+12\text{h}$, and $+24\text{h}$ lead times.

---

## 1. Array Dimensions & Files

| Split | Input Array $X$ | Target Array $Y$ | Sequences Count | Associated Files |
|---|---|---|---|---|
| **Train** | `(2275, 5, 7)` | `(2275, 3, 3)` | 2,275 | `train_sequences.npz`, `train_sequences_metadata.csv` |
| **Validation** | `(378, 5, 7)` | `(378, 3, 3)` | 378 | `val_sequences.npz`, `val_sequences_metadata.csv` |
| **Test** | `(423, 5, 7)` | `(423, 3, 3)` | 423 | `test_sequences.npz`, `test_sequences_metadata.csv` |

---

## 2. Feature Schema

### Input $X$ (5 timesteps: $t-24\text{h}, t-18\text{h}, t-12\text{h}, t-6\text{h}, t$ — 7 features):
1. `lat`: Cyclone center latitude ($^\circ\text{N}$)
2. `lon`: Cyclone center longitude ($^\circ\text{E}$)
3. `wind_speed`: Maximum sustained wind speed ($\text{km/h}$)
4. `pressure`: Minimum central pressure ($\text{hPa}$)
5. `sst`: Sea Surface Temperature ($^\circ\text{C}$) from ERA5
6. `wind_u`: Zonal 10m wind ($\text{m/s}$) from ERA5
7. `wind_v`: Meridional 10m wind ($\text{m/s}$) from ERA5

### Target $Y$ (3 lead times: $+6\text{h}, +12\text{h}, +24\text{h}$ — 3 features):
1. `lat`: Forecasted latitude ($^\circ\text{N}$)
2. `lon`: Forecasted longitude ($^\circ\text{E}$)
3. `wind_speed`: Forecasted wind speed ($\text{km/h}$)

---

## 3. Quickstart with PyTorch / Python

```python
from src.data.forecasting_dataset import CycloneForecastingDataset

# PyTorch Dataset
train_dataset = CycloneForecastingDataset(split="train")
x_seq, y_target = train_dataset[0]

print(x_seq.shape)    # (5, 7)
print(y_target.shape)  # (3, 3)
```

---

## 4. Baseline Formulation (Persistence)
To establish the non-AI baseline:
* **Position Persistence:** Extrapolate position at $t+\Delta t$ using recent velocity vector:
  $$\vec{v} = \frac{\text{pos}(t) - \text{pos}(t-6\text{h})}{6}$$
  $$\text{pos}(t+\Delta t) = \text{pos}(t) + \vec{v} \cdot \Delta t$$
* **Intensity Persistence:** Hold wind speed constant: $\text{wind}(t+\Delta t) = \text{wind}(t)$.

---

## 5. What Additional Data Is Required to Guarantee the Targets?

To reliably push the mean error under $100\text{ km}$ at 24h and under $50\text{ km}$ at 12h, the project needs:

1. **Expand ERA5 Historical Coverage back to 2010–2021:**
   - **Volume:** ~10 years of ERA5 reanalysis for the North Indian Ocean ($500\text{ hPa}, 850\text{ hPa}, 200\text{ hPa}$).
   - **Impact:** Eliminates zero-filled missing values across all 1,112 timesteps and adds another ~2,500 historical cyclone timesteps (Phailin, Fani, Hudhud, Amphan, Tauktae).
2. **Include Adjacent Basin Tracks (Western North Pacific / South Indian Ocean):**
   - **Physics:** Physics of steering flow and vertical wind shear are identical across ocean basins.
   - **Volume & Diversity:** Adding 5–10 years of Western North Pacific typhoon tracks (via IBTrACS + ERA5) multiplies training samples from $1,112 \to 15,000+$, exposing the network to hundreds of recurvature and stalling scenarios.
3. **Download 1,000+ INSAT-3D/3DR Satellite Frames from MOSDAC/NRL:**
   - **Visual Resolution:** Replaces padded image sequences with real thermal infrared (TIR-1 $10.8\mu\text{m}$) and water vapor (WV $6.9\mu\text{m}$) imagery for all storms since 2014.

