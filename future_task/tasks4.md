# Person 4 — Cyclone Trajectory & Intensity Forecasting: Implementation Guide
**SIH 2026 — PS 26070: Tropical Cyclone AI/ML System**  
**Role:** Person 4 (Cyclone Forecasting)

---

## 🎯 1. Your Role & Responsibility
You own the primary predictive component of the entire AI system:
1. **Multi-Horizon Trajectory Prediction:** Predict the future geographical position ($\text{Latitude}, \text{Longitude}$) of the cyclone at **+6h, +12h, and +24h** lead times.
2. **Multi-Horizon Intensity Prediction:** Predict future maximum sustained wind speed ($\text{km/h}$) at **+6h, +12h, and +24h**.
3. **Benchmark Against Persistence Baseline (CRITICAL):** Build and evaluate a physical movement-vector persistence baseline to scientifically prove that your neural network outperforms simple physics/extrapolation.
4. **Demonstrate Multi-Source Advantage:** Show that adding ERA5 atmospheric reanalysis (SST, MSLP, U/V wind) significantly reduces track and intensity error compared to track-only forecasting.

---

## 📦 2. What Person 1 Has Prepared for You

Person 1 has generated 3,076 temporal sequence arrays with **strict cyclone-level partitioning (zero data leakage)**:

### Dataset C Arrays (`data/processed/forecasting/`)

| Split | Input Tensor $X$ | Target Tensor $Y$ | Number of Sequences | Files |
|---|---|---|---|---|
| **Train** | `(2275, 5, 7)` | `(2275, 3, 3)` | 2,275 sequences | `train_sequences.npz`, `train_sequences_metadata.csv` |
| **Validation** | `(378, 5, 7)` | `(378, 3, 3)` | 378 sequences | `val_sequences.npz`, `val_sequences_metadata.csv` |
| **Test** | `(423, 5, 7)` | `(423, 3, 3)` | 423 sequences | `test_sequences.npz`, `test_sequences_metadata.csv` |

### Feature & Target Schema
* **Input $X$ (5 timesteps: $t-24\text{h}, t-18\text{h}, t-12\text{h}, t-6\text{h}, t$ — 7 features):**
  `[0: lat, 1: lon, 2: wind_speed, 3: pressure, 4: sst, 5: wind_u, 6: wind_v]`
* **Target $Y$ (3 lead times: $+6\text{h}, +12\text{h}, +24\text{h}$ — 3 features):**
  `[0: lat, 1: lon, 2: wind_speed]`

### Quickstart Loading:
```python
from src.data.forecasting_dataset import CycloneForecastingDataset, get_forecasting_data

# Option A: PyTorch Dataset
train_dataset = CycloneForecastingDataset(split="train")
x_tensor, y_tensor = train_dataset[0]

# Option B: Direct NumPy Arrays
X_train, Y_train, meta_train, features, targets = get_forecasting_data(split="train")
print(X_train.shape) # (2275, 5, 7)
print(Y_train.shape) # (2275, 3, 3)
```

---

## 🚀 3. Step-by-Step Implementation Workflow

### Step 3.1: Build the Persistence Baseline (Day 1–2)
Before training any neural network, implement the non-AI movement vector baseline.
* **Position Extrapolation:**
  $$\vec{v}_{\text{recent}} = \frac{\text{pos}(t) - \text{pos}(t-6\text{h})}{6\text{ hours}}$$
  $$\text{pos}(t + \Delta t) = \text{pos}(t) + \vec{v}_{\text{recent}} \cdot \Delta t$$
* **Intensity Persistence:** Hold wind speed constant: $\text{wind}(t + \Delta t) = \text{wind}(t)$.

```python
import numpy as np

def predict_persistence(X_sample):
    """
    X_sample: shape (5, 7) where index 4 is t=0, index 3 is t=-6h
    Returns: shape (3, 3) for +6h, +12h, +24h
    """
    curr_lat, curr_lon, curr_wind = X_sample[4, 0], X_sample[4, 1], X_sample[4, 2]
    prev_lat, prev_lon = X_sample[3, 0], X_sample[3, 1]
    
    # 6-hourly velocity
    v_lat = (curr_lat - prev_lat) / 6.0
    v_lon = (curr_lon - prev_lon) / 6.0
    
    forecasts = []
    for dt in [6, 12, 24]:
        pred_lat = curr_lat + v_lat * dt
        pred_lon = curr_lon + v_lon * dt
        pred_wind = curr_wind  # Intensity persistence
        forecasts.append([pred_lat, pred_lon, pred_wind])
        
    return np.array(forecasts, dtype=np.float32)
```

### Step 3.2: Build the Multi-Step LSTM / GRU Model (Day 3–5)
Implement a sequence-to-sequence or recurrent neural network:
* **Architecture:** 2-layer LSTM / GRU (hidden dim: 64 or 128) + Multi-Head Linear Decoder.

```python
import torch
import torch.nn as nn

class CycloneForecasterLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, num_steps=3, output_dim=3):
        super().__init__()
        self.num_steps = num_steps
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Decoder mapping hidden state -> 3 lead times x 3 features (9 outputs)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_steps * output_dim)
        )

    def forward(self, x):
        # x: (batch_size, 5, 7)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1] # (batch_size, hidden_dim)
        out = self.fc(last_hidden) # (batch_size, 9)
        return out.view(-1, self.num_steps, self.output_dim) # (batch_size, 3, 3)
```

### Step 3.3: Training & Multi-Task Loss Formulation
* **Loss Function:** Combined Mean Squared Error / Huber Loss for coordinates and wind speed:
  $$\mathcal{L} = \text{MSE}(\hat{\text{lat}}, \text{lat}) + \text{MSE}(\hat{\text{lon}}, \text{lon}) + \lambda \cdot \text{MSE}(\hat{\text{wind}}, \text{wind})$$
  *(Set $\lambda \approx 0.01$ to balance coordinate scale $\sim 10^1$ with wind scale $\sim 10^2$)*.
* **Optimizer:** Adam ($\text{lr} = 10^{-3}$ with `ReduceLROnPlateau`), 40–50 epochs.

---

## 📏 4. Evaluation Metrics (Great-Circle Distance)

You MUST evaluate track errors using the **Haversine Great-Circle Distance** formula (reporting error in kilometers):

$$d = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
*(where $R = 6371\text{ km}$)*.

### Required Reporting Table (Evaluated on Test Split):

| Lead Time | Persistence Baseline Track Error (km) | AI Model Track Error (km) | Track Error Reduction (%) | Wind Speed MAE ($\text{km/h}$) |
|:---:|:---:|:---:|:---:|:---:|
| **+6 Hours** | $XX.X\text{ km}$ | $YY.Y\text{ km}$ | $+ZZ.Z\%$ | $AA.A\text{ km/h}$ |
| **+12 Hours** | $XX.X\text{ km}$ | $YY.Y\text{ km}$ | $+ZZ.Z\%$ | $BB.B\text{ km/h}$ |
| **+24 Hours** | $XX.X\text{ km}$ | $YY.Y\text{ km}$ | $+ZZ.Z\%$ | $CC.C\text{ km/h}$ |

---

## 🔌 5. Inference Function & Integration Contract

### Required Inference Function (`src/forecasting/inference.py`):
```python
def forecast_cyclone(history_sequence):
    """
    Args:
        history_sequence: (5, 7) array or list of past 5 observations at 6h intervals.
    Returns:
        dict conforming to integration contract.
    """
    # Execute forward pass...
    
    return {
        "forecast": [
            {"hours": 6, "latitude": 17.10, "longitude": 81.60, "wind_speed": 150.0},
            {"hours": 12, "latitude": 17.80, "longitude": 80.90, "wind_speed": 155.0},
            {"hours": 24, "latitude": 19.10, "longitude": 79.50, "wind_speed": 160.0}
        ]
    }
```

---

## 📁 6. Required Handoff Package Checklist

- [ ] `models/forecasting/forecast_model.pt` (Trained PyTorch weights)
- [ ] `src/forecasting/forecaster.py` (Model architecture definition)
- [ ] `src/forecasting/inference.py` (Standardized inference function)
- [ ] `src/forecasting/baseline.py` (Persistence baseline implementation)
- [ ] `src/forecasting/evaluate.py` (Track error & intensity evaluation script)
- [ ] `models/forecasting/forecast_metrics.json` (Error metrics in km and km/h)
- [ ] `models/forecasting/track_predictions.png` (Visual map showing actual vs. predicted tracks)
- [ ] `src/forecasting/README.md` (Documentation and usage guide)

---

## 📅 7. Day-by-Day Roadmap (per Team Plan)

| Day | Target Milestone |
|---|---|
| **Day 1** | Inspect `data/processed/forecasting/`, test data loader, understand 7 features. |
| **Day 2** | Implement and evaluate physical Persistence Baseline across test sequences. |
| **Day 3** | Implement PyTorch LSTM / GRU architecture and verify forward pass tensor shapes. |
| **Day 4** | Train Track-Only LSTM (lat/lon) on training sequences. |
| **Day 5** | Add Multi-Source features (SST, MSLP, U/V wind) and train full environmental LSTM. |
| **Day 6** | Calculate Haversine track errors at +6h, +12h, and +24h; compare vs Baseline. |
| **Day 7** | Implement standardized `forecast_cyclone()` inference function and plot track maps. |
| **Day 8** | **FINAL HANDOFF:** Deliver model weights, inference scripts, metrics report, and README. |
