"""
src/classification/classifier.py
Classifiers for Multi-Source Cyclone Intensity & Category Estimation.
Includes:
1. MultispectralCycloneCNN: Deep CNN ingesting paired INSAT-3D Thermal IR + Visible imagery (2 channels).
2. MultisourceTabularModel: Gradient-boosted / Random Forest model ingesting ERA5 reanalysis + IBTrACS features.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

IMD_CLASSES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(IMD_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(IMD_CLASSES)}

IMD_SHORT_CODES = {
    "Depression": "D",
    "Deep Depression": "DD",
    "Cyclonic Storm": "CS",
    "Severe Cyclonic Storm": "SCS",
    "Very Severe Cyclonic Storm": "VSCS",
    "Extremely Severe Cyclonic Storm": "ESCS",
    "Super Cyclonic Storm": "SuCS"
}

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.conv(x))

class MultispectralCycloneCNN(nn.Module):
    """
    Dual-Channel Satellite CNN (Channel 0: Thermal IR, Channel 1: Visible).
    """
    def __init__(self, in_channels=2, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 128x128
            ConvBlock(32, 64),            # 64x64
            ConvBlock(64, 128),           # 32x32
            ConvBlock(128, 256),          # 16x16
            ConvBlock(256, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.wind_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        wind = self.wind_regressor(feat).squeeze(-1)
        return logits, wind

class MultisourceTabularModel:
    """
    Environmental Reanalysis + Location Tabular Model.
    Features: [lat, lon, sst, pressure_msl, wind_u, wind_v]
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.cls_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        self.wind_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        self.pres_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)

    def fit(self, X, y_cat, y_wind, y_pres):
        X_clean = np.nan_to_num(X, nan=28.0)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.cls_model.fit(X_scaled, y_cat)
        self.wind_model.fit(X_scaled, y_wind)
        valid_pres = (y_pres > 800.0)
        if np.any(valid_pres):
            self.pres_model.fit(X_scaled[valid_pres], y_pres[valid_pres])

    def predict(self, X):
        X_clean = np.nan_to_num(X, nan=28.0)
        X_scaled = self.scaler.transform(X_clean)
        pred_cat = self.cls_model.predict(X_scaled)
        pred_wind = self.wind_model.predict(X_scaled)
        pred_pres = self.pres_model.predict(X_scaled)
        probs = self.cls_model.predict_proba(X_scaled)
        confs = np.max(probs, axis=1)
        return pred_cat, pred_wind, pred_pres, confs

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
