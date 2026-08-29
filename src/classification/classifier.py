"""
src/classification/classifier.py
Model definitions for Person 3 (Cyclone Classification & Intensity Estimation).

Includes:
1. ImageOnlyIntensityModel: PyTorch CNN baseline (ResNet18 / MobileNetV3) for 133 INSAT IR crops.
2. MultisourceTabularModel: Gradient Boosted Decision Tree / MLP wrapper for 4,208 ERA5 + IBTrACS records.
3. MultimodalFusionModel: PyTorch multimodal feature fusion combining image CNN embeddings with tabular MLP embeddings.
"""

import os
import pickle
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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

if TORCH_AVAILABLE:
    class ImageOnlyIntensityModel(nn.Module):
        """
        Model A: Single-sensor baseline CNN for INSAT IR crops.
        Backbone: ResNet18 fine-tuned.
        Dual Output Heads:
          1. classifier: Logits for 7 IMD intensity categories.
          2. wind_regressor: Predicted maximum sustained wind speed (km/h).
        """
        def __init__(self, num_classes=7, backbone_name="resnet18"):
            super().__init__()
            self.backbone_name = backbone_name
            if backbone_name == "resnet18":
                weights = ResNet18_Weights.DEFAULT
                base_model = resnet18(weights=weights)
                in_feats = base_model.fc.in_features
                base_model.fc = nn.Identity()
                self.backbone = base_model
            else:
                weights = MobileNet_V3_Small_Weights.DEFAULT
                base_model = mobilenet_v3_small(weights=weights)
                in_feats = base_model.classifier[0].in_features
                base_model.classifier = nn.Identity()
                self.backbone = base_model

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_feats, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

            self.wind_regressor = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_feats, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            # x shape: (B, 3, H, W)
            feats = self.backbone(x)
            logits = self.classifier(feats)
            wind = self.wind_regressor(feats).squeeze(-1)
            return logits, wind

    class MultimodalFusionModel(nn.Module):
        """
        Model C: Fusion network combining Satellite Image CNN features with ERA5 Tabular MLP features.
        """
        def __init__(self, num_tabular_features=6, num_classes=7):
            super().__init__()
            # Image Branch
            weights = ResNet18_Weights.DEFAULT
            base_cnn = resnet18(weights=weights)
            in_feats_cnn = base_cnn.fc.in_features
            base_cnn.fc = nn.Identity()
            self.image_backbone = base_cnn
            self.image_proj = nn.Sequential(
                nn.Linear(in_feats_cnn, 64),
                nn.ReLU()
            )

            # Tabular Branch
            self.tabular_mlp = nn.Sequential(
                nn.Linear(num_tabular_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

            # Fused Classifier & Regressor (64 + 64 = 128)
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            self.wind_regressor = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, img_x, tab_x):
            img_emb = self.image_proj(self.image_backbone(img_x))
            tab_emb = self.tabular_mlp(tab_x)
            fused = torch.cat([img_emb, tab_emb], dim=1)
            logits = self.classifier(fused)
            wind = self.wind_regressor(fused).squeeze(-1)
            return logits, wind
else:
    class ImageOnlyIntensityModel:
        def __init__(self, *args, **kwargs):
            pass

    class MultimodalFusionModel:
        def __init__(self, *args, **kwargs):
            pass

class MultisourceTabularModel:
    """
    Model B: Multi-source environmental model (ERA5 reanalysis + IBTrACS location).
    Uses LightGBM (or RandomForest fallback) for classification, wind speed regression, and central pressure regression.
    """
    def __init__(self, use_lightgbm=True):
        self.use_lightgbm = use_lightgbm and LIGHTGBM_AVAILABLE
        self.feature_names = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
        self.scaler = StandardScaler()

        if self.use_lightgbm:
            self.clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            self.wind_reg = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            self.pres_reg = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        else:
            self.clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            self.wind_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
            self.pres_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)

    def fit(self, X, y_cat, y_wind, y_pres=None):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y_cat)
        self.wind_reg.fit(X_scaled, y_wind)
        if y_pres is not None and len(y_pres) > 0:
            valid_mask = y_pres > 0
            if np.any(valid_mask):
                self.pres_reg.fit(X_scaled[valid_mask], y_pres[valid_mask])

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        pred_cat = self.clf.predict(X_scaled)
        pred_wind = self.wind_reg.predict(X_scaled)
        try:
            pred_pres = self.pres_reg.predict(X_scaled)
        except Exception:
            pred_pres = np.full(len(X), 990.0)

        # Get probabilities for confidence
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X_scaled)
            confs = np.max(probs, axis=1)
        else:
            confs = np.full(len(X), 0.85)

        return pred_cat, pred_wind, pred_pres, confs

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
