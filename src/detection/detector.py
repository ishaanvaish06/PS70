"""
src/detection/detector.py
Multi-task Cyclone Detection & Structural Pattern Tagging Model.
Predicts:
  1. Cyclone Presence (Binary: Detected vs. No Cyclone)
  2. Structural Pattern (4-tier: no_cyclone, eye_visible, curved_band, shear_pattern)
  3. IMD Cyclone Category (8-tier: 7 IMD categories + No Cyclone)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PATTERN_CLASSES = ["no_cyclone", "eye_visible", "curved_band", "shear_pattern"]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_CLASSES)}
IDX_TO_PATTERN = {i: p for i, p in enumerate(PATTERN_CLASSES)}

CATEGORY_CLASSES = [
    "No Cyclone",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm"
]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORY_CLASSES)}
IDX_TO_CAT = {i: c for i, c in enumerate(CATEGORY_CLASSES)}

class CycloneDetector(nn.Module):
    def __init__(self, num_patterns=4, num_categories=8):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)
        in_feats = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        # Presence Head (Binary Logit)
        self.presence_head = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # Structural Pattern Head
        self.pattern_head = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_patterns)
        )

        # Auxiliary Category Head
        self.category_head = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_categories)
        )

    def forward(self, x):
        feat = self.backbone(x)
        presence_logit = self.presence_head(feat).squeeze(-1)
        pattern_logits = self.pattern_head(feat)
        category_logits = self.category_head(feat)

        return {
            "presence_logit": presence_logit,
            "pattern_logits": pattern_logits,
            "category_logits": category_logits
        }
