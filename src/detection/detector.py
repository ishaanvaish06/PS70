import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class CycloneDetector(nn.Module):

    def __init__(self, num_patterns=3, num_categories=7):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT

        self.backbone = mobilenet_v3_small(weights=weights)

        feature_size = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Identity()

        self.presence_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.pattern_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_patterns)
        )

        self.category_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_categories)
        )

    def forward(self, x):

        features = self.backbone(x)

        presence = self.presence_head(features)

        pattern = self.pattern_head(features)

        category = self.category_head(features)

        return {
            "presence": presence,
            "pattern": pattern,
            "category": category
        }
