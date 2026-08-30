import os
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.detection.detector import CycloneDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CycloneInference:

    def __init__(
        self,
        model_path="models/detection/model_weights.pt"
    ):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        self.pattern_to_idx = checkpoint["pattern_to_idx"]
        self.category_to_idx = checkpoint["category_to_idx"]

        self.idx_to_pattern = {
            idx: label
            for label, idx in self.pattern_to_idx.items()
        }

        self.idx_to_category = {
            idx: label
            for label, idx in self.category_to_idx.items()
        }

        self.model = CycloneDetector(
            num_patterns=len(self.pattern_to_idx),
            num_categories=len(self.category_to_idx)
        ).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()


    def detect_cyclone(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = self.transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

        presence_probability = torch.sigmoid(
            outputs["presence"]
        ).item()

        detected = (
            presence_probability >= 0.5
        )

        pattern_probabilities = torch.softmax(
            outputs["pattern"],
            dim=1
        )[0]

        pattern_index = torch.argmax(
            pattern_probabilities
        ).item()

        pattern_confidence = (
            pattern_probabilities[
                pattern_index
            ].item()
        )

        category_probabilities = torch.softmax(
            outputs["category"],
            dim=1
        )[0]

        category_index = torch.argmax(
            category_probabilities
        ).item()

        category_confidence = (
            category_probabilities[
                category_index
            ].item()
        )

        result = {

            "detected": detected,

            "confidence": round(
                presence_probability,
                4
            ),

            "structural_pattern": self.idx_to_pattern[
                pattern_index
            ],

            "pattern_confidence": round(
                pattern_confidence,
                4
            ),

            "category": self.idx_to_category[
                category_index
            ],

            "category_confidence": round(
                category_confidence,
                4
            )
        }

        return result


def detect_cyclone(image_path):

    detector = CycloneInference()

    return detector.detect_cyclone(
        image_path
    )


if __name__ == "__main__":

    print(
        "Cyclone inference module ready."
    )
