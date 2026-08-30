import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data.detection_dataset import CycloneDetectionDataset
from src.detection.detector import CycloneDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = CycloneDetectionDataset(
        split="test",
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    checkpoint = torch.load(
        "models/detection/model_weights.pt",
        map_location=DEVICE
    )

    pattern_to_idx = checkpoint["pattern_to_idx"]
    category_to_idx = checkpoint["category_to_idx"]

    model = CycloneDetector(
        num_patterns=len(pattern_to_idx),
        num_categories=len(category_to_idx)
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    pattern_true = []
    pattern_pred = []

    category_true = []
    category_pred = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)

            outputs = model(images)

            predicted_patterns = torch.argmax(
                outputs["pattern"],
                dim=1
            ).cpu().numpy()

            predicted_categories = torch.argmax(
                outputs["category"],
                dim=1
            ).cpu().numpy()

            actual_patterns = [
                pattern_to_idx[p]
                for p in batch["structural_pattern"]
            ]

            actual_categories = [
                category_to_idx[c]
                for c in batch["category"]
            ]

            pattern_true.extend(actual_patterns)
            pattern_pred.extend(predicted_patterns)

            category_true.extend(actual_categories)
            category_pred.extend(predicted_categories)

    results = {

        "pattern_accuracy": accuracy_score(
            pattern_true,
            pattern_pred
        ),

        "pattern_precision": precision_score(
            pattern_true,
            pattern_pred,
            average="weighted",
            zero_division=0
        ),

        "pattern_recall": recall_score(
            pattern_true,
            pattern_pred,
            average="weighted",
            zero_division=0
        ),

        "pattern_f1": f1_score(
            pattern_true,
            pattern_pred,
            average="weighted",
            zero_division=0
        ),

        "category_accuracy": accuracy_score(
            category_true,
            category_pred
        ),

        "category_precision": precision_score(
            category_true,
            category_pred,
            average="weighted",
            zero_division=0
        ),

        "category_recall": recall_score(
            category_true,
            category_pred,
            average="weighted",
            zero_division=0
        ),

        "category_f1": f1_score(
            category_true,
            category_pred,
            average="weighted",
            zero_division=0
        )
    }

    os.makedirs(
        "metrics",
        exist_ok=True
    )

    with open(
        "metrics/detection_metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nDetection Evaluation Results\n")

    for metric, value in results.items():

        print(
            f"{metric}: {value:.4f}"
        )


if __name__ == "__main__":
    evaluate()
