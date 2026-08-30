import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data.detection_dataset import CycloneDetectionDataset
from src.detection.detector import CycloneDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 10


def create_label_maps(dataset):

    patterns = sorted(
        dataset.df["structural_pattern"]
        .dropna()
        .unique()
    )

    categories = sorted(
        dataset.df["category"]
        .dropna()
        .unique()
    )

    pattern_to_idx = {
        pattern: idx
        for idx, pattern in enumerate(patterns)
    }

    category_to_idx = {
        category: idx
        for idx, category in enumerate(categories)
    }

    return pattern_to_idx, category_to_idx


def train():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CycloneDetectionDataset(
        split="train",
        transform=transform
    )

    val_dataset = CycloneDetectionDataset(
        split="val",
        transform=transform
    )

    pattern_to_idx, category_to_idx = create_label_maps(
        train_dataset
    )

    print("Pattern mapping:", pattern_to_idx)
    print("Category mapping:", category_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CycloneDetector(
        num_patterns=len(pattern_to_idx),
        num_categories=len(category_to_idx)
    ).to(DEVICE)

    presence_loss = nn.BCEWithLogitsLoss()
    pattern_loss = nn.CrossEntropyLoss()
    category_loss = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    os.makedirs(
        "models/detection",
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        progress = tqdm(train_loader)

        for batch in progress:

            images = batch["image"].to(DEVICE)

            detected = (
                batch["detected"]
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            pattern = torch.tensor(
                [
                    pattern_to_idx[p]
                    for p in batch["structural_pattern"]
                ],
                dtype=torch.long
            ).to(DEVICE)

            category = torch.tensor(
                [
                    category_to_idx[c]
                    for c in batch["category"]
                ],
                dtype=torch.long
            ).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss_presence = presence_loss(
                outputs["presence"],
                detected
            )

            loss_pattern = pattern_loss(
                outputs["pattern"],
                pattern
            )

            loss_category = category_loss(
                outputs["category"],
                category
            )

            loss = (
                loss_presence
                + loss_pattern
                + loss_category
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress.set_description(
                f"Epoch {epoch + 1}/{EPOCHS}"
            )

        train_loss = total_loss / len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(DEVICE)

                detected = (
                    batch["detected"]
                    .float()
                    .unsqueeze(1)
                    .to(DEVICE)
                )

                pattern = torch.tensor(
                    [
                        pattern_to_idx[p]
                        for p in batch["structural_pattern"]
                    ],
                    dtype=torch.long
                ).to(DEVICE)

                category = torch.tensor(
                    [
                        category_to_idx[c]
                        for c in batch["category"]
                    ],
                    dtype=torch.long
                ).to(DEVICE)

                outputs = model(images)

                loss_presence = presence_loss(
                    outputs["presence"],
                    detected
                )

                loss_pattern = pattern_loss(
                    outputs["pattern"],
                    pattern
                )

                loss_category = category_loss(
                    outputs["category"],
                    category
                )

                loss = (
                    loss_presence
                    + loss_pattern
                    + loss_category
                )

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "pattern_to_idx": pattern_to_idx,
                "category_to_idx": category_to_idx
            }

            torch.save(
                checkpoint,
                "models/detection/model_weights.pt"
            )

            print("Best model saved!")


if __name__ == "__main__":
    train()
