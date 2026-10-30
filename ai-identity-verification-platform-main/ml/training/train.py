"""
Trains the forgery detection CNN on the synthetic authentic/tampered dataset.

Run: python ml/training/train.py
Output: ml/training/checkpoints/forgery_cnn.pt
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from pathlib import Path
import logging

from model import ForgeryDetectionCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForgeryDataset(Dataset):
    """Loads authentic (label=0) and tampered (label=1) images."""

    def __init__(self, authentic_dir: str, tampered_dir: str):
        self.samples = []
        for f in Path(authentic_dir).glob("*.jpg"):
            self.samples.append((str(f), 0))
        for f in Path(tampered_dir).glob("*.jpg"):
            self.samples.append((str(f), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to CHW tensor format PyTorch expects
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor, torch.tensor([label], dtype=torch.float32)


def train():
    logger.info(f"Using device: {DEVICE}")

    dataset = ForgeryDataset("ml/training/data/authentic", "ml/training/data/tampered")
    logger.info(f"Total dataset size: {len(dataset)}")

    # 80/20 train/validation split — standard practice to catch overfitting
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ForgeryDetectionCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + binary cross-entropy, numerically stable
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # Validation pass
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("ml/training/checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "ml/training/checkpoints/forgery_cnn.pt")
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()