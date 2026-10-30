"""
Forgery detection CNN — a small convolutional network for binary
classification: authentic vs. tampered document image.

Architecture rationale: this is intentionally small (4 conv blocks).
Forgery detection here relies on local pixel-level inconsistencies
(compression artifacts, blending edges) rather than complex semantic
understanding, so a lightweight CNN is appropriate — a large pretrained
backbone (ResNet50+) would be overkill and slower for this task, and
risks overfitting on our synthetic dataset size.
"""
import torch
import torch.nn as nn


class ForgeryDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 256x256x3 -> 128x128x16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 128x128x16 -> 64x64x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), # global average pool -> 128x1x1, robust to input size variation
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),  # single logit: sigmoid gives P(tampered)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits — apply sigmoid at inference/loss time