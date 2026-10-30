"""
Evaluates the trained forgery detection model on the validation set,
producing standard classification metrics.

Run: python ml/training/evaluate.py
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

from model import ForgeryDetectionCNN
from train import ForgeryDataset, IMG_SIZE  # reuse the same dataset class

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")


def evaluate():
    dataset = ForgeryDataset("data/authentic", "data/tampered")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = ForgeryDetectionCNN().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/forgery_cnn.pt", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(np.atleast_1d(probs))
            all_preds.extend(np.atleast_1d(preds))
            all_labels.extend(labels.squeeze().numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info("=" * 50)
    logger.info("FORGERY DETECTION MODEL — EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Dataset size: {len(all_labels)} images (authentic + tampered)")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}  (of predicted 'tampered', how many truly were)")
    logger.info(f"Recall:    {rec:.4f}  (of truly 'tampered', how many were caught)")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                 Predicted Authentic  Predicted Tampered")
    logger.info(f"Actual Authentic        {cm[0][0]:>10}          {cm[0][1]:>10}")
    logger.info(f"Actual Tampered         {cm[1][0]:>10}          {cm[1][1]:>10}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=["Authentic", "Tampered"]))

    # Save results to a file for the README / portfolio
    with open("../../docs/evaluation_report.txt", "w") as f:
        f.write(f"Forgery Detection Model Evaluation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=["Authentic", "Tampered"]))

    logger.info("\nSaved report to docs/evaluation_report.txt")


if __name__ == "__main__":
    evaluate()