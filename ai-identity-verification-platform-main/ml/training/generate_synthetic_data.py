"""
Generates synthetic authentic/tampered image pairs for forgery detection training.

Authentic: base images with natural, uniform augmentation (compression, brightness).
Tampered: same base images with a localized synthetic edit (pasted patch, blended
region) simulating document tampering (e.g. altered number, swapped photo).

This is a standard bootstrapping technique used when real labeled forgery data
isn't available (which is virtually always true for identity documents, for
obvious privacy/legal reasons).
"""
import cv2
import numpy as np
import os
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
NUM_SAMPLES_PER_BASE = 20  # augmented variants generated per base image


def load_base_images(source_dir: str) -> list[np.ndarray]:
    """Load any images available to use as base templates."""
    images = []
    for f in Path(source_dir).glob("*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    return images


def generate_random_base_images(count: int) -> list[np.ndarray]:
    """
    Fallback: generate synthetic 'document-like' base images if no real
    samples are available — random colored rectangles with text-like noise
    patterns, simulating document structure for training purposes.
    """
    images = []
    for _ in range(count):
        img = np.full((IMG_SIZE, IMG_SIZE, 3), random.randint(200, 255), dtype=np.uint8)
        # Simulate text lines with dark rectangles
        for _ in range(random.randint(5, 10)):
            y = random.randint(0, IMG_SIZE - 20)
            x = random.randint(0, IMG_SIZE - 100)
            w = random.randint(50, 150)
            h = random.randint(8, 15)
            color = (random.randint(0, 80),) * 3
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        images.append(img)
    return images


def apply_authentic_augmentation(img: np.ndarray) -> np.ndarray:
    """Uniform, natural-looking augmentation — no localized editing."""
    quality = random.randint(80, 95)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    brightness = random.uniform(0.9, 1.1)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    angle = random.uniform(-3, 3)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, matrix, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REPLICATE)

    return img


def apply_tampering(img: np.ndarray) -> np.ndarray:
    """
    Simulates localized document tampering: a rectangular patch is cut,
    recompressed at a DIFFERENT quality than the rest of the image, and
    pasted back — mimicking a photoshopped edit (e.g. altered ID number).
    """
    img = img.copy()

    patch_w = random.randint(40, 100)
    patch_h = random.randint(20, 50)
    x = random.randint(0, IMG_SIZE - patch_w)
    y = random.randint(0, IMG_SIZE - patch_h)

    patch = img[y:y + patch_h, x:x + patch_w].copy()

    # Recompress the patch at a very different quality — this creates the
    # ELA-detectable inconsistency, and also apply a slight color/contrast
    # shift to simulate a mismatched paste source.
    tamper_quality = random.randint(30, 60)
    _, encoded = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, tamper_quality])
    patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    patch = np.clip(patch.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)

    img[y:y + patch_h, x:x + patch_w] = patch
    return img


def main():
    source_dir = "ml/training/source_images"
    authentic_out = "ml/training/data/authentic"
    tampered_out = "ml/training/data/tampered"

    base_images = []
    if os.path.exists(source_dir):
        base_images = load_base_images(source_dir)
        logger.info(f"Loaded {len(base_images)} real base images from {source_dir}")

    if not base_images:
        logger.info("No source images found — generating synthetic base templates instead.")
        base_images = generate_random_base_images(30)

    count = 0
    for base_idx, base in enumerate(base_images):
        for i in range(NUM_SAMPLES_PER_BASE):
            authentic = apply_authentic_augmentation(base)
            cv2.imwrite(f"{authentic_out}/auth_{base_idx}_{i}.jpg", authentic)

            tampered = apply_tampering(apply_authentic_augmentation(base))
            cv2.imwrite(f"{tampered_out}/tamp_{base_idx}_{i}.jpg", tampered)

            count += 2

    logger.info(f"Generated {count} total images ({count // 2} authentic, {count // 2} tampered)")


if __name__ == "__main__":
    main()