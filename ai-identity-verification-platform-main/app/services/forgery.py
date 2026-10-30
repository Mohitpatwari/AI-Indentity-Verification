"""Forgery detection using a trained CNN, run via ONNX Runtime.

Note: we deliberately use onnxruntime here instead of raw PyTorch —
PaddlePaddle (used for OCR) and PyTorch load conflicting versions of
shared C++ runtime DLLs on Windows when loaded in the same process,
causing crashes. Exporting the trained model to ONNX and running it
through onnxruntime (already used for face verification) avoids this
entirely and is also standard practice for production inference.
"""
import onnxruntime as ort
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

IMG_SIZE = 256
MODEL_PATH = "app/ml_models/forgery_cnn.onnx"

_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def check_forgery(image_bgr: np.ndarray) -> dict:
    """Run the trained CNN (via ONNX Runtime) on an image and return a tampering probability."""
    try:
        img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # add batch dimension

        outputs = _session.run(None, {_input_name: img})
        logit = float(outputs[0][0][0])
        probability = _sigmoid(logit)

        return {
            "status": "success",
            "risk_score": round(float(probability), 4),
            "suspicious": bool(probability > 0.5),
            "note": "Score from trained CNN via ONNX Runtime (0=authentic, 1=likely tampered).",
        }
    except Exception as e:
        logger.error(f"Forgery detection failed: {e}")
        return {"status": "error", "note": str(e), "risk_score": None}