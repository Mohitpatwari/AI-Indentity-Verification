"""OCR service using PaddleOCR (pretrained)."""
import logging
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Loaded once at import time — explicitly disabling mkldnn to avoid Windows CPU crash
_ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", enable_mkldnn=False)


def extract_text(image_bgr: np.ndarray) -> dict:
    """Run OCR and return full text plus per-line results with confidence scores."""
    try:
        # PaddleOCR 3.7.0 uses the predict() pipeline API
        result = _ocr_engine.predict(image_bgr)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {"full_text": "", "lines": []}

    lines = []
    try:
        for res in result:
            # PaddleOCR 3.7.0 result iterator format
            if hasattr(res, "text") and hasattr(res, "score"):
                text = res.text
                confidence = float(res.score)
                lines.append({"text": text, "confidence": round(confidence, 3)})
            elif isinstance(res, dict) and "rec_texts" in res:
                for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
                    lines.append({"text": text, "confidence": round(float(score), 3)})
            elif isinstance(res, (list, tuple)) and len(res) >= 2:
                # Fallback for legacy nested formats
                for line in res:
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        text, confidence = line[1][0], line[1][1]
                        lines.append({"text": text, "confidence": round(float(confidence), 3)})
    except Exception as parse_err:
        logger.error(f"Error parsing OCR results structure: {parse_err}")

    return {"full_text": " ".join(l["text"] for l in lines), "lines": lines}