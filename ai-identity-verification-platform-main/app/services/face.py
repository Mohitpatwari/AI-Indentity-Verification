"""Face extraction and verification using InsightFace (pretrained ArcFace)."""


import sys
import types

# Stub out insightface's optional mask_renderer submodule to avoid an
# unnecessary torch/albumentations import chain that's unstable on this
# Windows environment. We don't use mask rendering — only face detection
# and embedding extraction — so this is safe to skip.
_stub = types.ModuleType("insightface.app.mask_renderer")
_stub.MaskRenderer = None
sys.modules["insightface.app.mask_renderer"] = _stub


import logging
import numpy as np
from numpy.linalg import norm
import insightface
from app.core.config import settings

logger = logging.getLogger(__name__)

_face_app = insightface.app.FaceAnalysis(name="buffalo_sc")
_face_app.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 = CPU; use 0 for GPU


def get_face_embedding(image_bgr: np.ndarray) -> np.ndarray | None:
    """Detect the largest face and return its 512-d ArcFace embedding."""
    try:
        faces = _face_app.get(image_bgr)
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return None

    if not faces:
        return None

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding


def verify_faces(id_image_bgr: np.ndarray, selfie_image_bgr: np.ndarray) -> dict:
    """Compare face in ID document against a selfie via cosine similarity."""
    emb1 = get_face_embedding(id_image_bgr)
    emb2 = get_face_embedding(selfie_image_bgr)

    if emb1 is None or emb2 is None:
        return {"match": False, "similarity": 0.0, "error": "Face not detected in one or both images"}

    similarity = float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
    is_match = similarity > settings.face_match_threshold

    return {"match": is_match, "similarity": round(similarity, 4), "error": None}