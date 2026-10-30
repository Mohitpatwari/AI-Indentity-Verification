"""Verification endpoint — orchestrates the full pipeline."""
import logging
from fastapi import APIRouter, UploadFile, File
from app.utils.image_utils import decode_image
from app.services.ocr import extract_text
from app.services.classification import classify_document
from app.services.face import verify_faces
from app.services.extraction import extract_fields
from app.services.forgery import check_forgery
from app.services.report import generate_report
from sqlalchemy.orm import Session
from fastapi import Depends
from app.db.database import get_db
from app.db.models import VerificationRecord

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/verify")
async def verify(id_document: UploadFile = File(...), selfie: UploadFile = File(...),
    db: Session = Depends(get_db)):
    """
    Full identity verification pipeline:
    OCR -> classification -> extraction (stub) -> face verification ->
    forgery check (stub) -> report generation.
    """
    id_bytes = await id_document.read()
    selfie_bytes = await selfie.read()

    id_img = decode_image(id_bytes)
    selfie_img = decode_image(selfie_bytes)

    if id_img is None or selfie_img is None:
        return {"error": "Could not decode one or both images. Please upload JPG/PNG files."}

    try:
        ocr_result = extract_text(id_img)
        classification_result = classify_document(ocr_result["full_text"])
        extraction_result = extract_fields(ocr_result["full_text"])
        face_result = verify_faces(id_img, selfie_img)
        forgery_result = check_forgery(id_img)
        report = generate_report(classification_result, ocr_result, face_result, forgery_result)
        overall_status = "VERIFIED" if face_result.get("match") and classification_result["confidence"] > 0.3 else "REVIEW REQUIRED"

        record = VerificationRecord(
            document_type=classification_result["document_type"],
            classification_confidence=classification_result["confidence"],
            face_match=face_result.get("match"),
            face_similarity=face_result.get("similarity"),
            forgery_risk_score=forgery_result.get("risk_score"),
            forgery_suspicious=forgery_result.get("suspicious"),
            overall_status=overall_status,
            report_text=report,
        )
        db.add(record)
        db.commit()

        return {
            "classification": classification_result,
            "ocr": ocr_result,
            "extraction": extraction_result,
            "face_verification": face_result,
            "forgery_check": forgery_result,
            "report": report,
        }
    except Exception as e:
        logger.exception("Pipeline failed")
        return {"error": f"Verification pipeline failed: {str(e)}"}