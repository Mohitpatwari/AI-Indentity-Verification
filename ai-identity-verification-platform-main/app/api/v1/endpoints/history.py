"""Endpoint for retrieving past verification records."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import VerificationRecord

router = APIRouter()


@router.get("/history")
async def get_history(limit: int = 20, db: Session = Depends(get_db)):
    """Return the most recent verification records, newest first."""
    records = (
        db.query(VerificationRecord)
        .order_by(VerificationRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "document_type": r.document_type,
            "classification_confidence": r.classification_confidence,
            "face_match": r.face_match,
            "face_similarity": r.face_similarity,
            "forgery_risk_score": r.forgery_risk_score,
            "overall_status": r.overall_status,
        }
        for r in records
    ]