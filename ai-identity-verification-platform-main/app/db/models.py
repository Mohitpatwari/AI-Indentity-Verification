"""SQLAlchemy ORM models — defines the database schema."""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from datetime import datetime, timezone
from app.db.database import Base


class VerificationRecord(Base):
    """
    Stores a summary of each verification request. Deliberately does NOT
    store the uploaded images themselves — only derived metadata — to
    avoid retaining sensitive identity document images long-term.
    """
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document_type = Column(String, nullable=True)
    classification_confidence = Column(Float, nullable=True)

    face_match = Column(Boolean, nullable=True)
    face_similarity = Column(Float, nullable=True)

    forgery_risk_score = Column(Float, nullable=True)
    forgery_suspicious = Column(Boolean, nullable=True)

    overall_status = Column(String, nullable=True)
    report_text = Column(Text, nullable=True)