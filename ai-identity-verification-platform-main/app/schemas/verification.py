"""Pydantic schemas for the verification API — enables request validation
and auto-generated OpenAPI docs at /docs."""
from pydantic import BaseModel


class ClassificationResult(BaseModel):
    document_type: str
    confidence: float


class OCRLine(BaseModel):
    text: str
    confidence: float


class OCRResult(BaseModel):
    full_text: str
    lines: list[OCRLine]


class FaceVerificationResult(BaseModel):
    match: bool
    similarity: float
    error: str | None = None


class VerificationResponse(BaseModel):
    classification: ClassificationResult
    ocr: OCRResult
    face_verification: FaceVerificationResult
    extraction: dict  # stub payload today, structured schema tomorrow
    forgery_check: dict  # stub payload today
    report: str


class ErrorResponse(BaseModel):
    error: str