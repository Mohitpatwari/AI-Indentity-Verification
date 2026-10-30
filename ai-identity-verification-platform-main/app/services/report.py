"""Generates a human-readable verification report using Gemini (LLM),
with a template-based fallback if the API call fails."""
import logging
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)


def _template_report(classification: dict, ocr: dict, face_result: dict, forgery_result: dict) -> str:
    doc_type = classification["document_type"]
    doc_conf = classification["confidence"]
    face_status = "MATCH" if face_result.get("match") else "NO MATCH"
    similarity = face_result.get("similarity", 0.0)
    forgery_score = forgery_result.get("risk_score", "N/A")
    overall = "VERIFIED" if face_result.get("match") and doc_conf > 0.3 else "REVIEW REQUIRED"

    return f"""IDENTITY VERIFICATION REPORT
{'=' * 40}
Document Type: {doc_type} (confidence: {doc_conf})
Extracted Text Preview: {ocr['full_text'][:150]}...

Face Verification: {face_status}
Similarity Score: {similarity}

Forgery Risk Score: {forgery_score}

Overall Status: {overall}
"""


def generate_report(classification: dict, ocr: dict, face_result: dict, forgery_result: dict) -> str:
    if not settings.gemini_api_key:
        logger.warning("No Gemini API key configured — using template report.")
        return _template_report(classification, ocr, face_result, forgery_result)

    prompt = f"""You are an identity verification analyst. Write a concise, professional
verification report (under 150 words) based on this structured data. Be factual
and note any concerns clearly. Do not invent information not present below.

Document type detected: {classification['document_type']} (confidence: {classification['confidence']})
OCR extracted text: {ocr['full_text'][:500]}
Face verification match: {face_result.get('match')}
Face similarity score: {face_result.get('similarity')}
Forgery risk score (0=authentic, 1=likely tampered): {forgery_result.get('risk_score')}

Write the report now."""

    try:
        # Updated to the current active flash model
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini report generation failed, falling back to template: {e}")
        return _template_report(classification, ocr, face_result, forgery_result)