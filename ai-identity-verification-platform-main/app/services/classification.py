"""
Document type classification — MVP uses keyword matching on OCR text.

TODO (post-MVP): replace with a fine-tuned image classifier (ResNet/ViT)
trained on labeled document images for higher accuracy and to work even
when OCR quality is poor.
"""

DOC_KEYWORDS = {
    "AADHAAR": ["aadhaar", "unique identification", "government of india", "uidai"],
    "PAN": ["income tax department", "permanent account number", "pan"],
    "PASSPORT": ["passport", "republic of india", "type p"],
}


def classify_document(ocr_text: str) -> dict:
    text_lower = ocr_text.lower()
    scores = {
        doc_type: sum(1 for kw in kws if kw in text_lower) / len(kws)
        for doc_type, kws in DOC_KEYWORDS.items()
    }

    best_type = max(scores, key=scores.get)
    confidence = scores[best_type]

    if confidence == 0:
        return {"document_type": "UNKNOWN", "confidence": 0.0}
    return {"document_type": best_type, "confidence": round(confidence, 2)}