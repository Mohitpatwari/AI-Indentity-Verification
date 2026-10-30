"""Extracts structured fields (name, DOB, ID numbers) from OCR text using regex."""
import re

AADHAAR_PATTERN = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
PASSPORT_PATTERN = re.compile(r"\b([A-Z]\d{7})\b")
DATE_PATTERN = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b")


def extract_fields(ocr_text: str) -> dict:
    fields = {}

    aadhaar_match = AADHAAR_PATTERN.search(ocr_text)
    if aadhaar_match:
        fields["aadhaar_number"] = aadhaar_match.group(1).replace(" ", "")

    pan_match = PAN_PATTERN.search(ocr_text.upper())
    if pan_match:
        fields["pan_number"] = pan_match.group(1)

    passport_match = PASSPORT_PATTERN.search(ocr_text.upper())
    if passport_match:
        fields["passport_number"] = passport_match.group(1)

    dates_found = DATE_PATTERN.findall(ocr_text)
    if dates_found:
        fields["dates_found"] = dates_found

    return {
        "status": "success" if fields else "no_fields_matched",
        "fields": fields,
    }