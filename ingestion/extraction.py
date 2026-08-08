"""
Extracts clean text from a PDF file, page by page.

Primary path: pdfplumber, which preserves layout well enough for prose
documents like policy handbooks and interview guides.

Fallback: if a page yields no extractable text (e.g. it's a scanned image
rather than real text), we OCR that page with pytesseract. This mirrors
the OCR-fallback approach already used in the Pakistan Rights Assistant
project, so scanned company policy PDFs (a very real possibility for HR
documents) are still ingestible.
"""
import pdfplumber
from pdf2image import convert_from_path
import pytesseract


def extract_text_by_page(pdf_path: str) -> list[str]:
    """Returns a list of strings, one per page, in order."""
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
            else:
                # Empty page from pdfplumber usually means it's a scanned
                # image with no text layer -- fall back to OCR just for
                # this page rather than failing the whole document.
                ocr_text = _ocr_single_page(pdf_path, i)
                pages_text.append(ocr_text.strip())

    return pages_text


def _ocr_single_page(pdf_path: str, page_index: int) -> str:
    images = convert_from_path(
        pdf_path, first_page=page_index + 1, last_page=page_index + 1
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def extract_full_text(pdf_path: str) -> str:
    """Convenience wrapper: joins all pages into one string with page breaks marked,
    which the chunker uses as soft split hints."""
    pages = extract_text_by_page(pdf_path)
    return "\n\n".join(pages)