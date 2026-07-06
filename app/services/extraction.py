import io
import base64
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image
import docx
from google import genai
from google.genai import types
from app.core.config import settings

# Minimum characters on a page to consider text-based (below = image-only)
_MIN_TEXT_CHARS = 30


def _page_to_base64_png(pdf_bytes: bytes, page_index: int) -> str:
    """Render a single PDF page to a base64-encoded PNG using pypdfium2."""
    doc = pdfium.PdfDocument(pdf_bytes)
    page = doc[page_index]
    bitmap = page.render(scale=2.0)          # 2× scale → ~144 dpi, good for OCR
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ocr_page_with_gemini(pdf_bytes: bytes, page_index: int, page_num: int) -> str:
    """Send a rendered PDF page image to Gemini Vision and return extracted text."""
    try:
        b64 = _page_to_base64_png(pdf_bytes, page_index)
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=base64.b64decode(b64),
                    mime_type="image/png",
                ),
                (
                    f"This is page {page_num} of a PDF document. "
                    "Extract ALL text exactly as it appears, preserving structure "
                    "(paragraphs, bullet points, headings, tables). "
                    "Return only the extracted text, no commentary."
                ),
            ],
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"[OCR] Gemini Vision failed for page {page_num}: {e}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text from PDF. Falls back to Gemini Vision OCR for image-only pages."""
    pages_data = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()

            if len(text) < _MIN_TEXT_CHARS:
                # Image-only or scanned page — use Gemini Vision OCR
                print(f"[OCR] Page {i + 1} has <{_MIN_TEXT_CHARS} chars, using Vision OCR…")
                text = _ocr_page_with_gemini(file_bytes, i, i + 1)

            if text:
                pages_data.append({
                    "page_number": i + 1,
                    "text": text,
                })

    return pages_data


def extract_text_from_txt(file_bytes: bytes) -> list[dict]:
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [{"page_number": 1, "text": text}] if text else []


def extract_text_from_docx(file_bytes: bytes) -> list[dict]:
    """Extract text from DOCX, grouping paragraphs into synthetic pages."""
    doc = docx.Document(io.BytesIO(file_bytes))
    pages_data: list[dict] = []
    current_lines: list[str] = []
    current_len = 0
    page_num = 1
    PAGE_CHAR_LIMIT = 3000

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        current_lines.append(text)
        current_len += len(text)
        if current_len >= PAGE_CHAR_LIMIT:
            pages_data.append({
                "page_number": page_num,
                "text": "\n".join(current_lines),
            })
            page_num += 1
            current_lines = []
            current_len = 0

    if current_lines:
        pages_data.append({
            "page_number": page_num,
            "text": "\n".join(current_lines),
        })

    return pages_data
