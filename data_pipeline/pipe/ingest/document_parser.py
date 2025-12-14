import io
import logging
from typing import Optional

# Optional imports - fallback if not installed
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:
    pytesseract = None
    convert_from_bytes = None

logger = logging.getLogger(__name__)

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
        """
        Extracts text from PDF bytes. Tries pypdf first (text layer), 
        then fallback to OCR (tesseract) if text is minimal/empty.
        """
        if not PdfReader:
            raise ImportError("pypdf not installed.")
            
        text = ""
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF with pypdf: {e}")
            
        # OCR Fallback Check
        if len(text.strip()) < 50 and pytesseract and convert_from_bytes:
            logger.info("Minimal text found. Attempting OCR...")
            try:
                images = convert_from_bytes(file_bytes)
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img) + "\n"
                return ocr_text
            except Exception as e:
                logger.error(f"OCR failed: {e}")
        
        return text

    @staticmethod
    def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
        if not docx:
            raise ImportError("python-docx not installed.")
            
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return ""

    @staticmethod
    def parse_file(file_bytes: bytes, filename: str) -> str:
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return DocumentParser.extract_text_from_pdf_bytes(file_bytes)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            return DocumentParser.extract_text_from_docx_bytes(file_bytes)
        elif filename.endswith(".txt"):
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {filename}")
