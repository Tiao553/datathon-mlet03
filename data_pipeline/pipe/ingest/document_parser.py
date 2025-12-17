import io
import logging
from typing import Optional, Protocol

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

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

logger = logging.getLogger(__name__)

class OCRProvider(Protocol):
    def extract_text(self, images: list) -> str:
        ...

class TesseractAdapter:
    def extract_text(self, images: list) -> str:
        if not pytesseract:
            logger.warning("Tesseract not installed.")
            return ""
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text

class PaddleOCRAdapter:
    def __init__(self, lang='pt'):
        if not PaddleOCR:
            raise ImportError("PaddleOCR not installed")
        # Initialize only once ideally, but for now per-adapter is okay as it singleton-izes internally usually
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract_text(self, images: list) -> str:
        text_results = []
        for i, img in enumerate(images):
            # PaddleOCR expects path or numpy array. PDF2Image returns PIL images.
            import numpy as np
            img_np = np.array(img)
            
            result = self.ocr.ocr(img_np, cls=True)
            if result and result[0]:
                # result structure: [[[[x,y],..], (text, conf)], ...]
                page_text = "\n".join([line[1][0] for line in result[0]])
                text_results.append(page_text)
        return "\n".join(text_results)

class DocumentParser:
    @staticmethod
    def get_ocr_provider() -> OCRProvider:
        """
        Factory to get the best available OCR provider.
        Prioritizes PaddleOCR, falls back to Tesseract.
        """
        if PaddleOCR:
            logger.info("Using PaddleOCR provider")
            return PaddleOCRAdapter()
        elif pytesseract:
            logger.info("Using Tesseract provider")
            return TesseractAdapter()
        else:
            logger.warning("No OCR provider available")
            return None

    @staticmethod
    def extract_text_from_pdf_bytes(file_bytes: bytes, force_ocr: bool = False) -> str:
        """
        Extracts text from PDF bytes. 
        If force_ocr is True or pypdf extraction is minimal, uses OCR.
        """
        if not PdfReader:
            raise ImportError("pypdf not installed.")
            
        text = ""
        if not force_ocr:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                logger.error(f"Error reading PDF with pypdf: {e}")
            
        # Decision logic for OCR
        should_ocr = force_ocr or len(text.strip()) < 50
        
        if should_ocr:
            if not convert_from_bytes:
                logger.warning("pdf2image not installed, cannot convert PDF to images for OCR.")
                return text
                
            logger.info(f"Triggering OCR (Force={force_ocr}, TextLen={len(text.strip())})")
            
            try:
                images = convert_from_bytes(file_bytes)
                provider = DocumentParser.get_ocr_provider()
                if provider:
                    ocr_text = provider.extract_text(images)
                    # If we forced OCR, we return OCR text. 
                    # If it was fallback, we prefer OCR if it found something, else keep original (maybe empty)
                    if ocr_text.strip():
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
    def parse_file(file_bytes: bytes, filename: str, use_ocr: bool = False) -> str:
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return DocumentParser.extract_text_from_pdf_bytes(file_bytes, force_ocr=use_ocr)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            return DocumentParser.extract_text_from_docx_bytes(file_bytes)
        elif filename.endswith(".txt"):
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {filename}")
