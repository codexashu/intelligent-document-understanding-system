"""
OCR module entry point and factory.
"""

import logging
from idu_core.ocr.base import BaseOCREngine, BoundingBox, WordToken, DocumentPage, OCRResult
from idu_core.ocr.tesseract_engine import TesseractOCREngine
from idu_core.ocr.easyocr_engine import EasyOCREngine
from idu_core.ocr.synthetic_ocr import SyntheticOCREngine

logger = logging.getLogger(__name__)

def get_ocr_engine(engine_type: str = "auto", **kwargs) -> BaseOCREngine:
    """
    Factory function to retrieve an appropriate OCR engine.
    Options: 'tesseract', 'easyocr', 'synthetic', or 'auto'.
    """
    engine_type = engine_type.lower()
    
    if engine_type == "tesseract":
        return TesseractOCREngine(**kwargs)
        
    if engine_type == "easyocr":
        return EasyOCREngine(**kwargs)
        
    if engine_type == "synthetic":
        return SyntheticOCREngine(**kwargs)
        
    # Auto selection: try tesseract -> easyocr -> synthetic
    try:
        tess = TesseractOCREngine(**kwargs)
        if tess.available:
            return tess
    except Exception as e:
        logger.debug(f"Tesseract unavailable: {e}")
        
    try:
        easy = EasyOCREngine(**kwargs)
        if easy.available:
            return easy
    except Exception as e:
        logger.debug(f"EasyOCR unavailable: {e}")
        
    logger.info("Using Synthetic/Lightweight OCR engine as default fallback.")
    return SyntheticOCREngine(**kwargs)

__all__ = [
    "BaseOCREngine",
    "BoundingBox",
    "WordToken",
    "DocumentPage",
    "OCRResult",
    "TesseractOCREngine",
    "EasyOCREngine",
    "SyntheticOCREngine",
    "get_ocr_engine"
]
