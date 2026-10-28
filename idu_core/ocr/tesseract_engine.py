"""
Tesseract OCR Engine wrapper with 0-1000 coordinate normalization.
Extracts token words, confidence scores, line numbers, and block structures.
"""

from typing import Any, List
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from idu_core.ocr.base import BaseOCREngine, DocumentPage, WordToken, BoundingBox

class TesseractOCREngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: str = None, lang: str = "eng", psm: int = 3):
        self.lang = lang
        self.psm = psm
        try:
            import pytesseract
            self.pytesseract = pytesseract
            if tesseract_cmd:
                self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            self.available = True
        except ImportError:
            self.pytesseract = None
            self.available = False

    def extract_page(self, image: Any, page_num: int = 1) -> DocumentPage:
        if not self.available or self.pytesseract is None:
            raise RuntimeError("pytesseract is not installed or not configured in system.")
            
        w, h = 800, 1000
        if HAS_PIL and Image and isinstance(image, Image.Image):
            w, h = image.size
        elif hasattr(image, "size"):
            w, h = image.size
        elif hasattr(image, "shape"):
            h, w = image.shape[:2]
                
        custom_config = f"--oem 3 --psm {self.psm}"
        
        try:
            data = self.pytesseract.image_to_data(
                image, lang=self.lang, config=custom_config, output_type=self.pytesseract.Output.DICT
            )
        except Exception as e:
            raise RuntimeError(f"Tesseract execution failed: {str(e)}")
            
        tokens: List[WordToken] = []
        full_text_parts = []
        n_boxes = len(data["text"])
        
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 0:
                x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                box = BoundingBox(x0=x, y0=y, x1=x + bw, y1=y + bh)
                norm_box = box.normalize(img_width=w, img_height=h)
                token = WordToken(
                    text=text,
                    box=box,
                    normalized_box=norm_box,
                    confidence=conf / 100.0,
                    line_num=data["line_num"][i],
                    block_num=data["block_num"][i]
                )
                tokens.append(token)
                full_text_parts.append(text)
                
        return DocumentPage(
            page_num=page_num,
            image=image,
            width=w,
            height=h,
            tokens=tokens,
            raw_text=" ".join(full_text_parts)
        )
