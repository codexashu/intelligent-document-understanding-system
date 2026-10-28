"""
Synthetic & Lightweight OCR engine for standalone execution, mock testing,
and offline development environments.
"""

from typing import Any, List
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from idu_core.ocr.base import BaseOCREngine, DocumentPage, WordToken, BoundingBox

class SyntheticOCREngine(BaseOCREngine):
    """
    A lightweight OCR simulator that generates structured tokens.
    Can be used for mock testing, unit tests, or demo pipelines without external binaries.
    """
    def __init__(self, predefined_tokens: List[WordToken] = None):
        self.predefined_tokens = predefined_tokens

    def extract_page(self, image: Any, page_num: int = 1) -> DocumentPage:
        w, h = 800, 1000
        if HAS_PIL and Image and isinstance(image, Image.Image):
            w, h = image.size
        elif hasattr(image, "size"):
            w, h = image.size
        elif hasattr(image, "shape"):
            h, w = image.shape[:2]
                
        if self.predefined_tokens:
            tokens = []
            for t in self.predefined_tokens:
                norm_box = t.box.normalize(w, h)
                tokens.append(WordToken(
                    text=t.text,
                    box=t.box,
                    normalized_box=norm_box,
                    confidence=t.confidence,
                    line_num=t.line_num,
                    block_num=t.block_num
                ))
            raw_text = " ".join([t.text for t in tokens])
            return DocumentPage(
                page_num=page_num,
                image=image,
                width=w,
                height=h,
                tokens=tokens,
                raw_text=raw_text
            )

        # Standard tokens representing an invoice/form layout
        mock_data = [
            ("INVOICE", 50, 40, 180, 80, 0, 0),
            ("INVOICE", 50, 110, 120, 130, 1, 0),
            ("NUMBER:", 125, 110, 200, 130, 1, 0),
            ("INV-2026-8891", 210, 110, 360, 130, 1, 0),
            ("DATE:", 50, 140, 100, 160, 2, 0),
            ("2026-10-28", 110, 140, 220, 160, 2, 0),
            ("VENDOR:", 50, 190, 130, 210, 3, 0),
            ("Acme", 140, 190, 190, 210, 3, 0),
            ("Solutions", 195, 190, 280, 210, 3, 0),
            ("Inc.", 285, 190, 320, 210, 3, 0),
            ("CUSTOMER:", 50, 240, 150, 260, 4, 0),
            ("TechCorp", 160, 240, 250, 260, 4, 0),
            ("Global", 255, 240, 320, 260, 4, 0),
            ("DESCRIPTION", 50, 310, 170, 330, 5, 1),
            ("QTY", 400, 310, 440, 330, 5, 1),
            ("PRICE", 500, 310, 550, 330, 5, 1),
            ("AMOUNT", 650, 310, 720, 330, 5, 1),
            ("Cloud", 50, 350, 105, 370, 6, 1),
            ("Server", 110, 350, 175, 370, 6, 1),
            ("Hosting", 180, 350, 250, 370, 6, 1),
            ("2", 415, 350, 425, 370, 6, 1),
            ("$500.00", 495, 350, 560, 370, 6, 1),
            ("$1000.00", 645, 350, 725, 370, 6, 1),
            ("TOTAL", 500, 440, 560, 465, 7, 2),
            ("AMOUNT:", 565, 440, 640, 465, 7, 2),
            ("$1,000.00", 650, 440, 740, 465, 7, 2)
        ]
        
        tokens = []
        words = []
        for text, x0, y0, x1, y1, line_num, block_num in mock_data:
            box = BoundingBox(x0, y0, x1, y1)
            norm_box = box.normalize(w, h)
            tokens.append(WordToken(
                text=text,
                box=box,
                normalized_box=norm_box,
                confidence=0.98,
                line_num=line_num,
                block_num=block_num
            ))
            words.append(text)
            
        return DocumentPage(
            page_num=page_num,
            image=image,
            width=w,
            height=h,
            tokens=tokens,
            raw_text=" ".join(words)
        )
