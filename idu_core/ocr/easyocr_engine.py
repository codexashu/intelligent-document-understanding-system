"""
EasyOCR Engine implementation for deep-learning-based scene/document OCR.
"""

from typing import Any, List
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    np = None

from idu_core.ocr.base import BaseOCREngine, DocumentPage, WordToken, BoundingBox

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None
        try:
            import easyocr
            self.easyocr = easyocr
            self.available = True
        except ImportError:
            self.easyocr = None
            self.available = False

    def _get_reader(self):
        if self.reader is None and self.available:
            self.reader = self.easyocr.Reader(self.languages, gpu=self.gpu)
        return self.reader

    def extract_page(self, image: Any, page_num: int = 1) -> DocumentPage:
        if not self.available:
            raise RuntimeError("easyocr package is not installed.")
            
        reader = self._get_reader()
        w, h = 800, 1000
        if HAS_PIL and Image and isinstance(image, Image.Image):
            w, h = image.size
            img_np = np.array(image.convert("RGB")) if HAS_NP else image
        elif hasattr(image, "size"):
            w, h = image.size
            img_np = image
        elif hasattr(image, "shape"):
            h, w = image.shape[:2]
            img_np = image
        else:
            img_np = image

        results = reader.readtext(img_np)
        
        tokens: List[WordToken] = []
        full_text_parts = []
        
        for idx, (coord, text, conf) in enumerate(results):
            text = text.strip()
            if not text:
                continue
                
            xs = [pt[0] for pt in coord]
            ys = [pt[1] for pt in coord]
            box = BoundingBox(
                x0=int(min(xs)),
                y0=int(min(ys)),
                x1=int(max(xs)),
                y1=int(max(ys))
            )
            norm_box = box.normalize(img_width=w, img_height=h)
            
            words = text.split()
            if len(words) <= 1:
                tokens.append(WordToken(
                    text=text,
                    box=box,
                    normalized_box=norm_box,
                    confidence=float(conf),
                    line_num=idx,
                    block_num=0
                ))
            else:
                total_chars = max(1, sum(len(wrd) for wrd in words))
                cur_x = box.x0
                for wrd in words:
                    wrd_w = int((len(wrd) / total_chars) * box.width)
                    w_box = BoundingBox(x0=cur_x, y0=box.y0, x1=min(cur_x + wrd_w, box.x1), y1=box.y1)
                    w_norm = w_box.normalize(img_width=w, img_height=h)
                    tokens.append(WordToken(
                        text=wrd,
                        box=w_box,
                        normalized_box=w_norm,
                        confidence=float(conf),
                        line_num=idx,
                        block_num=0
                    ))
                    cur_x += wrd_w
                    
            full_text_parts.append(text)
            
        return DocumentPage(
            page_num=page_num,
            image=image,
            width=w,
            height=h,
            tokens=tokens,
            raw_text=" ".join(full_text_parts)
        )
