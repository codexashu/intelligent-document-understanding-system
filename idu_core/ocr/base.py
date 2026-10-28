"""
Base OCR abstractions and geometric bounding box primitives.
Provides normalized 0-1000 coordinate mapping standard for Multimodal Document Transformers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import time

@dataclass
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int
    
    def __post_init__(self):
        # Ensure valid coordinate bounds
        if self.x1 < self.x0:
            self.x0, self.x1 = self.x1, self.x0
        if self.y1 < self.y0:
            self.y0, self.y1 = self.y1, self.y0
            
    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)
        
    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)
        
    @property
    def area(self) -> int:
        return self.width * self.height
        
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
        
    def to_list(self) -> List[int]:
        return [self.x0, self.y0, self.x1, self.y1]
        
    def to_dict(self) -> Dict[str, int]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height
        }
        
    def normalize(self, img_width: int, img_height: int, scale: int = 1000) -> "BoundingBox":
        """Normalizes pixel coordinates to [0, scale] for transformer architectures."""
        if img_width <= 0 or img_height <= 0:
            return BoundingBox(0, 0, 0, 0)
        nx0 = int(min(max(0, (self.x0 / img_width) * scale), scale))
        ny0 = int(min(max(0, (self.y0 / img_height) * scale), scale))
        nx1 = int(min(max(0, (self.x1 / img_width) * scale), scale))
        ny1 = int(min(max(0, (self.y1 / img_height) * scale), scale))
        return BoundingBox(nx0, ny0, nx1, ny1)

    def denormalize(self, img_width: int, img_height: int, scale: int = 1000) -> "BoundingBox":
        """Converts normalized [0, scale] coordinates back to original pixel dimensions."""
        px0 = int((self.x0 / scale) * img_width)
        py0 = int((self.y0 / scale) * img_height)
        px1 = int((self.x1 / scale) * img_width)
        py1 = int((self.y1 / scale) * img_height)
        return BoundingBox(px0, py0, px1, py1)

    def iou(self, other: "BoundingBox") -> float:
        """Computes Intersection-over-Union (IoU) with another box."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        intersection = iw * ih
        
        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

@dataclass
class WordToken:
    text: str
    box: BoundingBox
    normalized_box: BoundingBox
    confidence: float = 1.0
    line_num: int = 0
    block_num: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "box": self.box.to_dict(),
            "normalized_box": self.normalized_box.to_dict(),
            "confidence": round(self.confidence, 4),
            "line_num": self.line_num,
            "block_num": self.block_num
        }

@dataclass
class DocumentPage:
    page_num: int
    image: Any  # PIL Image or numpy array
    width: int
    height: int
    tokens: List[WordToken] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "token_count": len(self.tokens),
            "tokens": [t.to_dict() for t in self.tokens],
            "raw_text": self.raw_text
        }

@dataclass
class OCRResult:
    pages: List[DocumentPage] = field(default_factory=list)
    total_tokens: int = 0
    execution_time: float = 0.0
    
    def get_full_text(self) -> str:
        return "\n\n--- Page Break ---\n\n".join(p.raw_text for p in self.pages)

class BaseOCREngine(ABC):
    """Abstract interface for Optical Character Recognition engines."""
    
    @abstractmethod
    def extract_page(self, image: Any, page_num: int = 1) -> DocumentPage:
        """Extract text tokens and bounding boxes from a single document image."""
        pass
        
    def extract_document(self, images: List[Any]) -> OCRResult:
        """Processes multiple document pages and compiles complete OCR results."""
        start_time = time.time()
        pages = []
        total_tokens = 0
        
        for idx, img in enumerate(images, start=1):
            page_data = self.extract_page(img, page_num=idx)
            pages.append(page_data)
            total_tokens += len(page_data.tokens)
            
        elapsed = time.time() - start_time
        return OCRResult(pages=pages, total_tokens=total_tokens, execution_time=elapsed)
