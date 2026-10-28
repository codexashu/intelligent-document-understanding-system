"""
Visual Bounding Box & Annotation Overlay Generator.
Produces human-interpretable visualizations of OCR tokens, entities, and tables.
"""

from typing import List, Dict, Any, Union, Optional
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageDraw = None
    ImageFont = None

import os

from idu_core.ocr.base import WordToken, BoundingBox
from idu_core.extraction.entity_extractor import ExtractedEntity
from idu_core.models.table_extractor import ExtractedTable

COLOR_PALETTE = {
    "HEADER": (147, 51, 234),        # Purple
    "QUESTION": (59, 130, 246),      # Blue
    "ANSWER": (16, 185, 129),        # Emerald Green
    "TOTAL_AMOUNT": (239, 68, 68),   # Red / Crimson
    "DATE": (14, 165, 233),          # Sky Blue
    "INVOICE_NUMBER": (245, 158, 11),# Amber / Orange
    "VENDOR_NAME": (16, 185, 129),   # Green
    "ADDRESS": (107, 114, 128),      # Gray
    "LINE_ITEM": (168, 85, 247),     # Violet
    "TABLE": (234, 179, 8),          # Yellow Gold
    "DEFAULT": (99, 102, 241)        # Indigo
}

class DocumentVisualizer:
    """
    Renders bounding boxes and entity tags onto document images.
    """
    def __init__(self):
        self.font = self._load_default_font()

    def _load_default_font(self):
        if not HAS_PIL or ImageFont is None:
            return None
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def get_color_for_label(self, label: str):
        clean_label = label.replace("B-", "").replace("I-", "").upper()
        return COLOR_PALETTE.get(clean_label, COLOR_PALETTE["DEFAULT"])

    def draw_entities(
        self,
        image: Any,
        entities: List[ExtractedEntity],
        tables: List[ExtractedTable] = None,
        draw_labels: bool = True
    ) -> Optional[Any]:
        if not HAS_PIL or Image is None or not isinstance(image, Image.Image):
            return image

        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        if tables:
            for t in tables:
                tb = t.box
                draw.rectangle(
                    [tb.x0, tb.y0, tb.x1, tb.y1],
                    outline=COLOR_PALETTE["TABLE"],
                    width=3
                )
                if draw_labels and self.font:
                    draw.text(
                        (tb.x0 + 4, max(0, tb.y0 - 15)),
                        f"TABLE ({len(t.rows)} rows)",
                        fill=COLOR_PALETTE["TABLE"],
                        font=self.font
                    )

        for ent in entities:
            box = ent.box
            color = self.get_color_for_label(ent.label)
            draw.rectangle(
                [box.x0, box.y0, box.x1, box.y1],
                outline=color,
                width=2
            )
            if draw_labels and self.font:
                tag_text = f"{ent.label} ({int(ent.confidence * 100)}%)"
                text_bbox = draw.textbbox((box.x0, max(0, box.y0 - 14)), tag_text, font=self.font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((box.x0, max(0, box.y0 - 14)), tag_text, fill=(255, 255, 255), font=self.font)

        return annotated

    def draw_tokens(self, image: Any, tokens: List[WordToken]) -> Optional[Any]:
        if not HAS_PIL or Image is None or not isinstance(image, Image.Image):
            return image

        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        for t in tokens:
            b = t.box
            draw.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(156, 163, 175), width=1)
            
        return annotated
