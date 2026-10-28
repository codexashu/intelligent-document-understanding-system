"""
Synthetic Document & Annotation Generator.
Generates realistic high-resolution document images (Invoices, Receipts, Forms)
complete with bounding boxes and ground-truth BIO entity labels.
"""

from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import os
import random

from idu_core.ocr.base import WordToken, BoundingBox

class SyntheticDocumentGenerator:
    """
    Renders realistic synthetic documents for model training, benchmarking, and unit testing.
    """
    def __init__(self, width: int = 800, height: int = 1000):
        self.width = width
        self.height = height
        self.font = ImageFont.load_default()

    def generate_invoice(self) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Creates a synthetic corporate invoice with headers, metadata, table items, and totals.
        """
        img = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        annotations = []

        def add_text_token(text: str, x: int, y: int, label: str, font_size: int = 12):
            # Approximate bounding box
            w = len(text) * 8
            h = 16
            draw.text((x, y), text, fill=(0, 0, 0), font=self.font)
            annotations.append({
                "text": text,
                "label": label,
                "box": [x, y, x + w, y + h]
            })
            return x + w

        # 1. Header
        add_text_token("TAX", 60, 50, "B-HEADER")
        add_text_token("INVOICE", 100, 50, "I-HEADER")

        # 2. Invoice Number
        add_text_token("Invoice", 60, 110, "B-QUESTION")
        add_text_token("Number:", 120, 110, "I-QUESTION")
        add_text_token("INV-2026-9041", 190, 110, "B-INVOICE_NUMBER")

        # 3. Invoice Date
        add_text_token("Date:", 60, 140, "B-QUESTION")
        add_text_token("2026-10-28", 110, 140, "B-DATE")

        # 4. Vendor Info
        add_text_token("Vendor:", 60, 180, "B-QUESTION")
        add_text_token("Apex", 130, 180, "B-VENDOR_NAME")
        add_text_token("Cloud", 175, 180, "I-VENDOR_NAME")
        add_text_token("Technologies", 225, 180, "I-VENDOR_NAME")
        add_text_token("LLC", 335, 180, "I-VENDOR_NAME")

        # 5. Customer Info
        add_text_token("Bill", 60, 220, "B-QUESTION")
        add_text_token("To:", 95, 220, "I-QUESTION")
        add_text_token("Global", 130, 220, "O")
        add_text_token("Enterprises", 185, 220, "O")

        # 6. Table Separator Line & Headers
        draw.line([(50, 280), (750, 280)], fill=(180, 180, 180), width=2)
        add_text_token("ITEM", 60, 290, "O")
        add_text_token("DESCRIPTION", 130, 290, "O")
        add_text_token("QTY", 450, 290, "O")
        add_text_token("UNIT_PRICE", 530, 290, "O")
        add_text_token("AMOUNT", 660, 290, "O")
        draw.line([(50, 315), (750, 315)], fill=(180, 180, 180), width=1)

        # 7. Table Rows
        items = [
            ("1", "Enterprise AI Cloud GPU Instance", "4", "$120.00", "$480.00"),
            ("2", "Document Intelligence API Subscription", "1", "$250.00", "$250.00"),
            ("3", "Dedicated Support & SLA Tier 1", "1", "$150.00", "$150.00"),
        ]
        
        y_cursor = 330
        for item_idx, desc, qty, rate, amt in items:
            add_text_token(item_idx, 60, y_cursor, "O")
            add_text_token(desc, 130, y_cursor, "B-LINE_ITEM")
            add_text_token(qty, 460, y_cursor, "O")
            add_text_token(rate, 540, y_cursor, "O")
            add_text_token(amt, 660, y_cursor, "O")
            y_cursor += 30

        draw.line([(50, y_cursor + 10), (750, y_cursor + 10)], fill=(180, 180, 180), width=2)

        # 8. Totals
        y_cursor += 30
        add_text_token("Subtotal:", 530, y_cursor, "B-QUESTION")
        add_text_token("$880.00", 660, y_cursor, "O")
        
        y_cursor += 25
        add_text_token("Tax", 530, y_cursor, "B-QUESTION")
        add_text_token("(10%):", 565, y_cursor, "I-QUESTION")
        add_text_token("$88.00", 660, y_cursor, "O")
        
        y_cursor += 30
        draw.line([(500, y_cursor - 5), (750, y_cursor - 5)], fill=(0, 0, 0), width=2)
        add_text_token("TOTAL", 530, y_cursor, "B-QUESTION")
        add_text_token("DUE:", 585, y_cursor, "I-QUESTION")
        add_text_token("$968.00", 660, y_cursor, "B-TOTAL_AMOUNT")
        draw.line([(500, y_cursor + 25), (750, y_cursor + 25)], fill=(0, 0, 0), width=2)

        return img, annotations

    def generate_receipt(self) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Creates a synthetic retail store receipt.
        """
        img = Image.new("RGB", (450, 700), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        annotations = []

        def add_tok(text: str, x: int, y: int, label: str):
            w = len(text) * 7
            h = 15
            draw.text((x, y), text, fill=(20, 20, 20), font=self.font)
            annotations.append({"text": text, "label": label, "box": [x, y, x + w, y + h]})

        # Header
        add_tok("SUPERMARKET", 160, 40, "B-HEADER")
        add_tok("ORGANIC", 185, 65, "B-VENDOR_NAME")
        add_tok("MARKET", 240, 65, "I-VENDOR_NAME")

        add_tok("Receipt#", 40, 110, "B-QUESTION")
        add_tok("REC-5582", 110, 110, "B-INVOICE_NUMBER")

        add_tok("Date:", 40, 135, "B-QUESTION")
        add_tok("2026-10-28", 90, 135, "B-DATE")

        draw.line([(30, 165), (420, 165)], fill=(120, 120, 120), width=1)

        items = [
            ("Organic Milk 1L", "$3.50"),
            ("Whole Wheat Bread", "$2.80"),
            ("Avocado Bag 5pk", "$5.99"),
            ("Dark Chocolate 85%", "$4.20"),
        ]
        
        y = 185
        for item, price in items:
            add_tok(item, 40, y, "B-LINE_ITEM")
            add_tok(price, 360, y, "O")
            y += 25

        draw.line([(30, y + 10), (420, y + 10)], fill=(120, 120, 120), width=1)
        y += 25
        add_tok("TOTAL:", 280, y, "B-QUESTION")
        add_tok("$16.49", 360, y, "B-TOTAL_AMOUNT")

        return img, annotations

    def generate_medical_form(self) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Creates a synthetic medical patient intake form.
        """
        img = Image.new("RGB", (800, 950), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        annotations = []

        def add_tok(text: str, x: int, y: int, label: str):
            w = len(text) * 8
            h = 16
            draw.text((x, y), text, fill=(0, 0, 0), font=self.font)
            annotations.append({"text": text, "label": label, "box": [x, y, x + w, y + h]})

        add_tok("PATIENT", 320, 40, "B-HEADER")
        add_tok("INTAKE", 390, 40, "I-HEADER")
        add_tok("FORM", 450, 40, "I-HEADER")

        add_tok("Patient", 50, 110, "B-QUESTION")
        add_tok("Name:", 115, 110, "I-QUESTION")
        add_tok("John", 170, 110, "B-ANSWER")
        add_tok("Doe", 215, 110, "I-ANSWER")

        add_tok("DOB:", 450, 110, "B-QUESTION")
        add_tok("1990-05-15", 495, 110, "B-DATE")

        add_tok("Medical", 50, 150, "B-QUESTION")
        add_tok("Record", 120, 150, "I-QUESTION")
        add_tok("Number:", 180, 150, "I-QUESTION")
        add_tok("MRN-992014", 250, 150, "B-INVOICE_NUMBER")

        add_tok("Provider:", 50, 190, "B-QUESTION")
        add_tok("St.", 130, 190, "B-VENDOR_NAME")
        add_tok("Jude", 160, 190, "I-VENDOR_NAME")
        add_tok("Medical", 205, 190, "I-VENDOR_NAME")
        add_tok("Center", 275, 190, "I-VENDOR_NAME")

        return img, annotations

def generate_sample_files(target_dir: str):
    """Generates all standard sample images and saves to target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    gen = SyntheticDocumentGenerator()
    
    inv_img, _ = gen.generate_invoice()
    inv_path = os.path.join(target_dir, "sample_invoice.png")
    inv_img.save(inv_path)
    
    rec_img, _ = gen.generate_receipt()
    rec_path = os.path.join(target_dir, "sample_receipt.png")
    rec_img.save(rec_path)
    
    form_img, _ = gen.generate_medical_form()
    form_path = os.path.join(target_dir, "sample_medical_form.png")
    form_img.save(form_path)
    
    return [inv_path, rec_path, form_path]

if __name__ == "__main__":
    generate_sample_files("samples")
    print("Sample documents generated successfully.")
