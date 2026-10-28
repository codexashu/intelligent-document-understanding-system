"""
Entity Post-processing & Normalization.
Cleans raw text spans into structured values (floats, ISO-8601 dates, standard IDs).
"""

import re
from datetime import datetime
from typing import Optional, Any
from idu_core.extraction.entity_extractor import ExtractedEntity

class EntityPostprocessor:
    """
    Cleans and standardizes extracted textual values into machine-actionable data types.
    """
    def normalize_amount(self, text: str) -> Optional[float]:
        """Converts strings like '$ 1,234.50 USD' to float 1234.50."""
        # Strip currency symbols and whitespace
        clean = re.sub(r"[^\d.,\-]", "", text.strip())
        if not clean:
            return None
        # Handle European decimal formats (1.234,50) vs US formats (1,234.50)
        if "," in clean and "." in clean:
            if clean.rfind(",") > clean.rfind("."):
                clean = clean.replace(".", "").replace(",", ".")
            else:
                clean = clean.replace(",", "")
        elif "," in clean:
            # Check if comma is used as decimal separator
            parts = clean.split(",")
            if len(parts) == 2 and len(parts[1]) == 2:
                clean = clean.replace(",", ".")
            else:
                clean = clean.replace(",", "")
                
        try:
            return float(clean)
        except ValueError:
            return None

    def normalize_date(self, text: str) -> Optional[str]:
        """Normalizes diverse date formats into standard ISO-8601 (YYYY-MM-DD)."""
        clean = text.strip()
        # Common date patterns
        date_formats = [
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %b %Y",
            "%d %B %Y"
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(clean, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Regex fallback for embedded dates
        match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", clean)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        match = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", clean)
        if match:
            p1, p2, y = match.groups()
            return f"{int(y):04d}-{int(p1):02d}-{int(p2):02d}"

        return None

    def normalize_identifier(self, text: str) -> str:
        """Strips noise words like 'INV #', 'No.', 'Ref:' from IDs."""
        clean = re.sub(r"(?i)^(invoice|inv|bill|ref|number|no|#|:|\s)+", "", text).strip()
        return clean

    def process_entity(self, entity: ExtractedEntity) -> ExtractedEntity:
        """Applies type-specific normalization to an entity."""
        label = entity.label.upper()
        if "TOTAL_AMOUNT" in label or "AMOUNT" in label or "PRICE" in label:
            entity.normalized_value = self.normalize_amount(entity.text)
        elif "DATE" in label:
            entity.normalized_value = self.normalize_date(entity.text)
        elif "INVOICE_NUMBER" in label or "ID" in label:
            entity.normalized_value = self.normalize_identifier(entity.text)
        else:
            entity.normalized_value = entity.text.strip()
        return entity
