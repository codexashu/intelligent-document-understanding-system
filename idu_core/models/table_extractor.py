"""
Table and Cell Structure Detection & Extraction.
Detects tabular regions in documents, reconstructs grid cells, and exports structured tables.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from idu_core.ocr.base import WordToken, BoundingBox

@dataclass
class TableCell:
    row: int
    col: int
    text: str
    box: BoundingBox
    confidence: float = 1.0

@dataclass
class ExtractedTable:
    table_id: int
    box: BoundingBox
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    cells: List[TableCell] = field(default_factory=list)
    
    def to_dataframe(self) -> Optional[Any]:
        if not HAS_PANDAS or pd is None:
            return None
            
        if self.headers and self.rows:
            max_cols = max(len(self.headers), max(len(r) for r in self.rows))
            headers = self.headers + [f"Col_{i}" for i in range(len(self.headers), max_cols)]
            padded_rows = [r + [""] * (max_cols - len(r)) for r in self.rows]
            return pd.DataFrame(padded_rows, columns=headers[:max_cols])
        elif self.rows:
            return pd.DataFrame(self.rows)
        return pd.DataFrame()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "bounding_box": self.box.to_dict(),
            "headers": self.headers,
            "rows": self.rows,
            "row_count": len(self.rows),
            "column_count": len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)
        }

class TableExtractor:
    """
    Heuristic and Spatial Clustering Engine for Document Table Extraction.
    """
    def __init__(self, y_threshold: int = 15, x_threshold: int = 25):
        self.y_threshold = y_threshold
        self.x_threshold = x_threshold

    def extract_tables(self, tokens: List[WordToken]) -> List[ExtractedTable]:
        if not tokens:
            return []

        table_tokens = [t for t in tokens if 250 <= t.normalized_box.y0 <= 750]
        if len(table_tokens) < 4:
            return []

        sorted_tokens = sorted(table_tokens, key=lambda t: t.box.y0)
        lines: List[List[WordToken]] = []
        current_line: List[WordToken] = []
        current_y = None

        for t in sorted_tokens:
            if current_y is None:
                current_y = t.box.center[1]
                current_line.append(t)
            elif abs(t.box.center[1] - current_y) <= self.y_threshold:
                current_line.append(t)
            else:
                lines.append(sorted(current_line, key=lambda x: x.box.x0))
                current_line = [t]
                current_y = t.box.center[1]

        if current_line:
            lines.append(sorted(current_line, key=lambda x: x.box.x0))

        table_lines = [line for line in lines if len(line) >= 2]
        if len(table_lines) < 2:
            return []

        all_x0 = min(t.box.x0 for line in table_lines for t in line)
        all_y0 = min(t.box.y0 for line in table_lines for t in line)
        all_x1 = max(t.box.x1 for line in table_lines for t in line)
        all_y1 = max(t.box.y1 for line in table_lines for t in line)
        table_box = BoundingBox(all_x0, all_y0, all_x1, all_y1)

        headers = [t.text for t in table_lines[0]]
        rows = []
        cells = []
        
        for row_idx, line in enumerate(table_lines[1:], start=1):
            row_vals = [t.text for t in line]
            rows.append(row_vals)
            for col_idx, t in enumerate(line):
                cells.append(TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=t.text,
                    box=t.box,
                    confidence=t.confidence
                ))

        extracted = ExtractedTable(
            table_id=1,
            box=table_box,
            headers=headers,
            rows=rows,
            cells=cells
        )
        return [extracted]
