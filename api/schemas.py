"""
Pydantic schemas and serialization models for the REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BoundingBoxModel(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int

class EntityModel(BaseModel):
    label: str
    text: str
    confidence: float
    bounding_box: BoundingBoxModel
    normalized_value: Optional[Any] = None

class LinkedPairModel(BaseModel):
    question: EntityModel
    answer: EntityModel
    distance: float

class TableModel(BaseModel):
    table_id: int
    bounding_box: BoundingBoxModel
    headers: List[str]
    rows: List[List[str]]
    row_count: int
    column_count: int

class PageExtractionModel(BaseModel):
    page_num: int
    width: int
    height: int
    token_count: int
    raw_text: str
    entities: List[EntityModel]
    linked_pairs: List[LinkedPairModel]
    tables: List[TableModel]

class ExtractionSummaryModel(BaseModel):
    total_pages: int
    total_tokens: int
    total_entities: int
    total_tables: int
    execution_time_seconds: float

class ExtractionResponse(BaseModel):
    status: str = "success"
    document_type: str
    type_confidence: float
    classification_probabilities: Dict[str, float]
    summary: ExtractionSummaryModel
    extracted_entities: Dict[str, List[Dict[str, Any]]]
    pages: List[PageExtractionModel]
    metadata: Dict[str, Any]

class ClassificationResponse(BaseModel):
    status: str = "success"
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    top_classes: List[Dict[str, Any]]

class DocVQAResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str
    confidence: float
    bounding_box: Optional[Dict[str, int]] = None

class TableExtractionResponse(BaseModel):
    status: str = "success"
    total_tables: int
    tables: List[TableModel]

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    supported_classes: List[str]
    supported_labels: List[str]
