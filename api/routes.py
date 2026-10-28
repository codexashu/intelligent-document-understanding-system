"""
FastAPI route handlers for Intelligent Document Understanding.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from typing import Optional
import io

from idu_core.pipeline import DocumentUnderstandingPipeline
from idu_core.config import config
from api.schemas import (
    ExtractionResponse,
    ClassificationResponse,
    DocVQAResponse,
    TableExtractionResponse,
    HealthResponse
)

router = APIRouter(prefix="/api/v1")

# Lazy-loaded pipeline singleton
_pipeline: Optional[DocumentUnderstandingPipeline] = None

def get_pipeline(ocr_engine: str = "auto") -> DocumentUnderstandingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentUnderstandingPipeline(ocr_engine=ocr_engine)
    return _pipeline

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns system status, active compute device, and label schema."""
    return HealthResponse(
        status="healthy",
        app_name=config.app_name,
        version=config.version,
        device=config.model.device,
        supported_classes=config.labels.document_classes,
        supported_labels=config.labels.entity_labels
    )

@router.post("/extract", response_model=ExtractionResponse, tags=["Document Understanding"])
async def extract_document(
    file: UploadFile = File(..., description="Document file (PDF, PNG, JPG, TIFF)"),
    ocr_engine: str = Query("auto", description="OCR Engine ('auto', 'tesseract', 'easyocr', 'synthetic')")
):
    """
    Performs full end-to-end document understanding:
    OCR + Classification + Key Information Extraction + Tables + QA Linking.
    """
    try:
        content = await file.read()
        pipeline = get_pipeline(ocr_engine=ocr_engine)
        result = pipeline.process(content, generate_visualizations=False)
        return ExtractionResponse(**result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_document(
    file: UploadFile = File(...)
):
    """
    Classifies a document into categories:
    [INVOICE, RECEIPT, FORM, RESUME, CONTRACT, SCIENTIFIC_REPORT, ID_DOCUMENT, OTHER].
    """
    try:
        content = await file.read()
        pipeline = get_pipeline()
        result = pipeline.process(content, generate_visualizations=False)
        return ClassificationResponse(
            predicted_class=result.document_type,
            confidence=result.type_confidence,
            probabilities=result.classification_probabilities,
            top_classes=[
                {"class": c, "confidence": p}
                for c, p in sorted(
                    result.classification_probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/tables", response_model=TableExtractionResponse, tags=["Table Extraction"])
async def extract_tables(
    file: UploadFile = File(...)
):
    """
    Extracts tabular structures, grid rows, headers, and cell coordinates from the document.
    """
    try:
        content = await file.read()
        pipeline = get_pipeline()
        result = pipeline.process(content, generate_visualizations=False)
        
        all_tables = []
        for p in result.pages:
            all_tables.extend([t.to_dict() for t in p.tables])

        return TableExtractionResponse(
            total_tables=len(all_tables),
            tables=all_tables
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table extraction failed: {str(e)}")

@router.post("/qa", response_model=DocVQAResponse, tags=["DocVQA"])
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(..., description="Natural language question about the document")
):
    """
    Answers arbitrary questions about the uploaded document using multimodal spatial reading.
    """
    try:
        content = await file.read()
        pipeline = get_pipeline()
        result = pipeline.process(content, generate_visualizations=False)
        qa_result = pipeline.ask(question, result)
        return DocVQAResponse(
            question=question,
            answer=qa_result["answer"],
            confidence=qa_result["confidence"],
            bounding_box=qa_result.get("bounding_box")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DocVQA failed: {str(e)}")

@router.post("/visualize", tags=["Visualization"])
async def visualize_document(
    file: UploadFile = File(...)
):
    """
    Processes the document and returns the page 1 annotated image with color-coded bounding boxes.
    """
    try:
        content = await file.read()
        pipeline = get_pipeline()
        result = pipeline.process(content, generate_visualizations=True)
        if not result.pages or result.pages[0].annotated_image is None:
            raise HTTPException(status_code=400, detail="No visual output generated.")
            
        buf = io.BytesIO()
        result.pages[0].annotated_image.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
