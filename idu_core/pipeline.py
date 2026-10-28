"""
Unified End-to-End Intelligent Document Understanding Pipeline.
Connects Preprocessing, OCR, Classification, Multimodal Token Tagging,
Table Extraction, Entity Normalization, and Visualization.
"""

from typing import Union, List, Dict, Any, Optional
import time
import io
from dataclasses import dataclass, field

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from idu_core.config import config
from idu_core.preprocessing.image_processor import ImagePreprocessor
from idu_core.preprocessing.pdf_converter import PDFConverter
from idu_core.ocr.base import BaseOCREngine, OCRResult, DocumentPage, WordToken
from idu_core.ocr import get_ocr_engine
from idu_core.models.multimodal_transformer import MultimodalDocumentTransformer
from idu_core.models.document_classifier import DocumentClassifier
from idu_core.models.table_extractor import TableExtractor, ExtractedTable
from idu_core.models.doc_vqa import DocumentVQAReader
from idu_core.extraction.entity_extractor import EntityExtractor, ExtractedEntity, LinkedPair
from idu_core.extraction.postprocessor import EntityPostprocessor
from idu_core.extraction.visualizer import DocumentVisualizer

@dataclass
class PageUnderstandingResult:
    page_num: int
    width: int
    height: int
    raw_text: str
    tokens: List[WordToken]
    entities: List[ExtractedEntity]
    linked_pairs: List[LinkedPair]
    tables: List[ExtractedTable]
    annotated_image: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "token_count": len(self.tokens),
            "raw_text": self.raw_text,
            "entities": [e.to_dict() for e in self.entities],
            "linked_pairs": [p.to_dict() for p in self.linked_pairs],
            "tables": [t.to_dict() for t in self.tables]
        }

@dataclass
class DocumentUnderstandingResult:
    document_type: str
    type_confidence: float
    total_pages: int
    total_tokens: int
    total_entities: int
    total_tables: int
    execution_time_seconds: float
    classification_probabilities: Dict[str, float]
    pages: List[PageUnderstandingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        aggregated_entities = {}
        for p in self.pages:
            for ent in p.entities:
                val = ent.normalized_value if ent.normalized_value is not None else ent.text
                if ent.label not in aggregated_entities:
                    aggregated_entities[ent.label] = []
                aggregated_entities[ent.label].append({
                    "text": ent.text,
                    "normalized": val,
                    "confidence": ent.confidence,
                    "box": ent.box.to_dict()
                })

        return {
            "document_type": self.document_type,
            "type_confidence": self.type_confidence,
            "classification_probabilities": self.classification_probabilities,
            "summary": {
                "total_pages": self.total_pages,
                "total_tokens": self.total_tokens,
                "total_entities": self.total_entities,
                "total_tables": self.total_tables,
                "execution_time_seconds": round(self.execution_time_seconds, 3)
            },
            "extracted_entities": aggregated_entities,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata
        }

class DocumentUnderstandingPipeline:
    """
    End-to-End Orchestrator for Deep Learning Document AI.
    """
    def __init__(
        self,
        ocr_engine: str = "auto",
        device: str = None,
        use_huggingface: bool = False
    ):
        self.device = device or config.model.device
        self.preprocessor = ImagePreprocessor()
        self.pdf_converter = PDFConverter()
        self.ocr = get_ocr_engine(engine_type=ocr_engine)
        self.classifier = DocumentClassifier(device=self.device)
        self.transformer = MultimodalDocumentTransformer(
            device=self.device, use_huggingface=use_huggingface
        )
        self.table_extractor = TableExtractor()
        self.vqa_reader = DocumentVQAReader()
        self.entity_extractor = EntityExtractor()
        self.postprocessor = EntityPostprocessor()
        self.visualizer = DocumentVisualizer()

    def process(
        self,
        document_input: Any,
        generate_visualizations: bool = True
    ) -> DocumentUnderstandingResult:
        start_time = time.time()

        if HAS_PIL and Image and isinstance(document_input, Image.Image):
            images = [document_input]
        elif isinstance(document_input, str) and (
            document_input.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ) and HAS_PIL and Image:
            images = [Image.open(document_input)]
        else:
            images = self.pdf_converter.convert_pdf_to_images(document_input)

        if not images:
            raise ValueError("No valid document images found to process.")

        ocr_pages = []
        for idx, img in enumerate(images, start=1):
            preprocessed_img = self.preprocessor.prepare_for_model(img)
            ocr_page = self.ocr.extract_page(preprocessed_img, page_num=idx)
            ocr_pages.append(ocr_page)

        full_doc_text = "\n".join(p.raw_text for p in ocr_pages)

        classification_result = self.classifier.classify(full_doc_text, images[0])
        doc_type = classification_result["predicted_class"]
        doc_conf = classification_result["confidence"]

        page_results = []
        total_tokens_count = 0
        total_entities_count = 0
        total_tables_count = 0

        for p in ocr_pages:
            total_tokens_count += len(p.tokens)
            
            token_preds = self.transformer.predict_tokens(p.tokens, p.image)
            raw_entities = self.entity_extractor.decode_bio_tokens(token_preds)
            entities = [self.postprocessor.process_entity(e) for e in raw_entities]
            total_entities_count += len(entities)

            linked_pairs = self.entity_extractor.link_question_answer_pairs(entities)
            tables = self.table_extractor.extract_tables(p.tokens)
            total_tables_count += len(tables)

            annotated_img = None
            if generate_visualizations:
                annotated_img = self.visualizer.draw_entities(
                    p.image, entities, tables=tables, draw_labels=True
                )

            page_results.append(PageUnderstandingResult(
                page_num=p.page_num,
                width=p.width,
                height=p.height,
                raw_text=p.raw_text,
                tokens=p.tokens,
                entities=entities,
                linked_pairs=linked_pairs,
                tables=tables,
                annotated_image=annotated_img
            ))

        elapsed = time.time() - start_time

        return DocumentUnderstandingResult(
            document_type=doc_type,
            type_confidence=doc_conf,
            total_pages=len(page_results),
            total_tokens=total_tokens_count,
            total_entities=total_entities_count,
            total_tables=total_tables_count,
            execution_time_seconds=elapsed,
            classification_probabilities=classification_result["probabilities"],
            pages=page_results,
            metadata={
                "ocr_engine": type(self.ocr).__name__,
                "device": self.device,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )

    def ask(self, question: str, result: DocumentUnderstandingResult) -> Dict[str, Any]:
        all_tokens = [t for p in result.pages for t in p.tokens]
        full_text = "\n".join(p.raw_text for p in result.pages)
        return self.vqa_reader.answer_question(question, all_tokens, raw_text=full_text)
