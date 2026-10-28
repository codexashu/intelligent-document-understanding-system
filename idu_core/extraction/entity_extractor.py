"""
Key Information Extraction (KIE) & Entity Assembler.
Decodes BIO sequence tokens into coherent multi-word entity spans
and performs 2D spatial Question-Answer linking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import math

from idu_core.ocr.base import WordToken, BoundingBox

@dataclass
class ExtractedEntity:
    label: str
    text: str
    box: BoundingBox
    confidence: float
    tokens: List[WordToken] = field(default_factory=list)
    normalized_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.box.to_dict(),
            "token_count": len(self.tokens),
            "normalized_value": self.normalized_value
        }

@dataclass
class LinkedPair:
    question: ExtractedEntity
    answer: ExtractedEntity
    spatial_distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "answer": self.answer.to_dict(),
            "distance": round(self.spatial_distance, 2)
        }

class EntityExtractor:
    """
    Assembles BIO tokens into complete entities and solves spatial entity linking.
    """
    def decode_bio_tokens(
        self, token_predictions: List[Tuple[WordToken, str, float]]
    ) -> List[ExtractedEntity]:
        """
        Groups sequential BIO tokens into unified entity objects.
        """
        entities: List[ExtractedEntity] = []
        current_entity_label = None
        current_tokens: List[WordToken] = []
        current_confs: List[float] = []

        def flush_current():
            nonlocal current_entity_label, current_tokens, current_confs
            if current_entity_label and current_tokens:
                text = " ".join(t.text for t in current_tokens)
                x0 = min(t.box.x0 for t in current_tokens)
                y0 = min(t.box.y0 for t in current_tokens)
                x1 = max(t.box.x1 for t in current_tokens)
                y1 = max(t.box.y1 for t in current_tokens)
                avg_conf = sum(current_confs) / len(current_confs)
                
                entities.append(ExtractedEntity(
                    label=current_entity_label,
                    text=text,
                    box=BoundingBox(x0, y0, x1, y1),
                    confidence=avg_conf,
                    tokens=list(current_tokens)
                ))
            current_entity_label = None
            current_tokens = []
            current_confs = []

        for token, bio_tag, conf in token_predictions:
            if bio_tag == "O":
                flush_current()
                continue
                
            prefix = bio_tag[:2]
            label = bio_tag[2:] if prefix in ("B-", "I-") else bio_tag

            if prefix == "B-":
                flush_current()
                current_entity_label = label
                current_tokens.append(token)
                current_confs.append(conf)
            elif prefix == "I-":
                if current_entity_label == label:
                    current_tokens.append(token)
                    current_confs.append(conf)
                else:
                    flush_current()
                    current_entity_label = label
                    current_tokens.append(token)
                    current_confs.append(conf)
            else:
                flush_current()

        flush_current()
        return entities

    def link_question_answer_pairs(
        self, entities: List[ExtractedEntity], max_horizontal_dist: float = 400
    ) -> List[LinkedPair]:
        """
        Links Question entities to their most probable Answer entities
        using 2D geometric reading-order proximity.
        """
        questions = [e for e in entities if "QUESTION" in e.label]
        answers = [e for e in entities if "ANSWER" in e.label or e.label in (
            "TOTAL_AMOUNT", "DATE", "INVOICE_NUMBER", "VENDOR_NAME"
        )]

        pairs: List[LinkedPair] = []
        used_answers = set()

        for q in questions:
            best_ans = None
            best_dist = float("inf")

            qx, qy = q.box.center
            for idx, a in enumerate(answers):
                if idx in used_answers:
                    continue
                ax, ay = a.box.center
                
                # Reading order constraint: Answer is usually to the right or immediately below Question
                dx = ax - qx
                dy = ay - qy
                
                # Check horizontal alignment (same line)
                if abs(dy) < 30 and dx > 0 and dx < max_horizontal_dist:
                    dist = dx + abs(dy) * 2
                    if dist < best_dist:
                        best_dist = dist
                        best_ans = (idx, a)
                # Check vertical alignment (answer below question)
                elif dy > 0 and dy < 80 and abs(dx) < 100:
                    dist = dy * 2 + abs(dx)
                    if dist < best_dist:
                        best_dist = dist
                        best_ans = (idx, a)

            if best_ans is not None:
                used_answers.add(best_ans[0])
                pairs.append(LinkedPair(
                    question=q,
                    answer=best_ans[1],
                    spatial_distance=best_dist
                ))

        return pairs
