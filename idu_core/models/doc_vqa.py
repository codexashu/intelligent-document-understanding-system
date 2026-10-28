"""
Document Visual Question Answering (DocVQA) Reader.
Enables natural language querying over document text and layout.
"""

from typing import List, Dict, Any, Optional
import re
from idu_core.ocr.base import WordToken, BoundingBox

class DocumentVQAReader:
    """
    Multimodal DocVQA answering engine.
    Finds exact answer spans from document tokens matching user questions.
    """
    def __init__(self):
        # Keyword and semantic association map
        self.intent_map = {
            "total": ["total", "amount", "balance", "$", "eur", "usd", "due", "sum"],
            "date": ["date", "issued", "due date", "billing date", "day"],
            "invoice": ["invoice", "number", "inv#", "inv-", "reference", "bill no"],
            "vendor": ["vendor", "seller", "from", "company", "inc", "corp", "ltd"],
            "customer": ["customer", "client", "bill to", "buyer", "to:"],
            "tax": ["tax", "vat", "gst", "%"],
            "quantity": ["qty", "quantity", "units", "count"]
        }

    def answer_question(
        self, question: str, tokens: List[WordToken], raw_text: str = ""
    ) -> Dict[str, Any]:
        """
        Locates the most relevant answer span for the given question.
        """
        q_lower = question.lower()
        if not tokens:
            return {
                "question": question,
                "answer": "No tokens detected in document.",
                "confidence": 0.0,
                "bounding_box": None
            }

        # 1. Total amount query
        if any(w in q_lower for w in ["total", "cost", "price", "how much", "amount"]):
            for i, t in enumerate(tokens):
                t_lower = t.text.lower()
                if "total" in t_lower:
                    # Look ahead for numeric / currency token
                    for j in range(i + 1, min(i + 6, len(tokens))):
                        candidate = tokens[j]
                        if any(c.isdigit() for c in candidate.text):
                            return {
                                "question": question,
                                "answer": candidate.text,
                                "confidence": 0.96,
                                "bounding_box": candidate.box.to_dict()
                            }
                if any(sym in t.text for sym in ["$", "€", "£", "₹"]) and any(c.isdigit() for c in t.text):
                    if t.normalized_box.y0 > 350:
                        return {
                            "question": question,
                            "answer": t.text,
                            "confidence": 0.92,
                            "bounding_box": t.box.to_dict()
                        }

        # 2. Date query
        if any(w in q_lower for w in ["date", "when", "issued", "day"]):
            for i, t in enumerate(tokens):
                if (
                    ("-" in t.text or "/" in t.text)
                    and sum(c.isdigit() for c in t.text) >= 4
                ):
                    return {
                        "question": question,
                        "answer": t.text,
                        "confidence": 0.95,
                        "bounding_box": t.box.to_dict()
                    }

        # 3. Invoice Number query
        if any(w in q_lower for w in ["invoice number", "invoice #", "id", "inv"]):
            for i, t in enumerate(tokens):
                if any(w in t.text.lower() for w in ["inv-", "invoice#", "inv_"]) or (
                    len(t.text) > 4 and "-" in t.text and sum(c.isdigit() for c in t.text) >= 3
                ):
                    return {
                        "question": question,
                        "answer": t.text,
                        "confidence": 0.94,
                        "bounding_box": t.box.to_dict()
                    }

        # 4. Vendor / Company query
        if any(w in q_lower for w in ["who", "vendor", "company", "issuer", "from"]):
            for i, t in enumerate(tokens):
                if "vendor:" in t.text.lower() or "from:" in t.text.lower():
                    # Take subsequent tokens on the same line
                    company_words = []
                    boxes = []
                    for j in range(i + 1, min(i + 4, len(tokens))):
                        if abs(tokens[j].box.center[1] - t.box.center[1]) < 20:
                            company_words.append(tokens[j].text)
                            boxes.append(tokens[j].box)
                    if company_words:
                        ans_box = BoundingBox(
                            min(b.x0 for b in boxes),
                            min(b.y0 for b in boxes),
                            max(b.x1 for b in boxes),
                            max(b.y1 for b in boxes)
                        )
                        return {
                            "question": question,
                            "answer": " ".join(company_words),
                            "confidence": 0.91,
                            "bounding_box": ans_box.to_dict()
                        }

        # 5. Customer query
        if any(w in q_lower for w in ["customer", "client", "bill to", "buyer"]):
            for i, t in enumerate(tokens):
                if "customer:" in t.text.lower() or "bill to" in t.text.lower():
                    cust_words = []
                    boxes = []
                    for j in range(i + 1, min(i + 4, len(tokens))):
                        if abs(tokens[j].box.center[1] - t.box.center[1]) < 20:
                            cust_words.append(tokens[j].text)
                            boxes.append(tokens[j].box)
                    if cust_words:
                        ans_box = BoundingBox(
                            min(b.x0 for b in boxes),
                            min(b.y0 for b in boxes),
                            max(b.x1 for b in boxes),
                            max(b.y1 for b in boxes)
                        )
                        return {
                            "question": question,
                            "answer": " ".join(cust_words),
                            "confidence": 0.90,
                            "bounding_box": ans_box.to_dict()
                        }

        # General Semantic Fallback: search for tokens matching words in question
        q_words = set(re.findall(r"\w+", q_lower))
        best_token = None
        best_score = 0.0

        for t in tokens:
            t_w = set(re.findall(r"\w+", t.text.lower()))
            common = len(q_words.intersection(t_w))
            if common > best_score:
                best_score = common
                best_token = t

        if best_token and best_score > 0:
            return {
                "question": question,
                "answer": best_token.text,
                "confidence": 0.65,
                "bounding_box": best_token.box.to_dict()
            }

        return {
            "question": question,
            "answer": "Answer span could not be determined with sufficient confidence.",
            "confidence": 0.20,
            "bounding_box": None
        }
