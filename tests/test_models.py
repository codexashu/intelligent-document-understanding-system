import unittest
from idu_core.ocr.base import BoundingBox, WordToken
from idu_core.models.multimodal_transformer import MultimodalDocumentTransformer
from idu_core.models.document_classifier import DocumentClassifier
from idu_core.models.table_extractor import TableExtractor
from idu_core.models.doc_vqa import DocumentVQAReader
from idu_core.extraction.entity_extractor import EntityExtractor
from idu_core.extraction.postprocessor import EntityPostprocessor

class TestIDUModels(unittest.TestCase):
    def setUp(self):
        self.tokens = [
            WordToken("INVOICE", BoundingBox(50, 40, 180, 80), BoundingBox(62, 40, 225, 80), 0.99, 0, 0),
            WordToken("INV-2026-001", BoundingBox(210, 110, 360, 130), BoundingBox(262, 110, 450, 130), 0.95, 1, 0),
            WordToken("2026-10-28", BoundingBox(110, 140, 220, 160), BoundingBox(137, 140, 275, 160), 0.96, 2, 0),
            WordToken("Acme", BoundingBox(140, 190, 190, 210), BoundingBox(175, 190, 237, 210), 0.94, 3, 0),
            WordToken("Solutions", BoundingBox(195, 190, 280, 210), BoundingBox(243, 190, 350, 210), 0.94, 3, 0),
            WordToken("TOTAL", BoundingBox(500, 440, 560, 465), BoundingBox(625, 440, 700, 465), 0.98, 4, 1),
            WordToken("$1,500.00", BoundingBox(650, 440, 740, 465), BoundingBox(812, 440, 925, 465), 0.99, 4, 1),
        ]

    def test_document_classifier(self):
        classifier = DocumentClassifier()
        text = "TAX INVOICE Invoice Number: INV-990 Amount Due: $1,500.00"
        res = classifier.classify(text)
        self.assertEqual(res["predicted_class"], "INVOICE")
        self.assertGreater(res["confidence"], 0.5)
        self.assertIn("INVOICE", res["probabilities"])

    def test_multimodal_transformer_prediction(self):
        transformer = MultimodalDocumentTransformer()
        preds = transformer.predict_tokens(self.tokens)
        self.assertEqual(len(preds), len(self.tokens))
        
        labels = [p[1] for p in preds]
        # Should identify invoice number, total amount, and date
        self.assertTrue(any("TOTAL_AMOUNT" in l for l in labels))
        self.assertTrue(any("INVOICE_NUMBER" in l for l in labels))
        self.assertTrue(any("DATE" in l for l in labels))

    def test_entity_extractor_and_linking(self):
        transformer = MultimodalDocumentTransformer()
        extractor = EntityExtractor()
        postprocessor = EntityPostprocessor()
        
        preds = transformer.predict_tokens(self.tokens)
        entities = extractor.decode_bio_tokens(preds)
        processed = [postprocessor.process_entity(e) for e in entities]
        
        # Verify normalized amount
        amount_ents = [e for e in processed if "TOTAL_AMOUNT" in e.label]
        self.assertGreaterEqual(len(amount_ents), 1)
        self.assertEqual(amount_ents[0].normalized_value, 1500.0)

    def test_doc_vqa_reader(self):
        reader = DocumentVQAReader()
        
        q1 = "What is the total amount?"
        a1 = reader.answer_question(q1, self.tokens)
        self.assertIn("1,500.00", a1["answer"])
        self.assertGreater(a1["confidence"], 0.7)

        q2 = "What is the invoice number?"
        a2 = reader.answer_question(q2, self.tokens)
        self.assertEqual(a2["answer"], "INV-2026-001")

    def test_table_extractor(self):
        extractor = TableExtractor()
        # Mock table grid tokens
        table_tokens = [
            WordToken("ITEM", BoundingBox(50, 300, 100, 320), BoundingBox(50, 300, 100, 320)),
            WordToken("PRICE", BoundingBox(200, 300, 250, 320), BoundingBox(200, 300, 250, 320)),
            WordToken("Server", BoundingBox(50, 340, 100, 360), BoundingBox(50, 340, 100, 360)),
            WordToken("$500", BoundingBox(200, 340, 250, 360), BoundingBox(200, 340, 250, 360)),
        ]
        tables = extractor.extract_tables(table_tokens)
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0].rows), 1)
        self.assertEqual(tables[0].headers, ["ITEM", "PRICE"])

if __name__ == "__main__":
    unittest.main()
