import unittest
import os
import json
from idu_core.pipeline import DocumentUnderstandingPipeline

class TestDocumentUnderstandingPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DocumentUnderstandingPipeline(ocr_engine="synthetic")

    def test_pipeline_end_to_end(self):
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "samples",
            "sample_invoice.png"
        )
        
        # Process document
        result = self.pipeline.process(sample_path, generate_visualizations=False)
        
        self.assertEqual(result.total_pages, 1)
        self.assertGreater(result.total_tokens, 0)
        self.assertGreater(result.total_entities, 0)
        self.assertIn(result.document_type, ["INVOICE", "RECEIPT", "FORM", "OTHER"])
        self.assertGreaterEqual(result.type_confidence, 0.5)

        # Test serializability
        res_dict = result.to_dict()
        self.assertIn("extracted_entities", res_dict)
        self.assertIn("summary", res_dict)
        json_output = json.dumps(res_dict)
        self.assertIsInstance(json_output, str)

    def test_pipeline_doc_vqa(self):
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "samples",
            "sample_invoice.png"
        )
        result = self.pipeline.process(sample_path, generate_visualizations=False)
        
        qa_ans = self.pipeline.ask("What is the total amount?", result)
        self.assertIsNotNone(qa_ans["answer"])
        self.assertGreater(qa_ans["confidence"], 0.5)

if __name__ == "__main__":
    unittest.main()
