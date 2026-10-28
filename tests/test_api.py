import unittest
import os
import asyncio

try:
    from fastapi.testclient import TestClient
    from api.app import create_app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        if not HAS_FASTAPI:
            self.skipTest("fastapi or testclient not installed")
        self.app = create_app()
        self.client = TestClient(self.app)
        self.sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "samples",
            "sample_invoice.png"
        )

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)

    def test_extract_endpoint(self):
        with open(self.sample_path, "rb") as f:
            response = self.client.post(
                "/api/v1/extract?ocr_engine=synthetic",
                files={"file": ("sample_invoice.png", f, "image/png")}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("extracted_entities", data)

    def test_qa_endpoint(self):
        with open(self.sample_path, "rb") as f:
            response = self.client.post(
                "/api/v1/qa",
                files={"file": ("sample_invoice.png", f, "image/png")},
                data={"question": "What is the total amount?"}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("answer", data)

if __name__ == "__main__":
    unittest.main()
