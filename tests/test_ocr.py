import unittest
from idu_core.ocr.base import BoundingBox, WordToken, DocumentPage
from idu_core.ocr.synthetic_ocr import SyntheticOCREngine

class TestOCRPrimitives(unittest.TestCase):
    def test_bounding_box_geometry(self):
        box = BoundingBox(x0=100, y0=50, x1=200, y1=150)
        self.assertEqual(box.width, 100)
        self.assertEqual(box.height, 100)
        self.assertEqual(box.area, 10000)
        self.assertEqual(box.center, (150.0, 100.0))

    def test_bounding_box_normalization(self):
        box = BoundingBox(x0=200, y0=100, x1=400, y1=300)
        # Image size 800 x 1000
        norm_box = box.normalize(img_width=800, img_height=1000, scale=1000)
        self.assertEqual(norm_box.x0, 250)
        self.assertEqual(norm_box.y0, 100)
        self.assertEqual(norm_box.x1, 500)
        self.assertEqual(norm_box.y1, 300)

        # Denormalize back
        denorm = norm_box.denormalize(img_width=800, img_height=1000, scale=1000)
        self.assertEqual(denorm.x0, 200)
        self.assertEqual(denorm.y0, 100)
        self.assertEqual(denorm.x1, 400)
        self.assertEqual(denorm.y1, 300)

    def test_bounding_box_iou(self):
        box_a = BoundingBox(0, 0, 10, 10)
        box_b = BoundingBox(5, 0, 15, 10)
        # Intersection is (5,0) to (10,10) = 5*10 = 50. Union is 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
        self.assertAlmostEqual(box_a.iou(box_b), 1.0 / 3.0, places=4)

        # Disjoint boxes
        box_c = BoundingBox(20, 20, 30, 30)
        self.assertEqual(box_a.iou(box_c), 0.0)

    def test_synthetic_ocr_engine(self):
        engine = SyntheticOCREngine()
        # Mock image with width 800, height 1000
        class MockImage:
            size = (800, 1000)
            
        page = engine.extract_page(MockImage(), page_num=1)
        self.assertEqual(page.page_num, 1)
        self.assertEqual(page.width, 800)
        self.assertEqual(page.height, 1000)
        self.assertGreater(len(page.tokens), 5)
        self.assertTrue(any("INV" in t.text or "INVOICE" in t.text for t in page.tokens))

if __name__ == "__main__":
    unittest.main()
