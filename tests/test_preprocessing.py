import unittest
from idu_core.preprocessing.image_processor import ImagePreprocessor
from idu_core.preprocessing.pdf_converter import PDFConverter

class TestPreprocessing(unittest.TestCase):
    def test_image_preprocessor_interface(self):
        processor = ImagePreprocessor()
        # Mock image object
        class MockImage:
            size = (800, 1000)
            def convert(self, mode):
                return self

        mock = MockImage()
        deskewed, angle = processor.deskew(mock)
        self.assertEqual(angle, 0.0)
        self.assertIsNotNone(deskewed)

        enhanced = processor.enhance_contrast(mock)
        self.assertIsNotNone(enhanced)

        binary = processor.binarize(mock)
        self.assertIsNotNone(binary)

    def test_pdf_converter_fallback(self):
        converter = PDFConverter()
        images = converter.convert_pdf_to_images(b"%PDF-1.4 mock pdf data")
        self.assertIsInstance(images, list)
        self.assertGreater(len(images), 0)

if __name__ == "__main__":
    unittest.main()
