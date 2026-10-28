"""
Document image preprocessing: deskewing, binarization, contrast enhancement,
and geometric normalization.
"""

from typing import Tuple, Union, Any, Optional

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    np = None

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (1000, 1000)):
        self.target_size = target_size
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            self.cv2 = None
            self.has_cv2 = False

    def to_cv2(self, image: Any) -> Any:
        if HAS_PIL and Image and isinstance(image, Image.Image):
            return np.array(image.convert("RGB")) if HAS_NP else image
        return image

    def to_pil(self, img_np: Any) -> Any:
        if not HAS_PIL or Image is None:
            return img_np
        if hasattr(img_np, "shape") and len(img_np.shape) == 2:
            return Image.fromarray(img_np).convert("RGB")
        if hasattr(img_np, "shape"):
            return Image.fromarray(img_np)
        return img_np

    def deskew(self, image: Any) -> Tuple[Any, float]:
        if not self.has_cv2 or self.cv2 is None or not HAS_NP:
            return image, 0.0
            
        cv2 = self.cv2
        img_np = self.to_cv2(image)
        if not hasattr(img_np, "shape"):
            return image, 0.0

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return self.to_pil(img_np), 0.0
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
            
        if abs(angle) < 0.5:
            return self.to_pil(img_np), 0.0
            
        (h, w) = img_np.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_np, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return self.to_pil(rotated), angle

    def enhance_contrast(self, image: Any) -> Any:
        if not self.has_cv2 or self.cv2 is None:
            return image
        cv2 = self.cv2
        img_np = self.to_cv2(image)
        if not hasattr(img_np, "shape"):
            return image
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return self.to_pil(enhanced)

    def binarize(self, image: Any, otsu: bool = True) -> Any:
        if not self.has_cv2 or self.cv2 is None:
            if HAS_PIL and Image and isinstance(image, Image.Image):
                pil_img = image.convert("L")
                thresh = 180
                fn = lambda x: 255 if x > thresh else 0
                return pil_img.point(fn, mode="1").convert("RGB")
            return image
            
        cv2 = self.cv2
        img_np = self.to_cv2(image)
        if not hasattr(img_np, "shape"):
            return image
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        if otsu:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        return self.to_pil(binary)

    def denoise(self, image: Any) -> Any:
        if not self.has_cv2 or self.cv2 is None:
            return image
        cv2 = self.cv2
        img_np = self.to_cv2(image)
        if not hasattr(img_np, "shape"):
            return image
        denoised = cv2.bilateralFilter(img_np, 9, 75, 75)
        return self.to_pil(denoised)

    def prepare_for_model(
        self, image: Any, do_deskew: bool = True, do_enhance: bool = True
    ) -> Any:
        processed = image
        if do_deskew:
            processed, _ = self.deskew(processed)
        if do_enhance:
            processed = self.enhance_contrast(processed)
        return processed
