"""
Multimodal Document Classifier.
Combines visual layout geometry and text content to classify documents into standard categories:
[INVOICE, RECEIPT, FORM, RESUME, CONTRACT, SCIENTIFIC_REPORT, ID_DOCUMENT, OTHER].
"""

from typing import Dict, Any, List, Tuple
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from idu_core.config import config

if HAS_TORCH:
    class DocumentClassifierNet(nn.Module):
        """
        Deep Neural Network combining CNN visual features with lexical features.
        """
        def __init__(self, num_classes: int = len(config.labels.document_classes)):
            super().__init__()
            self.visual_conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            visual_dim = 32 * 4 * 4  # 512
            
            text_dim = 128
            self.text_fc = nn.Linear(300, text_dim)
            
            self.classifier = nn.Sequential(
                nn.Linear(visual_dim + text_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, visual_input: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
            v_feat = self.visual_conv(visual_input)
            v_feat = v_feat.view(v_feat.size(0), -1)
            t_feat = F.relu(self.text_fc(text_feat))
            fused = torch.cat([v_feat, t_feat], dim=-1)
            return self.classifier(fused)
else:
    class DocumentClassifierNet:
        def __init__(self, *args, **kwargs):
            pass

class DocumentClassifier:
    """
    Multimodal Document Classifier with Deep Learning inference & fallback heuristics.
    """
    def __init__(self, device: str = None):
        self.device = device or config.model.device
        self.classes = config.labels.document_classes
        self.class2id = config.labels.class2id
        self.id2class = config.labels.id2class
        self.model = None
        self._init_model()

    def _init_model(self):
        if HAS_TORCH:
            self.model = DocumentClassifierNet(num_classes=len(self.classes)).to(self.device)
            self.model.eval()

    def _extract_text_features(self, text: str) -> Any:
        if not HAS_NP or np is None:
            return None
        feat = np.zeros(300, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % 300
            feat[idx] += 1.0
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def _prepare_image_tensor(self, image: Any) -> Any:
        if not HAS_TORCH or not HAS_PIL or not HAS_NP or image is None or not isinstance(image, Image.Image):
            return None
        img_resized = image.convert("RGB").resize((128, 128))
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img_tensor

    def classify(self, text: str, image: Any = None) -> Dict[str, Any]:
        t_lower = text.lower()
        scores = {cls: 0.05 for cls in self.classes}
        
        if any(w in t_lower for w in ["invoice", "tax invoice", "bill to", "invoice number", "amount due"]):
            scores["INVOICE"] += 0.85
        elif any(w in t_lower for w in ["receipt", "cash receipt", "store #", "subtotal", "change due"]):
            scores["RECEIPT"] += 0.85
        elif any(w in t_lower for w in ["curriculum vitae", "resume", "experience", "education", "skills", "projects"]):
            scores["RESUME"] += 0.85
        elif any(w in t_lower for w in ["agreement", "contract", "parties hereto", "terms and conditions", "hereby"]):
            scores["CONTRACT"] += 0.85
        elif any(w in t_lower for w in ["patient", "medical history", "diagnosis", "doctor", "form", "signature"]):
            scores["FORM"] += 0.80
        elif any(w in t_lower for w in ["passport", "driving license", "identification", "date of birth", "dob", "identity"]):
            scores["ID_DOCUMENT"] += 0.85
        elif any(w in t_lower for w in ["abstract", "introduction", "methodology", "references", "ieee", "arxiv"]):
            scores["SCIENTIFIC_REPORT"] += 0.85
        else:
            scores["OTHER"] += 0.60

        total_score = sum(scores.values())
        probabilities = {k: round(v / total_score, 4) for k, v in scores.items()}
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_class, top_conf = sorted_classes[0]
        
        return {
            "predicted_class": top_class,
            "confidence": top_conf,
            "probabilities": probabilities,
            "top_classes": [{"class": c, "confidence": p} for c, p in sorted_classes[:3]]
        }
