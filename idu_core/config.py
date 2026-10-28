"""
System Configuration for Intelligent Document Understanding.
Provides hardware acceleration detection, label maps, thresholds, and paths.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelConfig:
    device: str = "cpu"
    max_seq_length: int = 512
    image_size: int = 224
    coordinate_scale: int = 1000
    confidence_threshold: float = 0.5
    layoutlm_model_name: str = "microsoft/layoutlmv3-base"
    use_huggingface: bool = False  # Set to True when HF pretrained weights are downloaded

@dataclass
class LabelConfig:
    document_classes: List[str] = field(default_factory=lambda: [
        "INVOICE",
        "RECEIPT",
        "FORM",
        "RESUME",
        "CONTRACT",
        "SCIENTIFIC_REPORT",
        "ID_DOCUMENT",
        "OTHER"
    ])
    
    entity_labels: List[str] = field(default_factory=lambda: [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
        "B-TOTAL_AMOUNT",
        "I-TOTAL_AMOUNT",
        "B-DATE",
        "I-DATE",
        "B-INVOICE_NUMBER",
        "I-INVOICE_NUMBER",
        "B-VENDOR_NAME",
        "I-VENDOR_NAME",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-LINE_ITEM",
        "I-LINE_ITEM"
    ])
    
    @property
    def id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.entity_labels)}
        
    @property
    def label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.entity_labels)}
        
    @property
    def class2id(self) -> Dict[str, int]:
        return {cls: i for i, cls in enumerate(self.document_classes)}
        
    @property
    def id2class(self) -> Dict[int, str]:
        return {i: cls for i, cls in enumerate(self.document_classes)}

@dataclass
class SystemConfig:
    app_name: str = "Intelligent Document Understanding System"
    version: str = "1.0.0"
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir: str = os.path.join(base_dir, "samples")
    models_dir: str = os.path.join(base_dir, "weights")
    output_dir: str = os.path.join(base_dir, "output")
    model: ModelConfig = field(default_factory=ModelConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    
    def __post_init__(self):
        # Auto-detect optimal compute device (CUDA -> MPS on Mac -> CPU)
        try:
            import torch
            if torch.cuda.is_available():
                self.model.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.model.device = "mps"
            else:
                self.model.device = "cpu"
        except ImportError:
            self.model.device = "cpu"
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

# Global configuration singleton
config = SystemConfig()
