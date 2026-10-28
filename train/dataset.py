"""
PyTorch Dataset and Collate Functions for Multimodal Document Understanding.
Prepares token IDs, 2D normalized bounding boxes, visual patches, and label IDs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

from idu_core.config import config
from idu_core.ocr.base import WordToken, BoundingBox

class DocumentUnderstandingDataset(Dataset):
    """
    Dataset loader for token-level Key Information Extraction (KIE).
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        max_seq_length: int = 512,
        scale: int = 1000
    ):
        self.samples = samples
        self.max_seq_length = max_seq_length
        self.scale = scale
        self.label2id = config.labels.label2id

    def __len__(self) -> int:
        return len(self.samples)

    def _hash_word(self, word: str) -> int:
        return (abs(hash(word.lower())) % 30000) + 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tokens = sample.get("tokens", [])
        
        # Truncate to max sequence length
        tokens = tokens[:self.max_seq_length]
        
        input_ids = []
        bboxes = []
        labels = []
        
        for tok in tokens:
            word = tok["text"]
            lbl = tok.get("label", "O")
            box = tok["box"]  # [x0, y0, x1, y1]
            
            input_ids.append(self._hash_word(word))
            bboxes.append([
                int(min(max(0, box[0]), self.scale)),
                int(min(max(0, box[1]), self.scale)),
                int(min(max(0, box[2]), self.scale)),
                int(min(max(0, box[3]), self.scale)),
            ])
            labels.append(self.label2id.get(lbl, self.label2id.get("O", 0)))

        seq_len = len(input_ids)
        attention_mask = [1] * seq_len
        
        # Pad sequence up to max_seq_length
        pad_len = self.max_seq_length - seq_len
        if pad_len > 0:
            input_ids.extend([0] * pad_len)
            bboxes.extend([[0, 0, 0, 0]] * pad_len)
            labels.extend([-100] * pad_len)  # Ignore index for loss
            attention_mask.extend([0] * pad_len)

        if HAS_TORCH:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "bbox": torch.tensor(bboxes, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        return {
            "input_ids": np.array(input_ids),
            "bbox": np.array(bboxes),
            "attention_mask": np.array(attention_mask),
            "labels": np.array(labels)
        }
