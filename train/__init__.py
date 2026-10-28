"""
Training and synthetic data generation tools.
"""

from train.synthetic_generator import SyntheticDocumentGenerator, generate_sample_files
from train.dataset import DocumentUnderstandingDataset

__all__ = ["SyntheticDocumentGenerator", "generate_sample_files", "DocumentUnderstandingDataset"]
