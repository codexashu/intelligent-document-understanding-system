"""
Training and Fine-tuning script for Multimodal Document Transformer (KIE).
Trains token-level BIO classification with cross-entropy loss over 2D spatial layouts.
"""

import os
import argparse
from typing import List, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from idu_core.config import config
from idu_core.models.multimodal_transformer import NativeMultimodalTransformer
from train.dataset import DocumentUnderstandingDataset
from train.synthetic_generator import SyntheticDocumentGenerator

def train_kie_model(epochs: int = 3, batch_size: int = 4, lr: float = 3e-4):
    """
    Trains the Multimodal Transformer on token classification tasks.
    """
    if not HAS_TORCH:
        print("PyTorch is required for deep learning training. Please install torch.")
        return

    device = config.model.device
    print(f"Initiating Multimodal Transformer KIE Training on device: {device}")
    
    num_labels = len(config.labels.entity_labels)
    model = NativeMultimodalTransformer(
        hidden_size=256,
        num_layers=3,
        nhead=4,
        num_classes=num_labels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Generate synthetic training samples
    gen = SyntheticDocumentGenerator()
    training_data = []
    for _ in range(16):
        _, annot = gen.generate_invoice()
        tokens = [{"text": a["text"], "box": a["box"], "label": a["label"]} for a in annot]
        training_data.append({"tokens": tokens})

    dataset = DocumentUnderstandingDataset(training_data, max_seq_length=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
            
            # Reshape for cross entropy: (B * S, num_classes)
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch}/{epochs} - KIE Token Loss: {avg_loss:.4f}")

    # Save model weights checkpoint
    save_path = os.path.join(config.models_dir, "multimodal_transformer_kie.pth")
    torch.save(model.state_dict(), save_path)
    print(f"KIE Model weights saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train_kie_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
