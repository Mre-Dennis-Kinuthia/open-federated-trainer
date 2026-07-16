"""
Dataset Loader Module

Loads local datasets for LoRA fine-tuning.
Supports various formats and provides a simple interface.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    max_samples: Optional[int] = None  # Limit number of samples
    shuffle: bool = True
    seed: Optional[int] = None


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for LoRA fine-tuning.
    
    For MVP, generates synthetic data. In production, this would
    load real local datasets.
    """
    
    def __init__(self, texts: List[str], max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def load_local_dataset(
    dataset_path: Optional[str] = None,
    config: Optional[DatasetConfig] = None
) -> Tuple[List[str], int]:
    """
    Load local dataset for training.
    
    For MVP, generates synthetic data. In production, this would:
    1. Load from dataset_path
    2. Parse format (JSON, CSV, text, etc.)
    3. Apply preprocessing
    4. Return tokenized data
    
    Args:
        dataset_path: Path to dataset file (optional, for future use)
        config: Dataset configuration
        
    Returns:
        Tuple of (texts, num_samples)
    """
    if config is None:
        config = DatasetConfig()
    
    # For MVP: Generate synthetic training data
    # In production, load from dataset_path
    synthetic_texts = [
        "This is a sample training text for federated learning.",
        "LoRA adapters enable efficient fine-tuning of large language models.",
        "Federated learning preserves data privacy by keeping data local.",
        "The coordinator aggregates adapter weights from multiple clients.",
        "Each client trains on their own local dataset.",
    ] * 20  # Repeat to get more samples
    
    if config.max_samples:
        synthetic_texts = synthetic_texts[:config.max_samples]
    
    num_samples = len(synthetic_texts)
    
    return synthetic_texts, num_samples

