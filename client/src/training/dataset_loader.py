"""
Dataset Loader Module

Loads local private datasets for LoRA fine-tuning via the shared datasets package.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

from datasets import load_local_dataset as load_private_dataset


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: Optional[int] = None


def load_local_dataset(
    dataset_path: Optional[str] = None,
    config: Optional[DatasetConfig] = None
) -> Tuple[List[str], int]:
    """
    Load local private dataset for LoRA training.

    Uses DATASET_PATH / DATASET_FORMAT env vars when dataset_path is omitted.
    """
    if config is None:
        config = DatasetConfig()

    path = dataset_path or os.getenv("DATASET_PATH") or None
    ds = load_private_dataset(path=path)
    texts = list(ds.texts)
    if not texts and ds.rows:
        texts = [str(r.get("text", r)) for r in ds.rows]

    if config.max_samples:
        texts = texts[: config.max_samples]

    # Ensure non-empty for LoRA demos
    if not texts:
        texts = [
            "Federated LoRA keeps base weights fixed and shares adapters only.",
            "Private local text never leaves the volunteer or edge node.",
        ] * 10

    return texts, len(texts)
