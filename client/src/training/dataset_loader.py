"""
Dataset Loader Module

Loads local private datasets for LoRA fine-tuning.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

from private_datasets import load_local_dataset as load_private_dataset


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

    if config.max_samples:
        texts = texts[: config.max_samples]

    if not texts:
        raise ValueError(
            "LoRA training requires text samples. Set DATASET_TEXT_COLUMN to "
            "the text field in the configured dataset."
        )

    return texts, len(texts)
