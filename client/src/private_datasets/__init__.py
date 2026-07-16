"""Private, client-local dataset loading and tensor conversion.

No records are uploaded by this module. Synthetic fallback is opt-in through
``ALLOW_SYNTHETIC_DATA=true`` and is disabled by default.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


class DatasetConfigurationError(ValueError):
    """Raised when a real local dataset is missing or unusable."""


@dataclass
class LocalDataset:
    texts: List[str] = field(default_factory=list)
    labels: List[Any] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "empty"
    format: str = "unknown"

    @property
    def num_samples(self) -> int:
        return max(len(self.rows), len(self.texts), len(self.labels))

    def validate(self, *, require_labels: bool = False) -> None:
        if self.num_samples == 0:
            raise DatasetConfigurationError(f"Dataset is empty: {self.source}")
        if require_labels and len(self.labels) != self.num_samples:
            raise DatasetConfigurationError(
                f"Supervised training requires one label per sample "
                f"({len(self.labels)}/{self.num_samples})"
            )

    def as_tensors(
        self,
        kind: str = "tabular",
        channels: int = 1,
        size: int = 8,
        input_dim: int = 10,
    ):
        """Create tensors from actual local rows, text, or image files."""
        import torch

        self.validate(require_labels=True)
        if kind == "image":
            return self._image_tensors(channels=channels, size=size)

        matrix = _numeric_feature_matrix(
            self.rows,
            exclude={
                os.getenv("DATASET_TEXT_COLUMN", "text"),
                os.getenv("DATASET_LABEL_COLUMN", "label"),
                "path",
                "type",
            },
        )
        if matrix:
            x = torch.tensor(matrix, dtype=torch.float32)
            if x.size(1) < input_dim:
                x = torch.cat(
                    [x, torch.zeros(x.size(0), input_dim - x.size(1))],
                    dim=1,
                )
            elif x.size(1) > input_dim:
                x = x[:, :input_dim]
        elif self.texts:
            x = _hashed_text_features(self.texts, input_dim)
        else:
            raise DatasetConfigurationError(
                "No numeric columns or text values are available for tabular training"
            )

        y_values: List[List[float]] = []
        for value in self.labels:
            try:
                y_values.append([float(value)])
            except (TypeError, ValueError) as exc:
                raise DatasetConfigurationError(
                    f"MLP regression labels must be numeric; got {value!r}"
                ) from exc
        return x, torch.tensor(y_values, dtype=torch.float32)

    def _image_tensors(self, *, channels: int, size: int):
        import numpy as np
        import torch

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Image datasets require Pillow: pip install Pillow") from exc

        image_rows = [
            row for row in self.rows
            if row.get("type") == "image" and row.get("path")
        ]
        if len(image_rows) != self.num_samples:
            raise DatasetConfigurationError(
                "CNN training requires an image path for every dataset sample"
            )

        mode = "L" if channels == 1 else "RGB"
        images = []
        for row in image_rows:
            with Image.open(row["path"]) as image:
                image = image.convert(mode).resize((size, size))
                array = np.asarray(image, dtype=np.float32) / 255.0
            if channels == 1:
                array = array[None, :, :]
            else:
                array = array.transpose(2, 0, 1)
            images.append(array)
        labels = [_coerce_class_label(value) for value in self.labels]
        return (
            torch.tensor(np.stack(images), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


def _stable_bucket(value: str, width: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % width


def _hashed_text_features(texts: Sequence[str], width: int):
    """Deterministic hashing-vectorizer features without exporting text."""
    import torch

    if width < 1:
        raise DatasetConfigurationError("input_dim must be positive")
    features = torch.zeros(len(texts), width, dtype=torch.float32)
    for row_index, text in enumerate(texts):
        for token in str(text).lower().split():
            features[row_index, _stable_bucket(token, width)] += 1.0
        norm = features[row_index].norm()
        if norm > 0:
            features[row_index] /= norm
    return features


def _coerce_class_label(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigurationError(
            f"Image class labels must be integer encoded; got {value!r}"
        ) from exc


def _numeric_feature_matrix(
    rows: List[Dict[str, Any]],
    exclude: Optional[Set[str]] = None,
) -> Optional[List[List[float]]]:
    if not rows:
        return None
    excluded = exclude or set()
    keys: List[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        try:
            float(value)
            keys.append(key)
        except (TypeError, ValueError):
            continue
    if not keys:
        return None
    matrix: List[List[float]] = []
    for index, row in enumerate(rows):
        try:
            matrix.append([float(row[key]) for key in keys])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetConfigurationError(
                f"Row {index} has missing or non-numeric feature values"
            ) from exc
    return matrix


def _extract(
    rows: List[Dict[str, Any]],
    *,
    text_column: str,
    label_column: Optional[str],
    source: str,
    fmt: str,
) -> LocalDataset:
    texts: List[str] = []
    labels: List[Any] = []
    for row in rows:
        if text_column in row and row[text_column] is not None:
            texts.append(str(row[text_column]))
        if label_column and label_column in row and row[label_column] is not None:
            labels.append(row[label_column])
    return LocalDataset(
        texts=texts,
        labels=labels,
        rows=rows,
        source=source,
        format=fmt,
    )


def load_csv(path: Path, text_column: str, label_column: Optional[str]) -> LocalDataset:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return _extract(
        rows,
        text_column=text_column,
        label_column=label_column,
        source=str(path),
        fmt="csv",
    )


def load_jsonl(path: Path, text_column: str, label_column: Optional[str]) -> LocalDataset:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DatasetConfigurationError(
                    f"JSONL row {line_number} must be an object"
                )
            rows.append(value)
    return _extract(
        rows,
        text_column=text_column,
        label_column=label_column,
        source=str(path),
        fmt="jsonl",
    )


def load_json(path: Path, text_column: str, label_column: Optional[str]) -> LocalDataset:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DatasetConfigurationError("JSON dataset must be a list of objects")
    return _extract(
        value,
        text_column=text_column,
        label_column=label_column,
        source=str(path),
        fmt="json",
    )


def load_folder(path: Path) -> LocalDataset:
    rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    labels: List[Any] = []
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    class_dirs = sorted(item for item in path.iterdir() if item.is_dir())

    if class_dirs:
        class_index = {directory.name: index for index, directory in enumerate(class_dirs)}
        for directory in class_dirs:
            for file_path in sorted(directory.rglob("*")):
                if file_path.suffix.lower() in image_suffixes:
                    rows.append(
                        {
                            "path": str(file_path),
                            "label": class_index[directory.name],
                            "class_name": directory.name,
                            "type": "image",
                        }
                    )
                    labels.append(class_index[directory.name])
                elif file_path.suffix.lower() in {".txt", ".md"}:
                    text = file_path.read_text(encoding="utf-8", errors="strict")
                    rows.append(
                        {
                            "path": str(file_path),
                            "text": text,
                            "label": class_index[directory.name],
                            "class_name": directory.name,
                            "type": "text",
                        }
                    )
                    texts.append(text)
                    labels.append(class_index[directory.name])
    else:
        for file_path in sorted(path.rglob("*")):
            if file_path.suffix.lower() in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8", errors="strict")
                rows.append({"path": str(file_path), "text": text, "type": "text"})
                texts.append(text)
            elif file_path.suffix.lower() == ".jsonl":
                part = load_jsonl(file_path, "text", "label")
                rows.extend(part.rows)
                texts.extend(part.texts)
                labels.extend(part.labels)
    return LocalDataset(
        texts=texts,
        labels=labels,
        rows=rows,
        source=str(path),
        format="folder",
    )


def load_huggingface(
    name: str,
    *,
    split: str,
    text_column: str,
    label_column: Optional[str],
) -> LocalDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets require: pip install datasets"
        ) from exc
    dataset = load_dataset(name, split=split)
    rows = [dict(row) for row in dataset]
    return _extract(
        rows,
        text_column=text_column,
        label_column=label_column,
        source=f"hf:{name}:{split}",
        fmt="huggingface",
    )


def _synthetic_dataset() -> LocalDataset:
    texts = [
        "Federated learning keeps data on device.",
        "Volunteer compute contributes model updates.",
        "Edge nodes train locally and share deltas.",
    ]
    return LocalDataset(
        texts=texts,
        labels=[0, 1, 0],
        rows=[{"text": text, "label": index % 2} for index, text in enumerate(texts)],
        source="synthetic",
        format="synthetic",
    )


def load_local_dataset(
    path: Optional[str] = None,
    fmt: Optional[str] = None,
) -> LocalDataset:
    """Load and validate a private dataset configured on this client."""
    configured_path = (path or os.getenv("DATASET_PATH", "")).strip()
    data_format = (fmt or os.getenv("DATASET_FORMAT", "auto")).strip().lower()
    text_column = os.getenv("DATASET_TEXT_COLUMN", "text")
    label_column = os.getenv("DATASET_LABEL_COLUMN", "label") or None

    if not configured_path:
        if os.getenv("ALLOW_SYNTHETIC_DATA", "false").lower() in {"1", "true", "yes"}:
            return _synthetic_dataset()
        raise DatasetConfigurationError(
            "DATASET_PATH is required. To run the explicit demo only, set "
            "ALLOW_SYNTHETIC_DATA=true."
        )

    if data_format == "huggingface" or configured_path.startswith("hf:"):
        name = configured_path[3:] if configured_path.startswith("hf:") else configured_path
        result = load_huggingface(
            name,
            split=os.getenv("HF_SPLIT", "train"),
            text_column=text_column,
            label_column=label_column,
        )
        result.validate()
        return result

    local_path = Path(configured_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"DATASET_PATH not found: {local_path}")
    if data_format in {"", "auto"}:
        if local_path.is_dir():
            data_format = "folder"
        else:
            data_format = {
                ".csv": "csv",
                ".jsonl": "jsonl",
                ".json": "json",
            }.get(local_path.suffix.lower(), "")

    loaders = {
        "csv": lambda: load_csv(local_path, text_column, label_column),
        "jsonl": lambda: load_jsonl(local_path, text_column, label_column),
        "json": lambda: load_json(local_path, text_column, label_column),
        "folder": lambda: load_folder(local_path),
    }
    if data_format not in loaders:
        raise DatasetConfigurationError(f"Unsupported dataset format: {data_format!r}")
    result = loaders[data_format]()
    result.validate()
    return result
