"""
Private local dataset loaders.

Data never leaves the client. Configure via:
  DATASET_PATH   — file or directory
  DATASET_FORMAT — auto | csv | jsonl | json | folder | huggingface
  DATASET_TEXT_COLUMN / DATASET_LABEL_COLUMN — column names
  HF_DATASET / HF_SPLIT — for format=huggingface (local cache / hub)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class LocalDataset:
    """In-memory private dataset held only on the client."""

    texts: List[str] = field(default_factory=list)
    labels: List[Any] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "empty"
    format: str = "unknown"

    @property
    def num_samples(self) -> int:
        if self.rows:
            return len(self.rows)
        if self.texts:
            return len(self.texts)
        return len(self.labels)

    def as_tensors(
        self,
        kind: str = "tabular",
        channels: int = 1,
        size: int = 8,
        input_dim: int = 10,
    ):
        """Best-effort tensor view for built-in trainers (optional torch)."""
        import torch

        n = max(self.num_samples, 1)
        if kind == "image":
            # Prefer path-backed images when present; else deterministic noise seeded by label/text
            X = torch.zeros(n, channels, size, size)
            for i in range(n):
                seed = hash(str(self.texts[i] if i < len(self.texts) else i)) % (2**31)
                g = torch.Generator().manual_seed(seed)
                X[i] = torch.randn(channels, size, size, generator=g)
            if self.labels:
                y = torch.tensor(
                    [_coerce_int_label(l) for l in self.labels[:n]],
                    dtype=torch.long,
                )
                if len(y) < n:
                    y = torch.cat([y, torch.zeros(n - len(y), dtype=torch.long)])
            else:
                y = torch.randint(0, 2, (n,))
            return X, y

        # Prefer numeric feature columns from rows when available
        feature_matrix = _numeric_feature_matrix(self.rows, exclude={"text", "label", "path", "type"})
        if feature_matrix is not None and feature_matrix:
            width = len(feature_matrix[0])
            X = torch.tensor(feature_matrix, dtype=torch.float32)
            if width < input_dim:
                pad = torch.zeros(X.size(0), input_dim - width)
                X = torch.cat([X, pad], dim=1)
            elif width > input_dim:
                X = X[:, :input_dim]
        elif self.texts:
            # Hash-bag features from private text — data-dependent, stays on device
            X = torch.zeros(n, input_dim)
            for i, text in enumerate(self.texts[:n]):
                for tok in str(text).lower().split():
                    idx = hash(tok) % input_dim
                    X[i, idx] += 1.0
                # normalize
                norm = X[i].norm()
                if norm > 0:
                    X[i] /= norm
        else:
            X = torch.randn(n, input_dim)

        if self.labels:
            vals = []
            for l in self.labels[:n]:
                try:
                    vals.append([float(l)])
                except (TypeError, ValueError):
                    vals.append([float(hash(str(l)) % 100) / 100.0])
            y = torch.tensor(vals, dtype=torch.float32)
            if y.size(0) < n:
                pad = torch.zeros(n - y.size(0), 1)
                y = torch.cat([y, pad], dim=0)
        else:
            y = torch.sum(X, dim=1, keepdim=True)
        return X, y


def _coerce_int_label(label: Any) -> int:
    try:
        return int(label) % 1000
    except (TypeError, ValueError):
        return abs(hash(str(label))) % 2


def _numeric_feature_matrix(
    rows: List[Dict[str, Any]],
    exclude: Optional[set] = None,
) -> Optional[List[List[float]]]:
    """Extract aligned numeric columns from row dicts, or None if none found."""
    if not rows:
        return None
    exclude = exclude or set()
    keys: List[str] = []
    for key in rows[0].keys():
        if key in exclude:
            continue
        try:
            float(rows[0][key])
            keys.append(key)
        except (TypeError, ValueError):
            continue
    if not keys:
        return None
    matrix: List[List[float]] = []
    for row in rows:
        try:
            matrix.append([float(row.get(k, 0.0)) for k in keys])
        except (TypeError, ValueError):
            matrix.append([0.0] * len(keys))
    return matrix


def _detect_format(path: Path) -> str:
    if path.is_dir():
        return "folder"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    return "auto"


def load_csv(path: Path, text_col: str, label_col: Optional[str]) -> LocalDataset:
    texts: List[str] = []
    labels: List[Any] = []
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            if text_col in row:
                texts.append(str(row[text_col]))
            elif row:
                # first string-ish column
                texts.append(str(next(iter(row.values()))))
            if label_col and label_col in row:
                labels.append(row[label_col])
    return LocalDataset(texts=texts, labels=labels, rows=rows, source=str(path), format="csv")


def load_jsonl(path: Path, text_col: str, label_col: Optional[str]) -> LocalDataset:
    texts: List[str] = []
    labels: List[Any] = []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
            if text_col in obj:
                texts.append(str(obj[text_col]))
            elif "text" in obj:
                texts.append(str(obj["text"]))
            elif "content" in obj:
                texts.append(str(obj["content"]))
            if label_col and label_col in obj:
                labels.append(obj[label_col])
            elif "label" in obj:
                labels.append(obj["label"])
    return LocalDataset(texts=texts, labels=labels, rows=rows, source=str(path), format="jsonl")


def load_json(path: Path, text_col: str, label_col: Optional[str]) -> LocalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list):
        raise ValueError(f"JSON dataset must be a list or {{data: [...]}}: {path}")
    # write through jsonl logic via temp structure
    texts, labels, rows = [], [], []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        rows.append(obj)
        if text_col in obj:
            texts.append(str(obj[text_col]))
        elif "text" in obj:
            texts.append(str(obj["text"]))
        if label_col and label_col in obj:
            labels.append(obj[label_col])
        elif "label" in obj:
            labels.append(obj["label"])
    return LocalDataset(texts=texts, labels=labels, rows=rows, source=str(path), format="json")


def load_folder(path: Path) -> LocalDataset:
    """Load text files or class-subfolder image labels from a directory."""
    texts: List[str] = []
    labels: List[Any] = []
    rows: List[Dict[str, Any]] = []
    # class subfolders
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if subdirs:
        for sub in sorted(subdirs):
            for fp in sorted(sub.rglob("*")):
                if fp.suffix.lower() in {".txt", ".md"}:
                    texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
                    labels.append(sub.name)
                    rows.append({"path": str(fp), "label": sub.name})
                elif fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    labels.append(sub.name)
                    rows.append({"path": str(fp), "label": sub.name, "type": "image"})
                    texts.append(str(fp))
    else:
        for fp in sorted(path.rglob("*.txt")):
            texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
            rows.append({"path": str(fp)})
        for fp in sorted(path.rglob("*.jsonl")):
            part = load_jsonl(fp, "text", "label")
            texts.extend(part.texts)
            labels.extend(part.labels)
            rows.extend(part.rows)
    return LocalDataset(texts=texts, labels=labels, rows=rows, source=str(path), format="folder")


def load_huggingface(
    name: str,
    split: str = "train",
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> LocalDataset:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets package required for huggingface format. pip install datasets"
        ) from e
    ds = load_dataset(name, split=split)
    texts, labels, rows = [], [], []
    for row in ds:
        obj = dict(row)
        rows.append(obj)
        if text_col in obj:
            texts.append(str(obj[text_col]))
        if label_col and label_col in obj:
            labels.append(obj[label_col])
    return LocalDataset(
        texts=texts,
        labels=labels,
        rows=rows,
        source=f"hf:{name}:{split}",
        format="huggingface",
    )


def load_local_dataset(
    path: Optional[str] = None,
    fmt: Optional[str] = None,
) -> LocalDataset:
    """
    Load a private local dataset.

    If no path is configured, returns a small synthetic fallback so demos
    keep working, with source='synthetic'.
    """
    path = path or os.getenv("DATASET_PATH", "").strip()
    fmt = (fmt or os.getenv("DATASET_FORMAT", "auto")).strip().lower()
    text_col = os.getenv("DATASET_TEXT_COLUMN", "text")
    label_col = os.getenv("DATASET_LABEL_COLUMN", "label") or None
    if label_col == "":
        label_col = None

    if not path:
        # synthetic fallback
        texts = [
            "Federated learning keeps data on device.",
            "Volunteer compute contributes model updates.",
            "Edge nodes train locally and share deltas.",
        ]
        return LocalDataset(
            texts=texts,
            labels=[0, 1, 0],
            rows=[{"text": t, "label": i % 2} for i, t in enumerate(texts)],
            source="synthetic",
            format="synthetic",
        )

    if fmt == "huggingface" or path.startswith("hf:"):
        name = path[3:] if path.startswith("hf:") else (os.getenv("HF_DATASET") or path)
        split = os.getenv("HF_SPLIT", "train")
        return load_huggingface(name, split=split, text_col=text_col, label_col=label_col)

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH not found: {p}")

    detected = _detect_format(p) if fmt in ("auto", "") else fmt
    if detected == "csv":
        return load_csv(p, text_col, label_col)
    if detected == "jsonl":
        return load_jsonl(p, text_col, label_col)
    if detected == "json":
        return load_json(p, text_col, label_col)
    if detected == "folder":
        return load_folder(p)
    raise ValueError(f"Unsupported dataset format: {detected}")
