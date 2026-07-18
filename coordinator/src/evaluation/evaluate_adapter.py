"""Dataset-backed evaluation for an aggregated LoRA adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("evaluation")


@dataclass
class AdapterEvaluationResult:
    round_id: int
    adapter_version: str
    evaluation_loss: Optional[float]
    num_eval_samples: int
    passed: bool
    evaluated: bool
    previous_loss: Optional[float] = None
    improvement: Optional[float] = None
    reason: Optional[str] = None
    task_type: str = "causal_lm"
    metrics: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = {}


def _load_eval_texts(path: Path, text_column: str) -> List[str]:
    texts, _ = _load_eval_examples(path, text_column, require_labels=False)
    return texts


def _load_eval_examples(
    path: Path,
    text_column: str,
    *,
    label_column: str = "label",
    require_labels: bool = False,
) -> tuple[List[str], List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"LoRA evaluation dataset not found: {path}")
    labels: List[int] = []
    if path.is_dir():
        texts = [
            item.read_text(encoding="utf-8")
            for item in sorted(path.rglob("*.txt"))
        ]
    elif path.suffix.lower() == ".jsonl":
        texts = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or text_column not in row:
                    raise ValueError(
                        f"JSONL evaluation row {line_number} lacks {text_column!r}"
                    )
                texts.append(str(row[text_column]))
                if label_column in row:
                    labels.append(int(row[label_column]))
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        if not isinstance(value, list):
            raise ValueError("Evaluation JSON must be a list or {'data': [...]}")
        texts = []
        for row in value:
            if not isinstance(row, dict) or text_column not in row:
                continue
            texts.append(str(row[text_column]))
            if label_column in row:
                labels.append(int(row[label_column]))
    else:
        texts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    max_samples = max(1, int(os.getenv("LORA_EVAL_MAX_SAMPLES", "64")))
    texts = texts[:max_samples]
    labels = labels[:max_samples]
    if not texts:
        raise ValueError("LoRA evaluation dataset contains no text samples")
    if require_labels:
        if len(labels) != len(texts):
            raise ValueError(
                f"seq_cls evaluation requires '{label_column}' on every row"
            )
    return texts, labels


def _resolve_task_type(task_type: Optional[str]) -> str:
    key = (task_type or os.getenv("LORA_TASK_TYPE", "causal_lm") or "causal_lm").strip().lower()
    aliases = {
        "causal_lm": "causal_lm",
        "causal-lm": "causal_lm",
        "lm": "causal_lm",
        "seq_cls": "seq_cls",
        "sequence_classification": "seq_cls",
        "classification": "seq_cls",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported LoRA task_type={task_type!r}; use causal_lm or seq_cls"
        )
    return aliases[key]


def evaluate_adapter(
    round_id: int,
    adapter_version: str,
    aggregated_adapter: Dict[str, Any],
    *,
    base_model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    max_seq_length: int,
    previous_adapter_loss: Optional[float] = None,
    dataset_path: Optional[str] = None,
    task_type: Optional[str] = None,
) -> AdapterEvaluationResult:
    """Evaluate an aggregated adapter on a holdout dataset.

    Task-aware:
    - ``causal_lm``: causal language-model loss (default)
    - ``seq_cls``: requires labeled JSON/JSONL with ``label``; reports loss + accuracy

    If no evaluation dataset is configured, evaluation is explicitly skipped;
    no parameter-norm proxy or fabricated loss is returned.
    """
    resolved_task = _resolve_task_type(task_type)
    configured_path = (
        dataset_path or os.getenv("LORA_EVAL_DATASET_PATH", "")
    ).strip()
    if not configured_path:
        reason = "LORA_EVAL_DATASET_PATH is not configured"
        if os.getenv("LORA_REQUIRE_EVALUATION", "false").lower() in {
            "1",
            "true",
            "yes",
        }:
            raise ValueError(reason)
        return AdapterEvaluationResult(
            round_id=round_id,
            adapter_version=adapter_version,
            evaluation_loss=None,
            num_eval_samples=0,
            passed=False,
            evaluated=False,
            previous_loss=previous_adapter_loss,
            reason=reason,
            task_type=resolved_task,
        )
    if not aggregated_adapter:
        raise ValueError("Cannot evaluate an empty adapter")

    try:
        import torch
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
        )
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Real LoRA evaluation requires torch, transformers, and peft"
        ) from exc

    from aggregation.merge import apply_adapter_on_peft_copy

    texts, labels = _load_eval_examples(
        Path(configured_path).expanduser().resolve(),
        os.getenv("LORA_EVAL_TEXT_COLUMN", "text"),
        label_column=os.getenv("LORA_EVAL_LABEL_COLUMN", "label"),
        require_labels=(resolved_task == "seq_cls"),
    )
    device_name = os.getenv(
        "LORA_EVAL_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(
        f"Evaluating adapter {adapter_version} on {len(texts)} samples "
        f"(task={resolved_task})",
        extra={
            "component": "evaluation",
            "event": "evaluation_started",
            "round_id": round_id,
            "base_model": base_model_name,
            "task_type": resolved_task,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if resolved_task == "seq_cls":
        num_labels = len({int(x) for x in labels}) if labels else 2
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            trust_remote_code=False,
            torch_dtype=dtype,
            num_labels=max(2, num_labels),
        )
        peft_task = TaskType.SEQ_CLS
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
        peft_task = TaskType.CAUSAL_LM

    # Isolated PEFT wrapper on this freshly loaded model only.
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=peft_task,
        ),
    )
    apply_adapter_on_peft_copy(peft_model, aggregated_adapter)
    peft_model.to(device)
    peft_model.eval()

    batch_size = max(1, int(os.getenv("LORA_EVAL_BATCH_SIZE", "2")))
    weighted_loss = 0.0
    evaluated = 0
    correct = 0
    with torch.no_grad():
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            if resolved_task == "seq_cls":
                batch_labels = torch.tensor(
                    [int(x) for x in labels[offset : offset + batch_size]],
                    device=device,
                )
                output = peft_model(**tokens, labels=batch_labels)
                preds = output.logits.argmax(dim=-1)
                correct += int((preds == batch_labels).sum().item())
            else:
                batch_labels = tokens["input_ids"].clone()
                batch_labels[tokens["attention_mask"] == 0] = -100
                output = peft_model(**tokens, labels=batch_labels)
            weighted_loss += float(output.loss.item()) * len(batch)
            evaluated += len(batch)

    loss = weighted_loss / evaluated
    metrics: Dict[str, Any] = {"loss": loss}
    if resolved_task == "seq_cls":
        metrics["accuracy"] = correct / evaluated if evaluated else 0.0
    improvement = (
        loss - previous_adapter_loss
        if previous_adapter_loss is not None
        else None
    )
    tolerance = float(os.getenv("LORA_EVAL_REGRESSION_TOLERANCE", "0.1"))
    passed = improvement is None or improvement <= tolerance
    return AdapterEvaluationResult(
        round_id=round_id,
        adapter_version=adapter_version,
        evaluation_loss=loss,
        num_eval_samples=evaluated,
        passed=passed,
        evaluated=True,
        previous_loss=previous_adapter_loss,
        improvement=improvement,
        task_type=resolved_task,
        metrics=metrics,
    )
